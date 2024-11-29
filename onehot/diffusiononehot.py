import argparse
import torch
import numpy as np
import einops
import os
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
#from LACE


def sample_t(size=(1,), t_max=None,device="cuda:0"):
    """Samples batches of time steps to use."""
    t = torch.randint(low=0, high=t_max, size=size, device=device)
    return t.to(device)

def rand_fix(batch_size, mask, ratio=0.2, n_elements=25, stochastic=True):#随机mask

    if stochastic:
        indices = (torch.rand([batch_size, n_elements]) <= torch.rand([1]).item() * ratio).to(mask.device) * mask.to(torch.bool)
    else:
        a = torch.tensor([False, False, True, False, False, True, True, False, False, False,
                          False, False, False, False, False, False, False, False, False, False,
                          False, False, False, False, False])
        indices = einops.repeat(a, "l -> n l", n=batch_size) * mask.to(torch.bool)

    return indices

def finalize(layout, num_class):
    layout[:, :, num_class:] = torch.clamp(layout[:, :, num_class:], min=0, max=1)# / 2 + 0.5 没有对bbox做中心变换就不需要这段
    bbox = layout[:, :, num_class:]
    label = torch.argmax(layout[:, :, :num_class], dim=2)
    mask = (label != num_class-1).clone().detach()
    return bbox, label, mask

def task_union(l_0_batch, l_t_noise, num_class, real_mask,fix_mask = None):#forward函数其实是将uncondition,c,cwh,complete四种任务的batch在batchsize的维度拼接起来，一次性传入model这样可以一次就训练到四种任务。
                                                            #相应的timestep和噪声e也要复制4份
    batch_size = l_0_batch.shape[0]

    # cond c
    l_t_input_c = l_0_batch.clone()
    l_t_input_c[:, :, num_class:] = l_t_noise[:, :, num_class:]  

    # cond cwh
    l_t_input_cwh = l_0_batch.clone()
    l_t_input_cwh[:, :, num_class:num_class+2] = l_t_noise[:, :, num_class:num_class+2]

    # cond complete,label换成离散的mask
    if fix_mask == None:
        fix_mask = rand_fix(batch_size, real_mask, ratio=0.2)
    l_t_input_complete = l_t_noise.clone()
    l_t_input_complete[fix_mask] = l_0_batch[fix_mask]

    l_t_input_all = torch.cat([l_t_noise, l_t_input_c, l_t_input_cwh, l_t_input_complete], dim=0)

    return l_t_input_all, fix_mask

def task_union_reverse(l_0_batch, l_t_pred, num_class, fix_mask):#
                                                            
    batch_size = l_0_batch.shape[0]
    #uncond
    l_t_input_uncond = l_t_pred[:batch_size,:,:]

    # cond c
    l_t_input_c = l_0_batch.clone()
    l_t_input_c[:, :, num_class:] = l_t_pred[batch_size:2*batch_size, :, num_class:]

    # cond cwh
    l_t_input_cwh = l_0_batch.clone()
    l_t_input_cwh[:, :, num_class:num_class+2] = l_t_pred[2*batch_size:3*batch_size, :, num_class:num_class+2]

    # cond complete
    l_t_input_complete = l_t_pred[3*batch_size:4*batch_size,:,:].clone()
    l_t_input_complete[fix_mask] = l_0_batch[fix_mask]

    l_t_input_all = torch.cat([l_t_input_uncond, l_t_input_c, l_t_input_cwh, l_t_input_complete], dim=0)

    return l_t_input_all


def task_union_diffusion(l_0_batch, l_t_noise, num_class, real_mask,t,max_t,fix_mask = None):#forward函数其实是将uncondition,c,cwh,complete四种任务的batch在batchsize的维度拼接起来，一次性传入model这样可以一次就训练到四种任务。
                                                            #相应的timestep和噪声e也要复制4份
    batch_size = l_0_batch.shape[0]
    t = t.clone().view(-1,1,1)
    # cond c
    l_t_input_c = l_0_batch.clone()
    l_t_input_c[:, :, num_class:] = l_t_noise[:, :, num_class:]
    l_t_input_c[:, :, :num_class] =  (1-t/max_t)*l_t_input_c[:, :, :num_class] + t/max_t*l_t_noise[:,:,:num_class]

    # cond cwh
    l_t_input_cwh = l_0_batch.clone()
    l_t_input_cwh[:, :, num_class:num_class+2] = l_t_noise[:, :, num_class:num_class+2]
    l_t_input_cwh[:, :, :num_class] = (1-t/max_t)*l_t_input_cwh[: ,: , :num_class] + t/max_t*l_t_noise[:, :, :num_class] 
    l_t_input_cwh[:, :, num_class+2:] = (1-t/max_t)*l_t_input_cwh[: ,: ,num_class+2:] + t/max_t*l_t_noise[:, :,num_class+2:] 

    # cond complete,label换成离散的mask
    if fix_mask == None:
        fix_mask = rand_fix(batch_size, real_mask, ratio=0.2)    
    l_t_input_complete = l_t_noise.clone()
    l_t_input_complete[fix_mask] = ((1-t/max_t)*l_0_batch)[fix_mask] + (t/max_t*l_t_input_complete)[fix_mask]
    #注意括号，先乘再取元素这样广播就不会出错

    l_t_input_all = torch.cat([l_t_noise, l_t_input_c, l_t_input_cwh, l_t_input_complete], dim=0)

    return l_t_input_all, fix_mask

def task_union_reverse_diffusion(l_0_batch, l_t_pred, num_class, t, max_t, fix_mask):#
                                                            
    batch_size = l_0_batch.shape[0]
    t = t.clone().view(-1,1,1)
    #uncond
    l_t_input_uncond = l_t_pred[:batch_size,:,:]

    # cond c
    l_t_input_c = l_0_batch.clone()
    l_t_input_c[:, :, num_class:] = l_t_pred[batch_size:2*batch_size, :, num_class:]
    l_t_input_c[:, :, :num_class] = (1-t/max_t)*l_t_input_c[:, :, :num_class]  + t/max_t * l_t_pred[batch_size:2*batch_size, :, :num_class]

    # cond cwh
    l_t_input_cwh = l_0_batch.clone()
    l_t_input_cwh[:, :, num_class:num_class+2] = l_t_pred[2*batch_size:3*batch_size, :, num_class:num_class+2]
    l_t_input_cwh[:, :, :num_class] = (1-t/max_t)*l_t_input_cwh[:, :, :num_class] + t/max_t * l_t_pred[2*batch_size:3*batch_size, :, :num_class]
    l_t_input_cwh[:, :, num_class+2:] = (1-t/max_t)*l_t_input_cwh[:, :, num_class+2:] + t/max_t * l_t_pred[2*batch_size:3*batch_size, :, num_class+2:]

    # cond complete
    l_t_input_complete = l_t_pred[3*batch_size:4*batch_size,:,:].clone()
    l_t_input_complete[fix_mask] = ((1-t/max_t)*l_0_batch)[fix_mask] + (t/max_t*l_t_input_complete)[fix_mask]

    l_t_input_all = torch.cat([l_t_input_uncond, l_t_input_c, l_t_input_cwh, l_t_input_complete], dim=0)

    return l_t_input_all

#from DDGAN
#%% Diffusion coefficients 

def var_func_vp(t, beta_min, beta_max):#计算的是论文中的σ^{2}(t')
    log_mean_coeff = -0.25 * t ** 2 * (beta_max - beta_min) - 0.5 * t * beta_min
    var = 1. - torch.exp(2. * log_mean_coeff)
    return var

def var_func_geometric(t, beta_min, beta_max):
    return beta_min * ((beta_max / beta_min) ** t)

def extract(input, t, shape):
    out = torch.gather(input, 0, t)
    reshape = [shape[0]] + [1] * (len(shape) - 1)
    out = out.reshape(*reshape)

    return out

def get_time_schedule(args, device):
    n_timestep = args.num_timesteps
    eps_small = 1e-3
    t = np.arange(0, n_timestep + 1, dtype=np.float64)
    t = t / n_timestep#把时间归一化到0，1之间
    t = torch.from_numpy(t) * (1. - eps_small)  + eps_small
    return t.to(device)

def get_sigma_schedule(args, device):
    n_timestep = args.num_timesteps
    beta_min = args.beta_min
    beta_max = args.beta_max
    eps_small = 1e-3
   
    t = np.arange(0, n_timestep + 1, dtype=np.float64)
    t = t / n_timestep
    t = torch.from_numpy(t) * (1. - eps_small) + eps_small
    
    if args.use_geometric:
        var = var_func_geometric(t, beta_min, beta_max)
    else:
        var = var_func_vp(t, beta_min, beta_max)
    alpha_bars = 1.0 - var
    betas = 1 - alpha_bars[1:] / alpha_bars[:-1]
    
    first = torch.tensor(1e-8)
    betas = torch.cat((first[None], betas)).to(device)
    betas = betas.type(torch.float32)
    sigmas = betas**0.5
    a_s = torch.sqrt(1-betas)
    return sigmas, a_s, betas

class Diffusion_Coefficients():
    def __init__(self, args, device):
                
        self.sigmas, self.a_s, _ = get_sigma_schedule(args, device=device)
        self.a_s_cum = np.cumprod(self.a_s.cpu())
        self.sigmas_cum = np.sqrt(1 - self.a_s_cum ** 2)#1-α^{_}
        self.a_s_prev = self.a_s.clone()
        self.a_s_prev[-1] = 1
        
        self.a_s_cum = self.a_s_cum.to(device)
        self.sigmas_cum = self.sigmas_cum.to(device)
        self.a_s_prev = self.a_s_prev.to(device)
    
def q_sample(coeff, x_start, t, *, noise=None):#p(xt|x0)
    """
    Diffuse the data (t == 0 means diffused for t step)
    """
    if noise is None:
      noise = torch.randn_like(x_start)
      
    x_t = extract(coeff.a_s_cum, t, x_start.shape) * x_start + \
          extract(coeff.sigmas_cum, t, x_start.shape) * noise
    
    return x_t

def q_sample_pairs(coeff,x_start, t):#构造真实的discriminator的输入,analogbits情形下num_class应该等于bin(num_class)-2
    """
    Generate a pair of disturbed images for training
    :param x_start: x_0
    :param t: time step t
    :return: x_t, x_{t+1}
    """
    noise = torch.randn_like(x_start)
    x_t = q_sample(coeff, x_start, t)
    x_t_plus_one = extract(coeff.a_s, t+1, x_start.shape) * x_t + \
                   extract(coeff.sigmas, t+1, x_start.shape) * noise
    
    return x_t, x_t_plus_one
#%% posterior sampling
class Posterior_Coefficients():#计算q(xt-1|xt,x0)的均值和方差系数
    def __init__(self, args, device):
        
        _, _, self.betas = get_sigma_schedule(args, device=device)
        
        #we don't need the zeros
        self.betas = self.betas.type(torch.float32)[1:]
        
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, 0)
        self.alphas_cumprod_prev = torch.cat(
                                    (torch.tensor([1.], dtype=torch.float32,device=device), self.alphas_cumprod[:-1]), 0
                                        )               
        self.posterior_variance = self.betas * (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod)#β^{~}_{t},q(xt-1|xt,x0)的方差
        
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.rsqrt(self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1 / self.alphas_cumprod - 1)
        
        self.posterior_mean_coef1 = (self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1 - self.alphas_cumprod))#x0对应的系数
        self.posterior_mean_coef2 = ((1 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1 - self.alphas_cumprod))#xt对应的系数
        
        self.posterior_log_variance_clipped = torch.log(self.posterior_variance.clamp(min=1e-20))
        
def sample_posterior(coefficients, x_0,x_t, t):#根据q(xt-1|xt,x0)进行x_{t-1}的采样
    
    def q_posterior(x_0, x_t, t):
        mean = (
            extract(coefficients.posterior_mean_coef1, t, x_t.shape) * x_0
            + extract(coefficients.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        var = extract(coefficients.posterior_variance, t, x_t.shape)
        log_var_clipped = extract(coefficients.posterior_log_variance_clipped, t, x_t.shape)
        return mean, var, log_var_clipped
    
  
    def p_sample(x_0, x_t, t):
        mean, _, log_var = q_posterior(x_0, x_t, t)
        
        noise = torch.randn_like(x_t)
        
        nonzero_mask = (1 - (t == 0).type(torch.float32))#t=0时候mask为全0

        return mean + nonzero_mask[:,None,None] * torch.exp(0.5 * log_var) * noise#t=0时没有引入噪声,这地方相比原DDGAN的代码，删去了一个nonzero_mask的维度，因为layout是三维，原论文是处理的四维图像数据
            
    sample_x_pos = p_sample(x_0, x_t, t)
    
    return sample_x_pos


def c_sample_from_model(coefficients, generator, n_time, label, b_t_1, mask, opt,latent_z):
    b_t_1 = b_t_1
    with torch.no_grad():
        for i in reversed(range(n_time)):      
            t = torch.full((b_t_1.size(0),), i, dtype=torch.int64).to(b_t_1.device)
            layout_0 = generator(label, b_t_1, ~mask, latent_z)#这里还没有添加时间t
            b_new = sample_posterior(coefficients, layout_0[:,:,opt.num_label:], b_t_1, t)
            b_t_1 = b_new.detach()
        
    return torch.cat((label,b_t_1),dim=-1)

def sample_diffusion(coefficients, generator, layout_input, mask, cond, opt):
    B = layout_input.size(0)
    l_t_noise = torch.randn_like(layout_input).to(layout_input.device)
    z = torch.randn(layout_input.size(0), layout_input.size(1), opt.latent_dim,device=layout_input.device)
    t = torch.full((B,), opt.num_timesteps-1, dtype=torch.int64).to(layout_input.device)
    temp, fix_mask= task_union(layout_input, l_t_noise, opt.num_label, mask,fix_mask = None)
    if cond == 'c':
        l_t_1 = temp[B:2*B,:,:]
    elif cond == 'cwh':
        l_t_1 = temp[2*B:3*B,:,:]
    else:   
        l_t_1 = temp[3*B:4*B:,:]    
    with torch.no_grad():
        for i in reversed(range(opt.num_timesteps)):      
            t = torch.full((l_t_1.size(0),), i, dtype=torch.int64).to(l_t_1.device)
            layout_0 = generator(l_t_1[:,:,:opt.num_label],l_t_1[:,:,opt.num_label:], ~mask, z)#这里还没有添加时间t
            l_t_pred = sample_posterior(coefficients, layout_0, l_t_1, t)
            l_t_1 = task_union_reverse(layout_input, l_t_pred.repeat(4,1,1), opt.num_label, fix_mask).detach()
            if cond == 'c':
                l_t_1 = l_t_1[B:2*B,:,:]
            elif cond == 'cwh':
                l_t_1 = l_t_1[2*B:3*B,:,:]
            else:   
                l_t_1 = l_t_1[3*B:4*B:,:]    
        
    return l_t_1

def uncond_sample_from_model(coefficients, generator, n_time, x_t_1, mask, opt,latent_z):
    x_t_1 = x_t_1
    with torch.no_grad():
        for i in reversed(range(n_time)):            
            t = torch.full((x_t_1.size(0),), i, dtype=torch.int64).to(x_t_1.device)
            layout_0 = generator(x_t_1[:,:,:opt.num_label],x_t_1[:,:,opt.num_label:], None,latent_z)#这里还没有添加时间t
            x_new = sample_posterior(coefficients, layout_0, x_t_1, t)
            x_t_1 = x_new.detach()
        
    return x_t_1

def analogbits(x, num_class, device):
    embed_length = len(bin(num_class - 1)) - 2 
    x_flat = x.view(-1)  
    embed_matrix = torch.tensor([[int(digit) for digit in format(i, '0{}b'.format(embed_length))] for i in range(num_class)]).to(device)  
    b_t = embed_matrix[x_flat.long()]
    b_t = b_t.view(*x.shape, embed_length)
    b_t[b_t == 0] = -1
    return b_t

def bits2int(b_t, device):
    # 将 -1 替换为 0
    b_t = torch.where(torch.abs(b_t - (-1)) < torch.abs(b_t - 1), 0, 1)

    # 将 b_t 的形状从 [batch_size, seq_length, embed_length] 调整为 [batch_size * seq_length, embed_length]
    b_flat = b_t.view(-1, b_t.shape[-1])

    # 计算每个二进制表示的权重
    weights = torch.tensor([2**i for i in range(b_flat.shape[1]-1, -1, -1)], device=device).float()

    # 将二进制转换为整数
    x_flat = torch.matmul(b_flat.float(), weights)

    # 将 x_flat 的形状调整回原始 x 的形状
    x = x_flat.view(*b_t.shape[:-1])

    return x