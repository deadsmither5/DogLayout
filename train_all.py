import os
import argparse
os.environ['OMP_NUM_THREADS'] = '1'  # noqa

import torch
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from torch_geometric.data import DataLoader
from torch_geometric.utils import to_dense_batch
from torch.utils.tensorboard import SummaryWriter
from util.datasets.load_data import init_dataset
from test import test_fid_feat
from util.seq_util import sparse_to_dense, pad_until
from util.metric import compute_generative_model_scores, compute_maximum_iou, compute_overlap, compute_alignment
from model.onehotganZ import Generator, Discriminator_label,Generator_rico25
from fid.model import load_fidnet_v3
from ganutil import init_experiment, save_image, save_checkpoint
from onehot.diffusiononehot import *

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--name', type=str, default='',
                        help='experiment name')
    parser.add_argument("--dataset", default='rico25',
                        help="choose from [publaynet, rico13, rico25, magazine, crello]", type=str)
    parser.add_argument("--data_dir", default='./datasets', help="dir of datasets", type=str)    
    parser.add_argument('--batch_size', type=int, default=128,
                        help='batch size')
    parser.add_argument('--iteration', type=int, default=int(2e+5),
                        help='number of iterations to train for')
    parser.add_argument('--seed', type=int, help='manual seed')

    # General
    parser.add_argument('--b_size', type=int, default=4,
                        help='latent size')
    parser.add_argument('--lr', type=float, default=1e-5,
                        help='learning rate')
    parser.add_argument('--aug_flip', action='store_true',
                        help='use horizontal flip for data augmentation.')

    # Generator
    parser.add_argument('--G_d_model', type=int, default=256,
                        help='d_model for generator')
    parser.add_argument('--G_nhead', type=int, default=4,
                        help='nhead for generator')
    parser.add_argument('--G_num_layers', type=int, default=8,
                        help='num_layers for generator')

    # Discriminator
    parser.add_argument('--D_d_model', type=int, default=256,
                        help='d_model for discriminator')
    parser.add_argument('--D_nhead', type=int, default=4,
                        help='nhead for discriminator')
    parser.add_argument('--D_num_layers', type=int, default=8,
                        help='num_layers for discriminator')
    
    parser.add_argument('--num_label', type=int, default=26)
    parser.add_argument('--num_timesteps', type=int, default=4)
    parser.add_argument("--latent_dim", type=int, default=4)
    parser.add_argument('--beta_min', type=float, default= 0.1, help='beta_min for diffusion')
    parser.add_argument('--beta_max', type=float, default=20., help='beta_max for diffusion')
    parser.add_argument('--use_geometric', action='store_true',default=False)
    args = parser.parse_args()
    print(args)

    out_dir = init_experiment(args, "LayoutGAN++")
    writer = SummaryWriter(out_dir)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # load dataset
    train_dataset, train_dataloader = init_dataset(args.dataset, args.data_dir, batch_size=args.batch_size,
                                               split='train', shuffle=True, transform=None)

    val_dataset, val_dataloader =init_dataset(args.dataset, args.data_dir, batch_size=args.batch_size,
                                               split='test', shuffle=False, transform=None)#用test,不要开shuffle

    num_label = train_dataset.num_classes + 1 
    args.num_label = num_label
    print("num_label:{} onehot:{}".format(num_label,True))

    # setup model
    if args.dataset == 'publaynet':
        netG = Generator(args.b_size,num_label,
                        d_model=args.G_d_model,
                        nhead=args.G_nhead,
                        num_layers=args.G_num_layers,
                        z_dim = args.latent_dim
                        ).to(device)
    elif args.dataset == 'rico25':
        netG = Generator_rico25(args.b_size,num_label,
                        d_model=args.G_d_model,
                        nhead=args.G_nhead,
                        num_layers=args.G_num_layers,
                        z_dim = args.latent_dim
                        ).to(device)    

    netD = Discriminator_label(num_label,
                         d_model=args.D_d_model,
                         nhead=args.D_nhead,
                         num_layers=args.D_num_layers,
                         ).to(device)

    # prepare for evaluation
    fid_val = load_fidnet_v3(val_dataset, './fid/FIDNetV3', device=device)
    feats_test = test_fid_feat(args.dataset, device=device, batch_size=20)


    fixed_label = None
    val_layouts = [(data.x.numpy(), data.y.numpy()) for data in val_dataset]

    # setup optimizer
    optimizerD = optim.Adam(netD.parameters(), lr=args.lr)
    optimizerG = optim.Adam(netG.parameters(), lr=args.lr)

    # ckpt = torch.load('/home/ganzhaoxing/diffusion_layout_gan/output/publaynet/LayoutGAN++/20240402051532997015/checkpoint.pth.tar', map_location=device)
    # netG.load_state_dict(ckpt['netG'])
    # netD.load_state_dict(ckpt['netD'])
    # optimizerG.load_state_dict(ckpt['optimizerG'])
    # optimizerD.load_state_dict(ckpt['optimizerD'])

    iteration = 0
    last_eval, best_iou, best_fid= -1e+8, -1e+8, 1e+8#注意best_fid得是从很大的数初始化

    # max_epoch = args.iteration * args.batch_size / len(train_dataset)
    # max_epoch = int(torch.ceil(torch.tensor(max_epoch)).item())
    max_epoch =3000

    coeff = Diffusion_Coefficients(args, device)
    pos_coeff = Posterior_Coefficients(args, device)
    T_nums_step = get_time_schedule(args, device)
    
    for epoch in range(max_epoch):
        netG.train(), netD.train()
        for i, data in enumerate(train_dataloader):
            data = data.to(device)
            bbox, label, _, mask = sparse_to_dense(data)
            label, bbox, mask = pad_until(label, bbox, mask, max_seq_length=25)
            label, bbox, mask = label.to(device), bbox.to(device), mask.to(device)

            # shift to center

            # set mask to label 5
            label[mask==False] = num_label - 1
            # print("label:{}".format(label))

            label_oh = torch.nn.functional.one_hot(label, num_classes=num_label).to(torch.float32) 
            padding_mask = ~mask
            layout_input = torch.cat((label_oh, bbox), dim=2).to(device)
    
            # Update G network
            netG.zero_grad()
            t = sample_t([bbox.shape[0]], t_max=args.num_timesteps,device=device)
            t_all = torch.cat([t, t, t, t], dim=0)
            l_t, l_tp1 = q_sample_pairs(coeff, layout_input, t)
            l_t, fix_mask = task_union(layout_input, l_t, num_label, mask,fix_mask=None)
            l_tp1, fix_mask = task_union(layout_input, l_tp1, num_label, mask,fix_mask)
            latent_z = torch.randn(4*label.size(0),label.size(1),args.latent_dim,device=device)
    
            layout_fake= netG(l_tp1[:,:,:num_label],l_tp1[:,:,num_label:], padding_mask.repeat(4,1),latent_z)
            l_pos_sample = sample_posterior(pos_coeff, layout_fake, l_tp1, t_all)
            l_pos_sample = task_union_reverse(layout_input, l_pos_sample, num_label, fix_mask)
            D_fake = netD(torch.cat((l_pos_sample[:,:,:num_label], l_tp1[:,:,:num_label].detach()),dim = -1), torch.cat((l_pos_sample[:,:,num_label:], l_tp1[:,:,num_label:].detach()),dim =-1), padding_mask.repeat(4,1))#label和padding_mask b_t和 b_t+1应该共享
            loss_G = F.softplus(-D_fake).mean() 
            B = layout_input.size(0)#最后一个batch_size不一定有定义的大小
            # loss_G_recl = F.cross_entropy(layout_fake[B:3*B,:, :num_label].view(-1,num_label),label.view(-1).repeat(2)) +\
            #     F.cross_entropy((layout_fake[3*B:4*B,:,:num_label])[fix_mask].view(-1,num_label),label[fix_mask].view(-1))
            loss_G_total= loss_G #+ loss_G_recl + 10 *loss_G_recb
            loss_G_total.backward()
            # loss_G.backward()
            optimizerG.step()

            # Update D network
            #train with fake
            netD.zero_grad()
            D_fake = netD(torch.cat((l_pos_sample[:,:,:num_label].detach(), l_tp1[:,:,:num_label].detach()),dim = -1), torch.cat((l_pos_sample[:,:,num_label:].detach(), l_tp1[:,:,num_label:].detach()),dim =-1), padding_mask.repeat(4,1))#要考虑detach怎么设置
            loss_D_fake = F.softplus(D_fake).mean()

            #train with real
            D_real, logit_cls, bbox_recon = \
                netD(torch.cat((l_t[:,:,:num_label].detach(),l_tp1[:,:,:num_label].detach()),dim = -1), torch.cat((l_t[:,:,num_label:], l_tp1[:,:,num_label:]),dim =-1), padding_mask.repeat(4,1), reconst=True)
            # print("D_real:{}, logit_cls:{}, bbox_recon:{}".format(D_real.shape, logit_cls.shape, bbox_recon.shape))
            # print("data.y:{},data.x:{}".format(data.y.shape,data.x.shape))
            loss_D_real = F.softplus(-D_real).mean()
            loss_D_recl = F.cross_entropy(logit_cls, (label.view(-1)).repeat(4).to(device))
            loss_D_recb = F.mse_loss(bbox_recon, bbox.repeat(4,1,1).to(device))
            # print("loss_D_real:{}loss_D_recl:{}loss_D_recb:{}".format(loss_D_real,loss_D_recl,loss_D_recb))
            loss_D = loss_D_real + loss_D_fake
            loss_D += loss_D_recl + 10 * loss_D_recb
            loss_D.backward()
            optimizerD.step()

            if iteration % 50 == 0:
                D_real = torch.sigmoid(D_real).mean().item()
                D_fake = torch.sigmoid(D_fake).mean().item()
                loss_D, loss_G = loss_D.item(), loss_G.item()
                loss_D_fake, loss_D_real = loss_D_fake.item(), loss_D_real.item()
                loss_D_recl, loss_D_recb = loss_D_recl.item(), loss_D_recb.item()

                print('\t'.join([
                    f'[{epoch}/{max_epoch}][{i}/{len(train_dataloader)}]',
                    f'Loss_D: {loss_D:E}', f'Loss_G: {loss_G:E}',
                    f'Real: {D_real:.3f}', f'Fake: {D_fake:.3f}',
                ]))

                # add data to tensorboard
                tag_scalar_dict = {'real': D_real, 'fake': D_fake}
                writer.add_scalars('Train/D_value', tag_scalar_dict, iteration)
                writer.add_scalar('Train/Loss_D', loss_D, iteration)
                writer.add_scalar('Train/Loss_D_fake', loss_D_fake, iteration)
                writer.add_scalar('Train/Loss_D_real', loss_D_real, iteration)
                writer.add_scalar('Train/Loss_D_recl', loss_D_recl, iteration)
                writer.add_scalar('Train/Loss_D_recb', loss_D_recb, iteration)
                writer.add_scalar('Train/Loss_G', loss_G, iteration)

            if iteration % 5000 == 0:
                out_path = out_dir / f'real_samples.png'
                if not out_path.exists():
                    save_image(bbox, label, mask,
                               train_dataset.colors, out_path)
                    print("111111111111111")

                if fixed_label is None:#这里用fixed的label是为了使得每次保存的fake图像都是由同一组输入产生的，由此可以看到变化
                    fixed_label = label
                    fixed_b_t_1 = torch.randn(label.size(0), label.size(1), args.b_size, device=device)
                    fixed_mask = mask

                with torch.no_grad():
                    netG.eval()
                    out_path = out_dir / f'fake_samples_{iteration:07d}.png'
                    latent_z = torch.randn(label.size(0),label.size(1),args.latent_dim,device=device)
                    l_t_1 = torch.randn(label.size(0), label.size(1), args.b_size+num_label, device=device)
                    layout_fake = uncond_sample_from_model(pos_coeff, netG, args.num_timesteps, l_t_1, mask , args, latent_z)
                    # bbox_fake = netG(fixed_z, fixed_label, ~fixed_mask)
                    bbox_generated, label_generated, mask_generated = finalize(layout_fake,num_class=num_label)
                    save_image(bbox_generated, label_generated, mask_generated,
                               train_dataset.colors, out_path)
                    print(out_path)
                    print("00000000000000")
                    netG.train()


            iteration += 1

        if epoch != max_epoch - 1:
            if iteration - last_eval < 1000:
                continue

        # validation
        last_eval = iteration
        fake_layouts = []
        feats_generate = []
        netG.eval(), netD.eval()
        with torch.no_grad():
             for i, data in enumerate(val_dataloader):
                data = data.to(device)
                bbox, label, _, mask = sparse_to_dense(data)
                label, bbox, mask = pad_until(label, bbox, mask, max_seq_length=25)
                label, bbox, mask = label.to(device), bbox.to(device), mask.to(device)
                l_t_1 = torch.randn(label.size(0), label.size(1), args.b_size+num_label, device=device)

                latent_z = torch.randn(label.size(0),label.size(1),args.latent_dim,device=device)
                layout_fake = uncond_sample_from_model(pos_coeff, netG, args.num_timesteps, l_t_1, mask , args, latent_z)

                bbox_generated, label_generated, mask_generated = finalize(layout_fake,num_class=num_label)
                padding_mask = ~mask_generated
                label_generated[mask_generated==False] = 0

                feat = fid_val.extract_features(bbox_generated, label_generated, padding_mask)
                feats_generate.append(feat.cpu())

                # collect generated layouts
                for j in range(label.size(0)):
                    _mask = mask_generated[j]
                    b = bbox_generated[j][_mask].cpu().numpy()
                    l = label[j][_mask].cpu().numpy()
                    fake_layouts.append((b, l))

        result = compute_generative_model_scores(feats_test, feats_generate)
        fid_score_val = result['fid']

        max_iou_val = compute_maximum_iou(val_layouts, fake_layouts)
        writer.add_scalar('Epoch', epoch, iteration)
        tag_scalar_dict = {'val': fid_score_val}
        writer.add_scalars('Score/Layout FID', tag_scalar_dict, iteration)
        writer.add_scalar('Epoch', epoch, iteration)
        writer.add_scalar('Score/Maximum IoU', max_iou_val, iteration)

        # do checkpointing
        # is_best = best_iou < max_iou_val
        # best_iou = max(max_iou_val, best_iou)
        is_best = best_fid > fid_score_val
        best_fid = min(best_fid, fid_score_val)

        save_checkpoint({
            'args': vars(args),
            'epoch': epoch + 1,
            'netG': netG.state_dict(),
            'netD': netD.state_dict(),
            'best_fid': best_fid,
            'optimizerG': optimizerG.state_dict(),
            'optimizerD': optimizerD.state_dict(),
        }, is_best, out_dir)


if __name__ == "__main__":
    main()
