import torch
import torch.nn as nn

from model.util import TransformerWithToken
from model.backbone import*

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class Generator(nn.Module):#其实generator不应该由mask，因为生成的时候不知道哪些位置要mask，因此用一个额外的token会更好
    def __init__(self, b_size, label_size,
                 d_model=512, nhead=8, num_layers=4,z_dim=4):
        super().__init__()
        self.label_size = label_size
        self.num_layers = num_layers
        self.fc_b = nn.Linear(b_size, d_model // 2)
        self.emb_label = nn.Linear(label_size,d_model // 2)
        self.fc_z = nn.Linear(z_dim, 32)
        self.fc_in = nn.Linear(d_model + 32, d_model)

        self.pos_encoder = SinusoidalPosEmb(num_steps=25, dim=d_model)
        pos_i = torch.tensor([i for i in range(25)])
        self.pos_embed = self.pos_encoder(pos_i)

        encoder_layer = Block(d_model=d_model, nhead=nhead, dim_feedforward=2048, diffusion_step=4)
        self.layers = _get_clones(encoder_layer, num_layers)

        # te = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
        #                                 dim_feedforward=2048)#原本是d_model // 2，根据layoutdm修改一下
        # self.transformer = nn.TransformerEncoder(te, num_layers=num_layers)

        self.fc_out = nn.Linear(d_model, label_size + 4)

    def forward(self, label, bbox, padding_mask, latent_z, timestep):
        padding_mask =  None
        mask = None
        #self.fc_z(z)
        l = self.emb_label(label)
        b = self.fc_b(bbox)
        z = self.fc_z(latent_z)
        x = torch.cat([l,b,z], dim=-1)
        x = torch.relu(self.fc_in(x))

        # x = self.transformer(x, src_key_padding_mask=padding_mask)
        for i, mod in enumerate(self.layers):
            x = mod(
                x,
                src_mask=mask,
                src_key_padding_mask=padding_mask,
                timestep=timestep,
            )

            if i < self.num_layers - 1:
                x = torch.relu(x)
        x = self.fc_out(x)
        #logit_cls = x[:,:,:self.label_size]
        x = torch.sigmoid(x)

        return x #logit_cls.view(-1,self.label_size)

class Discriminator(nn.Module):#对于myout_onehot7.txt的实验来说dimfeedforward是d_model // 2 ，貌似保持判别器结构不变是最好的
    def __init__(self, label_size, d_model=512,
                 nhead=8, num_layers=4, max_bbox=50):
        super().__init__()

        # encoder
        self.emb_label = nn.Linear(label_size*2, d_model)
        self.fc_bbox = nn.Linear(8, d_model)
        self.enc_fc_in = nn.Linear(d_model * 2, d_model)

        self.enc_transformer = TransformerWithToken(d_model=d_model,
                                                    dim_feedforward=d_model // 2,
                                                    nhead=nhead, num_layers=num_layers)

        self.fc_out_disc = nn.Linear(d_model, 1)

        # decoder
        self.pos_token = nn.Parameter(torch.rand(max_bbox, 1, d_model))
        self.dec_fc_in = nn.Linear(d_model * 2, d_model)

        te = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                        dim_feedforward=d_model // 2)
        self.dec_transformer = nn.TransformerEncoder(te,
                                                     num_layers=num_layers)

        self.fc_out_cls = nn.Linear(d_model, label_size)
        self.fc_out_bbox = nn.Linear(d_model, 4)

    def forward(self, label, bbox, padding_mask, reconst=False):
        B, N, _ = bbox.size()
        l = self.emb_label(label)
        b = self.fc_bbox(bbox)
        x = self.enc_fc_in(torch.cat([l, b], dim=-1))
        #print("d x:{}".format(x.shape))
        x = torch.relu(x).permute(1, 0, 2)
        #print("d x permute:{}".format(x.shape))
        x = self.enc_transformer(x, src_key_padding_mask=padding_mask)
        #print("x_transform:{}".format(x.shape))
        x = x[0]
        #print("x[0]:{}".format(x.shape))

        # logit_disc: [B,]
        logit_disc = self.fc_out_disc(x).squeeze(-1)

        if not reconst:
            return logit_disc
        
        else:
            x = x.unsqueeze(0).expand(N, -1, -1)
            t = self.pos_token[:N].expand(-1, B, -1)
            x = torch.cat([x, t], dim=-1)
            x = torch.relu(self.dec_fc_in(x))
            #print("x.unsqueeze:{}".format(x.shape))
            x = self.dec_transformer(x, src_key_padding_mask=padding_mask)
            #print("padding_mask:{}".format(padding_mask.shape))
            x = x.permute(1, 0, 2)
            #print("x_before mask:{}".format(x.shape))
            x = x[~padding_mask]
            #print("x_final:{}".format(x.shape))
            # logit_cls: [M, L]    bbox_pred: [M, 4]
            logit_cls = self.fc_out_cls(x)
            #print("logit_cls:{}".format(logit_cls.shape))
            bbox_pred = torch.sigmoid(self.fc_out_bbox(x))

            return logit_disc, logit_cls, bbox_pred

