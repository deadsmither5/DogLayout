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
        self.num_layers = num_layers
        self.fc_in = nn.Linear(b_size + label_size, d_model)
        self.fc_z = nn.Linear(z_dim, 32)
        self.fc_all = nn.Linear(d_model + 32, d_model)

        self.pos_encoder = SinusoidalPosEmb(num_steps=25, dim=d_model)
        pos_i = torch.tensor([i for i in range(25)])
        self.pos_embed = self.pos_encoder(pos_i)

        encoder_layer = Block(d_model=d_model, nhead=nhead, dim_feedforward=2048, diffusion_step=4)
        self.layers = _get_clones(encoder_layer, num_layers)

        # te = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
        #                                 dim_feedforward=2048)#原本是d_model // 2，根据layoutdm修改一下
        # self.transformer = nn.TransformerEncoder(te, num_layers=num_layers)

        self.fc_bbox = nn.Linear(d_model, label_size)
        self.fc_label = nn.Linear(d_model, 4)

    def forward(self, layout , padding_mask, latent_z, timestep):
        padding_mask =  None
        mask = None
        #self.fc_z(z)
        x = self.fc_in(layout)
        z = self.fc_z(latent_z)
        x = torch.cat([x,z], dim=-1)
        x = torch.relu(self.fc_all(x))
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

        logit_bbox = self.fc_bbox(x)
        bbox_pred = torch.sigmoid(logit_bbox)

        logit_cls = self.fc_label(x)
        cls_pred = torch.sigmoid(logit_cls)
        # cls_pred = nn.functional.gumbel_softmax(logit_cls, tau=0.9 , hard=True)
        return torch.cat((cls_pred , bbox_pred),dim=-1)

class Discriminator(nn.Module):#对于myout_onehot7.txt的实验来说dimfeedforward是d_model // 2 ，貌似保持判别器结构不变是最好的
    def __init__(self, label_size, d_model=512,
                 nhead=8, num_layers=4, max_bbox=50):
        super().__init__()
        #只输入一个label如何呢，避免判别器根据label不同进行判断，我只对bbox进行判断
        self.num_layers = num_layers
        self.fc_in = nn.Linear(8 + 2*label_size, 2*d_model)
        self.fc_all = nn.Linear(2*d_model, d_model)

        self.pos_encoder = SinusoidalPosEmb(num_steps=25, dim=d_model)
        pos_i = torch.tensor([i for i in range(25)])
        self.pos_embed = self.pos_encoder(pos_i)

        encoder_layer = Block(d_model=d_model, nhead=nhead, dim_feedforward=2048, diffusion_step=4)
        self.layers = _get_clones(encoder_layer, num_layers)

        # te = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
        #                                 dim_feedforward=2048)#原本是d_model // 2，根据layoutdm修改一下
        # self.transformer = nn.TransformerEncoder(te, num_layers=num_layers)

        self.fc_outdisc = nn.Linear(d_model, 1)
        self.fc_out_cls = nn.Linear(d_model, label_size)
        self.fc_out_bbox = nn.Linear(d_model, 4)

    def forward(self, layout, padding_mask, timestep, reconst=False):
        padding_mask =  None
        mask = None
        #self.fc_z(z)
        x = self.fc_in(layout)

        x = torch.relu(self.fc_all(x))

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
        logit_disc = self.fc_outdisc(x)
    
        if reconst == False:
            return logit_disc
        else :
            bbox_pred = torch.sigmoid(self.fc_out_bbox(x))
            x = x.view(-1,x.size(2))
            logit_cls = self.fc_out_cls(x)
            return logit_disc, logit_cls, bbox_pred
