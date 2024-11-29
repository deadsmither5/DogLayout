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
        self.fc_b = nn.Linear(b_size, d_model // 2)
        self.emb_label = nn.Linear(label_size,d_model // 2)
        self.fc_z = nn.Linear(z_dim, 64)
        self.fc_in = nn.Linear(d_model + 64, d_model)

        te = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                        dim_feedforward=2048)#原本是d_model // 2，根据layoutdm修改一下
        self.transformer = nn.TransformerEncoder(te, num_layers=num_layers)

        self.fc_out = nn.Linear(d_model, label_size + 4)

    def forward(self, label, bbox, padding_mask,latent_z):
        # padding_mask =  None
        #self.fc_z(z)
        if padding_mask!=None:
            padding_mask = torch.zeros_like(padding_mask.clone())
        l = self.emb_label(label)
        b = self.fc_b(bbox)
        z = self.fc_z(latent_z)
        x = torch.cat([l,b,z], dim=-1)
        x = torch.relu(self.fc_in(x)).permute(1, 0, 2)

        x = self.transformer(x, src_key_padding_mask=padding_mask)

        x = self.fc_out(x.permute(1, 0, 2))
        logit_cls = x[:,:,:self.label_size]
        bbox_pred = torch.sigmoid(x[:,:,self.label_size:])
        #x[:,:,self.label_size:] = torch.sigmoid(x[:,:,self.label_size:])

        return torch.cat((logit_cls,bbox_pred),dim=-1) #logit_cls.view(-1,self.label_size)

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

        self.fc_out_disc = nn.Linear(d_model, label_size + 4)

        # decoder
        self.pos_token = nn.Parameter(torch.rand(max_bbox, 1, d_model))
        self.dec_fc_in = nn.Linear(d_model * 2, d_model)

        te = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                        dim_feedforward=2048)
        self.dec_transformer = nn.TransformerEncoder(te,
                                                     num_layers=num_layers)

        self.fc_out_cls = nn.Linear(d_model, label_size)
        self.fc_out_bbox = nn.Linear(d_model, 4)

    def forward(self, label, bbox, padding_mask, reconst=False):
        B, N, _ = bbox.size()
        padding_mask = torch.zeros_like(padding_mask.clone())
        l = self.emb_label(label)
        b = self.fc_bbox(bbox)
        x = self.enc_fc_in(torch.cat([l, b], dim=-1))
        x = torch.relu(x).permute(1, 0, 2)
        x = self.enc_transformer(x, src_key_padding_mask=padding_mask)
        x = x[0]
        logit_disc = self.fc_out_disc(x).squeeze(-1)

        if not reconst:
            return logit_disc
        
        else:
            x = x.unsqueeze(0).expand(N, -1, -1)
            t = self.pos_token[:N].expand(-1, B, -1)
            x = torch.cat([x, t], dim=-1)
            x = torch.relu(self.dec_fc_in(x))
            x = self.dec_transformer(x, src_key_padding_mask=padding_mask)
            x = x.permute(1, 0, 2)
            bbox_pred = torch.sigmoid(self.fc_out_bbox(x))
            x = x.reshape(-1,x.size(2))
            logit_cls = self.fc_out_cls(x)
            return logit_disc, logit_cls, bbox_pred

class Generator_rico25(nn.Module):#其实generator不应该由mask，因为生成的时候不知道哪些位置要mask，因此用一个额外的token会更好
    def __init__(self, b_size, label_size,
                 d_model=512, nhead=8, num_layers=4,z_dim=4):
        super().__init__()
        self.label_size = label_size
        self.fc_b = nn.Linear(b_size, d_model // 2)
        self.emb_label = nn.Linear(label_size,d_model // 2)
        self.fc_z = nn.Linear(z_dim, 32)
        self.fc_in = nn.Linear(d_model + 32, d_model)

        te = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                        dim_feedforward=2048)#原本是d_model // 2，根据layoutdm修改一下
        self.transformer = nn.TransformerEncoder(te, num_layers=num_layers)

        self.fc_out = nn.Linear(d_model, label_size + 4)

    def forward(self, label, bbox, padding_mask,latent_z):
        # padding_mask =  None
        #self.fc_z(z)
        if padding_mask!=None:
            padding_mask = torch.zeros_like(padding_mask.clone())
        l = self.emb_label(label)
        b = self.fc_b(bbox)
        z = self.fc_z(latent_z)
        x = torch.cat([l,b,z], dim=-1)
        x = torch.relu(self.fc_in(x)).permute(1, 0, 2)

        x = self.transformer(x, src_key_padding_mask=padding_mask)

        x = self.fc_out(x.permute(1, 0, 2))
        logit_cls = x[:,:,:self.label_size]
        bbox_pred = torch.sigmoid(x[:,:,self.label_size:])
        #x[:,:,self.label_size:] = torch.sigmoid(x[:,:,self.label_size:])

        return torch.cat((logit_cls,bbox_pred),dim=-1) #logit_cls.view(-1,self.label_size)

class Discriminator_label(nn.Module):#对于myout_onehot7.txt的实验来说dimfeedforward是d_model // 2 ，貌似保持判别器结构不变是最好的
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
        padding_mask = torch.zeros_like(padding_mask.clone())
        l = self.emb_label(label)
        b = self.fc_bbox(bbox)
        x = self.enc_fc_in(torch.cat([l, b], dim=-1))
        x = torch.relu(x).permute(1, 0, 2)
        x = self.enc_transformer(x, src_key_padding_mask=padding_mask)
        x = x[0]
        logit_disc = self.fc_out_disc(x).squeeze(-1)

        if not reconst:
            return logit_disc
        
        else:
            x = x.unsqueeze(0).expand(N, -1, -1)
            t = self.pos_token[:N].expand(-1, B, -1)
            x = torch.cat([x, t], dim=-1)
            x = torch.relu(self.dec_fc_in(x))
            x = self.dec_transformer(x, src_key_padding_mask=padding_mask)
            x = x.permute(1, 0, 2)
            bbox_pred = torch.sigmoid(self.fc_out_bbox(x))
            x = x.reshape(-1,x.size(2))
            logit_cls = self.fc_out_cls(x)
            return logit_disc, logit_cls, bbox_pred

class Discriminator_cat1(nn.Module):#对于myout_onehot7.txt的实验来说dimfeedforward是d_model // 2 ，貌似保持判别器结构不变是最好的
    def __init__(self, label_size, d_model=512,
                 nhead=8, num_layers=4, max_bbox=50):
        super().__init__()

        # encoder
        self.emb_label = nn.Linear(label_size, d_model)
        self.fc_bbox = nn.Linear(4, d_model)
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
        N = int(N/2)
        key_padding_mask = torch.zeros_like(padding_mask.clone().repeat(1,2))
        l = self.emb_label(label)
        b = self.fc_bbox(bbox)
        x = self.enc_fc_in(torch.cat([l, b], dim=-1))
        x = torch.relu(x).permute(1, 0, 2)
        x = self.enc_transformer(x, src_key_padding_mask=key_padding_mask)
        x = x[0]
        logit_disc = self.fc_out_disc(x).squeeze(-1)
        if not reconst:
            return logit_disc
        
        else:
            x = x.unsqueeze(0).expand(N, -1, -1)
            t = self.pos_token[:N].expand(-1, B, -1)
            x = torch.cat([x, t], dim=-1)
            x = torch.relu(self.dec_fc_in(x))
            key_padding_mask = torch.zeros_like(padding_mask.clone())
            x = self.dec_transformer(x, src_key_padding_mask=key_padding_mask)
            x = x.permute(1, 0, 2)
            bbox_pred = torch.sigmoid(self.fc_out_bbox(x))
            x = x.reshape(-1,x.size(2))
            logit_cls = self.fc_out_cls(x)
            return logit_disc, logit_cls, bbox_pred
        

class Generator_rico25_c_sp(nn.Module):#其实generator不应该由mask，因为生成的时候不知道哪些位置要mask，因此用一个额外的token会更好
    def __init__(self, b_size, label_size,
                 d_model=512, nhead=8, num_layers=4,z_dim=4):
        super().__init__()
        self.label_size = label_size
        self.fc_b = nn.Linear(b_size, d_model // 2)
        self.emb_label = nn.Linear(label_size,d_model // 2)
        self.fc_z = nn.Linear(z_dim, 32)
        self.fc_in = nn.Linear(d_model + 32, d_model)

        te = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                        dim_feedforward=2048)#原本是d_model // 2，根据layoutdm修改一下
        self.transformer = nn.TransformerEncoder(te, num_layers=num_layers)

        self.fc_out = nn.Linear(d_model, 4)

    def forward(self, label, bbox, padding_mask,latent_z):
        # padding_mask =  None
        #self.fc_z(z)
        label_t = label.clone()
        if padding_mask!=None:
            padding_mask = torch.zeros_like(padding_mask.clone())
        l = self.emb_label(label)
        b = self.fc_b(bbox)
        z = self.fc_z(latent_z)
        x = torch.cat([l,b,z], dim=-1)
        x = torch.relu(self.fc_in(x)).permute(1, 0, 2)

        x = self.transformer(x, src_key_padding_mask=padding_mask)

        x = self.fc_out(x.permute(1, 0, 2))
        bbox_pred = torch.sigmoid(x)
        #x[:,:,self.label_size:] = torch.sigmoid(x[:,:,self.label_size:])

        return torch.cat((label_t,bbox_pred),dim=-1) #logit_cls.view(-1,self.label_size)        

class Generator_rico25_ablation(nn.Module):#其实generator不应该由mask，因为生成的时候不知道哪些位置要mask，因此用一个额外的token会更好
    def __init__(self, b_size, label_size,
                 d_model=512, nhead=8, num_layers=4,z_dim=4):
        super().__init__()
        self.label_size = label_size
        self.fc_b = nn.Linear(b_size, d_model // 2)
        self.emb_label = nn.Linear(label_size,d_model // 2)
        self.fc_z = nn.Linear(z_dim, 32)
        self.fc_in = nn.Linear(d_model + 32, d_model)

        te = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                        dim_feedforward=2048)#原本是d_model // 2，根据layoutdm修改一下
        self.transformer = nn.TransformerEncoder(te, num_layers=num_layers)

        self.fc_out = nn.Linear(d_model, label_size + 4)

    def forward(self, label, bbox, padding_mask,latent_z):
        # padding_mask =  None
        #self.fc_z(z)
        if padding_mask!=None:
            padding_mask = torch.zeros_like(padding_mask.clone())
        l = self.emb_label(label)
        b = self.fc_b(bbox)
        z = self.fc_z(latent_z)
        x = torch.cat([l,b,z], dim=-1)
        x = torch.relu(self.fc_in(x)).permute(1, 0, 2)

        x = self.transformer(x, src_key_padding_mask=padding_mask)

        x = self.fc_out(x.permute(1, 0, 2))
        logit_cls = torch.softmax(x[:,:,:self.label_size],dim=-1)
        bbox_pred = torch.sigmoid(x[:,:,self.label_size:])
        #x[:,:,self.label_size:] = torch.sigmoid(x[:,:,self.label_size:])

        return torch.cat((logit_cls,bbox_pred),dim=-1) #logit_cls.view(-1,self.label_size)    