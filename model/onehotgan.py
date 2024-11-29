import torch
import torch.nn as nn

from model.util import TransformerWithToken


class Generator(nn.Module):
    def __init__(self, b_size, label_size,
                 d_model=512, nhead=8, num_layers=4):
        super().__init__()
        self.fc_b = nn.Linear(b_size, d_model // 2)
        self.emb_label = nn.Linear(label_size,d_model // 2)
        self.fc_in = nn.Linear(d_model, d_model)

        te = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                        dim_feedforward=2048)#原本是d_model // 2，根据layoutdm修改一下
        self.transformer = nn.TransformerEncoder(te, num_layers=num_layers)

        self.fc_out = nn.Linear(d_model, label_size + 4)

    def forward(self, label, bbox, padding_mask):
        #self.fc_z(z)
        l = self.emb_label(label)
        b = self.fc_b(bbox)
        x = torch.cat([l,b], dim=-1)
        x = torch.relu(self.fc_in(x)).permute(1, 0, 2)

        x = self.transformer(x, src_key_padding_mask=padding_mask)

        x = self.fc_out(x.permute(1, 0, 2))
        x = torch.sigmoid(x)

        return x

class Discriminator(nn.Module):
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
        # print("d x:{}".format(x.shape))
        x = torch.relu(x).permute(1, 0, 2)
        # print("d x permute:{}".format(x.shape))
        x = self.enc_transformer(x, src_key_padding_mask=padding_mask)
        # print("x_transform:{}".format(x.shape))
        x = x[0]
        # print("x[0]:{}".format(x[0].shape))

        # logit_disc: [B,]
        logit_disc = self.fc_out_disc(x).squeeze(-1)

        if not reconst:
            return logit_disc

        else:
            x = x.unsqueeze(0).expand(N, -1, -1)
            t = self.pos_token[:N].expand(-1, B, -1)
            x = torch.cat([x, t], dim=-1)
            x = torch.relu(self.dec_fc_in(x))

            x = self.dec_transformer(x, src_key_padding_mask=padding_mask)
            # print("padding_mask:{}".format(padding_mask.shape))
            x = x.permute(1, 0, 2)[~padding_mask]
            # print("x_final:{}".format(x.shape))
            # logit_cls: [M, L]    bbox_pred: [M, 4]
            logit_cls = self.fc_out_cls(x)
            # print("logit_cls:{}".format(logit_cls.shape))
            bbox_pred = torch.sigmoid(self.fc_out_bbox(x))

            return logit_disc, logit_cls, bbox_pred
