import torch
from torch import nn


class TGRU(nn.Module):
    def __init__(self, lab_dim, demo_dim, hidden_dim: int=32, feat_dim: int=8, act_layer=nn.GELU, drop=0.0, **kwargs):
        super().__init__()
        self.lab_dim = lab_dim
        self.demo_dim = demo_dim
        self.hidden_dim = hidden_dim
        self.feat_dim = feat_dim
        self.num_heads = 4
        self.act = act_layer()
        self.demo_proj = nn.Linear(demo_dim, hidden_dim)
        self.lab_proj = nn.Linear(lab_dim, lab_dim)
        self.grus = nn.ModuleList(
            [
                nn.GRU(1, feat_dim, num_layers=1, batch_first=True)
                for _ in range(lab_dim)
            ]
        )
        self.mhsas = nn.ModuleList(
            [   
                nn.MultiheadAttention(feat_dim, self.num_heads, dropout=drop, batch_first=True)
                for _ in range(lab_dim)
            ]
        )
        self.out_proj = nn.Linear(lab_dim * feat_dim + hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(drop)
    
    def forward(self, lab, demo, **kwargs):
        # lab: [bs, time_steps, lab_dim]
        # demo: [bs, demo_dim]
        bs, time_steps, _ = lab.shape
        demo = self.demo_proj(demo) # [bs, hidden_dim]
        lab = self.lab_proj(lab)    # [bs, time_steps, lab_dim]
        out = torch.zeros(bs, time_steps, self.lab_dim, self.feat_dim).to(lab.device)
        attention = torch.zeros(bs, time_steps, self.lab_dim, lab.shape[1]).to(lab.device)
        for i, (gru, mhsa) in enumerate(zip(self.grus, self.mhsas)):
            cur_feat = lab[:, :, i].unsqueeze(-1)   # [bs, time_steps, 1]
            cur_feat = gru(cur_feat)[0]             # [bs, time_steps, feat_dim]
            out[:, :, i] = cur_feat                 # [bs, time_steps, lab_dim, feat_dim]
            
            attn = mhsa(cur_feat, cur_feat, cur_feat, average_attn_weights=True)[1]
            attention[:, :, i] = attn

        out = out.flatten(2)                            # b t l f -> b t (l f)
        # concat demo and out
        out = torch.cat([demo.unsqueeze(1).repeat(1, time_steps, 1), out], dim=-1)
        out = self.out_proj(out)
        return out, attention


if __name__ == '__main__':
    bs = 2
    time = 8
    lab_dim = 10
    demo_dim = 2
    model = TGRU(lab_dim, demo_dim)

    lab = torch.randn(bs, time, lab_dim)
    demo = torch.randn(bs, demo_dim)
    out, attention = model(lab, demo)

    # print(out.shape, attention.shape)

    patient = 0
    patient_res = attention[patient].reshape(attention.shape[1], -1).abs().sum(-1).squeeze(-1)
    print(patient_res)
    patient_res = patient_res / patient_res.sum()
    print(patient_res)

    # for i in range(bs):
    #     for t in range(time):
    #         y = attention[i, t, :].abs().sum()
    # y = attention[:, :, :].abs() / attention[:, :, :].abs().sum(-1, keepdim=True)
    # print(y.shape)
    attention = attention.transpose(0, 2).reshape(lab_dim, -1).abs().sum(-1).squeeze(-1)
    print(attention)
    y = attention / attention.sum()
    print(y)
