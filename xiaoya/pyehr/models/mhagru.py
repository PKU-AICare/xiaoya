import torch
from torch import nn


class MHAGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim: int=32, feat_dim: int=8, act_layer=nn.GELU, drop=0.0, **kwargs):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.feat_dim = feat_dim
        self.num_heads = 4
        self.act = act_layer()
        self.input_proj = nn.Linear(input_dim, input_dim)
        self.grus = nn.ModuleList(
            [
                nn.GRU(1, feat_dim, num_layers=1, batch_first=True)
                for _ in range(input_dim)
            ]
        )
        self.mha = nn.MultiheadAttention(feat_dim, self.num_heads, dropout=drop, batch_first=True)
        self.out_proj = nn.Linear(input_dim * feat_dim, hidden_dim)
        self.dropout = nn.Dropout(drop)
    
    def forward(self, x, **kwargs):
        # x: [bs, time_steps, input_dim]
        bs, time_steps, _ = x.shape
        x = self.input_proj(x)   # [bs, time_steps, input_dim] -> [bs, time_steps, hidden_dim]
        out = torch.zeros(bs, time_steps, self.input_dim, self.feat_dim).to(x.device)
        attention = torch.zeros(bs, time_steps, self.input_dim, self.feat_dim).to(x.device)
        for i, gru in enumerate(self.grus):
            cur_feat = x[:, :, i].unsqueeze(-1)     # [bs, time_steps, 1]
            cur_feat = gru(cur_feat)[0]             # [bs, time_steps, feat_dim]
            out[:, :, i] = cur_feat                 # [bs, time_steps, input_dim, feat_dim]
            
            attn = self.mha(cur_feat, cur_feat, cur_feat, average_attn_weights=True)
            attention[:, :, i] = attn[0]

        out = out.flatten(2)        # [bs, time, input, feat] -> [bs, time, input * feat]
        out = self.out_proj(out)    # [bs, time, input * feat] -> [bs, time, hidden_dim]
        return out, attention
