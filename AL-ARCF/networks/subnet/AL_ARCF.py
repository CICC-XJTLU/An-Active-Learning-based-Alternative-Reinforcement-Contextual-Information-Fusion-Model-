import torch
import torch.nn as nn
import torch.nn.functional as F
from config.global_configs import *
import math
from EGCE import EGCE
# from .HCL_Module import CL3, HCLModule_3, HCLModule

class AL_ARCF(nn.Module):
    def __init__(self, beta_shift_a=0.5, beta_shift_v=0.5, dropout_prob=0.2, name=""):
        super(AL_ARCF, self).__init__()
        self.visual_embedding = nn.Embedding(label_size + 1, TEXT_DIM, padding_idx=label_size)
        self.acoustic_embedding = nn.Embedding(label_size + 1, TEXT_DIM, padding_idx=label_size)
        self.cat_connect = nn.Linear(3 * TEXT_DIM, TEXT_DIM)
        self.cross_ATT_visual = CrossAttention(dim=TEXT_DIM)
        self.cross_ATT_acoustic = CrossAttention(dim=TEXT_DIM)
        self.layer_norm = nn.LayerNorm(TEXT_DIM)
        self.dropout = nn.Dropout(dropout_prob)
        self.lstm = nn.LSTM(TEXT_DIM, TEXT_DIM, batch_first=True)
        self.fc1 = nn.Linear(TEXT_DIM, hidden_size)
        self.fc2 = nn.Linear(hidden_size, TEXT_DIM)
        self.cat_connect_two = nn.Linear(2 * TEXT_DIM, TEXT_DIM)
        self.egce = EGCE(TEXT_DIM)
        self.afi = AFI(TEXT_DIM, heads=8, dropout=dropout_prob)


    def forward(self, text_embedding, visual=None, acoustic=None, visual_ids=None, acoustic_ids=None):
        visual_ = self.visual_embedding(visual_ids)
        acoustic_ = self.acoustic_embedding(acoustic_ids)

        # textual
        b, n, c = text_embedding.size()
        text_embedding_re = text_embedding.view(b, c, h, w)
        text_embedding = self.egce(text_embedding_re)
        b, c, h, w = text_embedding.size()
        text_embedding = text_embedding.view(b, n, c)

        #visual
        visual_ = torch.relu(visual_)
        h, w = 1, 50
        b, n, c = visual_.size()
        visual_re = visual_.view(b, c, 1, n)
        visual_ = self.egce(visual_re)
        b, c, h, w = visual_.size()
        visual_ = visual_.view(b, n, c)
        visual_lstm, _ = self.lstm(visual_)
    
        # acoustic
        acoustic_lstm, _ = self.lstm(acoustic_)
        b, n, c = acoustic_.size()
        acoustic_re = acoustic_.view(b, c, h, w)
        acoustic_ = self.egce(acoustic_re)
        b, c, h, w = acoustic_.size()
        acoustic_ = acoustic_.view(b, n, c)

        fusion = self.afi(text_embedding, visual_, acoustic_)
        shift = self.layer_norm(fusion + text_embedding)

        output, _ = self.lstm(shift)
        output += text_embedding
        
        return output

class AFI(nn.Module):
    def __init__(self, dim, heads=8, dropout=0.2):
        super(AFI, self).__init__()
        self.dim = dim
        self.cross_att_t = CrossAttention(dim, heads=heads)
        self.cross_att_v = CrossAttention(dim, heads=heads)
        self.cross_att_a = CrossAttention(dim, heads=heads)

        # hybrid fusion
        self.hybrid_fc = nn.Linear(3 * dim, dim)

        # direct concat (case 3)
        self.concat_fc = nn.Linear(3 * dim, dim)

        self.dropout = nn.Dropout(dropout)

    def compute_correlation(self, t, v, a):
        def cos_corr(x, y):
            x_mean = x.mean(dim=1)  # [B, D]
            y_mean = y.mean(dim=1)
            return F.cosine_similarity(x_mean, y_mean, dim=-1)  # [B]

        rho_TV = cos_corr(t, v)
        rho_TA = cos_corr(t, a)
        rho_VA = cos_corr(v, a)
        delta = (rho_TV + rho_TA + rho_VA) / 3.0  # [B]
        return rho_TV, rho_TA, rho_VA, delta

    def forward(self, t, v, a):
        rho_TV, rho_TA, rho_VA, delta = self.compute_correlation(t, v, a)
        joint_corr = (rho_TV + rho_TA + rho_VA) / 3.0  # [B]

        outputs = []
        for i in range(t.size(0)):  # batch-wise 决策
            # Case 3: if all correlations exceed δ
            if rho_TV[i] > delta[i] and rho_TA[i] > delta[i] and rho_VA[i] > delta[i]:
                fused = torch.cat([t[i], v[i], a[i]], dim=-1)  # [N, 3D]
                fused = self.concat_fc(fused)

            # Case 1: joint corr > δ but individual corr low
            elif joint_corr[i] > delta[i]:
                fused = torch.cat([t[i], v[i], a[i]], dim=-1)  # [N, 3D]
                fused = self.hybrid_fc(fused)

            # Case 2: joint corr ≤ δ
            else:
                tv = self.cross_att_t(t[i].unsqueeze(0), v[i].unsqueeze(0), v[i].unsqueeze(0))
                ta = self.cross_att_a(t[i].unsqueeze(0), a[i].unsqueeze(0), a[i].unsqueeze(0))
                va = self.cross_att_v(v[i].unsqueeze(0), a[i].unsqueeze(0), a[i].unsqueeze(0))
                fused = (tv + ta + va).squeeze(0) / 3.0  # average fusion

            outputs.append(fused)

        fusion = torch.stack(outputs, dim=0)  # [B, N, D]
        return self.dropout(fusion)



class CrossAttention(nn.Module):
    def __init__(self, dim, heads=8):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.dim_per_head = dim // heads
        self.scale = self.dim_per_head ** -0.5

        assert dim % heads == 0, "Dimension must be divisible by the number of heads."

        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)

    def forward(self, queries, keys, values, mask=None):
        b, n, d = queries.size()
        h = self.heads

        queries = self.query(queries).view(b, n, h, self.dim_per_head).transpose(1, 2)
        keys = self.key(keys).view(b, -1, h, self.dim_per_head).transpose(1, 2)
        values = self.value(values).view(b, -1, h, self.dim_per_head).transpose(1, 2)

        dots = torch.einsum('bhqd,bhkd->bhqk', queries, keys) * self.scale

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value=True)
            assert mask.shape[-1] == dots.shape[-1], 'Mask has incorrect dimensions'
            mask = mask[:, None, :].expand(-1, h, -1)
            dots.masked_fill_(~mask, float('-inf'))

        attn = dots.softmax(dim=-1)
        out = torch.einsum('bhqk,bhvd->bhqd', attn, values)
        out = out.transpose(1, 2).contiguous().view(b, n, d)

        return out
