import torch
from torch import nn
import math
class AttentionDecoder(nn.Module):
    def __init__(self, d_model, n_heads):
            super(AttentionDecoder, self).__init__()

            d_keys = d_model // n_heads
            d_values = d_model // n_heads
            self.norm = nn.LayerNorm(d_model)
            self.query_projection = nn.Linear(d_model,
                                            d_keys * n_heads)
            self.key_projection = nn.Linear(d_model,
                                            d_keys * n_heads)
            self.value_projection = nn.Linear(d_model,
                                            d_values * n_heads)
            self.out_projection = nn.Linear(d_values * n_heads, d_model)
            self.n_heads = n_heads
            self.attn = SelfAttention()

    def forward(self, input):
        B, L, _ = input.shape
        _, S, _ = input.shape
        H = self.n_heads

        queries = self.query_projection(input).view(B, L, H, -1)
        keys = self.key_projection(input).view(B, S, H, -1)
        values = self.value_projection(input).view(B, S, H, -1)

        out = self.attn(queries,keys,values)
        out = out.view(B, L, -1)

        return self.out_projection(out)

class SelfAttention(nn.Module):
    def __init__(self, attention_dropout=0.0):
        super(SelfAttention, self).__init__()
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = 1. / math.sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        attn = scale * scores

        series = self.dropout(torch.softmax(attn, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", series, values)
        return V.contiguous()