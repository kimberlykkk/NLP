"""任务二：decoder-only block（Pre-LN：LayerNorm -> 子层 -> 残差）。"""
import torch.nn as nn

from .attention import CausalSelfAttention


class GPTBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff=None, dropout=0.1):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, positions, past_kv=None):
        """x: (B,T,D)；positions: (T,)；past_kv: {"K","V"} dict 或 None
        （模型层缓存用 dict 存）；返回 (x, k, v)，k/v 是该层拼接后的完整 K/V。"""
        if past_kv is not None:
            past_kv = (past_kv["K"], past_kv["V"])   # attention 收 (K, V) 元组
        xn = self.ln1(x)
        a, k, v = self.attn(xn, positions, past_kv)
        x = x + self.dropout(a)

        xn = self.ln2(x)
        x = x + self.dropout(self.ffn(xn))
        return x, k, v
