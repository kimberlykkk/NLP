"""任务二 M3 核心：causal 多头自注意力 + KV cache + RoPE。

- causal 掩码统一写法：query 行 t（绝对位置 pos_t），只能看 key 列 j <= pos_t；
  这个写法同时覆盖「全量前向（无历史）」和「增量解码（有历史）」两种情形，
  KV cache 开/关的 logits 才能严格一致（自检 kv_cache_equivalence 就是这么比的）。
- KV cache 里存的 K/V 是「拼接后的完整序列」，形状 (B, H, L, D)，
  增量解码时新 token 的位置 = 历史长度 L0 + 当前批内偏移。
"""
import math

import torch
import torch.nn as nn

from .rope import apply_rope, precompute_freqs


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1, rope_base=10000.0):
        super().__init__()
        assert d_model % n_heads == 0, "d_model 必须能被 n_heads 整除"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("freqs", precompute_freqs(self.d_k, rope_base))

    def _rotate(self, x, positions):
        """RoPE：x (B,H,T,D) -> 旋转后同形状（只用于 Q/K）。"""
        return apply_rope(x, positions, self.freqs)

    def forward(self, x, positions, past_kv=None):
        """x: (B, T, D)；positions: (T,) 绝对位置（含历史偏移）；
        past_kv: (K, V) 或 None；K/V 形状 (B, H, L, D) 是历史。
        返回 (out, k_new, v_new)：k_new/v_new 为拼接后的完整 K/V (B,H,L+T,D)。"""
        B, T, _ = x.shape

        def split(y):
            return y.view(B, T, self.n_heads, self.d_k).transpose(1, 2).contiguous()

        q = split(self.w_q(x))
        k = split(self.w_k(x))
        v = split(self.w_v(x))

        # RoPE：只加 Q/K、只加新来的 token（历史的 K/V 在入缓存时已旋转）
        q = self._rotate(q, positions)
        k = self._rotate(k, positions)

        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=2)   # 拼接维度 = 序列长度 T
            v = torch.cat([past_v, v], dim=2)
        L_total = k.shape[2]

        scores = q @ k.transpose(-2, -1) / math.sqrt(self.d_k)   # (B,H,T,L_total)

        # causal 掩码：pos_t = L0 + t 可见 key 位置 <= pos_t
        L0 = L_total - T
        rows = torch.arange(T, device=x.device)[:, None]     # (T,1)
        cols = torch.arange(L_total, device=x.device)[None, :]  # (1,L_total)
        causal = cols > (L0 + rows)                          # True = 屏蔽
        scores = scores.masked_fill(causal, float("-inf"))

        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = attn @ v                                       # (B,H,T,D_k)

        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        out = self.w_o(out)
        return out, k, v
