"""任务一 M1/M2：手写 scaled dot-product attention 与 MultiHeadAttention。

约定（与 eval/run.py 一致）：
- Q/K/V 形状 (B, H, T, D)
- mask 为 bool 张量，True = 被屏蔽（不参与 softmax），可广播到 (B, H, T, T)
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _attention(Q, K, V, mask=None, return_weights=False):
    """核心实现：缩放点积 + mask + softmax + 加权求和。

    scores = Q K^T / sqrt(d_k)
    mask 处填 -inf，softmax 后概率为 0（用 masked_fill 而不是乘 0，
    否则 softmax 后仍会泄漏概率）。
    """
    d_k = Q.shape[-1]
    # (B, H, T, D) x (B, H, D, T) -> (B, H, T, T)
    scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask.to(torch.bool), float("-inf"))
    attn = F.softmax(scores, dim=-1)  # (B, H, T, T)
    out = attn @ V                     # (B, H, T, D)
    if return_weights:
        return out, attn
    return out


def scaled_dot_product_attention(Q, K, V, mask=None):
    """自检契约入口：与 F.scaled_dot_product_attention 数值一致（误差 < 1e-5）。"""
    return _attention(Q, K, V, mask)


class MultiHeadAttention(nn.Module):
    """多 head 注意力：Q/K/V 线性投影 -> 分头 -> 缩放点积 -> 拼接 -> 输出投影。

    输入 x: (B, T, d_model)；输出 (out, attn)：
    - out:  (B, T, d_model)
    - attn: (B, n_heads, T, T)  注意力权重，供可视化
    """

    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model 必须能被 n_heads 整除"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        # 无 bias：注意力投影一般不配 bias（与原论文一致）
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        B, T, _ = q.shape

        def _split(x):
            # (B, T, d_model) -> (B, H, T, D)；reshape 后必须 contiguous 才能 transpose
            return x.view(B, T, self.n_heads, self.d_k).transpose(1, 2).contiguous()

        Q = _split(self.w_q(q))
        K = _split(self.w_k(k))
        V = _split(self.w_v(v))

        out, attn = _attention(Q, K, V, mask, return_weights=True)
        out = self.dropout(out)

        # (B, H, T, D) -> (B, T, H, D) -> (B, T, d_model)
        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        out = self.w_o(out)
        return out, attn
