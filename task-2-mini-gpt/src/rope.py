"""任务二：旋转位置编码（RoPE）。

要点：
- RoPE 只加到 Q 和 K 上，**不加到 V**（V 只代表"被取走的信息"，不含位置语义）；
- 用绝对位置角频率旋转：q_i 与 k_j 的注意力分数只取决于相对位置 (i - j)，
  这就是 RoPE 的外推优势——训练长度之外的相对位置同样有定义；
- 维度配对方式：相邻两维一组 (x0,x1),(x2,x3)... 旋转；
  实现与 KV cache 拼接必须只用这一种约定（否则数值对不上）。
"""
import torch


def precompute_freqs(dim: int, base: float = 10000.0):
    """返回角频率张量 (D/2,)。dim 必须是偶数。"""
    assert dim % 2 == 0, "RoPE 要求 d_k 为偶数"
    freqs = [base ** (-2.0 * i / dim) for i in range(dim // 2)]
    return torch.tensor(freqs, dtype=torch.float32)


def apply_rope(x, positions, freqs):
    """把 RoPE 施加到 x 上。

    x:        (B, H, T, D)  已拆好 head 的 Q 或 K
    positions:(T,)          每个位置的绝对 index（含历史长度偏移）
    freqs:    (D/2,)        角频率张量
    返回同形状的旋转后张量。
    """
    T = x.shape[2]
    # 角度 = 位置 × 频率: (T, D/2)
    ang = positions.to(torch.float32)[:, None] * freqs.to(x.device)[None, :]
    cos, sin = ang.cos(), ang.sin()

    x1 = x[..., 0::2]      # 偶数维 (B,H,T,D/2)
    x2 = x[..., 1::2]      # 奇数维
    out = torch.empty_like(x)
    out[..., 0::2] = x1 * cos - x2 * sin
    out[..., 1::2] = x1 * sin + x2 * cos
    return out
