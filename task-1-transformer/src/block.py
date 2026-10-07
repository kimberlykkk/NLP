"""Transformer encoder block：attention + FFN + 两个 residual + 两个 LayerNorm。

本实现采用 Pre-LN（先 LayerNorm 再进子层）：
- Pre-LN 梯度更稳定，训练时对学习率不那么敏感，小数据上更容易收敛；
- 原论文《Attention Is All You Need》用的是 Post-LN（先子层后 LayerNorm）。
两者数学等价的写法是 x = x + SubLayer(Norm(x))，区别只是在哪做归一化。
"""
import torch.nn as nn

from .attention import MultiHeadAttention


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff=None, dropout=0.1,
                 use_residual=True, use_norm=True):
        """use_residual/use_norm 是消融实验（S2）用的开关：
        - use_residual=False：去掉两个残差连接（x = SubLayer(x)，不做加法）
        - use_norm=False：去掉两个 LayerNorm（层内用 Identity 占位）
        默认都保留，行为与原来完全一致。
        """
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.use_residual = use_residual
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model) if use_norm else nn.Identity()
        self.norm2 = nn.LayerNorm(d_model) if use_norm else nn.Identity()
        self.dropout = nn.Dropout(dropout)
        self.last_attn = None  # (B, H, T, T)，可视化用

    def forward(self, x, mask=None):
        """x: (B, T, d_model)；mask: (B, T) bool，True = pad（被屏蔽）。"""
        # 把 (B, T) 的 padding mask 扩成 (B, 1, 1, T)，广播到 (B, H, T, T) 的 scores 上
        attn_mask = mask.unsqueeze(1).unsqueeze(1) if mask is not None else None

        # ---- 子层 1：多头自注意力 + residual ----
        xn = self.norm1(x)
        a, attn = self.attn(xn, xn, xn, mask=attn_mask)
        self.last_attn = attn
        x = self.dropout(a) if not self.use_residual else x + self.dropout(a)

        # ---- 子层 2：前馈网络 + residual ----
        xn = self.norm2(x)
        f = self.dropout(self.ffn(xn))
        x = f if not self.use_residual else x + f
        return x
