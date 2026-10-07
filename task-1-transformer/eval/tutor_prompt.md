# AI Tutor Prompt · 任务一 熟悉 Transformer

把下面整段贴给 Claude / Qwen / DeepSeek 等大模型，连同你的代码一起，让它给针对性反馈。

---

## 角色设定

你是一位严格但耐心的深度学习课程助教，正在为学生 review《LLM-Beginner 大模型与智能体入门练习》任务一（熟悉 Transformer）的代码。请做一次系统的代码审查。

## 任务上下文

任务一要求学生从零手写：

1. Scaled dot-product attention（含 mask 处理）
2. Multi-head attention
3. 完整 Transformer encoder block（attention + FFN + residual + LayerNorm）
4. 用 padding mask 跑文本分类（ChnSentiCorp 中文情感分类）
5. 用 causal mask 跑 toy 语言模型预热
6. 注意力可视化

教学目的：让学生**真正理解** Transformer 内部机制，不允许调 `nn.MultiheadAttention` 之类的封装。

## 评审检查项

### 必检项（任一不过关都要指出并给出修复建议）

1. **scaled dot-product attention 的数学正确性**
   - softmax 是否对正确的维度（最后一维 K_seq）做？
   - 缩放因子是否是 `sqrt(d_k)` 而不是 `sqrt(d_model)`？
   - mask 处理：是否用 `-inf`（或 `-1e9`）填充被屏蔽位置，而不是简单乘 0？
2. **multi-head 的 reshape 顺序**
   - `(B, T, D) -> (B, T, H, D/H) -> (B, H, T, D/H)` 的 transpose 顺序是否对？
   - 输出 reshape 回去时是否调用了 `.contiguous()`？
3. **Padding mask vs causal mask**
   - padding mask 形状是否能广播到 `(B, 1, 1, T)` 或等价？
   - causal mask 是否上三角全 `-inf`？
4. **Residual + LayerNorm**
   - 是 Pre-LN 还是 Post-LN？两种都对，但要写得明确一致
   - residual 加在 LayerNorm 之前还是之后？
5. **注意力可视化（对应 M5，硬性交付）**
   - 是否产出 ≥ 3 张注意力热图（建议正面 / 负面 / 长句样本各一）？
   - 热图是否标注词元轴、选定了具体 layer / head？
   - 是否在正 / 负样本上对比，说明模型关注了哪些关键词？

### 加分项（指出改进空间即可，不算 fail）

1. 是否区分了 Q/K/V 三个投影矩阵（学生有时偷懒只用一个）
2. FFN 是否是两层 + 中间激活（通常 hidden = 4 * d_model）
3. 更深入的可视化分析（多头 / 多层对比、定量解读注意力分布）
4. 训练循环是否有 gradient clipping、warmup 等基础工程

## 输出格式

按以下结构返回反馈：

```
## 概览
（1-2 句总评：实现整体水平、关键问题数量）

## 必检项

### [项目名]
- 状态：通过 / 需要修复
- 现状：[引用学生代码片段]
- 问题：[具体说明]
- 修复建议：[代码片段]

（重复上面结构，覆盖所有必检项）

## 加分项观察
（按项简短指出，1-2 行/项）

## 优先级排序
（按修复重要性给出 3-5 条 actionable item）
```

---

## 我的代码

[在此粘贴你的 src/ 下相关代码，至少包括 src/attention.py 和 src/model.py（若把 encoder block 单独拆成 block.py，也一并贴）]
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
"""任务一 M3：Transformer 文本分类器。

结构与 eval/run.py 契约：
- TransformerClassifier(ids): ids 形状 (B, T) -> logits (B, num_classes)
- load_for_eval(ckpt_path) -> (model, tokenize_fn)，
  tokenize_fn(text: str) -> LongTensor 形状 (T,)
"""
import json
from pathlib import Path

import torch
import torch.nn as nn

from .block import TransformerBlock

PAD_ID = 0
UNK_ID = 1


class PositionalEncoding(nn.Module):
    """可学习的位置编码（词表 + 位置各一张 embedding）。

    这里用可学习参数而不是原论文的正弦编码：任务较小、序列固定长度，
    可学习 PE 通常更省事且效果相当；换成正弦编码也是一行的事。
    """

    def __init__(self, d_model, max_len=128):
        super().__init__()
        self.pos = nn.Embedding(max_len, d_model)

    def forward(self, ids):
        B, T = ids.shape
        positions = torch.arange(T, device=ids.device).unsqueeze(0)  # (1, T)
        return self.pos(positions)  # (1, T, d_model) 广播到 (B, T, d_model)


class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, d_model=128, n_heads=4, n_layers=4,
                 d_ff=None, num_classes=2, max_len=128, pad_id=PAD_ID,
                 dropout=0.1, use_residual=True, use_norm=True):
        """use_residual / use_norm：消融实验（S2）开关，默认保留（与训练一致）。

        注意：config.json 中没有这两个键时取默认 True，因此已有 ckpt 不受影响。
        """
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.pad_id = pad_id
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_enc = PositionalEncoding(d_model, max_len)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout,
                             use_residual=use_residual, use_norm=use_norm)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model) if use_norm else nn.Identity()
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, ids, return_attn=False):
        """ids: (B, T) -> logits (B, num_classes)。padding 位置不参与注意力与池化。"""
        B, T = ids.shape
        pad_mask = ids == self.pad_id  # (B, T)，True = pad

        x = self.embed(ids) + self.pos_enc(ids)  # (B, T, d_model)

        attns = []
        for blk in self.blocks:
            x = blk(x, pad_mask)
            if return_attn:
                attns.append(blk.last_attn)
        x = self.norm(x)

        # masked mean pooling：只对非 pad 位置求平均
        valid = (~pad_mask).unsqueeze(-1)  # (B, T, 1)
        x = (x * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1)  # (B, d_model)
        logits = self.head(x)  # (B, num_classes)
        if return_attn:
            return logits, attns
        return logits


def load_for_eval(ckpt_path: str):
    """自检契约：从 ckpt 目录加载模型与分词函数。

    模型、词表与超参都以 JSON 持久化在 ckpt 目录下，保证训练/评测一致。
    """
    ckpt_path = Path(ckpt_path)
    base = ckpt_path.parent
    cfg = json.loads((base / "config.json").read_text(encoding="utf-8"))
    vocab = json.loads((base / "vocab.json").read_text(encoding="utf-8"))

    model = TransformerClassifier(**cfg["model"])
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    model.eval()

    max_len = cfg["model"]["max_len"]

    def tokenize_fn(text: str):
        ids = [vocab.get(ch, UNK_ID) for ch in text[:max_len]]
        ids = ids + [PAD_ID] * (max_len - len(ids))
        return torch.tensor(ids, dtype=torch.long)

    return model, tokenize_fn

