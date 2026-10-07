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
