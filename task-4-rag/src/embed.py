"""BGE 编码（transformers 手动池化：mean pooling + L2 normalize）。

sentence-transformers 5.2.3 在本机（transformers 5.2 + sm_120）批量编码会段错误，
因此统一走这条纯 transformers 路径，行为与 BGE 官方要求一致：
- query 侧由调用方加检索前缀（见 retriever.py）
- 文档侧不加前缀
"""
import torch
import torch.nn.functional as F
import torch.nn as nn


def encode_hf(model, tokenizer, texts, batch_size=64, max_length=512, device=None):
    device = device or next(model.parameters()).device
    outs = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            enc = tokenizer(texts[i:i + batch_size], padding=True, truncation=True,
                            max_length=max_length, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            out = model(**enc)[0]                              # (B, T, D)
            mask = enc["attention_mask"].unsqueeze(-1).float()
            pooled = (out * mask).sum(1) / mask.sum(1).clamp(min=1)
            outs.append(F.normalize(pooled, p=2, dim=-1).float().cpu())
    return torch.cat(outs).numpy()
