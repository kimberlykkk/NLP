"""任务二 M5：采样策略（greedy / top-k / top-p / temperature）。

约定：
- logits: (B, V)，返回 (B, 1) 的采样下标（LongTensor）
- temperature <= 0 -> greedy（argmax），避免除零
- top_k / top_p 传 0 表示不启用
- top-p：按概率降序累积，超过阈值 p 的尾部剪掉，**重新归一化**后采样；
  保留"刚越过阈值"的那一个候选（标准做法），否则 p 很小时会丢光。
"""
import torch


def sample_next(logits, top_k=0, top_p=0.0, temperature=1.0):
    if temperature <= 0:
        return logits.argmax(-1, keepdim=True)

    logits = logits / temperature
    if top_k and top_k > 0:
        k = min(int(top_k), logits.shape[-1])
        v, _ = logits.topk(k, dim=-1)
        logits[logits < v[..., -1:]] = float("-inf")

    probs = torch.softmax(logits, dim=-1)

    if top_p and 0.0 < top_p < 1.0:
        sorted_probs, sorted_idx = probs.sort(descending=True, dim=-1)
        cum = sorted_probs.cumsum(-1)
        # 删掉「越过后仍继续累积」的尾部：保留 cum 超过 p 的那一项
        remove = cum - sorted_probs > top_p
        sorted_probs = sorted_probs.masked_fill(remove, 0.0)
        renormalized = torch.zeros_like(probs)
        renormalized.scatter_(-1, sorted_idx, sorted_probs)
        probs = renormalized / renormalized.sum(-1, keepdim=True).clamp(min=1e-12)

    return torch.multinomial(probs, num_samples=1)
