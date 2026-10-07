"""任务三 M2：Qwen chat template + loss masking。

- format_messages(messages) -> str：
  Qwen 官方模板 <|im_start|>{role}\\n{content}<|im_end|>\\n
- build_labels(input_ids, messages) -> labels：
  与 input_ids 同形状；-100 表示不参与 loss。规则：
  * template 控制符（<|im_start|> / <|im_end|> / 换行）与 role 标记 -> -100
  * user / system 的全部内容 -> -100
  * assistant 的**内容** -> 保留真实 token id（teacher forcing）
  （assistant 前面的 "<|im_start|>assistant\\n" 控制符仍为 -100）
"""
import re
from functools import lru_cache
from pathlib import Path

MODEL_DIR = Path(__file__).parent.parent / "models" / "Qwen2.5-0.5B"


@lru_cache(maxsize=1)
def _tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(str(MODEL_DIR))


def format_messages(messages):
    """把消息列表套成 Qwen 官方 chat 模板字符串。"""
    parts = []
    for m in messages:
        role = str(m.get("role", "user"))
        content = str(m.get("content", ""))
        parts.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
    return "".join(parts)


def _keep_spans(text, messages):
    """返回 [(start, end)] 保留为训练目标的字符区间。

    assistant 段：**开头到内容结束**都保留（含 `<|im_start|>assistant\n` 头部——
    否则小模型学不会「用户说完 -> 生成 assistant 头部」这个关键转换，生成会退化成
    复读换行/死循环）；user/system 段整体不保留。"""
    spans, pos = [], 0
    for m in messages:
        role = str(m.get("role", "user"))
        head = f"<|im_start|>{role}\n"
        body = f"{m.get('content', '')}<|im_end|>\n"
        if role == "assistant":
            spans.append((pos, pos + len(head) + len(m.get("content", ""))))
        pos += len(head) + len(body)
    return spans


def build_labels(input_ids, messages):
    """input_ids: (T,) 或 (1, T) 的 tensor / list；返回同形状 labels。"""
    tok = _tokenizer()
    text = format_messages(messages)
    if torch_is_tensor(input_ids):
        n = input_ids.shape[-1]
        shape = input_ids.shape
        ids = input_ids.reshape(-1).tolist()
    else:
        ids = list(input_ids)
        n, shape = len(ids), (len(ids),)

    # 方式一（首选）：用与 input_ids 相同的 tokenization 拿字符 offset，
    # 标记「完全落在保留区间内」的 token。
    labels = None
    try:
        enc = tok(text, return_offsets_mapping=True)
        enc_ids = enc["input_ids"]
        if len(enc_ids) == n:
            spans = _keep_spans(text, messages)
            lab = []
            for tid, (s, e) in zip(enc_ids, enc["offset_mapping"]):
                keep = any(s >= a and e <= b and e > s for a, b in spans)
                lab.append(tid if keep else -100)
            labels = lab
    except Exception:
        labels = None

    # 方式二（回退）：前缀法
    if labels is None:
        keep_ranges = _prefix_token_ranges(text, messages, tok)
        labels = [-100] * n
        for start, end in keep_ranges:
            labels[start:end] = ids[start:end]

    if torch_is_tensor(input_ids):
        out = torch_tensor_like(input_ids, labels)
        return out
    return list(labels)


def _prefix_token_ranges(text, messages, tok):
    """返回 [(token_start, token_end)]：用前缀 token 化定位保留区间。"""
    ranges, pos = [], 0
    for m in messages:
        head = f"<|im_start|>{m.get('role', 'user')}\n"
        body = f"{m.get('content', '')}<|im_end|>\n"
        if str(m.get("role", "")) == "assistant":
            n_start = len(tok(text[:pos], add_special_tokens=False)["input_ids"])
            n_end = len(tok(text[:pos + len(head) + len(m.get("content", ""))],
                            add_special_tokens=False)["input_ids"])
            ranges.append((n_start, n_end))
        pos += len(head) + len(body)
    return ranges


def format_and_mask(messages, max_len=384):
    """训练侧一行式：messages -> (input_ids, labels)，已截断到 max_len 个 token。
    保留范围：assistant 头部 + 内容；其余（user/system/模板符）全部 -100。"""
    tok = _tokenizer()
    text = format_messages(messages)
    try:
        enc = tok(text, add_special_tokens=False, truncation=True, max_length=max_len,
                  return_offsets_mapping=True)
        ids, offs = enc["input_ids"], enc["offset_mapping"]
        spans = _keep_spans(text, messages)
        labels = []
        for tid, (s, e) in zip(ids, offs):
            keep = any(s >= a and e <= b and e > s for a, b in spans)
            labels.append(tid if keep else -100)
        return ids, labels
    except Exception:
        ids = tok(text, add_special_tokens=False)["input_ids"][:max_len]
        ranges = _prefix_token_ranges(text, messages, tok)
        labels = [-100] * len(ids)
        for start, end in ranges:
            labels[start:end] = ids[start:end]
        return ids, labels


def torch_is_tensor(x):
    import torch
    return isinstance(x, torch.Tensor)


def torch_tensor_like(ref, values):
    import torch
    return torch.tensor(values, dtype=ref.dtype if hasattr(ref, "dtype") else torch.long).view(ref.shape)
