"""任务二 M2/M3：decoder-only MiniGPT（RoPE + KV cache）。

契约（与 eval/run.py 一致）：
- MiniGPT(ids) 输入 (B, T) -> 返回 (B, T, vocab_size) 的 logits
- forward(ids, kv_cache=None, return_cache=False)：
    kv_cache 非空时按增量解码（输入可以是 (B,1)），位置从历史长度续算；
    return_cache=True 返回 (logits, cache)，cache["layers"][i] = {"K","V"}、
    cache["seq_len"] = 已处理 token 总数
- block_size / max_seq_len：训练上下文长度，自检按它切窗算困惑度
- generate(prompt_ids, max_new_tokens, top_k, top_p, temperature)：自回归生成
- load_for_eval(ckpt_path) -> (model, tokenizer)
"""
import json
from pathlib import Path

import torch
import torch.nn as nn

from .block import GPTBlock


class MiniGPT(nn.Module):
    def __init__(self, vocab_size, d_model=128, n_heads=4, n_layers=4,
                 d_ff=512, max_seq_len=128, dropout=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.block_size = max_seq_len       # 自检读 block_size（其次是 max_seq_len）
        self.max_seq_len = max_seq_len
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([
            GPTBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, ids, kv_cache=None, return_cache=False):
        """ids: (B, T) -> logits (B, T, vocab)。
        kv_cache 为 None 时是全量前向；否则增量解码（输入通常 (B,1)）。"""
        B, T = ids.shape
        past_layers = kv_cache["layers"] if kv_cache is not None else None
        L0 = kv_cache["seq_len"] if kv_cache is not None else 0

        # 绝对位置：新 token 接着历史长度续算（RoPE 角度必须连续，否则和全量对不上）
        positions = torch.arange(L0, L0 + T, device=ids.device)

        x = self.tok_emb(ids)                      # (B, T, d_model)
        new_layers = []
        for i, blk in enumerate(self.blocks):
            pkv = past_layers[i] if past_layers is not None else None
            x, k, v = blk(x, positions, pkv)
            new_layers.append({"K": k, "V": v})    # 每层都存完整 K/V
        logits = self.head(self.ln_f(x))           # (B, T, vocab)

        if return_cache:
            return logits, {"layers": new_layers, "seq_len": L0 + T}
        return logits

    def generate(self, prompt_ids, max_new_tokens=100, top_k=0, top_p=0.0,
                 temperature=1.0):
        """prompt_ids: List[int] -> 返回 prompt + 新生成 token 的 id 列表。
        全程带 KV cache（体现 M3 的价值：每步只算 1 个 token）。"""
        from .sampling import sample_next

        self.eval()
        ids = list(prompt_ids)
        cache = None
        with torch.no_grad():
            x = torch.tensor([ids], dtype=torch.long, device=next(self.parameters()).device)
            logits, cache = self(x, kv_cache=cache, return_cache=True)
            for _ in range(max_new_tokens):
                nxt = sample_next(logits[:, -1, :], top_k, top_p, temperature)  # (1,1)
                ids.append(int(nxt.item()))
                x = nxt
                logits, cache = self(x, kv_cache=cache, return_cache=True)
        return ids


def load_for_eval(ckpt_path: str):
    """自检契约：从 ckpt 目录加载模型与 tokenizer。"""
    from .tokenizer import BPETokenizer

    ckpt_path = Path(ckpt_path)
    base = ckpt_path.parent
    cfg = json.loads((base / "config.json").read_text(encoding="utf-8"))
    tok = BPETokenizer.from_pretrained(str(base / "tokenizer.json"))
    model = MiniGPT(**cfg["model"])
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    model.eval()
    return model, tok
