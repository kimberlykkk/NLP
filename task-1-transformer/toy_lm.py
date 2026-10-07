import random
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.attention import MultiHeadAttention
from src.model import PAD_ID, UNK_ID

ROOT = Path(__file__).parent
POETRY = ROOT.parent / "poetryFromTang.txt"

D_MODEL, N_HEADS, N_LAYERS, MAX_LEN = 64, 2, 2, 64
BATCH, STEPS, LR = 64, 400, 3e-4


class CausalLM(nn.Module):
    """极简 decoder-only：embedding + 多层因果自注意力 + FFN + 输出头。"""

    def __init__(self, vocab_size):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, D_MODEL, padding_idx=PAD_ID)
        self.pos = nn.Embedding(MAX_LEN, D_MODEL)
        self.blocks = nn.ModuleList([
            nn.ModuleDict({
                "attn": MultiHeadAttention(D_MODEL, N_HEADS),
                "ffn": nn.Sequential(nn.Linear(D_MODEL, 2 * D_MODEL), nn.GELU(),
                                     nn.Linear(2 * D_MODEL, D_MODEL)),
                "n1": nn.LayerNorm(D_MODEL), "n2": nn.LayerNorm(D_MODEL),
            }) for _ in range(N_LAYERS)
        ])
        self.head = nn.Linear(D_MODEL, vocab_size)

    def forward(self, ids):
        B, T = ids.shape
        causal = torch.triu(torch.ones(T, T, device=ids.device), diagonal=1).bool()  # True=屏蔽
        x = self.embed(ids) + self.pos(torch.arange(T, device=ids.device).unsqueeze(0))
        for blk in self.blocks:
            xn = blk["n1"](x)
            a, _ = blk["attn"](xn, xn, xn, mask=causal)
            x = x + a
            xn = blk["n2"](x)
            x = x + blk["ffn"](xn)
        return self.head(x)


def main():
    torch.manual_seed(0)
    text = Path(POETRY).read_text(encoding="utf-8")
    vocab = {"<pad>": PAD_ID, "<unk>": UNK_ID}
    for ch in text:
        if ch not in vocab and len(vocab) < 3000:
            vocab[ch] = len(vocab)
    ids = torch.tensor([vocab.get(c, UNK_ID) for c in text], dtype=torch.long)

    chunks = []
    for i in range(0, len(ids) - MAX_LEN, MAX_LEN):
        chunks.append(ids[i: i + MAX_LEN])

    model = CausalLM(len(vocab))
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    losses = []
    for step in range(STEPS):
        idx = random.randrange(len(chunks))
        x = chunks[idx].unsqueeze(0)
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, len(vocab)), x.view(-1))
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(loss.item())
        if step % 100 == 0:
            print(f"step {step:4d} | loss {loss.item():.4f}")

    # 采样：前 20 个字作为 prompt，temperature=0.8
    model.eval()
    prompt = "床前明月光"
    idx = [vocab.get(c, UNK_ID) for c in prompt]
    with torch.no_grad():
        for _ in range(30):
            x = torch.tensor([idx[-MAX_LEN:]], dtype=torch.long)
            logits = model(x)[0, -1] / 0.8
            logits = logits - logits.max()
            p = torch.softmax(logits, dim=-1)
            nxt = torch.multinomial(p, 1).item()
            if nxt in (PAD_ID, UNK_ID):
                break
            idx.append(nxt)
    id2ch = {v: k for k, v in vocab.items()}
    print("\n生成（toy，仅验证 causal mask 下的 next-token 学习）:")
    print("".join(id2ch.get(i, "?") for i in idx))


if __name__ == "__main__":
    main()
