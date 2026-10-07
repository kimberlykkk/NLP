"""训练 Transformer 文本分类器（ChnSentiCorp 中文情感二分类）。

用法：python train.py   （在 task-1-transformer 目录下执行）

产出：
- ckpt/best.pt        训练好的 state_dict（自检契约要求）
- ckpt/config.json    模型超参（load_for_eval 依赖）
- ckpt/vocab.json     字符词表（load_for_eval 依赖）
- figures/loss_curve.png  训练/验证曲线（报告用）
"""
import json
import math
import random
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.model import PAD_ID, UNK_ID, TransformerClassifier

ROOT = Path(__file__).parent
DATA = ROOT / "data"
CKPT = ROOT / "ckpt"
FIG = ROOT / "figures"

# ---- 超参（README 建议起点，可自行实验）----
D_MODEL = 128
N_HEADS = 4
N_LAYERS = 4
D_FF = 512
MAX_LEN = 128
MAX_VOCAB = 8000
BATCH = 32
EPOCHS = 8
LR = 3e-4
WEIGHT_DECAY = 0.01
WARMUP_STEPS = 100
DROPOUT = 0.1


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_vocab(texts):
    """字符级词表：<pad>=0, <unk>=1, 其余按训练集出现频次排序。"""
    counter = Counter()
    for t in texts:
        counter.update(t)
    vocab = {"<pad>": PAD_ID, "<unk>": UNK_ID}
    for ch, _ in counter.most_common(MAX_VOCAB):
        vocab[ch] = len(vocab)
    return vocab


class SentiDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        ids = [self.vocab.get(c, UNK_ID) for c in self.texts[i][: self.max_len]]
        ids = ids + [PAD_ID] * (self.max_len - len(ids))
        return torch.tensor(ids, dtype=torch.long), torch.tensor(self.labels[i], dtype=torch.long)


def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for ids, label in loader:
            ids, label = ids.to(device), label.to(device)
            logits = model(ids)
            correct += (logits.argmax(-1) == label).sum().item()
            total += label.numel()
    return correct / max(total, 1)


def train():
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")

    train_df = pd.read_parquet(DATA / "train.parquet")
    val_df = pd.read_parquet(DATA / "validation.parquet")
    print(f"训练 {len(train_df)} 条 / 验证 {len(val_df)} 条")

    vocab = build_vocab(train_df["text"])
    print(f"词表大小: {len(vocab)}")

    model = TransformerClassifier(
        vocab_size=len(vocab),
        d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS, d_ff=D_FF,
        num_classes=2, max_len=MAX_LEN, dropout=DROPOUT,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"参数量: {n_params / 1e6:.2f}M")

    train_ds = SentiDataset(train_df["text"].tolist(), train_df["label"].tolist(), vocab, MAX_LEN)
    val_ds = SentiDataset(val_df["text"].tolist(), val_df["label"].tolist(), vocab, MAX_LEN)
    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    total_steps = EPOCHS * len(train_loader)

    def lr_at(step):
        # 前 WARMUP_STEPS 线性升温，之后余弦退火到 0
        if step < WARMUP_STEPS:
            return step / WARMUP_STEPS
        t = (step - WARMUP_STEPS) / max(total_steps - WARMUP_STEPS, 1)
        return 0.5 * (1 + math.cos(math.pi * t))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    crit = nn.CrossEntropyLoss()

    CKPT.mkdir(exist_ok=True)
    FIG.mkdir(exist_ok=True)

    history = {"train_loss": [], "train_acc": [], "val_acc": [], "lr": []}
    best_acc, best_state, patience = 0.0, None, 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        tot_loss, tot_acc, n = 0.0, 0, 0
        for ids, label in train_loader:
            ids, label = ids.to(device), label.to(device)
            opt.zero_grad()
            logits = model(ids)
            loss = crit(logits, label)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot_loss += loss.item() * label.numel()
            tot_acc += (logits.argmax(-1) == label).sum().item()
            n += label.numel()
        train_acc = tot_acc / n
        val_acc = evaluate(model, val_loader, device)
        cur_lr = sched.get_last_lr()[0]
        history["train_loss"].append(tot_loss / n)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["lr"].append(cur_lr)
        print(f"epoch {epoch:2d} | loss {tot_loss / n:.4f} | train acc {train_acc:.4f} | val acc {val_acc:.4f} | lr {cur_lr:.2e}")

        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= 3:
                print("early stop")
                break

    assert best_state is not None, "训练未完成任何有效 epoch"
    torch.save(best_state, CKPT / "best.pt")
    cfg = {
        "model": {
            "vocab_size": len(vocab), "d_model": D_MODEL, "n_heads": N_HEADS,
            "n_layers": N_LAYERS, "d_ff": D_FF, "num_classes": 2,
            "max_len": MAX_LEN, "pad_id": PAD_ID, "dropout": DROPOUT,
        }
    }
    (CKPT / "config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    (CKPT / "vocab.json").write_text(json.dumps(vocab, ensure_ascii=False), encoding="utf-8")
    (ROOT / "history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"best val acc = {best_acc:.4f} -> 已保存 ckpt/best.pt")

    # 训练曲线
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(history["train_loss"], label="train loss")
    ax[0].set_xlabel("epoch"); ax[0].set_ylabel("loss"); ax[0].legend(); ax[0].set_title("Loss")
    ax[1].plot(history["train_acc"], label="train acc")
    ax[1].plot(history["val_acc"], label="val acc")
    ax[1].set_xlabel("epoch"); ax[1].set_ylabel("accuracy"); ax[1].legend(); ax[1].set_title("Accuracy")
    fig.tight_layout()
    fig.savefig(FIG / "loss_curve.png", dpi=150)
    print(f"训练曲线已保存 figures/loss_curve.png")


if __name__ == "__main__":
    train()
