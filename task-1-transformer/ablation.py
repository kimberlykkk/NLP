"""S1/S2 消融实验（叠加在完整训练流程上，一键跑全部配置）。

- S1（结构消融）：只改 head 数或层数，其余超参不动
  heads ∈ {2, 8}（基线 4），layers ∈ {2, 6}（基线 4）
- S2（组件消融）：拆掉 residual 或 LayerNorm，记录训练是否还能收敛

每组配置都用固定 seed=42、同一份数据/词表/优化器/调度器（与 train.py 完全一致），
只改被测的那一个变量，保证对比公平。

用法：
    python ablation.py               # 跑全部 7 组（基线 + S1×4 + S2×2）
    python ablation.py --subset S1   # 只跑 S1
    python ablation.py --subset S2   # 只跑 S2
    python ablation.py --epochs 5    # 覆盖训练轮数

产出：
- results/ablation_summary.csv     汇总表（提交报告用）
- results/ablation_detail.json     每组每轮 val acc 明细
- figures/ablation.png             柱状对比图
- figures/ablation_curves.png      训练曲线对比图
"""
import argparse
import json
import math
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 中文标签需要 CJK 字体，否则显示为方块
from matplotlib import font_manager

_avail = {f.name for f in font_manager.fontManager.ttflist}
for _name in ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC"]:
    if _name in _avail:
        matplotlib.rcParams["font.sans-serif"] = [_name, "DejaVu Sans"]
        matplotlib.rcParams["axes.unicode_minus"] = False
        break

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# 复用 train.py 里的数据/词表/评测逻辑，保证与正式训练完全同源
from train import (set_seed, build_vocab, SentiDataset, evaluate,
                   DATA, MAX_LEN, MAX_VOCAB, BATCH, EPOCHS, LR,
                   WEIGHT_DECAY, WARMUP_STEPS, DROPOUT)
from src.model import TransformerClassifier

ROOT = Path(__file__).parent
RESULTS = ROOT / "results"
FIG = ROOT / "figures"

# ---- 消融配置 ----
# 每组：(name, 中文描述, 传给 TransformerClassifier 的 kwargs)
BASELINE = (dict(d_model=128, n_heads=4, n_layers=4, d_ff=512))
EXPERIMENTS = [
    # S1：只改 head 数
    ("S1-heads2", "heads=2（基线4）",  dict(d_model=128, n_heads=2, n_layers=4, d_ff=512)),
    ("S1-heads8", "heads=8（基线4）",  dict(d_model=128, n_heads=8, n_layers=4, d_ff=512)),
    # S1：只改层数
    ("S1-layers2", "layers=2（基线4）", dict(d_model=128, n_heads=4, n_layers=2, d_ff=512)),
    ("S1-layers6", "layers=6（基线4）", dict(d_model=128, n_heads=4, n_layers=6, d_ff=512)),
    # S2：拆组件
    ("S2-no-residual", "无残差连接",    dict(d_model=128, n_heads=4, n_layers=4, d_ff=512, use_residual=False)),
    ("S2-no-norm",     "无LayerNorm",  dict(d_model=128, n_heads=4, n_layers=4, d_ff=512, use_norm=False)),
]


def train_one(name, desc, model_kwargs, vocab, train_loader, val_loader,
              device, epochs, seed=42, verbose=True):
    """训练一组配置，返回汇总 dict（含每轮 val acc 历史）。"""
    set_seed(seed)  # 每组固定同一种子，只让被测变量变化
    model = TransformerClassifier(vocab_size=len(vocab), num_classes=2,
                                  max_len=MAX_LEN, dropout=DROPOUT,
                                  **model_kwargs).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    total_steps = epochs * len(train_loader)

    def lr_at(step):
        if step < WARMUP_STEPS:
            return step / WARMUP_STEPS
        t = (step - WARMUP_STEPS) / max(total_steps - WARMUP_STEPS, 1)
        return 0.5 * (1 + math.cos(math.pi * t))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    crit = nn.CrossEntropyLoss()

    best_acc, patience, diverged = 0.0, 0, False
    hist = {"train_loss": [], "train_acc": [], "val_acc": []}
    t0 = time.time()

    if verbose:
        print(f"\n===== [{name}] {desc} | 参数量 {n_params / 1e6:.2f}M =====")

    for epoch in range(1, epochs + 1):
        model.train()
        tot_loss, tot_acc, n = 0.0, 0, 0
        for ids, label in train_loader:
            ids, label = ids.to(device), label.to(device)
            opt.zero_grad()
            logits = model(ids)
            loss = crit(logits, label)
            if not math.isfinite(loss.item()):  # NaN/Inf：训练发散，提前终止
                diverged = True
                break
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot_loss += loss.item() * label.numel()
            tot_acc += (logits.argmax(-1) == label).sum().item()
            n += label.numel()

        if diverged:
            if verbose:
                print(f"  epoch {epoch:2d} | loss 发散 (NaN/Inf)，训练终止")
            break

        train_acc = tot_acc / n
        val_acc = evaluate(model, val_loader, device)
        hist["train_loss"].append(tot_loss / n)
        hist["train_acc"].append(train_acc)
        hist["val_acc"].append(val_acc)
        if verbose:
            print(f"  epoch {epoch:2d} | loss {tot_loss / n:.4f} | train acc {train_acc:.4f} | val acc {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc, patience = val_acc, 0
        else:
            patience += 1
            if patience >= 3:
                if verbose:
                    print("  early stop")
                break

    return {
        "name": name, "desc": desc,
        "params": n_params, "best_val_acc": best_acc,
        "final_train_acc": hist["train_acc"][-1] if hist["train_acc"] else float("nan"),
        "epochs_done": len(hist["val_acc"]), "diverged": diverged,
        "elapsed_s": time.time() - t0,
        "hist": hist,
    }


def plot_summary(df):
    """柱状图：每组 best val acc，标出达标线 0.80 与参考基线 0.85。"""
    names = df["name"].tolist()
    accs = df["best_val_acc"].tolist()
    colors = []
    for n in names:
        if n == "baseline":
            colors.append("#4C72B0")
        elif n.startswith("S2"):
            colors.append("#C44E52")   # 组件消融用红，突出失败/变差
        else:
            colors.append("#55A868")

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(names, accs, color=colors)
    ax.axhline(0.80, color="red", ls="--", lw=1.2, label="达标线 0.80")
    ax.axhline(0.85, color="gray", ls=":", lw=1.2, label="参考基线 0.85")
    for b, a in zip(bars, accs):
        ax.text(b.get_x() + b.get_width() / 2, a + 0.004,
                f"{a:.4f}", ha="center", fontsize=9)
    ax.set_ylabel("best dev accuracy")
    ax.set_title("S1/S2 消融对比（一切其余超参固定，seed=42）")
    ymin = min(accs) - 0.06
    ax.set_ylim(max(0.0, ymin), 1.0)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "ablation.png", dpi=150)
    plt.close(fig)


def plot_curves(results):
    """每组 val acc 随 epoch 的曲线。"""
    fig, ax = plt.subplots(figsize=(9, 5))
    for r in results:
        ys = r["hist"]["val_acc"]
        xs = list(range(1, len(ys) + 1))
        ax.plot(xs, ys, marker="o", ms=4, label=f"{r['name']} ({r['desc']})")
    ax.set_xlabel("epoch")
    ax.set_ylabel("val accuracy")
    ax.set_title("训练曲线：各配置的 val accuracy")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "ablation_curves.png", dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", choices=["all", "S1", "S2"], default="all",
                    help="只跑 S1 或 S2（默认全部）")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")

    train_df = pd.read_parquet(DATA / "train.parquet")
    val_df = pd.read_parquet(DATA / "validation.parquet")
    print(f"训练 {len(train_df)} 条 / 验证 {len(val_df)} 条，epochs={args.epochs}")

    vocab = build_vocab(train_df["text"])
    train_ds = SentiDataset(train_df["text"].tolist(), train_df["label"].tolist(), vocab, MAX_LEN)
    val_ds = SentiDataset(val_df["text"].tolist(), val_df["label"].tolist(), vocab, MAX_LEN)
    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)

    runs = [("baseline", "基线 H=4/L=4", BASELINE)]
    runs += [(n, d, k) for n, d, k in EXPERIMENTS
             if args.subset == "all" or n.startswith(args.subset)]

    results = []
    for name, desc, kw in runs:
        r = train_one(name, desc, kw, vocab, train_loader, val_loader,
                      device, epochs=args.epochs)
        results.append(r)
        print(f"[{name}] {desc}: best val acc = {r['best_val_acc']:.4f} "
              f"({r['epochs_done']} epochs, {r['elapsed_s']:.0f}s, "
              f"{r['params'] / 1e6:.2f}M params)")

    # 汇总表（去掉 hist 列）
    df = pd.DataFrame([{k: v for k, v in r.items() if k != "hist"} for r in results])
    df["params_M"] = (df["params"] / 1e6).round(2)
    RESULTS.mkdir(exist_ok=True)
    df.to_csv(RESULTS / "ablation_summary.csv", index=False, encoding="utf-8-sig")

    detail = {
        r["name"]: {"desc": r["desc"], "best_val_acc": r["best_val_acc"],
                    "params": r["params"], "epochs_done": r["epochs_done"],
                    "diverged": r["diverged"], "history": r["hist"]}
        for r in results
    }
    (RESULTS / "ablation_detail.json").write_text(
        json.dumps(detail, ensure_ascii=False, indent=2), encoding="utf-8")

    FIG.mkdir(exist_ok=True)
    plot_summary(df)
    plot_curves(results)

    print("\n===== 汇总 =====")
    print(df[["name", "desc", "params_M", "best_val_acc", "final_train_acc",
              "epochs_done", "diverged"]].to_string(index=False))
    print(f"\n已保存 results/ablation_summary.csv、results/ablation_detail.json")
    print(f"已保存 figures/ablation.png、figures/ablation_curves.png")


if __name__ == "__main__":
    main()
