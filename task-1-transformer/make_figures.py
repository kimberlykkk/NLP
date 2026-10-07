"""M5：用训练好的模型生成注意力热图（>= 3 张）到 figures/。

挑样本：dev 集中的 1 个短正样本、1 个短负样本、1 个长句，
可视化某一层某个 head 的 (T, T) 注意力矩阵（T = 模型实际看到的词元数）。

注意：坐标轴只设数据范围内的刻度（每隔 step 个词元标一个），
否则 matplotlib 会把坐标范围撑到刻度范围、把热图挤成左上角一小块。
"""
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

from src.model import PAD_ID, load_for_eval

ROOT = Path(__file__).parent
CKPT = ROOT / "ckpt" / "best.pt"
FIG = ROOT / "figures"

# 可视化哪一层、哪个 head（层号 0 起）
LAYER, HEAD = 0, 0
PAD_CHAR = "·"  # padding 位置的占位符


def pick_samples():
    """返回 [(tag, text)]：短正样本、短负样本、最长句子。"""
    dev = pd.read_parquet(ROOT / "data" / "validation.parquet")
    dev["text"] = dev["text"].astype(str)

    def by_class(label):
        sub = dev[dev["label"] == label]
        sub = sub[sub["text"].str.len().between(10, 60)]
        return sub.iloc[0]

    longest = dev.sort_values("text", key=lambda s: s.str.len()).iloc[-1]
    return [
        ("positive", by_class(1)["text"]),
        ("negative", by_class(0)["text"]),
        ("longest", longest["text"]),
    ]


def main():
    FIG.mkdir(exist_ok=True)
    model, tokenize_fn = load_for_eval(str(CKPT))
    model.eval()
    max_len = 128  # 与训练一致；实际以模型接收的长度为准

    for tag, text in pick_samples():
        ids = tokenize_fn(text)
        with torch.no_grad():
            logits, attns = model(ids.unsqueeze(0), return_attn=True)
        pred = int(logits.argmax(-1).item())
        attn = attns[LAYER][0, HEAD]                    # (T, T)

        T = attn.shape[-1]
        n_real = min(len(text), T)
        tok_labels = list(text[:n_real]) + [PAD_CHAR] * (T - n_real)
        A = attn.cpu().numpy()

        step = max(1, T // 40)                          # 每 step 个词元标一个刻度
        tick_idx = list(range(0, T, step))
        tick_labels = [tok_labels[i] for i in tick_idx]

        fig, ax = plt.subplots(figsize=(11, 9))
        im = ax.imshow(A, cmap="viridis", aspect="auto", vmin=0.0, vmax=max(A.max(), 1e-9))
        ax.set_xticks(tick_idx, tick_labels, rotation=90, fontsize=8)
        ax.set_yticks(tick_idx, tick_labels, fontsize=8)
        ax.set_xlim(0, T - 1)
        ax.set_ylim(T - 1, 0)                           # 行 0 在顶部
        ax.set_xlabel("Key (被关注)"); ax.set_ylabel("Query (关注者)")
        ax.set_title(f"{tag} | layer {LAYER} head {HEAD} | pred={pred} | 实际词元数 {n_real}")
        fig.colorbar(im, fraction=0.046, pad=0.04)
        fig.tight_layout()

        out = FIG / f"attn_{tag}.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"已保存 {out}  (pred={pred}, 句子长度 {len(text)})")

    print("done")


if __name__ == "__main__":
    main()
