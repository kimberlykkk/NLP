"""任务二 M5：生成样例（四种采样策略对比），写 results/generation.md。

用法（在 task-2-mini-gpt 目录下）：python demo_generate.py
"""
import sys
from pathlib import Path

assert hasattr(sys.stdout, "reconfigure")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")   # 防 Windows GBK 控制台崩

from src.model import load_for_eval

ROOT = Path(__file__).parent
CKPT = ROOT / "ckpt" / "best.pt"
MAX_NEW_TOKENS = 40

PROMPTS = ["床前明月光", "春眠不觉晓"]

STRATEGIES = [
    ("greedy（temperature=0）",    dict(top_k=0,  top_p=0.0,  temperature=0.0)),
    ("top-k=20, t=0.8",            dict(top_k=20, top_p=0.0,  temperature=0.8)),
    ("top-p=0.9, t=1.0",           dict(top_k=0,  top_p=0.9,  temperature=1.0)),
    ("top-p=0.95, t=1.2",          dict(top_k=0,  top_p=0.95, temperature=1.2)),
]


def main():
    model, tok = load_for_eval(str(CKPT))
    lines = [f"# mini-GPT 生成样例（vocab={tok.vocab_size}，唐诗 quick-start）\n"]

    def clean(text):
        # byte-level 模型偶尔会生成不成 UTF-8 的字节序列（decode 会出 \ufffd），
        # 仅展示时剔除；不影响评估（评估只用合法文本的 encode-decode）。
        return text.replace("\ufffd", "")

    for prompt in PROMPTS:
        lines.append(f"\n## 提示：{prompt}\n")
        print(f"===== 提示：{prompt} =====")
        for name, kw in STRATEGIES:
            ids = model.generate(tok.encode(prompt), max_new_tokens=MAX_NEW_TOKENS, **kw)
            text = clean(tok.decode(ids))
            print(f"[{name}] {text}")
            lines.append(f"- **{name}**：{text}\n")

    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    (out / "generation.md").write_text("".join(lines), encoding="utf-8")
    print(f"\n已保存 {out / 'generation.md'}")


if __name__ == "__main__":
    main()
