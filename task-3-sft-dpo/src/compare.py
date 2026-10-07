"""任务三 M4：base / SFT / DPO 三模型同指令对比。

用法（在 task-3-sft-dpo 目录下）：python src/compare.py
产出：results/compare.md（打印 + 落盘，供提交）
"""
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # 任务目录进 path，兼容 python src/compare.py

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.chat import format_messages
from src.lora import load_lora

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models" / "Qwen2.5-0.5B"
CKPT = ROOT / "ckpt"
ALL_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj",
               "gate_proj", "up_proj", "down_proj"]

PROMPTS = [
    "用一句话介绍你自己。",
    "写一首关于春天的五言绝句。",
    "什么是反向传播？请简要解释。",
    "把「我爱深度学习」翻译成英文。",
]


def load_variant(name):
    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_DIR), torch_dtype=torch.bfloat16).to("cuda")
    if name != "base":
        lora_dir = CKPT / name
        if not (lora_dir / "adapter.bin").exists():
            print(f"[跳过] {name}：{lora_dir} 不存在")
            return None
        load_lora(model, lora_dir, target_modules=ALL_TARGETS)
    model.eval()
    return model


def chat(model, tok, prompt, max_new=96):
    text = format_messages([{"role": "user", "content": prompt}])
    enc = tok(text, add_special_tokens=False, return_tensors="pt")
    ids = enc.input_ids.to("cuda")
    attn = torch.ones_like(ids)
    with torch.no_grad():
        out = model.generate(
            ids, max_new_tokens=max_new, do_sample=True, temperature=0.7,
            top_p=0.9, attention_mask=attn,
            pad_token_id=tok.eos_token_id, eos_token_id=tok.eos_token_id,
            repetition_penalty=1.05)
    text = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()
    return text.replace("<|im_start|>assistant\n", "").strip()


def main():
    tok = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    models = {n: m for n, m in
              ((n, load_variant(n)) for n in ("base", "sft", "dpo")) if m is not None}
    lines = ["# base / SFT / DPO 对比样例\n"]
    for prompt in PROMPTS:
        lines.append(f"\n## Q：{prompt}\n")
        print(f"===== Q：{prompt} =====")
        for name in models:
            t0 = time.time()
            ans = chat(models[name], tok, prompt)
            lines.append(f"- **{name}**：{ans}\n")
            print(f"[{name}] {ans}")
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "compare.md").write_text("".join(lines), encoding="utf-8")
    print(f"已保存 results/compare.md")


if __name__ == "__main__":
    main()
