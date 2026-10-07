"""最小复现：训练 50 步 -> CE -> save_lora/load_lora -> 再 CE。"""
import sys
import math
import random
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.chat import format_and_mask
from src.lora import inject_lora, save_lora, load_lora
import train_sft

torch.manual_seed(0)
device = "cuda"
tok = AutoTokenizer.from_pretrained("models/Qwen2.5-0.5B")
pad_id = tok.eos_token_id
ALL_TARGETS = train_sft.ALL_TARGETS

items = train_sft.load_instruction_data("data/moss-sft-10w", 200, 42)
samples = []
for msgs in items:
    msgs = train_sft.keep_recent_turns(msgs)
    try:
        ids, labels = format_and_mask(msgs, max_len=384)
    except Exception:
        continue
    if len(ids) >= 4 and any(l != -100 for l in labels):
        samples.append((ids, labels))
print("样本数:", len(samples))

model = AutoModelForCausalLM.from_pretrained("models/Qwen2.5-0.5B", torch_dtype=torch.bfloat16).to(device)

def ce_on(m, batch):
    tot, n = 0.0, 0
    for ids, labels in batch:
        x = torch.tensor([ids], device=device)
        with torch.no_grad():
            logits = m(x).logits.float()
        mask = torch.tensor(labels[1:], device=device) != -100
        loss = F.cross_entropy(logits[0, :-1], torch.tensor(ids[1:], device=device), reduction="none")
        tot += loss[mask].sum().item()
        n += int(mask.sum().item())
    return tot / n

model.eval()
print("训练前 CE:", round(ce_on(model, samples[:10]), 3))

inject_lora(model, ALL_TARGETS, 16, 32)
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-4)
for step in range(50):
    batch = random.sample(samples, 8)
    L = max(len(x[0]) for x in batch)
    ids = torch.full((8, L), pad_id, dtype=torch.long, device=device)
    labels = torch.full((8, L), -100, dtype=torch.long, device=device)
    for j, (ids_j, labels_j) in enumerate(batch):
        ids[j, :len(ids_j)] = torch.tensor(ids_j)
        labels[j, :len(labels_j)] = torch.tensor(labels_j)
    model.train()
    opt.zero_grad()
    logits = model(ids).logits
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100)
    if step in (0, 9, 49):
        # 同 batch 的 float 精度 CE（逐 token）
        mask2 = labels != -100
        loss_float = F.cross_entropy(
            logits.float().view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100)
        print(f"  [step {step+1}] bf16_loss={loss.item():.4f} | float_loss={loss_float.item():.4f} | "
              f"labels 中 -100 占比={(labels == -100).float().mean().item():.3f} | "
              f"ids==labels 占比={(ids == labels).float().mean().item():.3f} | "
              f"labels.max()={labels.max().item()} | labels.min()={labels.min().item()}")
        # 用 batch 张量逐行重算（与 batch loss 同源）
        tot2, n2 = 0.0, 0
        for j in range(8):
            lj = logits[j:j + 1].float()
            mj = (labels[j:j + 1] != -100)
            l2 = F.cross_entropy(lj[0, :-1].reshape(-1, lj.size(-1)),
                                 labels[j:j + 1][0, 1:].reshape(-1), reduction="none")
            m2 = mj[0, 1:].reshape(-1)
            tot2 += l2[m2].sum().item(); n2 += int(m2.sum().item())
        print(f"  [step {step+1}] 张量逐行重算 CE={tot2 / max(n2, 1):.4f}")
        # 无 padding 的同一批序列
        tot, n = 0.0, 0
        per_sample_ce = []
        for j in range(8):
            ids_j, labels_j = batch[j]
            n_keep = sum(1 for l in labels_j if l != -100)
            xj = torch.tensor([ids_j], device=device)
            with torch.no_grad():
                lj = model(xj).logits.float()
            mj = torch.tensor(labels_j[1:], device=device) != -100
            l = F.cross_entropy(lj[0, :-1], torch.tensor(ids_j[1:], device=device), reduction="none")
            ce_j = l[mj].mean().item() if mj.any() else float("nan")
            per_sample_ce.append(round(ce_j, 2))
            tot += l[mj].sum().item(); n += int(mj.sum().item())
            if j == 0:
                print(f"  [step {step+1}] 样本0 保留数={n_keep} | 前12 labels={labels_j[:12]}")
                print(f"  [step {step+1}] 样本0 ids 前12: {ids_j[:12]}")
        print(f"  [step {step+1}] 逐样本 CE={per_sample_ce} | 无 padding 合计={tot / max(n, 1):.4f}")
    loss.backward()
    gnorm = nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
    opt.step()
    if step in (0, 9, 49):
        bmax = max(m.lora_B.abs().max().item() for m in model.modules() if hasattr(m, "lora_B"))
        print(f"step {step+1}: loss={loss.item():.4f} | grad_norm={gnorm:.3f} | |B|max={bmax:.4f}")

model.eval()
print("训练后 CE(前10样本):", round(ce_on(model, samples[:10]), 3))

save_lora(model, "ckpt/test_lora", 16, 32, ALL_TARGETS)
model2 = AutoModelForCausalLM.from_pretrained("models/Qwen2.5-0.5B", torch_dtype=torch.bfloat16).to(device)
load_lora(model2, "ckpt/test_lora", target_modules=ALL_TARGETS)
model2.eval()
print("重载后 CE(前10样本):", round(ce_on(model2, samples[:10]), 3))
print("DONE")
