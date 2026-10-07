"""任务三 M1：手写 LoRA 低秩注入（不用 peft）。

数学：对原线性层 W，训练时在旁路加低秩增量
    h = W x + (alpha / r) * B (A x)
其中 A: (r, in)、B: (out, r)，只有 A/B 可训练；W 冻结。
- A 用 kaiming 初始化、B 用零初始化（训练开始时增量 = 0，等价于原模型）；
- scaling = alpha / r：换 rank 时不改变等效学习率；
- merge_lora：把 (alpha/r) * B A 加回 W，并移除 LoRA 分支（推理时零开销）。

自检按 inject_lora(model, target_modules=['q_proj','v_proj'], r=8, alpha=16)
调用并检查可训练参数占比 < 5%。
"""
import json
import math
from pathlib import Path

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """包裹一个 nn.Linear：冻结原权重，旁路低秩分支。"""

    def __init__(self, linear, r, alpha):
        super().__init__()
        assert isinstance(linear, nn.Linear)
        self.linear = linear
        self.r = int(r)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.r
        in_f, out_f = linear.in_features, linear.out_features
        # A/B 必须和原权重同 device/dtype（模型可能已被 .to(cuda)）
        device, dtype = linear.weight.device, linear.weight.dtype
        self.lora_A = nn.Parameter(torch.empty(self.r, in_f, device=device, dtype=dtype))    # (r, in)
        self.lora_B = nn.Parameter(torch.zeros(out_f, self.r, device=device, dtype=dtype))   # (out, r)
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        # 原权重冻结（bias 也冻结，保持语义）
        self.linear.weight.requires_grad_(False)
        if self.linear.bias is not None:
            self.linear.bias.requires_grad_(False)

    def forward(self, x):
        # x @ A^T  ->  (B,T,r) ; 再 @ B^T -> (B,T,out)
        lora = (x @ self.lora_A.t()) @ self.lora_B.t()
        return self.linear(x) + lora * self.scaling


def _inject(module, target_modules, r, alpha, prefix=""):
    for child_name, child in list(module.named_children()):
        full = f"{prefix}{child_name}"
        if child_name in target_modules and isinstance(child, nn.Linear):
            setattr(module, child_name, LoRALinear(child, r, alpha))
        else:
            _inject(child, target_modules, r, alpha, full + ".")


def inject_lora(model, target_modules, r, alpha):
    """把 target_modules 中所有 nn.Linear 原地替换为 LoRALinear，并冻结除 LoRA 分支外的全部参数。

    标准 LoRA 语义：整个基座模型（W/bias/norm 等）全部冻结，只有 A/B 可训练；
    自检按可训练参数占比 < 5% 校验。返回同一个 model。
    """
    _inject(model, target_modules, r, alpha)
    for p in model.parameters():
        p.requires_grad_(False)
    for _, module in named_loras(model):
        module.lora_A.requires_grad_(True)
        module.lora_B.requires_grad_(True)
    return model


def _merge(module):
    for child_name, child in list(module.named_children()):
        if isinstance(child, LoRALinear):
            with torch.no_grad():
                w = child.lora_B @ child.lora_A        # (out, in)
                child.linear.weight.add_(w * child.scaling)
            setattr(module, child_name, child.linear)  # 换回普通 Linear
        else:
            _merge(child)


def merge_lora(model):
    """把 LoRA 增量合并进原权重并移除旁路。返回同一个 model（推理结果等价）。"""
    _merge(model)
    return model


# ---------- 适配器持久化（ckpt/sft、ckpt/dpo 的格式） ----------
def named_loras(model):
    """迭代 (模块名, LoRALinear)。"""
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            yield name, module


def save_lora(model, ckpt_dir, r, alpha, target_modules):
    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    state = {}
    for name, module in named_loras(model):
        state[f"{name}.lora_A"] = module.lora_A.detach().cpu().clone()
        state[f"{name}.lora_B"] = module.lora_B.detach().cpu().clone()
    torch.save(state, ckpt_dir / "adapter.bin")
    (ckpt_dir / "adapter_config.json").write_text(
        json.dumps({"r": r, "alpha": alpha, "target_modules": target_modules,
                    "num_params": sum(p.numel() for p in state.values()),
                    "num_layers": len(state) // 2},
                   ensure_ascii=False, indent=2),
        encoding="utf-8")
    return ckpt_dir


def load_lora(model, ckpt_dir, r=None, alpha=None, target_modules=None):
    """注入（若未注入）并加载适配器权重。r/alpha/target 缺省读 adapter_config.json。"""
    ckpt_dir = Path(ckpt_dir)
    cfg = json.loads((ckpt_dir / "adapter_config.json").read_text(encoding="utf-8"))
    r = r if r is not None else cfg["r"]
    alpha = alpha if alpha is not None else cfg["alpha"]
    target_modules = target_modules if target_modules is not None else cfg["target_modules"]
    if not any(isinstance(m, LoRALinear) for m in model.modules()):
        inject_lora(model, target_modules, r, alpha)
    state = torch.load(ckpt_dir / "adapter.bin", map_location="cpu")
    for name, module in named_loras(model):
        module.lora_A.data.copy_(state[f"{name}.lora_A"])
        module.lora_B.data.copy_(state[f"{name}.lora_B"])
    return model
