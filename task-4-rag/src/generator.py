"""任务四 M4：生成器。

优先调用 OpenAI 兼容的本地模型服务（Ollama / vLLM / llama.cpp，通过环境变量配置）；
服务不可达时回退到任务三 SFT 的 Qwen2.5-0.5B（本地 transformers 加载）。

环境变量：
- RAG_LLM_BASE_URL（默认 http://localhost:11434/v1）
- RAG_LLM_API_KEY（默认 ollama）
- RAG_LLM_MODEL（默认 qwen2.5:7b-instruct）
"""
import os
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]

BASE_URL = os.environ.get("RAG_LLM_BASE_URL", "http://localhost:11434/v1")
API_KEY = os.environ.get("RAG_LLM_API_KEY", "ollama")
MODEL = os.environ.get("RAG_LLM_MODEL", "qwen2.5:7b-instruct")


class LocalGenerator:
    """OpenAI 兼容 HTTP 客户端（轻量，不依赖 openai 包也可用）。"""

    def __init__(self, base_url=BASE_URL, api_key=API_KEY, model=MODEL, timeout=120):
        self.base_url, self.model, self.timeout = base_url, model, timeout
        self.api_key = api_key

    def available(self):
        try:
            import requests
            r = requests.get(self.base_url + "/models", timeout=5,
                             headers={"Authorization": f"Bearer {self.api_key}"})
            return r.ok
        except Exception:
            return False

    def chat(self, prompt: str):
        import requests
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.35,
            "max_tokens": 640,
        }
        for attempt in range(3):
            try:
                r = requests.post(
                    self.base_url + "/chat/completions",
                    json=payload, timeout=self.timeout,
                    headers={"Authorization": f"Bearer {self.api_key}"})
                r.raise_for_status()
                return r.json()["choices"][0]["message"]["content"].strip()
            except Exception as e:
                if attempt == 2:
                    raise RuntimeError(f"LLM 服务调用失败：{type(e).__name__} {str(e)[:100]}")
                time.sleep(3)


class _LoRA:
    """最小 LoRA 包装（与任务三 src/lora 等价），用于本地加载 SFT 适配器。"""

    @staticmethod
    def load_sft_adapter(model, adapter_dir, r, alpha):
        import torch
        import torch.nn as nn
        target_names = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
        state = torch.load(Path(adapter_dir) / "adapter.bin", map_location="cpu")
        state = {k.replace("model.", "", 1): v for k, v in state.items()}
        with torch.no_grad():
            for name, module in model.named_modules():
                if not isinstance(module, nn.Linear):
                    continue
                short = name.split(".")[-1]
                if short not in target_names or f"{name}.lora_A" not in state:
                    continue
                A = state[f"{name}.lora_A"].to(module.weight.device, module.weight.dtype)
                B = state[f"{name}.lora_B"].to(module.weight.device, module.weight.dtype)
                module.weight.add_(B @ A * (alpha / r))
        return model


class SFTGenerator:
    """本地 transformers 回退：优先 Qwen2.5-0.5B-Instruct；缺失则用任务三 SFT 模型。"""

    def __init__(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        instruct_dir = ROOT / "models" / "Qwen2.5-0.5B-Instruct"
        third_dir = ROOT.parent / "task-3-sft-dpo"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        if (instruct_dir / "model.safetensors").exists():
            model_dir, adapter_dir = instruct_dir, None
        else:
            model_dir = third_dir / "models" / "Qwen2.5-0.5B"
            adapter_dir = third_dir / "ckpt" / "sft"
        self.tok = AutoTokenizer.from_pretrained(str(model_dir))
        self.model = AutoModelForCausalLM.from_pretrained(
            str(model_dir), torch_dtype=dtype).to(device)
        if adapter_dir is not None and (adapter_dir / "adapter.bin").exists():
            _LoRA.load_sft_adapter(self.model, adapter_dir, r=16, alpha=32)
        self.model.eval()
        self.name = "Qwen2.5-0.5B-Instruct" if adapter_dir is None else "task3-SFT-Qwen2.5-0.5B"

    def chat(self, prompt: str):
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        text = (f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n")
        ids = self.tok(text, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
        with torch.no_grad():
            out = self.model.generate(
                ids, max_new_tokens=320, do_sample=True, temperature=0.55, top_p=0.92,
                eos_token_id=[151643, 151645], pad_token_id=self.tok.eos_token_id)
        ans = self.tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
        return ans.split("<|im_end|>")[0].replace("<|im_start|>assistant\n", "").strip()


def get_generator():
    http = LocalGenerator()
    if http.available():
        print("使用 OpenAI 兼容本地服务：", BASE_URL, "| model:", MODEL)
        return http
    gen = SFTGenerator()
    print(f"本地 LLM 服务不可达，回退到本地 transformers 生成器：{gen.name}")
    return gen
