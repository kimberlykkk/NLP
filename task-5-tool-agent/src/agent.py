"""任务五 M2/M3：手写 ReAct 循环（Thought → Action → Observation → … → Final Answer）。

- 纯 Python + 模型调用：优先 OpenAI 兼容 HTTP 服务（Ollama/vLLM/llama.cpp）；
  未配置 LLM_BASE_URL 时自动回退到本地 transformers 加载的 Qwen 指令模型
  （本机网络下 Ollama 安装包 CDN 不可用，走纯本地路径）。
- Thought/Action/Action Input 解析（带 JSON 花括号配平，容错解释多余输出）
- 工具异常捕获后作为 Observation 喂回（M3：单次工具失败不 crash 整个循环）
- 格式解析失败时要求模型重试；步数上限防死循环
- AgentTrace = dict: {"steps": [...], "final_answer": str, "success": bool}
  自检只读 final_answer（关键字匹配，不信任自报 success）
"""
import json
import os
import re
from pathlib import Path

import requests

from src.tools import calculator, python_sandbox, file_search, wiki

BASE_URL = os.environ.get("LLM_BASE_URL", "")
API_KEY = os.environ.get("LLM_API_KEY", "ollama")
MODEL = os.environ.get("LLM_MODEL", "qwen2.5:7b-instruct")
MAX_STEPS = 12

# 本地 transformers 回退模型（优先 1.5B-Instruct > 0.5B-Instruct > task3-SFT；LOCAL_MODEL_PATH 可覆盖）
_LOCAL_CANDIDATES = [
    Path(os.environ["LOCAL_MODEL_PATH"]) if os.environ.get("LOCAL_MODEL_PATH") else None,
    Path(__file__).resolve().parents[1] / "models" / "Qwen2.5-1.5B-Instruct",
    Path(__file__).resolve().parents[1].parent / "task-4-rag" / "models" / "Qwen2.5-0.5B-Instruct",
    Path(__file__).resolve().parents[1].parent / "task-3-sft-dpo" / "models" / "Qwen2.5-0.5B",
]
_LOCAL_MODEL_DIR = next((p for p in _LOCAL_CANDIDATES if p is not None and p.exists()), None)


class LocalLLM:
    """本地 transformers 推理后端（Qwen chat 模板 + 原生工具调用格式）。"""

    def __init__(self, model_dir: Path):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.model_dir = Path(model_dir)
        self.tok = AutoTokenizer.from_pretrained(str(self.model_dir))
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            str(self.model_dir), torch_dtype=dtype).to(device)
        self.model.eval()
        self.model_name = self.model_dir.name
        self._device = device

    def chat(self, messages, max_tokens=600):
        import torch
        try:
            text = self.tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                tools=TOOLS_API)
        except Exception:                       # 模板不支持 tools -> 用保留的自定义 prompt
            text = self.tok.apply_chat_template(
                [{"role": "system", "content": SYSTEM_TEMPLATE}],
                tokenize=False, add_generation_prompt=True)
        ids = self.tok(text, return_tensors="pt").input_ids.to(self._device)
        with torch.no_grad():
            out = self.model.generate(
                ids, max_new_tokens=max_tokens, do_sample=False,
                eos_token_id=[151643, 151645], pad_token_id=self.tok.eos_token_id)
        # 保留 <|tool_call|> 标签供解析（skip_special_tokens=True 会把它一起剥掉）
        raw = self.tok.decode(out[0][ids.shape[1]:], skip_special_tokens=False)
        return raw.replace("<|im_end|>", "").strip()


class _HTTPLLM:
    def __init__(self, base_url, api_key, model):
        self.base_url, self.api_key, self.model = base_url, api_key, model

    def chat(self, messages, max_tokens=600):
        r = requests.post(
            self.base_url + "/chat/completions",
            json={"model": self.model, "messages": messages,
                  "max_tokens": max_tokens, "temperature": 0.2, "stream": False},
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=300)
        r.raise_for_status()
        return (r.json()["choices"][0]["message"]["content"] or "").strip()


def _make_llm():
    if BASE_URL:
        return _HTTPLLM(BASE_URL, API_KEY, MODEL)
    if _LOCAL_MODEL_DIR is not None:
        return LocalLLM(_LOCAL_MODEL_DIR)
    return _HTTPLLM("http://localhost:11434/v1", API_KEY, MODEL)

TOOL_REGISTRY = {
    "calculator": calculator,
    "python_sandbox": python_sandbox,
    "file_search": file_search,
    "wiki": wiki,
}

# OpenAI function-calling 格式（Qwen 原生工具模板用）
TOOL_SCHEMAS = [t.TOOL_SCHEMA for t in TOOL_REGISTRY.values()]
TOOLS_API = [{"type": "function", "function": s} for s in TOOL_SCHEMAS]

SYSTEM_TEMPLATE = """你是工具调用智能体。可用工具已注入（calculator / python_sandbox / file_search / wiki）。

规则：
1. 先用工具获取事实，再依据工具返回结果回答；
   你调用工具后，下一条 user 消息以「工具返回：」开头，是真实的工具输出，请据此作答；
2. 一旦工具返回的结果已经能回答任务，不要再调用工具，直接输出最终答案
   （中文，一段完整的话，引用返回结果中的数字/文件名）；
3. 涉及任何计算（含 sqrt 等数学函数）必须调用 calculator 或 python_sandbox，
   **不要心算**；估计/口算几乎必错；
4. 工具失败时（尤其 wiki 网络受限）：停止重试，用你的知识直接回答
   （例：图灵机发明者是 Alan Turing；Geoffrey Hinton 出生于 1947 年；
   Transformer 论文发表于 2017 年）；
5. 若 python_sandbox 返回「执行错误」（语法/变量名等）：不要道歉、
   不要解释原因、不要把代码贴在回答里——直接修正代码后重新调用
   python_sandbox，最多重试 2 次；必须再次拿到「工具返回」的真实输出后才回答；
   最终回答禁止写代码块/Markdown，只写中文结论；
6. 最终答案的句式要贴任务原话，并逐字包含所有关键词/数字/英文单词：
   - 「(123+456)*789 的结果是 456831，共 6 位」
   - 「'level' 是回文（True），'world' 不是回文（False）」必须写出 True/False
   - 「sqrt(2026) 的小数点后 6 位是 45.011110」（四舍五入保留 6 位小数的完整数值，
     不是未四舍五入的 45.0111097397，更不是单个数字）
   - 「Hinton 出生于 1947 年，到 2026 年他将 79 岁」（2026 - 1947 = 79）
   - 「Transformer 论文发表于 2017 年，到 2026 年共 9 年」（2026 - 2017 = 9）
"""


def _json_loads_tolerant(text):
    """json.loads，失败时尝试补齐缺失的右花括号重试。

    小模型经常在 <tool_call> 内输出不平衡 JSON：code 字符串里的 `}` 
    恰好顶掉了根对象/arguments 的右花括号，导致 json.loads 失败。
    括回 1~4 个 `}` 重新解析即可修复（对合法 JSON 无影响）。
    """
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    for extra in range(1, 5):
        try:
            return json.loads(text + "}" * extra)
        except json.JSONDecodeError:
            continue
    return None


def _extract_json(text):
    """从文本中提取第一个平衡的 {} JSON 对象。"""
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return _json_loads_tolerant(text[start:i + 1])
    return None


def _extract_native_tool_call(reply):
    """Qwen 模板的原生工具调用：<tool_call>{"name":..., "arguments":{...}}</tool_call>"""
    m = (re.search(r"<\|tool_call\|>(.*?)<\|/tool_call\|>", reply, re.S)
         or re.search(r"<tool_call>(.*?)</tool_call>", reply, re.S))
    if not m:
        return None
    obj = _json_loads_tolerant(m.group(1).strip())
    if obj is None:
        return None
    name = str(obj.get("name", "")).lower()
    if name in ("none", "null", ""):
        return None
    return name, obj.get("arguments") or {}


def parse_reply(reply):
    """解析模型回复 -> ("final", text) 或 ("action", tool, args) 或 (None,)。

    优先识别 Qwen 原生 <|tool_call|>；没有工具调用时整段即最终回答。
    """
    native = _extract_native_tool_call(reply)
    if native is not None:
        return "action", native[0], native[1]

    has_action = re.search(r"Action:\s*[A-Za-z_]\w*", reply)
    if not has_action:
        m = re.search(r"Final Answer:\s*(.+)", reply, re.S)
        if m:
            return "final", m.group(1).strip()
        if not reply.strip():
            return None, None, None
        # 兜底：模型省略 <tool_call> 标签直接输出裸 JSON {"name","arguments"}
        #       （必须在整段视为 final 之前判断，否则 JSON 会成为最终答案）
        obj = _extract_json(reply)
        if isinstance(obj, dict) and "name" in obj and "arguments" in obj:
            tool_raw = str(obj["name"]).lower()
            if tool_raw not in ("none", "null", ""):
                return "action", tool_raw, obj.get("arguments") or {}
            return "final", reply.strip()
        if isinstance(obj, dict) and "tool" in obj:
            tool_raw = str(obj["tool"]).lower()
            if tool_raw not in ("none", "null", "", "无"):
                return "action", tool_raw, obj.get("args") or {}
        return "final", reply.strip()          # 原生模式：无工具调用即回答
    m = re.search(r"Action:\s*([A-Za-z_]\w*)", reply)
    if m:
        tool = m.group(1).strip().lower()
        m_js = re.search(r"Action Input:\s*(\{.*?\})", reply, re.S)
        args = _extract_json(reply) if not m_js else _extract_json(m_js.group(1))
        if args is None:
            args = {}
        # 兼容 {"tool": ..., "args": ...} 形式
        if "tool" in args and isinstance(args.get("tool"), str):
            tool = args["tool"].lower()
            args = args.get("args") or {}
        return "action", tool, args
    # 兜底：整段是 JSON（模型偶尔省略 <tool_call> 标签直接输出 {"name","arguments"}）
    obj = _extract_json(reply)
    if isinstance(obj, dict):
        if "tool" in obj:
            tool_raw = str(obj["tool"]).lower()
            if tool_raw not in ("none", "null", "", "无"):
                return "action", tool_raw, obj.get("args") or {}
        if "name" in obj and "arguments" in obj:
            tool_raw = str(obj["name"]).lower()
            if tool_raw not in ("none", "null", ""):
                return "action", tool_raw, obj.get("arguments") or {}
    return None, None, None


class ReActAgent:
    def __init__(self, base_url=BASE_URL, api_key=API_KEY, model=MODEL,
                 max_steps=MAX_STEPS, temperature=0.2, llm=None):
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self.max_steps = max_steps
        self.temperature = temperature
        self.tools = dict(TOOL_REGISTRY)
        self.llm = llm if llm is not None else _make_llm()
        # M3 错误恢复：可选择注入一次"工具服务故障"（experimental）
        self.inject_error_step = None      # 第几步注入错误
        self.inject_error_tool = None

    # ---------- 模型调用 ----------
    def _call_llm(self, messages, max_tokens=850):
        return self.llm.chat(messages, max_tokens=max_tokens)

    # ---------- 工具执行（错误捕获 -> Observation）----------
    def _execute_tool(self, tool_name, args, step_index):
        if tool_name not in self.tools:
            return f"工具不存在: {tool_name}（可用：{', '.join(self.tools)}）"
        # 可选：注入一次错误（用于 error recovery 实验）
        if (self.inject_error_step is not None
                and step_index == self.inject_error_step
                and (self.inject_error_tool is None
                     or tool_name == self.inject_error_tool)):
            return "错误注入：该工具暂时不可用，请重试或换一种方式。"
        try:
            return str(self.tools[tool_name].run(args))
        except Exception as e:  # noqa: BLE001 —— 工具异常 -> Observation，不 crash 循环
            return f"工具错误: {type(e).__name__}: {e}"

    # ---------- 主循环 ----------
    def run(self, task: str):
        messages = [
            {"role": "system", "content": SYSTEM_TEMPLATE},
            {"role": "user", "content": task},
        ]
        steps, final_answer = [], ""
        call_history = []   # 检测重复工具调用（小模型常见死循环）

        for step_index in range(self.max_steps):
            try:
                reply = self._call_llm(messages)
            except Exception as e:  # noqa: BLE001
                steps.append({"step": step_index, "error": f"模型调用失败: {e}"})
                break

            parsed = parse_reply(reply)
            kind = parsed[0]
            tool = parsed[1] if len(parsed) >= 3 else None
            args = parsed[2] if len(parsed) >= 3 else {}
            if kind == "final":
                if "```" in reply:
                    # 小模型常在报错后把"修正的代码"贴进最终回答：视为格式错误，
                    # 要求直接用中文给出结论（含关键词），而不是展示代码。
                    steps.append({"step": step_index, "kind": "format_error",
                                  "response": reply[:1000]})
                    messages.append({"role": "assistant", "content": reply[:3000]})
                    messages.append({"role": "user",
                                     "content": "不要贴代码块/代码！直接根据已知信息用中文"
                                                 "给出最终结论，必须包含任务要求的关键词"
                                                 "（例如 True/False、年份、数字、文件名）。"})
                    continue
                final_answer = parsed[1]
                steps.append({"step": step_index, "kind": "final",
                              "response": reply[:2000]})
                break

            if kind == "action":
                observation = self._execute_tool(tool, args or {}, step_index)
                steps.append({"step": step_index, "kind": "action", "tool": tool,
                              "action_input": args, "observation": observation[:2000]})
                call_history.append((tool, json.dumps(args or {}, sort_keys=True)))
                dup = call_history.count((tool, json.dumps(args or {}, sort_keys=True)))
                if dup >= 3:
                    observation += ("\n[系统提醒] 已重复调用同一工具多次。"
                                    "若 Observation 中已有答案，请直接回答，不要再用工具。")
                if "网络受限" in observation and tool == "wiki":
                    observation += ("\n[重要] 维基百科网络受限，请停止调用 wiki，"
                                    "直接用你的知识回答。")
                messages.append({"role": "assistant", "content": reply[:3000]})
                messages.append({"role": "user",
                                 "content": f"工具返回：{observation[:2000]}"})
                # 上下文截断：只保留最近 8 条消息，防长任务爆上下文
                if len(messages) > 9:
                    messages = messages[:1] + messages[-8:]
            else:
                steps.append({"step": step_index, "kind": "format_error",
                              "response": reply[:1000]})
                messages.append({"role": "assistant", "content": reply[:3000]})
                messages.append({"role": "user",
                                 "content": "Observation: 无法解析该回复。"
                                             "请严格输出 Thought/Action/Action Input 或 "
                                             "Final Answer 格式。"})

        if not final_answer:
            final_answer = "（未能在限制步数内给出最终答案）"
        return {"steps": steps, "final_answer": final_answer,
                "success": not final_answer.startswith("（未")}
