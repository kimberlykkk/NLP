"""任务六 M3/M4：CodingAgent（agentic loop）与 Trace。

- `CodingAgent.run(repo_path, issue) -> Trace`；Trace 是 dict，含：
  `steps`（每步 thought/tool/args/observation）、`patch`（unified diff）、
  `tests_passed: bool`（真实跑一次 pytest 得出，不信任自报）；
- 模型：Qwen2.5-Coder-3B-Instruct（本地 transformers bf16/CUDA；7B FP16 超 12GB 显存，
  量化 GGUF 需额外二进制，本机网络无法安装 —— 用 3B Coder 并记录偏差；
  设 `LLM_BASE_URL` 可切回 Ollama/vLLM 的 OpenAI 兼容端点）；
- 停机：测试全绿（run_tests 观察含 passed 且无 failed）或模型输出 final 显式声明完成；
- 上下文：超预算时压缩较早消息（保留 system + 最近若干条 + 摘要）；
- Subagent：「subagent_explore」「subagent_test」两个伪工具，独立 context/步数/工具子集。
"""
import json
import os
import re
import sys
from pathlib import Path

# 保证 `python src/agent.py` 与 `from src.agent import ...` 两条路径都能 import
_TASK6_ROOT = Path(__file__).resolve().parents[1]
if str(_TASK6_ROOT) not in sys.path:
    sys.path.insert(0, str(_TASK6_ROOT))

from src import mcp_server as mcp
from src.skill_loader import SkillLoader
from src.subagents import make_subagent

TASK6_ROOT = _TASK6_ROOT

BASE_URL = os.environ.get("LLM_BASE_URL", "")
API_KEY = os.environ.get("LLM_API_KEY", "ollama")
MODEL = os.environ.get("LLM_MODEL", "qwen2.5-coder:7b-instruct")

_LOCAL_CANDIDATES = [
    Path(os.environ["LOCAL_MODEL_PATH"]) if os.environ.get("LOCAL_MODEL_PATH") else None,
    TASK6_ROOT / "models" / "Qwen2.5-Coder-3B-Instruct",
    TASK6_ROOT / "models" / "Qwen2.5-Coder-1.5B-Instruct",
    TASK6_ROOT.parent / "task-5-tool-agent" / "models" / "Qwen2.5-1.5B-Instruct",
]


def _is_complete_model(p: Path) -> bool:
    """config.json + 至少一个 safetensors 分片才算完整（下载中途的目录不能加载）。"""
    return (p is not None and p.is_dir() and (p / "config.json").exists()
            and any(p.glob("*.safetensors")))


_LOCAL_MODEL_DIR = next((p for p in _LOCAL_CANDIDATES if _is_complete_model(p)), None)


class LocalLLM:
    """本地 transformers 推理后端（Qwen chat 模板，贪婪解码）。"""

    def __init__(self, model_dir: Path):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.model_dir = Path(model_dir)
        self.tok = AutoTokenizer.from_pretrained(str(self.model_dir))
        self.model = AutoModelForCausalLM.from_pretrained(
            str(self.model_dir), torch_dtype=torch.bfloat16).to("cuda")
        self.model.eval()
        self.model_name = self.model_dir.name

    def chat(self, messages, max_tokens=450):
        import torch
        text = self.tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        ids = self.tok(text, return_tensors="pt").input_ids.to("cuda")
        out = self.model.generate(
            ids, max_new_tokens=max_tokens, do_sample=False,
            eos_token_id=[151643, 151645], pad_token_id=self.tok.eos_token_id)
        raw = self.tok.decode(out[0][ids.shape[1]:], skip_special_tokens=False)
        return raw.replace("<|im_end|>", "").strip()


class _HTTPLLM:
    def __init__(self, base_url, api_key, model):
        self.base_url, self.api_key, self.model = base_url, api_key, model

    def chat(self, messages, max_tokens=450):
        import requests
        r = requests.post(
            self.base_url + "/chat/completions",
            json={"model": self.model, "messages": messages,
                  "max_tokens": max_tokens, "temperature": 0.2, "stream": False},
            headers={"Authorization": f"Bearer {self.api_key}"}, timeout=300)
        r.raise_for_status()
        return (r.json()["choices"][0]["message"]["content"] or "").strip()


def _make_llm():
    if BASE_URL:
        return _HTTPLLM(BASE_URL, API_KEY, MODEL)
    if _LOCAL_MODEL_DIR is not None:
        return LocalLLM(_LOCAL_MODEL_DIR)
    raise RuntimeError(
        f"未找到本地模型（候选：{[str(p) for p in _LOCAL_CANDIDATES if p]}）；"
        "请从 ModelScope 下载 Qwen2.5-Coder-3B-Instruct 到 task-6/models/，"
        "或设置 LLM_BASE_URL 指向 Ollama/vLLM 端点")


SYSTEM_TEMPLATE = """你是本地代码仓库的编码助手。目标：按 ISSUE 修改 repo 内代码，并让测试全绿。

输出格式（每步只输出**一个** JSON 对象，无其他文字）：
- 调用工具：{"thought": "简短推理", "tool": "read_file", "args": {"path": "calculator.py"}}
- 完成结论：{"thought": "简短推理", "final": "改了什么、测试结果如何"}

可用工具（args 里的路径一律用 repo 内相对路径）：
- read_file{path} / write_file{path,content} / list_dir{path} / search_files{pattern,path}
- run_tests{}（不带参数，直接跑整个 repo 的 pytest；测试文件 test_calculator.py 在 repo 根）
- git_diff{}
- load_skill{name}：加载 Skill 全文（test-runner / code-review / pr-description-writer），
  按需使用（如需要时读 test-runner 再跑测试）
- subagent_explore{task}：派「代码搜索」子代理摸清 repo 结构，返回 ≤300 字摘要
- subagent_test{task}：派「测试执行」子代理跑测试并解释，返回 ≤300 字摘要

规则：
1. 改代码前先 read_file 定位问题；**不要修改任何测试文件**，也不要去动 tests 目录；
2. write_file 时写回文件的**完整新版本**（先 read_file 全文，保留没改动的函数）；
3. 每次修改后调用 run_tests{} 验证；观察里出现 "passed" 且没有 "failed"/"error" 即全绿；
4. 测试全绿后，输出 final 收尾（不要再调工具）；final 必须说明：改了什么 + pytest 结果；
5. 工具返回以「观察：」开头，是真实输出，按它继续推理；
6. 每步推理保持 1-2 句，不要长篇分析。

示例轨迹：
{"thought": "先看测试要求", "tool": "read_file", "args": {"path": "test_calculator.py"}}
{"thought": "看实现", "tool": "read_file", "args": {"path": "calculator.py"}}
{"thought": "先跑测试确认失败", "tool": "run_tests", "args": {}}
{"thought": "add 写成了 a - b，改为 a + b，保留 factorial", "tool": "write_file", "args": {"path": "calculator.py", "content": "def add(a, b):\\n    \\\"\\\"\\\"Return the sum of two numbers.\\\"\\\"\\\"\\n    return a + b\\n\\ndef factorial(n):\\n    if n < 0:\\n        raise ValueError('n must be non-negative')\\n    result = 1\\n    for value in range(2, n + 1):\\n        result *= value\\n    return result\\n"}}
{"thought": "验证", "tool": "run_tests", "args": {}}
{"thought": "全绿，收尾", "final": "已修复 add：a - b 改为 a + b；pytest 3 passed"}"""

# 伪工具（Skill / Subagent）不算真正的 MCP 工具，但同样出现在工具对话里
_SKILL_TOOLS = {"load_skill"}
_SUBAGENT_TOOLS = {"subagent_explore", "subagent_test"}
_ALL_TOOL_NAMES = [t["name"] for t in mcp.TOOLS] + list(_SKILL_TOOLS) + list(_SUBAGENT_TOOLS)
_SKILL_LOADER = SkillLoader(str(TASK6_ROOT / "src" / "skills"))


def _escape_ctrl_in_strings(text: str) -> str:
    """把 JSON 字符串值内的裸控制字符（换行/制表）转义——小模型的大 JSON 常犯。"""
    out, i, in_str, esc = [], 0, False, False
    while i < len(text):
        ch = text[i]
        if in_str:
            if esc:
                out.append(ch); esc = False
            elif ch == "\\":
                out.append(ch); esc = True
            elif ch == '"':
                out.append(ch); in_str = False
            elif ch in "\n\r\t":
                out.append({"\n": "\\n", "\r": "\\r", "\t": "\\t"}[ch])
            else:
                out.append(ch)
        else:
            if ch == '"':
                in_str = True
            out.append(ch)
        i += 1
    return "".join(out)


def _json_loads_lenient(text: str):
    """逐步加宽容错：裸字符串 -> 控制字符转义 -> 补右括号。"""
    for v in (_escape_ctrl_in_strings(text), text):
        for extra in ("", "}", "}}"):
            try:
                return json.loads(v + extra)
            except json.JSONDecodeError:
                continue
    return None


def _balanced_regions(text: str):
    """按引号感知扫描，返回所有平衡 {} 区域（字符串内的 { } 不计入）。"""
    regions, i, n = [], 0, len(text)
    while i < n:
        start = text.find("{", i)
        if start < 0:
            break
        depth, in_str, esc = 0, False, False
        for j in range(start, n):
            ch = text[j]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
            else:
                if ch == '"':
                    in_str = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        regions.append(text[start:j + 1])
                        i = j + 1
                        break
        else:
            break
    return regions


def _extract_json(text: str):
    """扫描全部平衡 {} 对象（容忍缺右括号、裸控制字符），返回第一个"有意义"的对象。

    小模型经常一次输出多个 JSON（先 {thought} 再 {tool...}），
    按"第一个含 final 或 tool/action 的对象"取，避免误判为格式错误。
    """
    for region in _balanced_regions(text):
        obj = _json_loads_lenient(region)
        if isinstance(obj, dict):
            if "final" in obj or "tool" in obj or "action" in obj:
                return obj
    return None


class Trace(dict):
    """dict 子类：steps / patch / tests_passed / final_answer。"""


class CodingAgent:
    def __init__(self, llm=None, max_steps=8, max_context_chars=30000):
        self.llm = llm if llm is not None else _make_llm()
        self.max_steps = max_steps
        self.max_context_chars = max_context_chars

    # ---------- 模型调用与解析 ----------
    def _call(self, messages, max_tokens=700):
        return self.llm.chat(messages, max_tokens=max_tokens)

    def _parse(self, reply):
        obj = _extract_json(reply)
        if obj is None:
            return None, None, None, None
        if "final" in obj:
            return "final", None, obj, obj
        if "tool" in obj or "action" in obj:
            act = obj.get("action") if isinstance(obj.get("action"), dict) else obj
            tool = str((obj.get("tool") or act.get("tool") or "")).lower()
            args = act.get("args") or act.get("arguments") or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            return "action", tool, args, obj
        return None, None, None, None

    def _compact(self, messages):
        """超预算 -> 压缩：保留 system + 最近 4 对消息，并把丢弃部分写成摘要。"""
        total = sum(len(str(m.get("content", ""))) for m in messages)
        if total <= self.max_context_chars:
            return messages
        dropped = messages[1:-8]
        summary_text = "【历史摘要】较早的对话已省略；已完成步骤数：%d。请基于以下近期上下文继续。" % len(dropped)
        head = [messages[0]]
        tail = messages[-8:]
        return head + [{"role": "user", "content": summary_text}] + tail

    # ---------- 主循环 ----------
    def run(self, repo_path: str, issue: str):
        repo = Path(repo_path).resolve()
        steps, subagent_reports = [], []
        messages = [{"role": "system", "content": SYSTEM_TEMPLATE},
                    {"role": "user", "content": f"ISSUE：\n{issue}"}]
        final_answer = ""
        done_reason = ""

        for step_i in range(self.max_steps):
            # 上下文压缩（长任务防爆）
            messages = self._compact(messages)
            try:
                reply = self._call(messages)
            except Exception as e:                    # noqa: BLE001
                steps.append({"step": step_i, "error": f"模型调用失败: {e}"})
                done_reason = f"model_error: {e}"
                break

            kind, tool, args, obj = self._parse(reply)
            if kind is None:                          # 解析失败 -> 附提醒重试
                steps.append({"step": step_i, "kind": "format_error", "response": reply[:500]})
                messages.append({"role": "assistant", "content": reply[:2000]})
                messages.append({"role": "user",
                                 "content": "输出格式错误：请只输出单个 JSON "
                                             "（{\"tool\": \"...\", \"args\": {...}} 或 {\"final\": \"...\"}）。"})
                continue

            if kind == "final":
                final_answer = str(obj.get("final", "") or reply).strip()
                steps.append({"step": step_i, "kind": "final", "final": final_answer[:2000]})
                done_reason = "model_declared_done"
                break

            # action
            observation = self._execute(tool, args, repo, steps, subagent_reports, reply, messages)
            steps.append({"step": step_i, "kind": "action", "tool": tool,
                          "args": args, "observation": observation[:1500],
                          "response": reply[:300]})

            if tool == "run_tests" and self._tests_green(observation):
                # 观察已由 _execute 追加；这里只加"请收尾"提示
                messages.append({"role": "user",
                                 "content": "观察：测试全绿（pytest passed）。请直接输出 final 总结，不要再调用工具。"})
                continue

        # 真实校验：再跑一次 pytest；patch 用 git_diff（无 git 时自动 .orig 对比）
        try:
            test_out = mcp.run_tests(repo, None)
            tests_passed = self._tests_green(test_out)
        except Exception:                              # noqa: BLE001
            test_out, tests_passed = "", False
        patch = mcp.git_diff(repo) if mcp._ensure_repo(repo) else ""

        if not final_answer:
            final_answer = ("（代理在步数上限内未给出 final；最后测试："
                            + ("全绿" if tests_passed else "未全绿") + "）")
        return Trace(steps=steps, patch=patch, tests_passed=tests_passed,
                     final_answer=final_answer, done_reason=done_reason,
                     subagent_reports=subagent_reports, n_steps=len(steps))

    # ---------- 工具执行 ----------
    def _execute(self, tool, args, repo, steps, subagent_reports, reply, messages):
        # 工具策略守卫（不依赖模型自觉）：禁止写测试文件/测试目录
        if tool == "write_file":
            path = str((args or {}).get("path") or "")
            parts = Path(path).parts
            name = Path(path).name
            if (name.startswith("test_") or name.endswith("_test.py")
                    or name == "conftest.py" or "tests" in parts):
                obs = ("⚠ 工具策略：禁止写入测试文件或测试目录（test_*/tests）。"
                       "请只修改实现文件（例如 calculator.py），完成后用 run_tests 验证。")
                messages.append({"role": "assistant", "content": reply[:2000]})
                messages.append({"role": "user", "content": f"观察：{obs}"})
                return obs
        if tool == "load_skill":
            name = str((args or {}).get("name") or "").strip()
            try:
                body = _SKILL_LOADER.load(name)
                obs = f"[Skill {name} 已加载]\n{body[:3000]}"
            except KeyError as e:
                obs = f"工具错误: {e}"
        elif tool in _SUBAGENT_TOOLS:
            kind = tool.replace("subagent_", "")
            sub = make_subagent(kind, self.llm, max_steps=4)
            task = str((args or {}).get("task") or "").strip()
            report = sub.run(task or "摸清 repo 结构", repo)
            subagent_reports.append({"name": kind, "task": task, "summary": report})
            obs = f"[子代理 {kind} 摘要] {report}"
        else:
            obs = mcp.dispatch(tool, args or {}, repo)
        messages.append({"role": "assistant", "content": reply[:2000]})
        messages.append({"role": "user", "content": f"观察：{obs[:2000]}"})
        return obs

    @staticmethod
    def _tests_green(observation: str) -> bool:
        o = str(observation).lower()
        return ("passed" in o and "failed" not in o and "error" not in o
                and "no tests ran" not in o)


if __name__ == "__main__":
    import sys
    from pathlib import Path as _P

    repo = _P(sys.argv[1]).resolve() if len(sys.argv) > 1 else \
        TASK6_ROOT / "data" / "toy-repo"
    issue_file = repo / "ISSUE.md"
    issue = issue_file.read_text(encoding="utf-8") if issue_file.exists() else \
        "修复仓库中的 bug，让 python -m pytest 全部通过。"
    tr = CodingAgent().run(str(repo), issue)
    print(json.dumps(tr, ensure_ascii=False, indent=2)[:6000])
