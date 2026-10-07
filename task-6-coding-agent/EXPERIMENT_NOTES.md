# 任务六实验记录（Mini Coding Agent）

> 本机：RTX 5070 Ti Laptop（12GB，sm_120，torch cu128）+ Python 3.14
> 模型后端：**本地 transformers 加载 Qwen2.5-Coder-3B-Instruct（bf16 / CUDA）**
> （README 默认 Qwen2.5-Coder-7B-Instruct：FP16 ≈15GB 超显存；Q4_K_M 量化需要
> llama.cpp 二进制，本机 GitHub 下载被墙无法安装 → 用 3B Coder 代替并记录偏差；
> 设 `LLM_BASE_URL` 可切回 Ollama/vLLM 的 OpenAI 兼容端点）
> 数据：`data/toy-repo`（download.py 生成：buggy `calculator.add` + 3 个 pytest + ISSUE.md）

## 最终状态（自检 eval/run.py，从仓库根目录运行）

| 测试 | 预期 | 结果 |
|---|---|---|
| `mcp_server_lists_tools` | ≥5 工具（list_tools 顶层导出） | ✅ pass（7 工具） |
| `skill_loader_metadata` | SkillLoader.list_skills() 每项含 name+description（≥2） | ✅ pass（3 个 skill，无缺元数据） |
| `toy_repo_patch` | 修好 calculator.add，`python -m pytest` 全绿 | ✅ pass（3 passed；trace 含 patch） |
| `swebench_lite_sample` | 可选（S4，需 clone 对应 repo） | 跳过（本机 GitHub 被墙无法 clone） |

详细结果在 `eval/result.json`。toy-repo 上的完整轨迹（3B Coder，一次跑通，共 6 步 / 16s）：
read_file(test_calculator.py) → read_file(calculator.py) → run_tests（确认 FF 失败）→
write_file(calculator.py 完整重写：add 改 a + b、保留 factorial) → run_tests（3 passed）→
final（模型显式声明完成，`done_reason=model_declared_done`）。

## 分层实现（Tools / Skills / Subagents / Agent loop）

- **MCP server（`src/mcp_server.py`，M1）**：7 个工具（read_file / write_file /
  list_dir / search_files / run_tests / git_diff / git_apply）。
  - 顶层导出 `list_tools() -> List[dict]`（name/description/input_schema），
    与 stdio 注册的 MCP 工具**同名**；
  - `python src/mcp_server.py [--repo ...]` 可独立启动（mcp 2.x MCPServer；
    兼容 mcp<2 的 FastMCP）；
  - 安全：路径先 resolve、只允许 repo 内相对路径（拒 `..`/绝对路径/越界，实测
    `../README.md` 与绝对路径都被拦）；subprocess 一律 list 形式（无 shell=True、
    限定 cwd、限时）；git_apply 不用任何危险 git 命令；工具异常转结构化错误不 crash；
  - 本机无 git（download.py 的 git init 预期失败）：`git_diff` 自动回退 difflib，
    对比工作文件与 `*.orig` 快照（eval 重置基准），patch 照常产出。
- **Skill（`src/skill_loader.py` + `src/skills/`，M2）**：约 50 行加载器：扫描
  `skills/*/SKILL.md` 的 YAML front-matter 建索引（name/description/path/body），
  `list_skills()` 只给元数据、`load(name)` 命中后才给全文（渐进式披露）。
  3 个 Skill：code-review / pr-description-writer / test-runner。
- **Subagent（`src/subagents/`，M3 加分）**：`SubAgent` 有独立 message 列表、
  独立步数上限（默认 4）、工具子集（explore 只读三件套；test 只给
  run_tests/read_file/search_files）；主 agent 只接收 ≤500 字摘要。
  主循环里的伪工具 `subagent_explore` / `subagent_test`。
- **Agent loop（`src/agent.py`，M3/M4）**：模型输出单个 JSON
  （`{"thought","tool","args"}` 或 `{"thought","final"}`）；主循环
  模型 → 工具 → 「观察：」→ 循环；三层栈都能被调用：MCP 工具、`load_skill`（SkillLoader
  按名加载全文，渐进式披露）、`subagent_explore`/`subagent_test`（独立上下文子代理）；
  停机 = 测试全绿（run_tests 观察含 passed 且无 failed/error）或模型显式 final；
  步数上限 8 防死循环；上下文超 30k 字符压缩（system + 摘要 + 最近 8 条）；
  Trace 为 dict，含 steps / patch / tests_passed（最终 pytest 真实复跑，不信自报）/
  final_answer / done_reason / subagent_reports。

## 关键调试链（写报告可引）

1. **模型容量**：先用 task-5 的 Qwen2.5-1.5B-Instruct（非 Coder）开发——它能定位
   add 的 bug 并写对 `a + b`，但**执意"修正测试文件"**（认为测试写错了），
   死循环在写 test_calculator.py 上，最终 pytest 从未跑绿；0.5B/小模型在该任务上
   表现为"修测试而不是修实现"的锚定错误 → 换 Coder-3B（模型训练目标的差异）。
2. **多 JSON 输出**：模型一次回复先给 `{"thought":...}` 再给 `{"tool":...}`，
   解析器若只取第一个对象会全部判为格式错误 → 引号感知扫描全部平衡对象，
   取第一个含 final/tool/action 的；
3. **JSON 字符串内裸换行**：模型在 content 字段里写真实换行（非法 JSON）→
   `_escape_ctrl_in_strings`（引号感知地把字符串内的 \n\r\t 转义）+ 补右括号重试；
4. **content 含花括号**：`balanced_groups` 必须引号感知（`{}` 在字符串内不计深度），
   否则写代码文件时 JSON 切割错位；
5. **工具策略守卫**：即使 prompt 明令"不要改测试文件"，1.5B 仍会尝试 →
   Agent 层拦截 write_file 到 `test_*`/`tests` 的写入并回推提示（不依赖模型自觉）；
   该守卫对 SWE-bench 也安全（harness 自行应用测试 patch）；
6. **MCP SDK 版本坑**：pip 装到 mcp 2.x（FastMCP 改名 MCPServer，
   `mcp.server.mcpserver`）→ 兼容两代 API 的写法（try MCPServer except FastMCP），
   并把 stdio 注册名与 `list_tools()` 名字对齐；
7. **网络**：ModelScope aliyun 稳定但 ~600KB/s（6.2GB ≈ 1.5-3h）；hf-mirror.com
   API 可达但大文件 302 到被墙的 us.aws.cdn.hf.co（xet 后端），禁用 XET 仍可 →
   最终用 ModelScope，并验证"下载中途目录不可加载"。

## 复现命令

```bash
# 1) 生成 toy-repo
python task-6-coding-agent/data/download.py
# 2) 官方自检（仓库根目录）
python task-6-coding-agent/eval/run.py
# 3) 可选：单次 agent 轨迹
python task-6-coding-agent/data/run_agent.py
# 4) 可选：MCP stdio 握手验证
python task-6-coding-agent/data/test_mcp_stdio.py
```

模型：`task-6-coding-agent/models/Qwen2.5-Coder-3B-Instruct`（ModelScope 下载）。
换模型：目录进 models/ 或 `LOCAL_MODEL_PATH`；换 HTTP 后端：`LLM_BASE_URL`。
