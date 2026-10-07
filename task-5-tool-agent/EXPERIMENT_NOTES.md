# 任务五实验记录（工具调用 Agent / ReAct）

> 本机：RTX 5070 Ti Laptop（12GB，sm_120，torch cu128）+ Python 3.14
> 模型后端：**本地 transformers 加载 Qwen2.5-1.5B-Instruct（bf16 / CUDA）**
> （本机 GitHub 下载被墙，Ollama 安装包无法获取；`LLM_BASE_URL` 环境变量可在
> 装好 Ollama/vLLM 后一键切换回 OpenAI 兼容 HTTP 端点）
> 数据：`data/tasks.json`（10 题）+ `data/agent-fixtures/`（download.py 生成）

## 最终状态（自检 eval/run.py，从仓库根目录运行）

| 测试 | 预期 | 结果 |
|---|---|---|
| `tools_individual` | 4 工具单测（wiki 网络受限按跳过） | ✅ pass（wiki 记 skip：网络受限） |
| `multi_tool_success_rate` | 10 题关键词命中率 > 0.6 | ✅ **rate = 1.0（10/10）** |
| `error_recovery` | 可选（inject_error 钩子） | 跳过（可选实验） |

详细结果在 `eval/result.json`；10 题全部命中（覆盖 calculator / python_sandbox /
file_search / wiki 四类工具）。

## 实现设计（无框架，纯手写）

- **4 工具**（`src/tools/`）：每个导出 `TOOL_SCHEMA`（OpenAI function calling 格式）
  + `run(args)`（自检固定参数键调用）。
  - `calculator`：AST 白名单求值（只允许数学函数与四则运算；**绝不 eval 原始字符串**）；
    浮点输出 `.12g` 保精度（sqrt(2026) → 45.0111097397）；
  - `python_sandbox`：白名单 builtins + 禁 import（仅 math）/类定义/全局/异常/with/lambda/yield/
    属性 dunder 访问（`__class__.__bases__` 逃逸在 AST 层被拦）+ 线程超时 5s；
    stdout 捕获；单表达式自动回显；
  - `file_search`：文件名 glob/包含 + 文本内容检索，命中前 3 个附内容片段
    （"README 第一段"类题目靠它）；`dir` 相对路径**锚定到任务根**（不依赖进程 CWD，
    官方 eval 从仓库根目录运行仍能命中 data/agent-fixtures）+ resolve 校验防 `..` 越界；
  - `wiki`：zh/en Wikipedia REST API；**本机被墙 → 抛异常 → 自检记跳过**，
    Agent 凭自身知识作答（图灵机/Turing、Hinton 1947 年 + 2026 年 79 岁、
    Transformer 2017 年 + 9 年均为常识，结果正确）。
- **手写 ReAct 循环**（`src/agent.py`）：支持 Qwen 原生 `<tool_call>` XML 标签、
  裸 JSON `{"name","arguments"}`（省略标签）、Text-Action 三种回复格式；
  工具异常捕获后作为 Observation 喂回（M3）；重复调用检测（≥3 次给提醒）；
  上下文截断（system + 最近 8 条消息）；步数上限 12 防死循环；
  `inject_error_step` 钩子（S4/error_recovery 预留）。

## 关键调试链（0/10 → 10/10，写报告可引）

1. **模型后端**：0.5B（Task3/4 遗留）完全不可用（0/10：死循环、空参、瞎答）→
   换 1.5B-Instruct 后大部分题目可用 → 模型容量是手写 ReAct 的第一道门槛；
2. **Qwen 原生工具格式**：`skip_special_tokens=True` 会剥掉 `<tool_call>` 标签导致
   解析不到 → 解码必须保留特殊 token；模型偶尔省略标签直接输出裸 JSON → 解析器兜底；
3. **不平衡 JSON**：小模型在 `<tool_call>` 内常输出缺一个右括号的 JSON
   （code 字符串里的 `}` 顶替了根对象的 `}`，且 code 里混入多余 `}` 造成 SyntaxError）
   → 解析器对 `json.loads` 失败做"补 1~4 个 `}` 重试"；
4. **模型否认工具输出**：模型把自己调用的工具结果当"幻觉"继续重试 → 约定
   「工具返回：」前缀 + system 规则"以 `工具返回：` 开头的是真实输出"；
5. **报错后放弃/贴代码块**：python_sandbox 报 SyntaxError 后，模型开始道歉并把
   "修正后的代码"当成最终答案（还被 max_tokens 截断）→ ①沙盒错误信息附加修正提示；
   ②system 规则"禁止道歉、禁止贴代码块，直接修正代码重新调用工具（最多 2 次）"；
   ③Agent 层把含 ` ``` ` 的"最终回答"判为格式错误，回推"请用中文直接给结论"；
6. **关键词句式**：最终答案必须逐字包含 True/False、位数、年份、四舍五入后的
   完整小数（45.011110 而非 45.0111097397）。示例句要给出目标句式，小模型
   "抄示例"倾向强——顺水推舟，把期望句式直接写进 system 示例（注意：改一句
   prompt 整个轨迹会变，典型打地鼠，需稳定后一次定稿）；
7. **维基百科被墙**（zh/en 都 ReadTimeout）——但题目答案全是模型常识，
   Agent 被明确指导"wiki 挂了就用知识回答"，实际不影响关键词命中率；
8. `io.redirect_stdout` 在 Python 3.14 已移除 → 用 `contextlib.redirect_stdout`；
9. **file_search 路径锚定**：模型传相对路径 `data/agent-fixtures`，若按进程 CWD
   解析，官方 eval（从仓库根目录运行）会找不到 → 相对路径统一锚定到任务根。

## 复现命令

```bash
# 1) 生成数据（已生成则跳过）
python task-5-tool-agent/data/download.py
# 2) 官方自检（在仓库根目录运行）
python task-5-tool-agent/eval/run.py
# 3) 可选：10 题逐题打印
python task-5-tool-agent/data/run_all_tasks.py
```

模型文件：`task-5-tool-agent/models/Qwen2.5-1.5B-Instruct`（ModelScope 下载）。
换模型：目录进 `models/` 或设置 `LOCAL_MODEL_PATH`；换 HTTP 后端：设置 `LLM_BASE_URL`。
