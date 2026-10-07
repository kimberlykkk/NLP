"""任务六 M1：手写 MCP server（暴露 ≥ 5 个工具）。

- 顶层导出 `list_tools() -> List[dict]`（每项含 name/description/input_schema），
  供自检 `mcp_server_lists_tools` 枚举（>= 5 即过）；
- 可独立运行：`python src/mcp_server.py [--repo data/toy-repo]`（stdio 握手）；
- 安全（逐条不可省）：
  * read_file / write_file / list_dir / search_files 先 resolve 规整路径，
    校验落在目标 repo 内；拒绝绝对路径与 `..` 越界；
  * 跑 git / pytest 用 subprocess 的 list 形式（args=[...]，无 shell=True，限定 cwd），
    不把用户/模型文本拼进 shell 串；
  * git_apply 只在工作区操作，绝不执行 reset --hard / clean -fd 等危险命令。
- 工具统一返回字符串（Observation 风格）；异常被捕获成结构化错误文本，不 crash server。

注意：本机未装 git（download.py 的 git init 静默跳过是预期），git_diff 自动回退到
Python difflib：对比工作文件与其 `*.orig` 快照（eval 重置用的基准），一样能产出 patch。
"""
import argparse
import fnmatch
import os
import subprocess
import sys
from difflib import unified_diff
from pathlib import Path

TASK6_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPO = TASK6_ROOT / "data" / "toy-repo"
REPO_DIR = Path(os.environ.get("TASK6_REPO", DEFAULT_REPO)).resolve()

_READ_LIMIT = 200 * 1024          # read_file 单文件上限
_MAX_RESULTS = 20                 # 检索类工具的最大结果数
_TEXT_EXTS = {".py", ".md", ".txt", ".json", ".yml", ".yaml", ".toml", ".cfg", ".ini",
              ".csv", ".html", ".js", ".ts", ".rs", ".go", ".java", ".c", ".h", ".cpp"}


# ---------------- 路径安全 ----------------

def _safe_path(repo: Path, rel: str) -> Path:
    """相对路径 -> repo 内绝对路径；拒绝绝对路径与越界。"""
    p = Path(str(rel or "."))
    if p.is_absolute():
        raise ValueError(f"拒绝绝对路径: {rel}")
    cand = (repo / p).resolve()
    if not cand.is_relative_to(repo.resolve()):
        raise ValueError(f"路径越界（不在目标 repo 内）: {rel}")
    return cand


def _ensure_repo(repo: Path) -> Path:
    repo = Path(repo).resolve()
    if not repo.is_dir():
        raise ValueError(f"repo 目录不存在: {repo}")
    return repo


# ---------------- 子进程安全 ----------------

def _run(argv: list, cwd: Path, timeout: float = 60.0) -> str:
    """list 形式 + 限定 cwd，绝无 shell 注入面。"""
    try:
        r = subprocess.run(argv, cwd=cwd, text=True, capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return f"执行超时（>{timeout}s）: {' '.join(argv)}"
    except FileNotFoundError as e:
        return f"命令不存在: {e}"       # 如本机无 git —— 预期内，见文件头注释
    out = (r.stdout + r.stderr).strip()
    return f"exit={r.returncode}\n{out[-4000:]}"


# ---------------- 工具实现（repo 显式传入，可被 agent 复用的同一套实现） ----------------

def read_file(repo, path: str) -> str:
    """读取 repo 内文本文件。"""
    repo = _ensure_repo(repo)
    p = _safe_path(repo, path)
    if not p.is_file():
        raise ValueError(f"文件不存在: {path}")
    if p.stat().st_size > _READ_LIMIT:
        raise ValueError(f"文件过大（>{_READ_LIMIT} 字节），请用 search_files 定位后分段读取")
    text = p.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    return f"{p.relative_to(repo)}（{len(lines)} 行）:\n```\n{text}\n```"


def write_file(repo, path: str, content: str) -> str:
    """写入 repo 内文本文件（覆盖；目录自动创建）。"""
    repo = _ensure_repo(repo)
    p = _safe_path(repo, path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8", newline="\n")
    return f"已写入 {p.relative_to(repo)}（{len(content)} 字符）"


def list_dir(repo, path: str = ".") -> str:
    """列出 repo 内目录结构（最多 2 层 + 文件大小）。"""
    repo = _ensure_repo(repo)
    root = _safe_path(repo, path)
    if not root.is_dir():
        raise ValueError(f"目录不存在: {path}")
    lines = []
    for depth, (dirpath, dirnames, filenames) in enumerate(os.walk(root)):
        rel_dir = Path(dirpath).relative_to(repo)
        if depth > 2:
            dirnames[:] = []
            continue
        indent = "  " * depth
        lines.append(f"{indent}{rel_dir}/")
        for fname in sorted(filenames):
            fp = Path(dirpath) / fname
            lines.append(f"{indent}  {fname}（{fp.stat().st_size} 字节）")
    return "\n".join(lines[:200])


def search_files(repo, pattern: str, path: str = ".") -> str:
    """按文件名/通配符/内容关键词在 repo 内搜索，返回路径 + 内容片段。"""
    repo = _ensure_repo(repo)
    root = _safe_path(repo, path)
    if not root.is_dir():
        raise ValueError(f"目录不存在: {path}")
    hits = []
    for p in sorted(root.rglob("*")):
        if not p.is_file() or p.is_symlink():
            continue
        rel = p.relative_to(root)
        name_ok = (fnmatch.fnmatch(p.name, pattern)
                   or fnmatch.fnmatch(str(rel), pattern)
                   or pattern in p.name or pattern in str(rel))
        content_ok = False
        if p.suffix.lower() in _TEXT_EXTS and p.stat().st_size <= _READ_LIMIT:
            try:
                content_ok = pattern in p.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                pass
        if name_ok or content_ok:
            snippet = ""
            if p.suffix.lower() in _TEXT_EXTS:
                try:
                    snippet = p.read_text(encoding="utf-8", errors="ignore").strip()[:300]
                except OSError:
                    pass
            hits.append((rel, snippet))
        if len(hits) >= _MAX_RESULTS:
            break
    if not hits:
        return f"未找到与 {pattern!r} 匹配的文件/内容。"
    lines = [f"找到 {len(hits)} 个匹配："]
    for rel, snippet in hits:
        lines.append(f"- {rel}")
        if snippet:
            lines.append(f"    {snippet.splitlines()[0][:200]}")
    return "\n".join(lines)


def run_tests(repo, test_path: str | None = None) -> str:
    """在 repo 内运行 python -m pytest（可选指定测试文件）。"""
    repo = _ensure_repo(repo)
    argv = [sys.executable, "-m", "pytest", "-q"]
    if test_path:
        argv.append(str(_safe_path(repo, test_path)))
    return _run(argv, cwd=repo, timeout=120)


def _diff_against_orig(repo: Path) -> str:
    """git 不可用时回退：对比每个工作文件与其 *.orig 快照（difflib unified diff）。"""
    parts = []
    for snap in sorted(repo.rglob("*.orig")):
        target = snap.with_suffix("")                 # calculator.py.orig -> calculator.py
        if not target.is_file():
            continue
        old = snap.read_text(encoding="utf-8", errors="replace").splitlines()
        new = target.read_text(encoding="utf-8", errors="replace").splitlines()
        diff = list(unified_diff(old, new, fromfile=str(snap.relative_to(repo)),
                                 tofile=str(target.relative_to(repo)), lineterm=""))
        if diff:
            parts.append("\n".join(diff))
    return "\n\n".join(parts)


def git_diff(repo) -> str:
    """工作区 diff（git 优先，无 git 时用 *.orig 快照对比）。"""
    repo = _ensure_repo(repo)
    if (repo / ".git").exists():
        out = _run(["git", "diff"], cwd=repo)
        if "命令不存在" not in out and "exit=0" in out:
            return out if out.strip() != "exit=0" else "（工作区无改动）"
    fallback = _diff_against_orig(repo)
    if fallback:
        return fallback
    has_orig = any(repo.rglob("*.orig"))
    return "（无改动）" if has_orig else "（未找到 .orig 快照可对比）"


def git_apply(repo, patch: str) -> str:
    """把 unified diff 应用到工作区（仅在本 repo 内；不用危险 git 命令）。"""
    repo = _ensure_repo(repo)
    if not patch.strip():
        raise ValueError("patch 为空")
    if not (repo / ".git").exists():
        # 无 git：给出明确错误（agent 用 write_file 落盘即可）
        raise ValueError("本 repo 无 git；git_apply 不可用，请用 write_file 直接改文件")
    r = subprocess.run(["git", "apply", "-"], input=patch, cwd=repo,
                       text=True, capture_output=True, timeout=60)
    out = (r.stdout + r.stderr).strip()
    return f"exit={r.returncode}\n{out[-2000:]}" if out else f"exit={r.returncode}"


# ---------------- MCP 工具注册（schema 权威来源） ----------------

TOOLS = [
    {"name": "read_file",
     "description": "读取 repo 内文本文件的内容（自动拒绝外部路径）。",
     "input_schema": {"type": "object",
                      "properties": {"path": {"type": "string", "description": "repo 内相对路径"}},
                      "required": ["path"]}},
    {"name": "write_file",
     "description": "写入/覆盖 repo 内文本文件（自动创建目录；拒绝外部路径）。",
     "input_schema": {"type": "object",
                      "properties": {"path": {"type": "string"},
                                     "content": {"type": "string"}},
                      "required": ["path", "content"]}},
    {"name": "list_dir",
     "description": "列出 repo 内目录结构（文件名 + 大小，最多两层）。",
     "input_schema": {"type": "object",
                      "properties": {"path": {"type": "string", "description": "相对于 repo 的目录"}},
                      "required": []}},
    {"name": "search_files",
     "description": "按文件名/通配符/内容关键词在 repo 内搜索，返回路径和内容片段。",
     "input_schema": {"type": "object",
                      "properties": {"pattern": {"type": "string"},
                                     "path": {"type": "string", "description": "搜索起点目录"}},
                      "required": ["pattern"]}},
    {"name": "run_tests",
     "description": "在 repo 内运行 python -m pytest（可指定测试文件），返回 pytest 输出。",
     "input_schema": {"type": "object",
                      "properties": {"test_path": {"type": "string", "description": "可选测试文件"}},
                      "required": []}},
    {"name": "git_diff",
     "description": "查看工作区改动 diff（无 git 时自动用 *.orig 快照对比）。",
     "input_schema": {"type": "object", "properties": {}, "required": []}},
    {"name": "git_apply",
     "description": "将 unified diff 应用到工作区（仅限本 repo；无 git 时不可用，请改用 write_file）。",
     "input_schema": {"type": "object",
                      "properties": {"patch": {"type": "string"}},
                      "required": ["patch"]}},
]


def list_tools() -> list:
    """顶层导出（自检用）：返回工具 schema 列表，每项含 name。"""
    return [dict(t) for t in TOOLS]


def dispatch(name: str, args: dict, repo=None) -> str:
    """按名字调用工具实现（agent 与 MCP server 共用；异常转结构化错误文本）。"""
    repo = _ensure_repo(repo or REPO_DIR)
    args = args or {}
    try:
        if name == "read_file":
            return read_file(repo, args["path"])
        if name == "write_file":
            return write_file(repo, args["path"], args["content"])
        if name == "list_dir":
            return list_dir(repo, args.get("path", "."))
        if name == "search_files":
            return search_files(repo, args["pattern"], args.get("path", "."))
        if name == "run_tests":
            return run_tests(repo, args.get("test_path"))
        if name == "git_diff":
            return git_diff(repo)
        if name == "git_apply":
            return git_apply(repo, args["patch"])
        return f"未知工具: {name}（可用：{', '.join(t['name'] for t in TOOLS)}）"
    except KeyError as e:
        return f"参数缺失: {e}"
    except Exception as e:                      # 工具异常 -> 结构化错误，不 crash
        return f"工具错误: {type(e).__name__}: {e}"


# ---------------- MCP stdio server（独立运行路径） ----------------

def main(argv=None):
    ap = argparse.ArgumentParser(description="Mini Coding Agent MCP server (stdio)")
    ap.add_argument("--repo", default=str(DEFAULT_REPO),
                    help=f"目标 repo 目录（默认 {DEFAULT_REPO}）")
    ns = ap.parse_args(argv)
    global REPO_DIR                       # noqa: PLW0603 —— server 进程专用
    REPO_DIR = Path(ns.repo).resolve()

    # mcp>=2 用 MCPServer；mcp<2 用 FastMCP（API 几乎一致，均暴露 .tool()/.run()）
    try:
        from mcp.server.mcpserver import MCPServer as _Server
    except ImportError:
        try:
            from mcp.server.fastmcp import FastMCP as _Server
        except ImportError as e:
            sys.exit(f"[错误] 未安装 mcp SDK（pip install mcp）：{e}")

    mcp = _Server("mini-coding-agent")

    @mcp.tool(name="read_file")
    def read_file_tool(path: str) -> str:
        """读取 repo 内文本文件。"""
        return read_file(REPO_DIR, path)

    @mcp.tool(name="write_file")
    def write_file_tool(path: str, content: str) -> str:
        """写入 repo 内文本文件。"""
        return write_file(REPO_DIR, path, content)

    @mcp.tool(name="list_dir")
    def list_dir_tool(path: str = ".") -> str:
        """列出 repo 目录结构。"""
        return list_dir(REPO_DIR, path)

    @mcp.tool(name="search_files")
    def search_files_tool(pattern: str, path: str = ".") -> str:
        """按文件名/内容搜索。"""
        return search_files(REPO_DIR, pattern, path)

    @mcp.tool(name="run_tests")
    def run_tests_tool(test_path: str | None = None) -> str:
        """运行 pytest。"""
        return run_tests(REPO_DIR, test_path)

    @mcp.tool(name="git_diff")
    def git_diff_tool() -> str:
        """查看工作区 diff。"""
        return git_diff(REPO_DIR)

    @mcp.tool(name="git_apply")
    def git_apply_tool(patch: str) -> str:
        """应用 unified diff。"""
        return git_apply(REPO_DIR, patch)

    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
