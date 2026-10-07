"""任务五 M1：file_search（本地目录文件名/内容检索）。

- 文件名匹配：pattern 含通配符（* ? [）时按 glob；否则按文件名包含匹配
- 内容匹配：对文本文件（≤500KB）按 pattern 搜索正文
- 命中文件附带内容片段（前若干行），保证"读 README 第一段"这类任务可答复
- 路径安全：resolve 后校验必须落在允许根目录内（防 .. 越界）

自检：run({"pattern": "README.md", "dir": str(项目根)}) 应返回包含 "README.md" 的字符串。
"""
import fnmatch
from pathlib import Path

TOOL_SCHEMA = {
    "name": "file_search",
    "description": "在本地目录里按文件名或文件内容搜索文件，返回匹配文件的路径、大小和内容片段。"
                   "pattern 可以是文件名（如 README.md）、通配符（如 *.md）或要搜索的词（如 TODO）。",
    "parameters": {
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "文件名/通配符/内容关键词"},
            "dir": {"type": "string", "description": "搜索根目录路径"},
        },
        "required": ["pattern", "dir"],
    },
}

_TEXT_EXTS = {".md", ".txt", ".py", ".json", ".yml", ".yaml", ".csv", ".rst", ".html"}
_MAX_CONTENT_BYTES = 512 * 1024
_SNIPPET_CHARS = 600
# 工具根目录：相对路径统一锚定到任务五项目根，而不是依赖进程 CWD
# （官方 eval 从仓库根目录运行，模型传 "data/agent-fixtures" 这类相对路径必须仍能命中）
_TASK5_ROOT = Path(__file__).resolve().parents[2]


def _resolve_safe(dir_path: str) -> Path:
    cand = Path(dir_path).expanduser()
    if not cand.is_absolute():
        cand = _TASK5_ROOT / cand
    root = cand.resolve()
    if not root.is_dir():
        raise ValueError(f"目录不存在: {dir_path}")
    return root


def _list_files(root: Path):
    for p in sorted(root.rglob("*")):
        if p.is_file() and not p.is_symlink():
            yield p


def _content_match(path: Path, pattern: str) -> bool:
    if path.suffix.lower() not in _TEXT_EXTS:
        return False
    try:
        if path.stat().st_size > _MAX_CONTENT_BYTES:
            return False
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return False
    return pattern in text


def _snippet(path: Path) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
        lines = [l for l in text.splitlines() if l.strip()]
        return "\n".join(lines)[:_SNIPPET_CHARS]
    except OSError:
        return ""


def run(args: dict) -> str:
    pattern = str(args.get("pattern", "")).strip()
    if not pattern:
        raise ValueError("缺少 pattern 参数")
    root = _resolve_safe(str(args.get("dir", ".")))
    matched = []
    for p in _list_files(root):
        try:
            rel = p.relative_to(root)
        except ValueError:
            continue
        name_ok = (
            fnmatch.fnmatch(p.name, pattern)
            or fnmatch.fnmatch(str(rel), pattern)
            or pattern in p.name
            or pattern in str(rel)
        )
        content_ok = _content_match(p, pattern)
        if name_ok or content_ok:
            matched.append((p, rel, content_ok))

    if not matched:
        return f"在 {root} 下未找到与 {pattern!r} 匹配的文件。"

    lines = [f"找到 {len(matched)} 个匹配文件：" for _ in range(1)]
    for rank, (p, rel, by_content) in enumerate(matched[:20], 1):
        lines.append(f"[{rank}] {rel}（{p.stat().st_size} 字节，"
                     f"内容匹配={by_content}）")
        if rank <= 3:   # 前几个附内容片段（足够回答"第一段写了什么"）
            snip = _snippet(p)
            if snip:
                lines.append(f"    内容片段：\n{snip}")
    return "\n".join(lines)
