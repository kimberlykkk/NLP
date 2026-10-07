"""任务四 M4：端到端 RAG 流水线。

answer(query) -> dict(answer: str, sources: List[dict])
"""
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from .retriever import Retriever
from .generator import get_generator

_retriever = None
_generator = None


def _get_retriever():
    global _retriever
    if _retriever is None:
        _retriever = Retriever()
    return _retriever


def _get_generator():
    global _generator
    if _generator is None:
        _generator = get_generator()
    return _generator


def _build_prompt(query, docs):
    context = []
    seen = set()
    for d in docs:
        key = d["text"][:80]
        if key in seen:
            continue
        seen.add(key)
        context.append(f"[{d['source']}]\n{d['text']}")
    ctx = "\n\n".join(context)[:4000]
    return (
        "你是一个文档问答助手。只能根据下面提供的资料回答问题：\n\n"
        f"{ctx}\n\n"
        f"问题：{query}\n\n"
        "要求：1) 只使用资料中的信息；2) 如果资料中没有答案，"
        "直接回答“根据提供的资料无法确定”；3) 用中文、简洁回答。"
    )


def answer(query: str):
    """检索 + 生成，返回 {answer, sources}。"""
    docs = _get_retriever().retrieve(query, k=4)
    prompt = _build_prompt(query, docs)
    ans = _get_generator().chat(prompt)
    return {"answer": ans, "sources": docs[:4]}
