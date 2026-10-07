"""任务四 M1：手写 chunker（按字符切分 + 重叠）。

约定：chunk_size / overlap 均以【字符】计（自检按字符平均长度核验）。
"""
import re


def clean_text(text: str) -> str:
    """合并连续空白（PDF 抽取的分页/换行噪声）。"""
    return re.sub(r"[ \t\r\f\v]+", " ", text)


def chunk_text(text: str, chunk_size: int, overlap: int) -> list:
    """固定大小 + 重叠窗口切分。

    - 平均 chunk 长度 ≈ chunk_size（除最后一个），满足自检 (0.5x, 1.2x) 区间；
    - overlap>0 让跨边界的 anchor 至少有一块完整命中。
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size 必须为正数")
    overlap = max(0, min(overlap, chunk_size - 1))
    text = clean_text(text)
    chunks, start, n = [], 0, len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunks.append(text[start:end])
        if end >= n:
            break
        start = end - overlap
    return chunks


def chunk_documents(docs: list, chunk_size: int, overlap: int) -> list:
    """按文档（如 PDF 页）切分，每页内用 chunk_text，页边界优先保留。

    docs: [(text, source)]；返回 [(chunk_text, source)]。
    """
    out = []
    for text, source in docs:
        pieces = chunk_text(text, chunk_size, overlap)
        for p in pieces:
            if p.strip():
                out.append((p, source))
    return out
