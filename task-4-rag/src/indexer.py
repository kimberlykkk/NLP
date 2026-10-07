"""任务四 M2：从 data/kb.pdf 抽取文本 -> BGE embedding -> FAISS 索引。

产出（首次运行自动生成，之后直接加载）：
- data/index/chunks.json    [{text, source}]（source 形如 "page 12"）
- data/index/embeddings.npy (N, D) float32（已 L2 normalize）
- data/index/faiss.index    IndexFlatIP 内积索引（= cosine）
"""
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np

from . import chunker
from .chunker import chunk_documents

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
KB_PDF = DATA / "kb.pdf"
INDEX_DIR = DATA / "index"
# 最终切分策略（chunk 扫描后的最优）：按页为单元
PAGE_LIMIT = 1200      # 页文本 <=1200 字 -> 整页一块
CHUNK_SIZE = 768       # 更长页面按 768 细分
OVERLAP = 128


def extract_pdf_pages(pdf_path):
    from pypdf import PdfReader
    reader = PdfReader(str(pdf_path))
    pages = []
    for i, page in enumerate(reader.pages, 1):
        text = page.extract_text() or ""
        if text.strip():
            pages.append((text, f"page {i}"))
    print(f"PDF 共 {len(reader.pages)} 页，抽取到 {len(pages)} 页文本")
    return pages


def page_chunking(pages, page_limit=PAGE_LIMIT, chunk_size=CHUNK_SIZE, overlap=OVERLAP):
    """页级 chunk：整页优先，超长页再细分（页面边界是最自然的语义边界）。"""
    out = []
    for text, source in pages:
        t = chunker.clean_text(text)
        if not t.strip():
            continue
        if len(t) <= page_limit:
            out.append((t, source))
        else:
            for p in chunker.chunk_text(t, chunk_size, overlap):
                if p.strip():
                    out.append((p, source))
    return out


def build_index():
    assert KB_PDF.exists(), f"缺少 {KB_PDF}；先跑 data/download_pdf.py"
    pages = extract_pdf_pages(KB_PDF)
    chunks = page_chunking(pages)
    print(f"chunk 数：{len(chunks)}（page-aware: 整页单元 + 超页 768/128 细分）")

    # ---- BGE embedding ----
    # 注意：优先用 transformers 手动池化（本机 sentence-transformers 5.2.3
    # 与 transformers 5.2 + sm_120 组合在批量编码时原生段错误，HF 路径稳定）。
    model_dir = ROOT / "models" / "bge-small-zh-v1.5"
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
        from .embed import encode_hf
        tok = AutoTokenizer.from_pretrained(str(model_dir))
        device = "cuda" if torch.cuda.is_available() else "cpu"
        m = AutoModel.from_pretrained(str(model_dir)).to(device).eval()
        emb = encode_hf(m, tok, [c[0] for c in chunks], device=device)
        print(f"嵌入完成（transformers 池化）：{emb.shape}")
    except Exception as e:
        print(f"transformers 路径失败（{type(e).__name__}: {str(e)[:80]}），回退 sentence-transformers")
        from sentence_transformers import SentenceTransformer
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = SentenceTransformer(str(model_dir), device=device)
        emb = model.encode([c[0] for c in chunks], batch_size=64,
                           normalize_embeddings=True, show_progress_bar=True,
                           convert_to_numpy=True)
        print(f"嵌入完成：{emb.shape}")

    import faiss
    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb.astype("float32"))

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(INDEX_DIR / "faiss.index"))
    np.save(INDEX_DIR / "embeddings.npy", emb.astype("float32"))
    (INDEX_DIR / "chunks.json").write_text(
        json.dumps([{"text": t, "source": s} for t, s in chunks],
                   ensure_ascii=False), encoding="utf-8")
    print(f"索引已保存 {INDEX_DIR}（{len(chunks)} chunks）")
    return index, chunks


def load_index():
    if (INDEX_DIR / "faiss.index").exists():
        import faiss
        index = faiss.read_index(str(INDEX_DIR / "faiss.index"))
        chunks = json.loads((INDEX_DIR / "chunks.json").read_text(encoding="utf-8"))
        return index, chunks
    print("索引不存在，现场构建 ...")
    return build_index()


if __name__ == "__main__":
    build_index()
