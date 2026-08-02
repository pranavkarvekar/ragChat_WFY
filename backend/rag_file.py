"""
rag_file.py
===========
Production-grade, stateful RAG pipeline for uploaded documents.

Pipeline stages
---------------
1. PARSE      — PyMuPDF (PDF) / python-docx (DOCX) / plain text (TXT/MD)
2. CHUNK      — Structural (heading-aware) chunking, 900-char target with overlap
3. EMBED      — Dense: all-MiniLM-L6-v2  |  Sparse: TF-IDF (BM25-like)
4. STORE      — Milvus Lite HNSW + SPARSE_INVERTED_INDEX, tagged user_id/source_id
5. RETRIEVE   — Hybrid search (AnnSearchRequest × 2 → RRFRanker, top-20 candidates)
6. RERANK     — BAAI/bge-reranker-base CrossEncoder → top-3 chunks
7. GENERATE   — Groq LLaMA-3.3-70b with strict QA prompt, streamed via SSE

Caching: if a (user_id, source_id) pair already exists in Milvus, steps 1-4
are skipped entirely — subsequent queries are instant.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from typing import Generator

from dotenv import load_dotenv
from groq import Groq

# pyrefly: ignore [missing-import]
from .embeddings import (
    fit_and_save_tfidf,
    get_dense_embedder,
    load_tfidf,
    scipy_row_to_dict,
)
# pyrefly: ignore [missing-import]
from .milvus_client import hybrid_search as milvus_hybrid_search
# pyrefly: ignore [missing-import]
from .milvus_client import fetch_all_chunks, insert_chunks, source_exists

load_dotenv()
_groq = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ── Prompt template ───────────────────────────────────────────────────────────

_STRICT_QA_PROMPT = """\
You are a precise document analysis assistant.

Your ONLY task is to answer the user's question using EXCLUSIVELY the context \
blocks extracted from their uploaded document shown below.

STRICT RULES:
1. Base your answer solely on the provided context blocks.
2. Do NOT use any prior knowledge or make assumptions beyond the context.
3. If the answer cannot be confidently found in the context blocks, you MUST \
respond with exactly:
   "I cannot find that information in the provided source material."
4. Be concise, factual, and direct. Reference the block number when possible.

--- CONTEXT BLOCKS ---
{context}
--- END CONTEXT ---

Question: {question}

Answer:"""


# ═══════════════════════════════════════════════════════════════════════════════
# 1. TEXT EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════════

def _extract_text(file_path: str) -> str:
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        return _extract_pdf(file_path)
    if ext in (".docx", ".doc"):
        return _extract_docx(file_path)
    if ext in (".txt", ".md", ".rtf"):
        return _extract_plain(file_path)
    raise ValueError(f"Unsupported file type: '{ext}'. Supported: PDF, DOCX, TXT, MD")


def _extract_pdf(path: str) -> str:
    """High-fidelity extraction via PyMuPDF with reading-order sort."""
    # pyrefly: ignore [missing-import]
    import pymupdf  # fitz

    doc = pymupdf.open(path)
    pages: list[str] = []
    for page_num, page in enumerate(doc, 1):
        text = page.get_text(sort=True).strip()
        if text:
            pages.append(f"[Page {page_num}]\n{text}")
    doc.close()
    return "\n\n".join(pages)


def _extract_docx(path: str) -> str:
    """Extract paragraph text from a Word document."""
    from docx import Document

    doc = Document(path)
    paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
    return "\n\n".join(paragraphs)


def _extract_plain(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        return fh.read()


# ═══════════════════════════════════════════════════════════════════════════════
# 2. STRUCTURAL CHUNKING
# ═══════════════════════════════════════════════════════════════════════════════

def _structural_chunk(
    text: str,
    max_chars: int = 900,
    overlap: int = 150,
) -> list[str]:
    """
    Split text preserving semantic structure.

    Priority order:
      1. Markdown / document headings  (# / ## / ###)
      2. Paragraph boundaries          (double newline)
      3. Hard character limit          (max_chars with overlap)
    """
    # Split on heading markers — keep heading at start of each block
    raw_blocks = re.split(r"\n(?=#{1,6}\s)", text)

    chunks: list[str] = []
    for block in raw_blocks:
        block = block.strip()
        if not block:
            continue
        if len(block) <= max_chars:
            chunks.append(block)
            continue
        # Large block → split by paragraphs
        paragraphs = [p.strip() for p in block.split("\n\n") if p.strip()]
        current = ""
        for para in paragraphs:
            candidate = (current + "\n\n" + para).strip() if current else para
            if len(candidate) <= max_chars:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                if len(para) > max_chars:
                    # Hard split with character overlap
                    for i in range(0, len(para), max_chars - overlap):
                        piece = para[i : i + max_chars].strip()
                        if piece:
                            chunks.append(piece)
                    current = ""
                else:
                    current = para
        if current:
            chunks.append(current)

    return [c for c in chunks if len(c.strip()) > 20]


# ═══════════════════════════════════════════════════════════════════════════════
# 3 + 4. INGESTION  (embed → store in Milvus)
# ═══════════════════════════════════════════════════════════════════════════════

def _compute_source_id(file_path: str) -> str:
    """SHA-256 of file content — used as the stable, content-addressed source_id."""
    h = hashlib.sha256()
    with open(file_path, "rb") as fh:
        for block in iter(lambda: fh.read(65536), b""):
            h.update(block)
    return h.hexdigest()[:64]


def _ingest_file(file_path: str, user_id: str, source_id: str, status_cb=None) -> None:
    """Parse, chunk, embed, and persist a document into Milvus."""
    if status_cb: status_cb("📄 Extracting text from document...")
    text = _extract_text(file_path)
    if not text.strip():
        raise ValueError("No readable text was found in the uploaded file.")

    if status_cb: status_cb("✂️ Splitting content into chunks...")
    chunks = _structural_chunk(text)
    if not chunks:
        raise ValueError("Document content could not be split into indexable chunks.")

    if status_cb: status_cb("🧠 Generating dense embeddings...")
    embedder = get_dense_embedder()
    dense_vecs = embedder.encode(
        chunks, normalize_embeddings=True, show_progress_bar=False, batch_size=32
    )

    # Fit TF-IDF on this document's chunk corpus and persist for query time
    if status_cb: status_cb("📈 Building keyword index...")
    vectorizer = fit_and_save_tfidf(chunks, user_id, source_id)
    sparse_matrix = vectorizer.transform(chunks)

    if status_cb: status_cb("💾 Persisting index in Milvus...")
    records: list[dict] = []
    for i, chunk in enumerate(chunks):
        records.append(
            {
                "user_id":       user_id,
                "source_id":     source_id,
                "chunk_text":    chunk[:4000],          # guard against VARCHAR limit
                "dense_vector":  dense_vecs[i].tolist(),
                "sparse_vector": scipy_row_to_dict(sparse_matrix.getrow(i)),
            }
        )

    # Insert in batches to stay within Milvus request size limits
    BATCH = 100
    for i in range(0, len(records), BATCH):
        insert_chunks(records[i : i + BATCH])


# ═══════════════════════════════════════════════════════════════════════════════
# 5 + 6. RETRIEVAL + RERANKING
# ═══════════════════════════════════════════════════════════════════════════════

def _retrieve_top_chunks(
    question: str,
    user_id: str,
    source_id: str,
    top_k: int = 5,
) -> list[str]:
    """
    Hybrid search (dense HNSW + sparse TF-IDF via RRF) → top `top_k` chunks.

    The RRF ranker already fuses dense and sparse signals into a strong
    relevance ordering.  A heavy cross-encoder reranker is skipped to keep
    query latency under 3 seconds on CPU hardware.
    """
    embedder = get_dense_embedder()
    dense_q = embedder.encode(
        [question], normalize_embeddings=True, show_progress_bar=False
    )[0].tolist()

    try:
        vectorizer = load_tfidf(user_id, source_id)
    except FileNotFoundError:
        # bm25_models/ was wiped by Render's ephemeral FS — rebuild from Milvus chunks
        import logging as _logging
        _logging.getLogger(__name__).warning(
            "TF-IDF pkl missing for source %s — rebuilding from Milvus chunks.", source_id
        )
        stored_chunks = fetch_all_chunks(user_id, source_id)
        if not stored_chunks:
            # No chunks found — fall back to dense-only search
            return milvus_hybrid_search(dense_q, {}, user_id, source_id, limit=top_k)
        vectorizer = fit_and_save_tfidf(stored_chunks, user_id, source_id)

    sparse_q = scipy_row_to_dict(vectorizer.transform([question]).getrow(0))

    return milvus_hybrid_search(dense_q, sparse_q, user_id, source_id, limit=top_k)


# ═══════════════════════════════════════════════════════════════════════════════
# 7. STREAMING QUERY  (SSE generator)
# ═══════════════════════════════════════════════════════════════════════════════

def query_file_stream(
    file_path: str,
    question: str,
    user_id: str,
) -> Generator[str, None, None]:
    """
    Full RAG pipeline as an SSE generator.

    Yields lines in SSE format:
        data: {"status": "..."}\\n\\n   – progress updates
        data: {"token": "..."}\\n\\n    – individual LLM tokens
        data: {"error": "..."}\\n\\n    – error message
        data: [DONE]\\n\\n              – stream terminator

    Temp file cleanup is guaranteed in the finally block.
    """
    try:
        source_id = _compute_source_id(file_path)

        if not source_exists(user_id, source_id):
            yield _sse({"status": "📄 Extracting text and building index..."})
            _ingest_file(file_path, user_id, source_id)
            yield _sse({"status": "✅ Indexed! Searching for relevant passages..."})
        else:
            yield _sse({"status": "⚡ Cache hit — searching index instantly..."})

        top_chunks = _retrieve_top_chunks(question, user_id, source_id)

        if not top_chunks:
            yield _sse({"token": "I cannot find that information in the provided source material."})
            yield "data: [DONE]\n\n"
            return

        context_str = "\n\n---\n\n".join(
            f"[Block {i + 1}]\n{chunk}" for i, chunk in enumerate(top_chunks)
        )
        prompt = _STRICT_QA_PROMPT.format(context=context_str, question=question)

        # Stream LLM tokens
        stream = _groq.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            stream=True,
            temperature=0.1,
            max_tokens=1024,
        )
        for chunk in stream:
            content = chunk.choices[0].delta.content
            if content:
                yield _sse({"token": content})

        yield "data: [DONE]\n\n"

    except Exception as exc:
        yield _sse({"error": str(exc)})
        yield "data: [DONE]\n\n"
    finally:
        # Always clean up the temporary upload file
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except OSError:
            pass


# ── Helper ────────────────────────────────────────────────────────────────────

def _sse(payload: dict) -> str:
    """Format a dict as a single SSE data line."""
    return f"data: {json.dumps(payload)}\n\n"
