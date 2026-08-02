"""
rag_web.py
==========
Production-grade, stateful RAG pipeline for web pages.

Scraping strategy (no Firecrawl API key required)
--------------------------------------------------
Uses httpx + html2text to download and convert any public webpage into clean
Markdown.  This removes nav bars, ads, and boilerplate automatically.

Pipeline stages
---------------
1. SCRAPE     — httpx GET → html2text → clean Markdown
2. CHUNK      — Markdown-aware chunking (headings → paragraphs → chars)
3. EMBED      — Dense: all-MiniLM-L6-v2  |  Sparse: TF-IDF (BM25-like)
4. STORE      — Milvus Lite HNSW + SPARSE_INVERTED_INDEX, tagged user_id/source_id
5. RETRIEVE   — Hybrid search (AnnSearchRequest × 2 → RRFRanker, top-20)
6. RERANK     — BAAI/bge-reranker-base CrossEncoder → top-3 chunks
7. GENERATE   — Groq LLaMA-3.3-70b with strict QA prompt, streamed via SSE

Caching: source_id = SHA-256(url).  Same URL returns instant results on
second and subsequent queries without re-scraping or re-embedding.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from typing import Generator

from dotenv import load_dotenv
from groq import Groq

from .embeddings import (
    fit_and_save_tfidf,
    get_dense_embedder,
    load_tfidf,
    scipy_row_to_dict,
)
from .milvus_client import hybrid_search as milvus_hybrid_search
from .milvus_client import insert_chunks, source_exists

load_dotenv()
_groq = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ── Prompt template ───────────────────────────────────────────────────────────

_STRICT_QA_PROMPT = """\
You are a precise web content analysis assistant.

Your ONLY task is to answer the user's question using EXCLUSIVELY the context \
blocks extracted from the scraped webpage shown below.

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
# 1. SCRAPING
# ═══════════════════════════════════════════════════════════════════════════════

def _fetch_wikipedia(url: str) -> str:
    """
    Fetch a Wikipedia article using the MediaWiki Action API.

    Wikipedia blocks server-side scraping (403) from cloud IPs even with
    a real browser User-Agent.  The official REST API was also decommissioned.
    The Action API is designed for programmatic access and always works.

    Supports:
      - https://en.wikipedia.org/wiki/TITLE
      - https://en.m.wikipedia.org/wiki/TITLE
    """
    import re as _re
    import httpx

    # Extract the article title from the URL
    match = _re.search(r"/wiki/([^#?]+)", url)
    if not match:
        raise ValueError(f"Cannot extract Wikipedia article title from URL: {url}")

    title = match.group(1)  # e.g. "Machine_learning"
    lang = "en"
    lang_match = _re.match(r"https?://([a-z]+)\.(?:m\.)?wikipedia", url)
    if lang_match:
        lang = lang_match.group(1)

    # Use the reliable MediaWiki Action API
    api_url = f"https://{lang}.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "prop": "extracts",
        "explaintext": "1",  # return plain text, not HTML
        "titles": title,
        "format": "json"
    }
    
    headers = {
        "User-Agent": "ragChat-WFY/1.0 (https://github.com/pranavkarvekar/ragChat_WFY)",
    }

    resp = httpx.get(api_url, params=params, timeout=30, follow_redirects=True, headers=headers)
    resp.raise_for_status()
    data = resp.json()

    # The API returns data in: {"query": {"pages": {"<page_id>": {"extract": "..."}}}}
    pages = data.get("query", {}).get("pages", {})
    if not pages or "-1" in pages:
        raise ValueError(f"Wikipedia page not found for title: {title}")
    
    # Get the first page's extract
    page = next(iter(pages.values()))
    extract = page.get("extract", "")
    
    if not extract:
        raise ValueError(f"No text content found for Wikipedia page: {title}")

    return f"# {page.get('title', title)}\n\n{extract}"


def _scrape_url(url: str) -> str:
    """
    Fetch a webpage and convert its HTML to clean Markdown.

    Uses httpx for the HTTP request and html2text for conversion.
    Simulates a real browser User-Agent to avoid simple bot blocks.

    For Wikipedia URLs, uses the Wikipedia REST API to avoid 403 blocks
    from cloud server IPs.
    """
    import html2text
    import httpx

    # Route Wikipedia URLs through the REST API (never 403)
    if "wikipedia.org/wiki/" in url:
        return _fetch_wikipedia(url)

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept":          "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
    }

    try:
        response = httpx.get(url, timeout=30, follow_redirects=True, headers=headers)
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        status = exc.response.status_code
        if status == 403:
            raise ValueError(
                f"Access denied (403) for '{url}'. "
                "This site blocks automated access. Try a different source."
            ) from exc
        elif status == 404:
            raise ValueError(f"Page not found (404): '{url}'") from exc
        else:
            raise ValueError(
                f"HTTP {status} error fetching '{url}': {exc}"
            ) from exc

    converter = html2text.HTML2Text()
    converter.ignore_links   = True   # strip link URLs — reduces chunk noise
    converter.ignore_images  = True
    converter.ignore_tables  = False
    converter.body_width     = 0       # no line wrapping
    converter.single_line_break = False

    return converter.handle(response.text)



# ═══════════════════════════════════════════════════════════════════════════════
# 2. MARKDOWN-AWARE CHUNKING
# ═══════════════════════════════════════════════════════════════════════════════

def _markdown_chunk(
    text: str,
    max_chars: int = 900,
    overlap: int = 150,
) -> list[str]:
    """
    Split Markdown content preserving semantic boundaries.

    Priority:
      1. Heading boundaries  (# / ## / ###)
      2. Paragraph breaks    (\\n\\n)
      3. Hard character cap  (max_chars with overlap)

    Filters out fragments shorter than 40 characters (navigation artefacts).
    """
    raw_blocks = re.split(r"\n(?=#{1,6}\s)", text)

    chunks: list[str] = []
    for block in raw_blocks:
        block = block.strip()
        if not block:
            continue
        if len(block) <= max_chars:
            chunks.append(block)
            continue
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
                    for i in range(0, len(para), max_chars - overlap):
                        piece = para[i : i + max_chars].strip()
                        if piece:
                            chunks.append(piece)
                    current = ""
                else:
                    current = para
        if current:
            chunks.append(current)

    return [c for c in chunks if len(c.strip()) >= 40]


# ═══════════════════════════════════════════════════════════════════════════════
# 3 + 4. INGESTION  (embed → store in Milvus)
# ═══════════════════════════════════════════════════════════════════════════════

def _compute_source_id(url: str) -> str:
    """Content-addressed source_id: SHA-256 of the URL string."""
    return hashlib.sha256(url.encode("utf-8")).hexdigest()[:64]


def _ingest_url(url: str, user_id: str, source_id: str, status_cb=None) -> None:
    """Scrape, chunk, embed, and persist a webpage into Milvus."""
    if status_cb: status_cb("🌐 Scraping webpage content...")
    markdown = _scrape_url(url)
    if not markdown.strip():
        raise ValueError("No readable content found at the provided URL.")

    if status_cb: status_cb("✂️ Splitting content into chunks...")
    chunks = _markdown_chunk(markdown)
    if not chunks:
        raise ValueError("Could not extract meaningful content from the URL.")

    if status_cb: status_cb("🧠 Generating dense embeddings...")
    embedder = get_dense_embedder()
    dense_vecs = embedder.encode(
        chunks, normalize_embeddings=True, show_progress_bar=False, batch_size=32
    )

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
                "chunk_text":    chunk[:4000],
                "dense_vector":  dense_vecs[i].tolist(),
                "sparse_vector": scipy_row_to_dict(sparse_matrix.getrow(i)),
            }
        )

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
    """Hybrid search (dense HNSW + sparse TF-IDF via RRF) → top `top_k` chunks.

    The RRF ranker already fuses dense and sparse signals into a strong
    relevance ordering.  A heavy cross-encoder reranker is skipped to keep
    query latency under 3 seconds on CPU hardware.
    """
    embedder = get_dense_embedder()
    dense_q = embedder.encode(
        [question], normalize_embeddings=True, show_progress_bar=False
    )[0].tolist()

    vectorizer = load_tfidf(user_id, source_id)
    sparse_q = scipy_row_to_dict(vectorizer.transform([question]).getrow(0))

    return milvus_hybrid_search(dense_q, sparse_q, user_id, source_id, limit=top_k)


# ═══════════════════════════════════════════════════════════════════════════════
# 7. STREAMING QUERY  (SSE generator)
# ═══════════════════════════════════════════════════════════════════════════════

def query_website_stream(
    url: str,
    question: str,
    user_id: str,
) -> Generator[str, None, None]:
    """
    Full web RAG pipeline as an SSE generator.

    Yields SSE lines:
        data: {"status": "..."}\\n\\n   – progress updates
        data: {"token": "..."}\\n\\n    – individual LLM tokens
        data: {"error": "..."}\\n\\n    – error message
        data: [DONE]\\n\\n              – stream terminator
    """
    try:
        source_id = _compute_source_id(url)

        if not source_exists(user_id, source_id):
            yield _sse({"status": "🌐 Scraping webpage and building index..."})
            _ingest_url(url, user_id, source_id)
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
        try:
            with open(r"C:\Users\HP\.gemini\antigravity-ide\brain\21575277-9d72-4edc-b629-cfb7d93f8f28\scratch\last_web_prompt.txt", "w", encoding="utf-8") as f:
                f.write(prompt)
        except Exception:
            pass

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


# ── Helper ────────────────────────────────────────────────────────────────────

def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"
