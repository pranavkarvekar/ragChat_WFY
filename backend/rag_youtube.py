"""
rag_youtube.py
==============
Production-grade, stateful RAG pipeline for YouTube videos.

Transcript Extraction (3-tier, no audio download needed)
---------------------------------------------------------
1. youtube-transcript-api  – fast lightweight caption fetch (0.6.2 instance API)
2. yt-dlp subtitle-only    – downloads .vtt/.srt caption files, NO audio/video
3. Clear error             – tells the user the video needs English captions

Why no yt-dlp audio download?
  Cloud/datacenter IPs (Render, Railway, etc.) are blocked by YouTube's
  bot-detection for video/audio downloads. Subtitle files are a plain text
  HTTP request and are rarely blocked.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from typing import Generator

import yt_dlp
from dotenv import load_dotenv
from groq import Groq

# pyrefly: ignore [missing-import]
try:
    from youtube_transcript_api import YouTubeTranscriptApi
except Exception:
    YouTubeTranscriptApi = None

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
from .milvus_client import fetch_all_chunks, insert_chunks

load_dotenv()
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

_STRICT_QA_PROMPT = """\
You are an assistant that answers questions strictly based on the transcript of a YouTube video.

Your ONLY task is to answer the user's question using EXCLUSIVELY the context \
blocks extracted from the video transcript shown below.

STRICT RULES:
1. Base your answer solely on the provided context blocks.
2. Do NOT use any prior knowledge or make assumptions beyond the context.
3. If the answer cannot be confidently found in the context blocks, you MUST \
respond with exactly:
   "The transcript does not contain that information."
4. Be concise, factual, and direct. Reference the block number when possible.

--- CONTEXT BLOCKS ---
{context}
--- END CONTEXT ---

Question: {question}

Answer:"""


# ── Utilities ─────────────────────────────────────────────────────────────────

def _compute_source_id(url: str) -> str:
    """Content-addressed source_id: SHA-256 of the YouTube URL."""
    return hashlib.sha256(url.encode("utf-8")).hexdigest()[:64]


_YOUTUBE_ID_REGEX = re.compile(
    r"(?:youtube\.com\/(?:[^\/]+\/.+\/|(?:v|e(?:mbed)?)\/|.*[?&]v=)|youtu\.be\/)"
    r"([^\"&?\/\s]{11})"
)


def _extract_video_id(url: str) -> str | None:
    m = _YOUTUBE_ID_REGEX.search(url or "")
    return m.group(1) if m else None


def _clean_snippet_text(s) -> str:
    """Return text from either a dict snippet or a FetchedTranscriptSnippet object."""
    if isinstance(s, dict):
        return s.get("text", "")
    return getattr(s, "text", "")


def _parse_vtt_srt(content: str) -> str:
    """Convert a VTT or SRT subtitle file to plain text."""
    lines = content.splitlines()
    text_parts: list[str] = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith("WEBVTT") or line.startswith("NOTE"):
            continue
        # Skip timestamp lines: "00:00:00.000 --> 00:00:05.000" or "1" (SRT counter)
        if re.match(r"[\d:.,]+ --> [\d:.,]+", line):
            continue
        if re.match(r"^\d+$", line):
            continue
        # Strip inline HTML/VTT tags like <c>, <00:00:01.000>
        line = re.sub(r"<[^>]+>", "", line)
        if line:
            text_parts.append(line)
    return " ".join(text_parts).strip()


# ── Tier 1: youtube-transcript-api (0.6.2 instance-based API) ─────────────────

def _fetch_via_transcript_api(url: str) -> str | None:
    """Fast caption fetch using youtube-transcript-api. Returns plain text or None."""
    if YouTubeTranscriptApi is None:
        return None
    video_id = _extract_video_id(url)
    if not video_id:
        return None

    api = YouTubeTranscriptApi()

    # Try English language codes (manual + auto-generated)
    for lang_codes in (
        ["en", "en-US", "en-GB", "en-IN", "en-AU", "en-CA"],
        ["a.en"],
    ):
        try:
            fetched = api.fetch(video_id, languages=lang_codes)
            text = " ".join(_clean_snippet_text(s) for s in fetched).strip()
            if text:
                return text
        except Exception:
            continue

    # Last resort: list every available transcript and grab the first one
    try:
        for transcript_obj in api.list(video_id):
            try:
                fetched = transcript_obj.fetch()
                text = " ".join(_clean_snippet_text(s) for s in fetched).strip()
                if text:
                    return text
            except Exception:
                continue
    except Exception:
        pass

    return None


# ── Tier 2: yt-dlp subtitle-only download (NO audio, NO bot detection) ────────

def _fetch_via_ytdlp_subtitles(url: str) -> str | None:
    """
    Download caption/subtitle files ONLY using yt-dlp (skip_download=True).
    
    This is far lighter than audio download and rarely blocked because it's
    just fetching a small text file — not triggering YouTube's streaming limits.
    Returns plain text or None.
    """
    temp_dir = tempfile.mkdtemp()
    try:
        ydl_opts = {
            "writeautomaticsub": True,     # auto-generated captions
            "writesubtitles": True,         # manual captions
            "subtitleslangs": ["en", "en-US", "en-GB", "en-IN", "a.en"],
            "subtitlesformat": "vtt/srt/best",
            "skip_download": True,          # ← KEY: never download video/audio
            "outtmpl": os.path.join(temp_dir, "%(id)s.%(ext)s"),
            "quiet": True,
            "no_warnings": True,
            "nocheckcertificate": True,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

        # Find the downloaded subtitle file
        for fname in os.listdir(temp_dir):
            if fname.endswith((".vtt", ".srt", ".ttml", ".srv1", ".srv2", ".srv3")):
                fpath = os.path.join(temp_dir, fname)
                with open(fpath, "r", encoding="utf-8", errors="replace") as f:
                    content = f.read()
                text = _parse_vtt_srt(content)
                if text:
                    return text
    except Exception:
        pass
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
    return None


# ── Master ingestion pipeline ──────────────────────────────────────────────────

def _ingest_youtube(url: str, user_id: str, source_id: str, status_cb=None) -> None:
    """Fetch transcript (2 strategies), chunk, embed, and index into Milvus."""

    # Tier 1 — youtube-transcript-api
    if status_cb:
        status_cb("📺 Fetching video captions...")
    transcript = _fetch_via_transcript_api(url)

    # Tier 2 — yt-dlp subtitle-only (no audio download)
    if not transcript:
        if status_cb:
            status_cb("📝 Downloading subtitle file (no audio)...")
        transcript = _fetch_via_ytdlp_subtitles(url)

    if not transcript or not transcript.strip():
        raise ValueError(
            "Could not find captions for this video.\n"
            "This happens when:\n"
            "  • The video has no English captions or auto-generated subtitles\n"
            "  • The video is age-restricted or private\n\n"
            "💡 Try: paste a YouTube URL for a video that has CC captions enabled "
            "(look for the [CC] icon in the YouTube player)."
        )

    if status_cb:
        status_cb("✂️ Splitting content into chunks...")
    from .rag_file import _structural_chunk  # pyrefly: ignore [missing-import]
    chunks = _structural_chunk(transcript)
    if not chunks:
        raise ValueError("Transcription content could not be split into indexable chunks.")

    if status_cb:
        status_cb("🧠 Generating dense embeddings...")
    embedder = get_dense_embedder()
    dense_vecs = embedder.encode(
        chunks, normalize_embeddings=True, show_progress_bar=False, batch_size=32
    )

    if status_cb:
        status_cb("📈 Building keyword index...")
    vectorizer = fit_and_save_tfidf(chunks, user_id, source_id)
    sparse_matrix = vectorizer.transform(chunks)

    if status_cb:
        status_cb("💾 Persisting index in Milvus...")
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


# ── Retrieval ──────────────────────────────────────────────────────────────────

def _retrieve_top_chunks(
    question: str,
    user_id: str,
    source_id: str,
    top_k: int = 5,
) -> list[str]:
    """Hybrid dense+sparse search. Auto-rebuilds TF-IDF from Milvus if pkl is missing."""
    import logging as _logging

    embedder = get_dense_embedder()
    dense_q = embedder.encode(
        [question], normalize_embeddings=True, show_progress_bar=False
    )[0].tolist()

    try:
        tfidf = load_tfidf(user_id, source_id)
    except FileNotFoundError:
        # bm25_models/ was wiped by Render's ephemeral FS — rebuild from Milvus chunks
        _logging.getLogger(__name__).warning(
            "TF-IDF pkl missing for YouTube source %s — rebuilding from Milvus chunks.",
            source_id,
        )
        stored_chunks = fetch_all_chunks(user_id, source_id)
        if not stored_chunks:
            return milvus_hybrid_search(dense_q, {}, user_id, source_id, limit=top_k)
        tfidf = fit_and_save_tfidf(stored_chunks, user_id, source_id)

    sparse_q = scipy_row_to_dict(tfidf.transform([question]).getrow(0))
    return milvus_hybrid_search(dense_q, sparse_q, user_id, source_id, limit=top_k)


# ── Streaming query (SSE generator) ───────────────────────────────────────────

def query_youtube_stream(
    url: str,
    question: str,
    user_id: str,
) -> Generator[str, None, None]:
    """
    Full YouTube RAG pipeline as an SSE generator.

    Yields JSON-encoded SSE events:
      {"type": "status",  "message": "..."}
      {"type": "token",   "content": "..."}
      {"type": "error",   "message": "..."}
      {"type": "done"}
    """

    def _send(event_type: str, **kwargs) -> str:
        return "data: " + json.dumps({"type": event_type, **kwargs}) + "\n\n"

    def _status(msg: str):
        yield _send("status", message=msg)

    source_id = _compute_source_id(url)

    try:
        # ── Ingest (skip if already indexed) ─────────────────────────────────
        from .milvus_client import source_exists  # pyrefly: ignore [missing-import]

        if not source_exists(user_id, source_id):
            yield from _status("📺 Fetching video captions...")

            def status_cb(msg: str):
                pass  # status already sent; individual tier msgs go to logs

            _ingest_youtube(url, user_id, source_id, status_cb=status_cb)
        else:
            yield _send("status", message="⚡ Using cached video index...")

        # ── Retrieve ──────────────────────────────────────────────────────────
        yield from _status("🔍 Finding relevant transcript segments...")
        chunks = _retrieve_top_chunks(question, user_id, source_id)

        if not chunks:
            yield _send("error", message="No relevant transcript segments found for your question.")
            return

        # ── Generate ──────────────────────────────────────────────────────────
        context = "\n\n".join(f"[Block {i+1}]\n{c}" for i, c in enumerate(chunks))
        prompt = _STRICT_QA_PROMPT.format(context=context, question=question)

        yield from _status("💬 Generating answer...")

        stream = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            stream=True,
            temperature=0.1,
            max_tokens=1024,
        )

        for chunk in stream:
            delta = chunk.choices[0].delta
            if delta and delta.content:
                yield _send("token", content=delta.content)

        yield _send("done")

    except Exception as exc:
        yield _send("error", message=str(exc))
