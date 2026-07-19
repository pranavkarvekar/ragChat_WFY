"""
views.py
========
Django API views for the RAG platform.

Ingestion of files and URLs now happens asynchronously in background threads.
HTTP requests immediately return a 202 Accepted status with a unique source_id,
and the frontend polls the status endpoint to monitor ingestion progress.

Once ready, chat queries stream LLM tokens via a unified SSE chat endpoint.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import tempfile
import threading
from typing import Generator
from concurrent.futures import ThreadPoolExecutor

# pyrefly: ignore [missing-import]
from django.http import JsonResponse, StreamingHttpResponse
# pyrefly: ignore [missing-import]
from django.views.decorators.csrf import csrf_exempt
# pyrefly: ignore [missing-import]
from django.views.decorators.http import require_POST, require_GET
from dotenv import load_dotenv
from groq import Groq

# pyrefly: ignore [missing-import]
from .rag_file import _compute_source_id as _compute_file_source_id
# pyrefly: ignore [missing-import]
from .rag_web import _compute_source_id as _compute_web_source_id
# pyrefly: ignore [missing-import]
from .rag_youtube import _compute_source_id as _compute_youtube_source_id
# pyrefly: ignore [missing-import]
from .milvus_client import source_exists


load_dotenv()
_groq = Groq(api_key=os.getenv("GROQ_API_KEY"))
logger = logging.getLogger(__name__)

# ── Prompt templates ──────────────────────────────────────────────────────────

# Used for OPEN-ENDED questions: summaries, overviews, explanations.
# Encourages synthesis across blocks rather than refusing when the answer
# isn't stated verbatim.
_SYNTHESIS_PROMPT = """\
You are a helpful document assistant. Using ONLY the context blocks below,
which were extracted from the user's uploaded document, provide a thorough
and well-structured answer to their question.

RULES:
1. Base your answer ONLY on the provided context blocks.
2. Synthesize and connect information across multiple blocks where helpful.
3. If the context is sparse on a topic, share what IS available and note any gaps.
4. Be thorough, organised, and clear. Use bullet points or sections if helpful.

--- CONTEXT BLOCKS ---
{context}
--- END CONTEXT ---

Question: {question}

Answer:"""

# ── Background Task Runner & Tracker ──────────────────────────────────────────

INGESTION_STATUS = {}
_status_lock = threading.Lock()
INGESTION_EXECUTOR = ThreadPoolExecutor(max_workers=4)


def _get_status(user_id: str, source_id: str) -> dict | None:
    with _status_lock:
        return INGESTION_STATUS.get((user_id, source_id))


def _set_status(user_id: str, source_id: str, status: str, progress: str = "", error: str = ""):
    with _status_lock:
        INGESTION_STATUS[(user_id, source_id)] = {
            "status": status,
            "progress": progress,
            "error": error
        }


def _bg_ingest_file(file_path: str, user_id: str, source_id: str):
    # pyrefly: ignore [missing-import]
    from rag_file import _ingest_file
    
    def cb(progress_msg):
        _set_status(user_id, source_id, "processing", progress_msg)
        
    try:
        cb("📄 Extracting text...")
        _ingest_file(file_path, user_id, source_id, status_cb=cb)
        _set_status(user_id, source_id, "ready")
    except Exception as e:
        logger.exception("Background file ingestion failed for source_id: %s", source_id)
        _set_status(user_id, source_id, "failed", error=str(e))
    finally:
        try:
            if os.path.exists(file_path):
                os.unlink(file_path)
        except OSError:
            pass


def _bg_ingest_url(url: str, user_id: str, source_id: str):
    # pyrefly: ignore [missing-import]
    from rag_web import _ingest_url
    
    def cb(progress_msg):
        _set_status(user_id, source_id, "processing", progress_msg)
        
    try:
        cb("🌐 Scraping webpage...")
        _ingest_url(url, user_id, source_id, status_cb=cb)
        _set_status(user_id, source_id, "ready")
    except Exception as e:
        logger.exception("Background URL ingestion failed for source_id: %s", source_id)
        _set_status(user_id, source_id, "failed", error=str(e))


def _bg_ingest_youtube(url: str, user_id: str, source_id: str):
    # pyrefly: ignore [missing-import]
    from rag_youtube import _ingest_youtube
    
    def cb(progress_msg):
        _set_status(user_id, source_id, "processing", progress_msg)
        
    try:
        cb("🎬 Processing YouTube video...")
        _ingest_youtube(url, user_id, source_id, status_cb=cb)
        _set_status(user_id, source_id, "ready")
    except Exception as e:
        logger.exception("Background YouTube ingestion failed for source_id: %s", source_id)
        _set_status(user_id, source_id, "failed", error=str(e))


# ── Utilities ─────────────────────────────────────────────────────────────────

def _user_id(request) -> str:
    """
    Return a stable string identifier for the requesting user.

    Authenticated users → their DB primary key (str).
    Unauthenticated requests → a per-session anonymous ID (so vectors are
    still namespaced and don't bleed between anonymous visitors).
    """
    if request.user.is_authenticated:
        return str(request.user.id)
    anon_id = request.session.get("anon_id")
    if not anon_id:
        import uuid
        anon_id = f"anon_{uuid.uuid4().hex[:12]}"
        request.session["anon_id"] = anon_id
    return anon_id


def _sse(payload: dict) -> str:
    """Format a dict as a single SSE data line."""
    return f"data: {json.dumps(payload)}\n\n"


def _sse_response(generator) -> StreamingHttpResponse:
    """Wrap a generator in a StreamingHttpResponse with correct SSE headers."""
    response = StreamingHttpResponse(
        streaming_content=generator,
        content_type="text/event-stream",
    )
    response["X-Accel-Buffering"] = "no"   # disable Nginx buffering
    response["Cache-Control"]     = "no-cache"
    return response


# ── Query helpers ─────────────────────────────────────────────────────────────

# Keywords that signal the user wants a synthesis / overview rather than a
# specific fact — triggers the softer _SYNTHESIS_PROMPT.
_SUMMARY_TRIGGERS = {
    "summary", "summarize", "summarise", "summarization",
    "overview", "outline", "brief", "briefing",
    "explain", "explanation", "describe", "description",
    "detail", "details", "elaborate",
    "about", "tell me", "what is", "what are", "what does",
    "key points", "main points", "highlights", "takeaway",
    "introduction", "conclusion",
}

_TOC_LINE_RE = re.compile(
    r'\.{3,}\s*\d+\s*$'          # "Chapter 3 ............ 42"
    r'|\b(?:page|pg\.?)\s*\d+'   # "page 42" / "pg. 42"
    r'|^\d+\s*$',                 # lines that are just a page number
    re.IGNORECASE,
)


def _is_synthesis_query(question: str) -> bool:
    """Return True when the question wants a summary/overview rather than a fact."""
    q = question.lower()
    return any(trigger in q for trigger in _SUMMARY_TRIGGERS)


def _filter_toc_chunks(chunks: list[str]) -> list[str]:
    """
    Remove table-of-contents chunks from the candidate list.

    A chunk is considered a TOC entry when >55% of its non-empty lines
    match the TOC pattern (trailing dots + page number, or bare page number).
    These chunks contain only references to where content lives, not the
    content itself, and confuse the LLM.
    """
    good: list[str] = []
    for chunk in chunks:
        lines = [l.strip() for l in chunk.split("\n") if l.strip()]
        if not lines:
            continue
        toc_lines = sum(1 for l in lines if _TOC_LINE_RE.search(l))
        if len(lines) > 0 and toc_lines / len(lines) > 0.55:
            continue  # skip this TOC chunk
        good.append(chunk)
    # If filtering removed everything, fall back to original list so we
    # always have something to give the LLM.
    return good if good else chunks


# ── Chat streaming generator ──────────────────────────────────────────────────

def query_chat_stream(
    source_id: str,
    question: str,
    user_id: str,
    source_type: str,
) -> Generator[str, None, None]:
    try:
        is_synthesis = _is_synthesis_query(question)
        # Use more candidates for open-ended/summary queries so we look
        # past table-of-contents entries and find actual content.
        top_k = 15 if is_synthesis else 8

        if source_type == "web":
            from .rag_web import _retrieve_top_chunks, _STRICT_QA_PROMPT
        elif source_type == "youtube":
            from .rag_youtube import _retrieve_top_chunks, _STRICT_QA_PROMPT
        else:
            from .rag_file import _retrieve_top_chunks, _STRICT_QA_PROMPT

        top_chunks = _retrieve_top_chunks(question, user_id, source_id, top_k=top_k)

        # Strip table-of-contents chunks — they reference page numbers but
        # contain no actual content, causing "cannot find" false negatives.
        top_chunks = _filter_toc_chunks(top_chunks)

        if not top_chunks:
            yield _sse({"token": "I could not find relevant content in the indexed document. "
                                 "Please try rephrasing your question."})
            yield "data: [DONE]\n\n"
            return

        context_str = "\n\n---\n\n".join(
            f"[Block {i + 1}]\n{chunk}" for i, chunk in enumerate(top_chunks)
        )

        # Pick prompt based on whether the question wants synthesis or a fact
        if is_synthesis:
            prompt = _SYNTHESIS_PROMPT.format(context=context_str, question=question)
        else:
            prompt = _STRICT_QA_PROMPT.format(context=context_str, question=question)

        stream = _groq.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            stream=True,
            temperature=0.2 if is_synthesis else 0.1,
            max_tokens=1536 if is_synthesis else 1024,
        )
        for chunk in stream:
            content = chunk.choices[0].delta.content
            if content:
                yield _sse({"token": content})

        yield "data: [DONE]\n\n"

    except Exception as exc:
        yield _sse({"error": str(exc)})
        yield "data: [DONE]\n\n"


# ── API Views ─────────────────────────────────────────────────────────────────

@csrf_exempt
@require_POST
def api_web_chat(request):
    """
    POST /api/web/
    Form fields: url (str)
    Returns: JSON {status, source_id}
    """
    url = request.POST.get("url", "").strip()
    if not url:
        return JsonResponse({"error": "'url' is required."}, status=400)

    user_uid = _user_id(request)
    source_id = _compute_web_source_id(url)

    # Cache hit
    if source_exists(user_uid, source_id):
        _set_status(user_uid, source_id, "ready")
        return JsonResponse({"status": "ready", "source_id": source_id})

    # Already processing check
    status_info = _get_status(user_uid, source_id)
    if status_info and status_info["status"] in ("processing", "ready"):
        return JsonResponse({"status": status_info["status"], "source_id": source_id})

    # Start ingestion
    _set_status(user_uid, source_id, "processing", "🌐 Scraping webpage...")
    INGESTION_EXECUTOR.submit(_bg_ingest_url, url, user_uid, source_id)
    
    return JsonResponse({"status": "processing", "source_id": source_id}, status=202)


@csrf_exempt
@require_POST
def api_file_chat(request):
    """
    POST /api/files/
    Form fields: file (multipart upload)
    Returns: JSON {status, source_id}
    """
    upload = request.FILES.get("file")
    if not upload:
        return JsonResponse({"error": "'file' (upload) is required."}, status=400)

    # Save to a temporary file to compute SHA256 and process
    suffix = os.path.splitext(upload.name)[1]
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        h = hashlib.sha256()
        for chunk in upload.chunks():
            tmp.write(chunk)
            h.update(chunk)
        tmp.close()
        source_id = h.hexdigest()[:64]
    except Exception as exc:
        tmp.close()
        try:
            os.unlink(tmp.name)
        except OSError:
            pass
        return JsonResponse({"error": f"File upload failed: {exc}"}, status=500)

    user_uid = _user_id(request)

    # Cache hit
    if source_exists(user_uid, source_id):
        try:
            os.unlink(tmp.name)
        except OSError:
            pass
        _set_status(user_uid, source_id, "ready")
        return JsonResponse({"status": "ready", "source_id": source_id})

    # Already processing check
    status_info = _get_status(user_uid, source_id)
    if status_info and status_info["status"] in ("processing", "ready"):
        try:
            os.unlink(tmp.name)
        except OSError:
            pass
        return JsonResponse({"status": status_info["status"], "source_id": source_id})

    # Start background ingestion
    _set_status(user_uid, source_id, "processing", "📄 Uploading and parsing file...")
    INGESTION_EXECUTOR.submit(_bg_ingest_file, tmp.name, user_uid, source_id)

    return JsonResponse({"status": "processing", "source_id": source_id}, status=202)


@csrf_exempt
@require_GET
def api_status(request):
    """
    GET /api/status/?source_id=<id>
    Returns: JSON status information
    """
    source_id = request.GET.get("source_id", "").strip()
    if not source_id:
        return JsonResponse({"error": "'source_id' query parameter is required."}, status=400)

    user_uid = _user_id(request)
    
    # Check global tracker first
    status_info = _get_status(user_uid, source_id)
    if status_info:
        return JsonResponse(status_info)

    # If not in tracker, query Milvus to check if it has already been loaded previously
    if source_exists(user_uid, source_id):
        _set_status(user_uid, source_id, "ready")
        return JsonResponse({"status": "ready", "progress": "", "error": ""})

    # Default to not found/failed
    return JsonResponse({"status": "failed", "progress": "", "error": "Source not found or failed initialization."}, status=404)


@csrf_exempt
@require_POST
def api_chat(request):
    """
    POST /api/chat/
    Form fields: source_id (str), question (str), source_type (str: 'file' or 'web')
    Returns: SSE stream of tokens
    """
    source_id = request.POST.get("source_id", "").strip()
    question = request.POST.get("question", "").strip()
    source_type = request.POST.get("source_type", "file").strip()

    if not source_id or not question:
        return JsonResponse({"error": "Both 'source_id' and 'question' are required."}, status=400)

    user_uid = _user_id(request)
    
    # Confirm ready
    if not source_exists(user_uid, source_id):
        status_info = _get_status(user_uid, source_id)
        if status_info and status_info["status"] == "failed":
            return JsonResponse({"error": f"Ingestion failed: {status_info['error']}"}, status=400)
        return JsonResponse({"error": "Document is still indexing. Please wait."}, status=400)

    return _sse_response(query_chat_stream(source_id, question, user_uid, source_type))


@csrf_exempt
@require_POST
def api_youtube_chat(request):
    """
    POST /api/youtube/
    Form fields: url (str)
    Returns: JSON {status, source_id}
    """
    url = request.POST.get("url", "").strip()
    if not url:
        return JsonResponse({"error": "'url' is required."}, status=400)

    user_uid = _user_id(request)
    source_id = _compute_youtube_source_id(url)

    # Cache hit
    if source_exists(user_uid, source_id):
        _set_status(user_uid, source_id, "ready")
        return JsonResponse({"status": "ready", "source_id": source_id})

    # Already processing check
    status_info = _get_status(user_uid, source_id)
    if status_info and status_info["status"] in ("processing", "ready"):
        return JsonResponse({"status": status_info["status"], "source_id": source_id})

    # Start background ingestion
    _set_status(user_uid, source_id, "processing", "🎬 Processing YouTube video...")
    INGESTION_EXECUTOR.submit(_bg_ingest_youtube, url, user_uid, source_id)

    return JsonResponse({"status": "processing", "source_id": source_id}, status=202)
