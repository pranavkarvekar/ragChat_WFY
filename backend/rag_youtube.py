import os
import tempfile
import yt_dlp
import hashlib
from dotenv import load_dotenv
from groq import Groq
import re

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

def _compute_source_id(url: str) -> str:
    """Content-addressed source_id: SHA-256 of the YouTube URL."""
    return hashlib.sha256(url.encode("utf-8")).hexdigest()[:64]

def download_youtube_audio_temp(url: str) -> str:
    temp_dir = tempfile.mkdtemp()
    temp_file_tmpl = os.path.join(temp_dir, "audio.%(ext)s")
    
    # Optimized low-bandwidth options to bypass throttling and minimize latency
    ydl_opts = {
        "format": "ba[ext=m4a]/ba[ext=webm]/worstaudio",
        "outtmpl": temp_file_tmpl,
        "quiet": True,
        "nokeepalive": True,
        "concurrent_fragment_downloads": 5,
        "buffersize": 1024 * 16,
        "nocheckcertificate": True,
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
    except Exception as exc:
        msg = str(exc)
        if "Sign in to confirm" in msg or "bot" in msg.lower() or "cookies" in msg.lower():
            raise RuntimeError(
                "YouTube is blocking audio download from this server because it requires "
                "browser sign-in verification. The video may not have English captions. "
                "Please try a different YouTube video that has English captions enabled."
            ) from exc
        raise
    
    files = os.listdir(temp_dir)
    if not files:
        raise RuntimeError("No audio file downloaded by yt-dlp.")
    return os.path.join(temp_dir, files[0])

def transcribe_audio_with_groq(audio_path: str) -> str:
    with open(audio_path, "rb") as fh:
        transcription = groq_client.audio.transcriptions.create(
            file=fh,
            model="whisper-large-v3-turbo",
            response_format="text",
            language="en"
        )
    return transcription

_YOUTUBE_ID_REGEX = re.compile(r"(?:youtube\.com\/(?:[^\/]+\/.+\/|(?:v|e(?:mbed)?)\/|.*[?&]v=)|youtu\.be\/)([^\"&?\/\s]{11})")

def _extract_video_id(url: str) -> str | None:
    m = _YOUTUBE_ID_REGEX.search(url or "")
    return m.group(1) if m else None

def _fetch_transcript_fast(url: str) -> str | None:
    """Try to fetch captions quickly without downloading audio. Returns text or None."""
    if YouTubeTranscriptApi is None:
        return None
    video_id = _extract_video_id(url)
    if not video_id:
        return None
    try:
        # First try common English locale codes
        try:
            transcripts = YouTubeTranscriptApi.get_transcript(
                video_id, languages=["en", "en-US", "en-GB", "en-IN", "a.en"]
            )
        except Exception:
            # Fallback to listing transcripts and grabbing the first available
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
            transcript_obj = None
            for t in transcript_list:
                transcript_obj = t
                break
            if not transcript_obj:
                return None
            transcripts = transcript_obj.fetch()

        text = " ".join([seg.get("text", "") for seg in transcripts if seg.get("text")])
        return text.strip() or None
    except Exception:
        return None

def _ingest_youtube(url: str, user_id: str, source_id: str, status_cb=None) -> None:
    """Scrape captions or download audio, chunk, embed, and index YouTube video content into Milvus."""
    if status_cb: status_cb("📺 Checking for video captions...")
    transcript = _fetch_transcript_fast(url)
    
    if not transcript:
        if status_cb: status_cb("⚡ Downloading low-bandwidth audio stream...")
        audio_file = None
        try:
            audio_file = download_youtube_audio_temp(url)
            if status_cb: status_cb("🎙️ Transcribing audio with Groq Whisper...")
            transcript = transcribe_audio_with_groq(audio_file)
        finally:
            if audio_file and os.path.exists(audio_file):
                try:
                    os.remove(audio_file)
                    os.rmdir(os.path.dirname(audio_file))
                except Exception:
                    pass
                    
    if not transcript or not transcript.strip():
        raise ValueError("Could not extract or transcribe content from the YouTube video.")
        
    if status_cb: status_cb("✂️ Splitting content into chunks...")
    from .rag_file import _structural_chunk  # pyrefly: ignore [missing-import]
    chunks = _structural_chunk(transcript)
    if not chunks:
        raise ValueError("Transcription content could not be split into indexable chunks.")
        
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

def _retrieve_top_chunks(
    question: str,
    user_id: str,
    source_id: str,
    top_k: int = 5,
) -> list[str]:
    """Retrieve top chunks for YouTube RAG queries using dense-sparse hybrid search."""
    embedder = get_dense_embedder()
    dense_q = embedder.encode(
        [question], normalize_embeddings=True, show_progress_bar=False
    )[0].tolist()
    
    try:
        tfidf = load_tfidf(user_id, source_id)
    except FileNotFoundError:
        # bm25_models/ was wiped by Render's ephemeral FS — rebuild from Milvus chunks
        import logging as _logging
        _logging.getLogger(__name__).warning(
            "TF-IDF pkl missing for YouTube source %s — rebuilding from Milvus chunks.", source_id
        )
        stored_chunks = fetch_all_chunks(user_id, source_id)
        if not stored_chunks:
            return milvus_hybrid_search(dense_q, {}, user_id, source_id, limit=top_k)
        tfidf = fit_and_save_tfidf(stored_chunks, user_id, source_id)

    sparse_q = scipy_row_to_dict(tfidf.transform([question]).getrow(0))
    
    return milvus_hybrid_search(dense_q, sparse_q, user_id, source_id, limit=top_k)
