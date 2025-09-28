import os
import tempfile
import yt_dlp
from dotenv import load_dotenv
from groq import Groq
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
import re
try:
    from youtube_transcript_api import YouTubeTranscriptApi
except Exception:
    YouTubeTranscriptApi = None

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

groq_client = Groq(api_key=GROQ_API_KEY)
llm = ChatGroq(
    model_name="llama-3.3-70b-versatile",
    temperature=0.7,
    api_key=GROQ_API_KEY
)

def download_youtube_audio_temp(url: str) -> str:
    temp_dir = tempfile.mkdtemp()
    temp_file_path = os.path.join(temp_dir, "audio.m4a")
    ydl_opts = {"format": "bestaudio/best", "outtmpl": temp_file_path, "quiet": True}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    return temp_file_path

def transcribe_audio_with_groq(audio_path: str) -> str:
    with open(audio_path, "rb") as fh:
        transcription = groq_client.audio.transcriptions.create(
            file=fh,
            model="whisper-large-v3-turbo",
            response_format="text",
            language="en"
        )
    return transcription

def ask_question(transcript: str, question: str) -> str:
    template = PromptTemplate(
        input_variables=["transcript", "question"],
        template="""
You are an assistant that answers questions strictly based on the transcript of a YouTube video.

Transcript:
{transcript}

Question: {question}

Answer only from the transcript. If the answer is not present, say "The transcript does not contain that information."
"""
    )
    prompt = template.format(transcript=transcript, question=question)
    out = llm.invoke(prompt)
    return getattr(out, "content", str(out))

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
        transcripts = YouTubeTranscriptApi.get_transcript(video_id, languages=["en"])
        text = " ".join([seg.get("text", "") for seg in transcripts if seg.get("text")])
        return text.strip() or None
    except Exception:
        return None

def process_youtube(url: str):
    # Fast path: use captions if available
    fast = _fetch_transcript_fast(url)
    if fast:
        return {"transcript": fast}

    # Fallback: download audio and run Whisper via Groq
    audio_file = None
    try:
        audio_file = download_youtube_audio_temp(url)
        transcript = transcribe_audio_with_groq(audio_file)
        return {"transcript": transcript}
    finally:
        if audio_file and os.path.exists(audio_file):
            try:
                os.remove(audio_file)
            except Exception:
                pass

