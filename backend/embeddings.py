"""
embeddings.py
=============
Lightweight embedding engine using ONNX Runtime (no PyTorch needed).

Replaces the original sentence-transformers + PyTorch stack (~400 MB RAM)
with onnxruntime + tokenizers (~50 MB RAM) while producing identical
384-dim embeddings from the same all-MiniLM-L6-v2 model.

TF-IDF (sparse vectors)
  Each (user_id, source_id) pair has its own TfidfVectorizer fitted on the
  chunk corpus at ingest time and pickled to:
      <project_root>/bm25_models/<user_id>/<source_id>.pkl

The "bm25_models" naming is intentional — TF-IDF with sublinear_tf=True
produces BM25-like behaviour for sparse keyword retrieval.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger(__name__)

# bm25_models/ lives alongside manage.py (ragWFY/bm25_models/)
_BM25_DIR = Path(__file__).resolve().parent.parent / "bm25_models"

# ONNX model exported during build.sh (sentence-transformers/all-MiniLM-L6-v2)
_ONNX_MODEL_DIR = Path(__file__).resolve().parent.parent / "onnx_model"

# ── Lazy singletons ───────────────────────────────────────────────────────────

_ort_session = None
_tokenizer = None


def _download_file_if_missing(url: str, dest_path: Path) -> None:
    if dest_path.exists():
        return
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading missing model file from %s...", url)
    import urllib.request
    import shutil
    with urllib.request.urlopen(url) as response, open(dest_path, "wb") as out_file:
        shutil.copyfileobj(response, out_file)
    logger.info("Downloaded successfully to %s", dest_path)


def _load_onnx_model() -> None:
    """Load ONNX model and tokenizer lazily on first use (auto-downloading if missing)."""
    global _ort_session, _tokenizer
    if _ort_session is not None:
        return

    import onnxruntime as ort
    from tokenizers import Tokenizer

    model_path = _ONNX_MODEL_DIR / "model.onnx"
    tokenizer_path = _ONNX_MODEL_DIR / "tokenizer.json"

    if not model_path.exists() or not tokenizer_path.exists():
        logger.info("ONNX model files missing on disk. Auto-downloading from Hugging Face Hub...")
        _ONNX_MODEL_DIR.mkdir(parents=True, exist_ok=True)
        _download_file_if_missing(
            "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/onnx/model.onnx",
            model_path,
        )
        _download_file_if_missing(
            "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/tokenizer.json",
            tokenizer_path,
        )

    logger.info("Loading ONNX model from: %s", model_path)
    _ort_session = ort.InferenceSession(
        str(model_path),
        providers=["CPUExecutionProvider"],
    )

    logger.info("Loading tokenizer from: %s", tokenizer_path)
    _tokenizer = Tokenizer.from_file(str(tokenizer_path))
    _tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")
    _tokenizer.enable_truncation(max_length=512)

    logger.info("ONNX embedding model loaded successfully.")


def _mean_pooling(
    token_embeddings: np.ndarray,
    attention_mask: np.ndarray,
) -> np.ndarray:
    """Apply mean pooling over token embeddings, respecting the attention mask."""
    mask_expanded = np.expand_dims(attention_mask, axis=-1).astype(np.float32)
    summed = np.sum(token_embeddings * mask_expanded, axis=1)
    counts = np.clip(mask_expanded.sum(axis=1), a_min=1e-9, a_max=None)
    return summed / counts


def _l2_normalize(vectors: np.ndarray) -> np.ndarray:
    """L2 normalize each row vector."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.clip(norms, a_min=1e-9, a_max=None)
    return vectors / norms


class ONNXEmbedder:
    """
    Drop-in replacement for SentenceTransformer.

    Exposes the same ``.encode()`` interface so all existing code
    (rag_file.py, rag_web.py, rag_youtube.py) works without changes.
    """

    def encode(
        self,
        sentences,
        normalize_embeddings: bool = True,
        show_progress_bar: bool = False,
        batch_size: int = 32,
    ) -> np.ndarray:
        _load_onnx_model()

        if isinstance(sentences, str):
            sentences = [sentences]

        all_embeddings: list[np.ndarray] = []

        for i in range(0, len(sentences), batch_size):
            batch = sentences[i : i + batch_size]
            encoded = _tokenizer.encode_batch(batch)

            input_ids = np.array([e.ids for e in encoded], dtype=np.int64)
            attention_mask = np.array(
                [e.attention_mask for e in encoded], dtype=np.int64
            )
            token_type_ids = np.zeros_like(input_ids)

            # Run ONNX inference
            outputs = _ort_session.run(
                None,
                {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "token_type_ids": token_type_ids,
                },
            )

            # outputs[0] = last_hidden_state (batch, seq_len, 384)
            embeddings = _mean_pooling(outputs[0], attention_mask)

            if normalize_embeddings:
                embeddings = _l2_normalize(embeddings)

            all_embeddings.append(embeddings)

        return np.vstack(all_embeddings)


# ── Public interface (same function name as before) ───────────────────────────

_dense_model: ONNXEmbedder | None = None


def get_dense_embedder() -> ONNXEmbedder:
    """Return the cached ONNX embedding model (same API as SentenceTransformer)."""
    global _dense_model
    if _dense_model is None:
        _dense_model = ONNXEmbedder()
    return _dense_model


# ── TF-IDF persistence ────────────────────────────────────────────────────────

def _tfidf_path(user_id: str, source_id: str) -> Path:
    user_dir = _BM25_DIR / user_id
    user_dir.mkdir(parents=True, exist_ok=True)
    return user_dir / f"{source_id}.pkl"


def fit_and_save_tfidf(chunks: list[str], user_id: str, source_id: str) -> TfidfVectorizer:
    """
    Fit a TF-IDF vectorizer on `chunks`, persist it to disk, and return it.

    Parameters mirror BM25:  sublinear_tf=True converts raw term frequencies
    to log scale, approximating BM25 saturation.
    """
    vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),    # unigrams + bigrams for better phrase recall
        min_df=1,
        max_features=65536,    # cap vocabulary to keep sparse vectors bounded
        sublinear_tf=True,     # log(tf) — BM25-like term saturation
    )
    vectorizer.fit(chunks)
    path = _tfidf_path(user_id, source_id)
    with open(path, "wb") as fh:
        pickle.dump(vectorizer, fh, protocol=pickle.HIGHEST_PROTOCOL)
    return vectorizer


def load_tfidf(user_id: str, source_id: str) -> TfidfVectorizer:
    """Load and return a previously fitted TF-IDF vectorizer from disk."""
    path = _tfidf_path(user_id, source_id)
    if not path.exists():
        raise FileNotFoundError(
            f"TF-IDF model not found for source '{source_id}'. "
            "The document may need to be re-indexed."
        )
    with open(path, "rb") as fh:
        return pickle.load(fh)


# ── Sparse vector utilities ───────────────────────────────────────────────────

def scipy_row_to_dict(sparse_row) -> dict[int, float]:
    """
    Convert a single scipy sparse matrix row into a Milvus-compatible
    sparse vector dict: {feature_index: float_value, ...}.

    Zero values are excluded — Milvus SPARSE_FLOAT_VECTOR is stored in COO
    format and ignores zeros automatically, but we trim them here for clarity.
    """
    coo = sparse_row.tocoo()
    return {int(j): float(v) for j, v in zip(coo.col, coo.data) if v != 0.0}
