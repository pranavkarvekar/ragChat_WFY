"""
embeddings.py
=============
Singleton model management for all AI models used in the RAG pipeline.

Models (all loaded lazily and cached as module-level singletons):
  - Dense embedder  : sentence-transformers/all-MiniLM-L6-v2  (384-dim)
  - Cross-encoder   : BAAI/bge-reranker-base                  (reranking)

TF-IDF (sparse vectors)
  Each (user_id, source_id) pair has its own TfidfVectorizer fitted on the
  chunk corpus at ingest time and pickled to:
      <project_root>/bm25_models/<user_id>/<source_id>.pkl

The "bm25_models" naming is intentional — TF-IDF with sublinear_tf=True
produces BM25-like behaviour for sparse keyword retrieval.
"""

from __future__ import annotations

import pickle
from pathlib import Path

from sentence_transformers import CrossEncoder, SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

# bm25_models/ lives alongside manage.py (ragWFY/bm25_models/)
_BM25_DIR = Path(__file__).resolve().parent.parent / "bm25_models"

# ── Lazy singletons ───────────────────────────────────────────────────────────

_dense_model: SentenceTransformer | None = None
_reranker: CrossEncoder | None = None


def get_dense_embedder() -> SentenceTransformer:
    """Return the cached all-MiniLM-L6-v2 embedding model."""
    global _dense_model
    if _dense_model is None:
        _dense_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _dense_model


def get_reranker() -> CrossEncoder:
    """Return the cached BGE cross-encoder reranker model."""
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder("BAAI/bge-reranker-base", device="cpu")
    return _reranker


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
