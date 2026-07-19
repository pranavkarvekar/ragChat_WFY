"""
milvus_client.py
================
Centralised Milvus connection, schema management, and search utilities.

Uses Milvus Lite (file-based, no Docker) by default.  Set MILVUS_URI in .env
to 'http://localhost:19530' to switch to a full Milvus server.

Collection schema
-----------------
  pk            INT64 (auto-generated primary key)
  user_id       VARCHAR  – owner of the indexed content
  source_id     VARCHAR  – SHA-256 of the URL or file content (deduplicate)
  chunk_text    VARCHAR  – the actual text chunk stored for retrieval context
  dense_vector  FLOAT_VECTOR[384]  – all-MiniLM-L6-v2 cosine embeddings
  sparse_vector SPARSE_FLOAT_VECTOR – TF-IDF features for lexical search

Indexes
-------
  dense_vector  : HNSW  (M=16, efConstruction=200, metric=COSINE)
  sparse_vector : SPARSE_INVERTED_INDEX (metric=IP)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

# pyrefly: ignore [missing-import]
from pymilvus import AnnSearchRequest, DataType, MilvusClient, RRFRanker

COLLECTION_NAME = "rag_chunks"
DENSE_DIM = 384  # all-MiniLM-L6-v2

_client: MilvusClient | None = None


# ── Connection ────────────────────────────────────────────────────────────────

def _resolve_uri() -> str:
    """
    Resolve the Milvus connection URI at call time (NOT at import time).

    - If MILVUS_DB_PATH starts with 'http', treat it as a remote server URI.
    - Otherwise start a Milvus Lite gRPC server and return its local URI.

    We intentionally avoid calling load_dotenv() at module level so that
    pymilvus does not see MILVUS_URI in os.environ during its own import
    (pymilvus 3.0 reads os.environ['MILVUS_URI'] at import time and raises
    ConnectionConfigException if it contains a file path).
    """
    from dotenv import load_dotenv
    load_dotenv()  # safe here — pymilvus is already fully imported by now

    raw = os.getenv("MILVUS_DB_PATH", "./milvus_ragwfy.db")

    if raw.startswith("http"):
        return raw   # Remote / full Milvus server — use as-is

    # Resolve to absolute path relative to the Django project root
    db_path = Path(__file__).resolve().parent.parent / Path(raw).name
    abs_path = str(db_path)

    # milvus-lite 3.0: start embedded gRPC server, return http://127.0.0.1:PORT
    # On Windows, a stale lock file from a crashed/killed process blocks startup.
    # Safely remove it before calling start_and_get_uri().
    lock_file = Path(abs_path) / "data.lock"
    if lock_file.exists():
        try:
            lock_file.unlink()
        except OSError:
            pass  # another live process may hold it — let Milvus raise naturally

    try:
        # pyrefly: ignore [missing-import]
        from milvus_lite.server_manager import server_manager_instance
        uri = server_manager_instance.start_and_get_uri(abs_path)
        if uri:
            return uri
    except Exception as e:
        raise RuntimeError(
            f"Failed to start Milvus Lite server for '{abs_path}': {e}"
        ) from e

    raise RuntimeError(
        f"Milvus Lite server_manager returned None for path '{abs_path}'. "
        "Check that milvus-lite 3.0 is installed correctly."
    )


def get_client() -> MilvusClient:
    """Return (and lazily initialise) the singleton Milvus client."""
    global _client
    if _client is None:
        uri = _resolve_uri()
        _client = MilvusClient(uri=uri)
        _ensure_collection(_client)
    return _client


# ── Schema / Collection management ───────────────────────────────────────────

def _ensure_collection(client: MilvusClient) -> None:
    """Create the rag_chunks collection if it does not already exist."""
    if client.has_collection(COLLECTION_NAME):
        client.load_collection(COLLECTION_NAME)
        return

    schema = client.create_schema(auto_id=True, enable_dynamic_field=True)
    schema.add_field("pk",            DataType.INT64,           is_primary=True, auto_id=True)
    schema.add_field("user_id",       DataType.VARCHAR,         max_length=256)
    schema.add_field("source_id",     DataType.VARCHAR,         max_length=512)
    schema.add_field("chunk_text",    DataType.VARCHAR,         max_length=4096)
    schema.add_field("dense_vector",  DataType.FLOAT_VECTOR,    dim=DENSE_DIM)
    schema.add_field("sparse_vector", DataType.SPARSE_FLOAT_VECTOR)

    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="dense_vector",
        index_type="HNSW",
        metric_type="COSINE",
        params={"M": 8, "efConstruction": 100},
    )
    index_params.add_index(
        field_name="sparse_vector",
        index_type="SPARSE_INVERTED_INDEX",
        metric_type="IP",
    )

    client.create_collection(
        collection_name=COLLECTION_NAME,
        schema=schema,
        index_params=index_params,
    )


# ── Source helpers ────────────────────────────────────────────────────────────

def source_exists(user_id: str, source_id: str) -> bool:
    """Return True if vectors for this (user, source) already exist in Milvus."""
    try:
        res = get_client().query(
            collection_name=COLLECTION_NAME,
            filter=f'user_id == "{user_id}" and source_id == "{source_id}"',
            output_fields=["pk"],
            limit=1,
        )
        return len(res) > 0
    except Exception:
        return False


def delete_source(user_id: str, source_id: str) -> None:
    """Delete all vectors belonging to this (user, source) — useful for re-index."""
    get_client().delete(
        collection_name=COLLECTION_NAME,
        filter=f'user_id == "{user_id}" and source_id == "{source_id}"',
    )


# ── Insertion ─────────────────────────────────────────────────────────────────

def insert_chunks(records: list[dict]) -> None:
    """
    Bulk-insert chunk records into Milvus.

    Each record must contain:
        user_id       : str
        source_id     : str
        chunk_text    : str  (≤ 4096 chars)
        dense_vector  : list[float]  (len == DENSE_DIM)
        sparse_vector : dict[int, float]  (TF-IDF feature map)
    """
    if not records:
        return
    get_client().insert(collection_name=COLLECTION_NAME, data=records)


# ── Hybrid Search ─────────────────────────────────────────────────────────────

def hybrid_search(
    dense_query: list[float],
    sparse_query: dict[int, float],
    user_id: str,
    source_id: str,
    limit: int = 5,
) -> list[str]:
    """
    Run a hybrid dense + sparse search filtered to (user_id, source_id).

    Uses Reciprocal Rank Fusion (RRF) to merge dense and sparse result lists.
    Returns up to `limit` chunk text strings, ordered by fused relevance.
    """
    expr = f'user_id == "{user_id}" and source_id == "{source_id}"'

    dense_req = AnnSearchRequest(
        data=[dense_query],
        anns_field="dense_vector",
        param={"metric_type": "COSINE", "params": {"ef": 50}},
        limit=limit,
        expr=expr,
    )
    sparse_req = AnnSearchRequest(
        data=[sparse_query],
        anns_field="sparse_vector",
        param={"metric_type": "IP"},
        limit=limit,
        expr=expr,
    )

    client = get_client()
    try:
        results = client.hybrid_search(
            collection_name=COLLECTION_NAME,
            reqs=[dense_req, sparse_req],
            ranker=RRFRanker(),
            limit=limit,
            output_fields=["chunk_text"],
        )
    except Exception:
        # Fallback: pure dense search if sparse index is empty/unavailable
        results = client.search(
            collection_name=COLLECTION_NAME,
            data=[dense_query],
            anns_field="dense_vector",
            param={"metric_type": "COSINE", "params": {"ef": 50}},
            limit=limit,
            filter=expr,
            output_fields=["chunk_text"],
        )

    chunks: list[str] = []
    if not results:
        return chunks

    for hit in results[0]:
        text = ""
        if isinstance(hit, dict):
            # MilvusClient may return fields at top level or inside 'entity'
            text = hit.get("chunk_text") or hit.get("entity", {}).get("chunk_text", "")
        elif hasattr(hit, "entity"):
            text = hit.entity.get("chunk_text", "")
        if text:
            chunks.append(text)

    return chunks
