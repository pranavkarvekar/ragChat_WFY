from django.apps import AppConfig


class BackendConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "backend"

    def ready(self) -> None:
        """
        Optional model warmup. Disabled by default on Render/Gunicorn so the
        worker stays alive and responsive (loading ML models on 0.1 vCPU in a
        background thread starves Gunicorn and causes 500/timeout errors).
        Models load lazily on the first RAG request instead.

        Set ENABLE_WARMUP=1 to turn boot warmup back on.
        """
        import os
        import sys
        import threading
        import logging

        logger = logging.getLogger(__name__)

        # Never warm up in runserver parent reloader process
        if "runserver" in sys.argv and os.environ.get("RUN_MAIN") != "true":
            return

        # Skip on cloud/gunicorn unless explicitly enabled
        if os.environ.get("ENABLE_WARMUP", "").lower() not in ("1", "true", "yes"):
            return

        def _warmup() -> None:
            try:
                logger.info("Warmup: loading ONNX embedding model...")
                from .embeddings import get_dense_embedder
                get_dense_embedder()
                logger.info("Warmup: ONNX embedding model ready.")
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Warmup: ONNX model load failed (will retry on first request): %s", exc
                )

            try:
                if os.environ.get("ENABLE_MILVUS_WARMUP", "").lower() in ("1", "true", "yes"):
                    logger.info("Warmup: connecting to Milvus/Zilliz...")
                    from .milvus_client import get_client
                    get_client()
                    logger.info("Warmup: Milvus connection ready.")
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Warmup: Milvus connection failed (will retry on first request): %s", exc
                )

        threading.Thread(target=_warmup, daemon=True, name="rag-warmup").start()
