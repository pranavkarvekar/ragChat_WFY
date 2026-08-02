from django.apps import AppConfig


class BackendConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "backend"

    def ready(self) -> None:
        """
        Model warmup on startup.

        Previously disabled because torch + Milvus Lite caused OOM on Render Free tier.
        Now safe because we use ONNX Runtime (~50 MB) instead of PyTorch (~400 MB).

        The ONNX model is loaded in a background thread so the worker
        starts accepting requests immediately — warmup happens concurrently.
        This eliminates the 2-5 minute first-request delay.
        """
        import os
        import sys
        import threading
        import logging

        logger = logging.getLogger(__name__)

        # Never warm up in runserver parent reloader process
        if "runserver" in sys.argv and os.environ.get("RUN_MAIN") != "true":
            return

        def _warmup() -> None:
            try:
                # Always warm up ONNX embedder — only ~50 MB, safe on free tier
                logger.info("Warmup: loading ONNX embedding model...")
                from .embeddings import get_dense_embedder
                get_dense_embedder()
                logger.info("Warmup: ONNX embedding model ready.")
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Warmup: ONNX model load failed (will retry on first request): %s", exc
                )

            try:
                # Optionally warm up Milvus connection
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
