from django.apps import AppConfig


class BackendConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "backend"

    def ready(self) -> None:
        """
        Optional model warmup. Disabled by default on Render/Gunicorn so the
        worker stays alive (loading torch + Milvus at boot often OOMs → 502).
        Models still load lazily on the first RAG request.

        Set ENABLE_WARMUP=1 to turn boot warmup back on (needs enough RAM).
        """
        import os
        import sys
        import threading

        # Never warm up in runserver parent reloader
        if "runserver" in sys.argv and os.environ.get("RUN_MAIN") != "true":
            return

        # Skip on cloud/gunicorn unless explicitly enabled
        if os.environ.get("ENABLE_WARMUP", "").lower() not in ("1", "true", "yes"):
            return

        def _warmup() -> None:
            try:
                from .milvus_client import get_client
                get_client()
                from .embeddings import get_dense_embedder
                get_dense_embedder()
            except Exception as exc:  # noqa: BLE001
                import logging
                logging.getLogger(__name__).warning(
                    "Warmup failed (will retry on first request): %s", exc
                )

        threading.Thread(target=_warmup, daemon=True, name="rag-warmup").start()
