from django.apps import AppConfig


class BackendConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'backend'

    def ready(self) -> None:
        """
        Pre-load heavy singletons in a background thread so the first HTTP
        request is never blocked by model / DB initialisation.

        Guards against Django's StatReloader: the reloader forks a child
        worker process and sets RUN_MAIN='true' in it.  We only warm up in
        that child — never in the parent watcher — to avoid two processes
        fighting over the Milvus Lite lock file.
        """
        import os
        import sys
        import threading

        # Skip warmup in Django's StatReloader parent only (runserver watcher).
        # Gunicorn/production never sets RUN_MAIN — still warm up there.
        if "runserver" in sys.argv and os.environ.get("RUN_MAIN") != "true":
            return

        def _warmup() -> None:
            try:
                # 1. Boot Milvus Lite gRPC server + ensure collection exists
                # pyrefly: ignore [missing-import]
                from .milvus_client import get_client
                get_client()

                # 2. Load sentence-transformers dense embedder into memory
                # pyrefly: ignore [missing-import]
                from .embeddings import get_dense_embedder
                get_dense_embedder()

            except Exception as exc:          # noqa: BLE001
                import logging
                logging.getLogger(__name__).warning(
                    "Warmup failed (will retry on first request): %s", exc
                )

        t = threading.Thread(target=_warmup, daemon=True, name="rag-warmup")
        t.start()

