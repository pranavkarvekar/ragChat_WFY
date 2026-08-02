"""
WSGI config for ragWFY project.

It exposes the WSGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/5.2/howto/deployment/wsgi/
"""

import os

from django.core.wsgi import get_wsgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'ragWFY.settings')

application = get_wsgi_application()

# Automatically apply database migrations at startup on cloud environments
# (Render wipes .gitignore'd sqlite files between build and deploy, causing 500 errors on POST /login and /register)
try:
    import logging
    from django.core.management import call_command
    call_command("migrate", "--no-input", interactive=False)
except Exception as exc:  # noqa: BLE001
    logging.getLogger(__name__).warning("Startup migrate failed: %s", exc)
