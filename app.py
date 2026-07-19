"""
Compatibility entrypoint for hosts that default to `gunicorn app:app`.
Exposes the Django WSGI application as `app`.
"""
from ragWFY.wsgi import application

app = application
