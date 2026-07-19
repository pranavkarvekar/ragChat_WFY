# RAG Chat WFY

Django RAG app: upload PDFs, paste web/YouTube URLs, ask questions with Groq + Milvus Lite.

## Local run

```bash
python -m venv venv
# Windows: venv\Scripts\activate
# macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
```

Create `.env` in the project root:

```env
GROQ_API_KEY=your_key_here
MILVUS_DB_PATH=./milvus_ragwfy.db
DEBUG=True
```

```bash
python manage.py migrate
python manage.py runserver
```

Open http://127.0.0.1:8000

## Deploy on Render (simplest)

1. Push this repo to GitHub
2. Go to https://dashboard.render.com ? **New** ? **Blueprint**
3. Connect this repository (uses `render.yaml`)
4. Set `GROQ_API_KEY` in the Render dashboard when prompted
5. Deploy ? use at least a **Starter** plan (ML models need RAM)

Or manually: **New Web Service** ? connect repo ?

- Build: `chmod +x build.sh && ./build.sh`
- Start: `gunicorn ragWFY.wsgi:application --bind 0.0.0.0:$PORT --workers 1 --threads 4 --timeout 120`
- Env: `DEBUG=False`, `SECRET_KEY=...`, `GROQ_API_KEY=...`, `ALLOWED_HOSTS=.onrender.com`

## Stack

| Layer | Tech |
|-------|------|
| Web | Django 5.2 + templates |
| LLM | Groq |
| Vectors | Milvus Lite |
| Embeddings | sentence-transformers |
| Hosting | Gunicorn + WhiteNoise |
