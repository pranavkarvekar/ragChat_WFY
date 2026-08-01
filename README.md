# 🧠 RAG Chat WFY

A production-grade, multi-source **Retrieval-Augmented Generation (RAG)** application built with Django, Groq (LLaMA-3.3), and Milvus Lite (Hybrid Dense-Sparse Search).

Allows users to upload documents (PDF/Word/TXT), paste Web URLs, or process YouTube videos to ask precise context-restricted questions with streaming token responses.

---

## ✨ Features

- **🌐 Web Chat**: Scrape and index webpage content in clean Markdown format for lexical and semantic search.
- **🎬 YouTube Chat**: Fast path transcript extraction falling back to high-performance low-bandwidth audio download and Whisper-large transcription.
- **📄 File Chat**: Parse uploaded PDFs, DOCX, and TXT files instantly using PyMuPDF.
- **⚡ Real-time SSE Streaming**: Answers stream back token-by-token directly from the Groq LLaMA models.
- **🔒 Multi-Tenant Namespace Isolation**: All vectorized chunks are tagged with a stable `user_id`, guaranteeing users only query their own documents.
- **🎯 Hybrid Search & RRF**: Combines dense semantic vectors (`all-MiniLM-L6-v2`) and sparse lexical vectors (TF-IDF) using Reciprocal Rank Fusion (RRF) for precise keyword and contextual matching.

---

## 🚀 How to Run Locally

### 1. Prerequisites
- **Python 3.10** or higher
- **pip** (Python package manager)

### 2. Setup Virtual Environment
Clone the repository and initialize a local virtual environment:

```bash
# Clone the repository
cd rag_WFY

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows (PowerShell/CMD):
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies
Install all the required python packages:
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Create a file named `.env` in the root folder (`D:\rag_WFY\.env`):

```env
GROQ_API_KEY=your_groq_api_key_here
MILVUS_URI=./milvus_ragwfy.db
DEBUG=True
```
> Get your free Groq API key at [console.groq.com](https://console.groq.com/).

### 5. Apply Database Migrations
Create the SQLite tables for session management and user authentication:
```bash
python manage.py migrate
```

### 6. Create Admin Account (Optional)
If you want access to the Django Admin panel:
```bash
python manage.py createsuperuser
```

### 7. Run the Server
Launch the Django development server:
```bash
python manage.py runserver
```

Open **[http://127.0.0.1:8000](http://127.0.0.1:8000)** in your web browser.

---

## 🏛️ Project Architecture

| Layer | Technology | Role |
|---|---|---|
| **Web Framework** | Django 5.2 | Routing, View logic, Session auth, Admin panel |
| **Vector Database** | Milvus Lite | Embedded database storing text chunks, dense vectors, and sparse TF-IDF vectors |
| **Dense Embeddings** | `all-MiniLM-L6-v2` | Generates 384-dimensional semantic cosine representations |
| **Sparse Indexing** | Scikit-learn TF-IDF | Persists `bm25_models/` per-user/source to perform lexical keyword matching |
| **LLM Inference** | LLaMA-3.3-70b-versatile | High-velocity context-constrained answering via Groq API |
| **Static files** | WhiteNoise | Efficient static asset delivery |

---

## 📁 Key Directories

- **`backend/`**: Contains core RAG processing scripts (`rag_file.py`, `rag_web.py`, `rag_youtube.py`), `milvus_client.py` schema initialization, and API endpoints.
- **`frontend/`**: Manages HTML template files, authentication views, and frontend client scripts.
- **`bm25_models/`**: Auto-generated folder containing serialized lexical vocabulary mappings per-source.
- **`milvus_ragwfy.db`**: Local database containing the indexed chunks and vector spaces.
