# PolicyPilot

PolicyPilot is an agentic RAG assistant for company HR policies and employee support. It answers from a private Pinecone knowledge base first, grades the retrieved evidence, and only falls back to a web search when the private KB is not strong enough.

## How it works

The LangGraph workflow in `app/rag/workflow.py` runs these steps:

1. Route the question to the private KB or a direct chat response.
2. Retrieve the top `TOP_K` chunks from Pinecone.
3. Grade the KB evidence as `good` or `weak`.
4. If weak, search the web with Tavily.
5. Grade the web evidence; rewrite the query and retry when configured.
6. Generate a grounded answer with citations, or return an insufficient-evidence response.

Every chat request is written to a local SQLite audit table.

## Project layout

```
app/
  api/routes.py          FastAPI routes: /api/health, /api/chat, /api/ingest
  core/config.py         Pydantic settings and project paths
  core/logger.py         File logging
  rag/state.py           LangGraph state and structured-output schemas
  rag/vector_store.py    Gemini embeddings + Pinecone index management
  rag/workflow.py        LangGraph agent
  services/audit.py      SQLite query audit log
  services/ingestion.py  PDF/TXT/MD/DOCX loading and chunking
  main.py                FastAPI app and page rendering
static/                  Frontend CSS/JS
templates/index.html     Chat UI
ingest_sample_kb.py      CLI to index data/sample_kb
```

## Setup

Requires Python 3.12.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

Fill in `.env`:

| Variable | Purpose |
| --- | --- |
| `GEMINI_API_KEY` | Google Gemini key for chat and embeddings |
| `GOOGLE_API_KEY` | Optional separate key for embeddings; falls back to `GEMINI_API_KEY` |
| `GEMINI_MODEL` | Chat model, default `gemini-3.8-flash` |
| `EMBEDDING_MODEL` | Gemini embedding model, default `gemini-embedding-001` |
| `EMBEDDING_DIMENSION` | Optional embedding dimension override; must match the Pinecone index |
| `PINECONE_API_KEY` | Pinecone API key |
| `PINECONE_INDEX_NAME` | Index name, default `policypilot` |
| `PINECONE_NAMESPACE` | Namespace, default `company-hr-kb` |
| `PINECONE_CLOUD` | Serverless cloud, default `aws` |
| `PINECONE_REGION` | Serverless region, default `us-east-1` |
| `TAVILY_API_KEY` | Optional web fallback |
| `TOP_K` | Number of chunks to retrieve, default `4` |
| `MAX_WEB_REWRITES` | Query-rewrite retries after weak web evidence, default `1` (alias: `MAX_RETRIES`) |
| `ADMIN_API_KEY` | Required by `POST /api/ingest`; change from the default |
| `MAX_UPLOAD_MB` | Upload size limit, default `25` |
| `DATABASE_PATH` | SQLite audit log path, default `data/audit.db` (alias: `AUDIT_DB_PATH`) |

The Pinecone index is created automatically with the embedding dimension implied by `EMBEDDING_MODEL`, unless `EMBEDDING_DIMENSION` is set. Supported model aliases and dimensions are listed in `app/rag/vector_store.py`.

Use a dedicated index and namespace for PolicyPilot. If you point it at an index used by another application, retrieval will return unrelated documents. When the index already exists, its dimension must match `EMBEDDING_DIMENSION` (or the dimension implied by `EMBEDDING_MODEL`).

## Run

```powershell
python run.py
```

Open <http://127.0.0.1:8080>.

## Index documents

Use the "Add HR Document" button in the UI, or upload directly:

```powershell
curl.exe -X POST http://127.0.0.1:8080/api/ingest `
  -H "X-Admin-Key: change-me-in-production" `
  -F "file=@data/sample_kb/handbook.pdf"
```

To index every document in `data/sample_kb`:

```powershell
python ingest_sample_kb.py
```

## API

- `GET /api/health` – service health.
- `POST /api/chat` – JSON body `{"question": "..."}`.
- `POST /api/ingest` – multipart file upload with the `X-Admin-Key` header.

## Docker

```powershell
docker build -t policypilot .
docker run --rm -p 8080:8080 --env-file .env policypilot
```

## Current limitations

- No authentication or per-user authorization; the UI and `/api/chat` are open.
- `/api/ingest` uses a single shared admin key.
- No rate limiting, global request timeout, or streaming responses.
- `MAX_SUPPORT_RETRIES` and `MAX_RETRIEVAL_REWRITES` are not wired; only `MAX_WEB_REWRITES` (alias `MAX_RETRIES`) controls the retry loop.
- Re-uploading the same document creates duplicate vectors; there is no content hashing or deduplication.
- No automated tests yet.
- Audit records store the question, source, and trace, but not the answer or citations.
- The sample KB folder is empty; add your own `.pdf`, `.txt`, `.md`, or `.docx` files.
