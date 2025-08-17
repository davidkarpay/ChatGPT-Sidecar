
# Sidecar Context App (MVP)

Local-first context archive with semantic search over your ChatGPT export and other documents.

## Quickstart (MacBook Air M1)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# First run
uvicorn app.main:app --host 127.0.0.1 --port 8088 --reload
```

### Environment
Copy `.env.example` to `.env` and adjust values.

### Endpoints
- `GET /healthz`
- `POST /search` — body: `{ "query": "text", "k": 8 }`
- `POST /ingest/chatgpt-export` — body: `{ "root_path": "/ABS/PATH/TO/EXPORT", "project_id": 1, "chunk_chars": 1200 }`
- `POST /reindex` — rebuild FAISS from DB

### Notes
- Default index is **uncompressed** (`IndexFlatIP`).
- SQLite DB is `sidecar.db` at project root.
