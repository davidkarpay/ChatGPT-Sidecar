
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

## Web UI
Open `http://127.0.0.1:8088/` in a browser. Enter your API key, type a query, and click **Search** (FAISS top-k) or **Advanced** (MMR rerank).
- **Copy context** copies a JSON block ready to paste into ChatGPT as grounding.
- Advanced params:
  - `candidates` (default 50, max 200): FAISS candidate pool size
  - `lambda` (default 0.5, 0..1): MMR tradeoff (0=diversity, 1=similarity)

## API Endpoints
- `GET /healthz` — Health check
- `POST /search` — Semantic search (params: query, k, index_name)
- `POST /search/advanced` — MMR-enhanced search (params: query, k, candidates, lambda)
- `POST /ingest/chatgpt-export` — Import ChatGPT conversations
- `POST /reindex` — Rebuild FAISS index from database

### Search Examples
```bash
# Basic search
curl -X POST http://127.0.0.1:8088/search \
  -H "X-API-Key: your-key" \
  -H "Content-Type: application/json" \
  -d '{"query": "your search", "k": 8}'

# Advanced MMR search  
curl -X POST http://127.0.0.1:8088/search/advanced \
  -H "X-API-Key: your-key" \
  -H "Content-Type: application/json" \
  -d '{"query": "your search", "k": 8, "candidates": 50, "lambda": 0.5}'
```

## Deployment

### Prerequisites
- Python 3.11+
- 4GB+ RAM (for embedding models)
- Storage space for your documents and FAISS indexes

### Setup Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/davidkarpay/ChatGPT-Sidecar.git
   cd ChatGPT-Sidecar
   ```

2. **Set up Python environment**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Configure environment**
   ```bash
   cp .env.example .env
   # Edit .env and set your own API_KEY (generate a secure random string)
   ```

4. **Start the server**
   ```bash
   uvicorn app.main:app --host 127.0.0.1 --port 8088 --reload
   ```

5. **Import your data**
   - Visit `http://127.0.0.1:8088/` in your browser
   - Enter your API key
   - Use the import section to upload your ChatGPT export
   - Click "Rebuild Search Index" after import

### Production Deployment

For production use:

```bash
# Use a production WSGI server
pip install gunicorn
gunicorn app.main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8088

# Or use uvicorn in production mode
uvicorn app.main:app --host 0.0.0.0 --port 8088 --workers 4
```

**Important**: 
- Generate a strong, unique `API_KEY` for production
- Consider using HTTPS in production
- Backup your `sidecar.db` file regularly

### Docker Deployment (Optional)

Create a `Dockerfile`:
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8088
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8088"]
```

Build and run:
```bash
docker build -t sidecar .
docker run -p 8088:8088 -v $(pwd)/data:/app/data sidecar
```

### Performance Tuning
- Default index is **uncompressed** (`IndexFlatIP`) for accuracy
- For large datasets (>100k chunks), consider using compressed indexes
- Adjust `CHUNK_CHARS` and `CHUNK_OVERLAP` based on your content type
- Monitor memory usage with embedding model loading

## Security

### Best Practices
- **API Key**: Generate a strong, unique API key for the `API_KEY` environment variable
- **Local Only**: This application is designed for local/private use - avoid exposing to public internet without proper authentication
- **Data Privacy**: All data stays local - no external API calls for search operations
- **HTTPS**: Use HTTPS in production deployments
- **Backups**: Regularly backup your `sidecar.db` file containing your indexed content

### Generating a Secure API Key
```bash
# Generate a random 32-character API key
python3 -c "import secrets; print('sidecar-' + secrets.token_urlsafe(32))"
```

### Security Features Built-in
- ✅ SQL injection prevention (parameterized queries)
- ✅ XSS protection in web UI (HTML escaping)
- ✅ Input validation (Pydantic models)
- ✅ No external API dependencies for core search
- ✅ Local-first architecture

### Notes
- SQLite DB is `sidecar.db` at project root
- FAISS indexes are stored in `data/indexes/` (auto-created)
- All search operations are local-only (no external API calls)
- Sensitive files (`.env`, `*.db`, indexes) are automatically gitignored
