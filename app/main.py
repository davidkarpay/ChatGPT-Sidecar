import os
from pathlib import Path
from fastapi import FastAPI, Depends, HTTPException, Header
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv

from .db import DB
from .vectorstore import FaissStore
from .ingest_chatgpt import ingest_export
from .ingest_multi_layer import ingest_export_multi_layer
from .search_fusion import MultiLayerSearchFusion

load_dotenv()

API_KEY = os.getenv("API_KEY", "change-me")
DB_PATH = os.getenv("DB_PATH", "sidecar.db")
INDEX_PATH = Path(os.getenv("INDEX_PATH", "data/indexes/main.faiss"))
IDS_PATH = Path(os.getenv("IDS_PATH", "data/indexes/main.pkl"))
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

app = FastAPI(title="Sidecar Context API")

# Mount static files
static_path = Path(__file__).resolve().parent.parent / "static"
app.mount("/static", StaticFiles(directory=str(static_path)), name="static")

async def auth(x_api_key: str | None = Header(default=None)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

# ensure DB schema
with DB(DB_PATH) as db:
    db.init_schema(Path(__file__).resolve().parent.parent / "schema.sql")

# global FAISS store
store = FaissStore(INDEX_PATH, IDS_PATH, EMBED_MODEL)
store.load()

# Multi-layer search fusion
fusion_search = MultiLayerSearchFusion(DB_PATH, EMBED_MODEL)

class SearchReq(BaseModel):
    query: str
    k: int = 8
    index_name: str = "main"

class MultiLayerSearchReq(BaseModel):
    query: str
    k: int = 8

class IngestReq(BaseModel):
    root_path: str
    project_id: int | None = None
    chunk_chars: int = 1200

class ReindexReq(BaseModel):
    index_name: str = "main"

@app.get("/")
async def root():
    """Serve the web UI"""
    html_path = Path(__file__).resolve().parent.parent / "static" / "index.html"
    with open(html_path, "r") as f:
        content = f.read()
    return HTMLResponse(content=content)

@app.get("/healthz")
async def healthz():
    return {"ok": True}

@app.post("/search", dependencies=[Depends(auth)])
async def search(req: SearchReq):
    results = store.search(req.query, k=req.k)
    if not results:
        return {"query": req.query, "hits": []}
    faiss_ids = [faiss_idx for faiss_idx, _ in results]
    with DB(DB_PATH) as db:
        rows_by_fid = db.fetch_chunks_by_faiss_indices(faiss_ids, req.index_name)
    hits = []
    for rank, (faiss_idx, score) in enumerate(results, start=1):
        r = rows_by_fid.get(faiss_idx)
        if not r:
            continue
        hits.append({
            "rank": rank,
            "score": float(score),
            "chunk_id": r["chunk_id"],
            "document": {
                "id": r["doc_id"],
                "title": r["title"],
                "doc_type": r["doc_type"]
            },
            "snippet": r["text"],
            "offset": {"start": r["start_char"], "end": r["end_char"]}
        })
    return {"query": req.query, "hits": hits}

@app.post("/search-multi-layer", dependencies=[Depends(auth)])
async def search_multi_layer(req: MultiLayerSearchReq):
    return fusion_search.search_multi_layer(req.query, req.k)

@app.post("/ingest/chatgpt-export", dependencies=[Depends(auth)])
async def ingest_chatgpt(req: IngestReq):
    count = ingest_export(req.root_path, req.project_id, req.chunk_chars)
    return {"imported_conversations": count}

@app.post("/ingest/chatgpt-export-multi-layer", dependencies=[Depends(auth)])
async def ingest_chatgpt_multi_layer(req: IngestReq):
    result = ingest_export_multi_layer(req.root_path, req.project_id)
    return result

@app.post("/reindex", dependencies=[Depends(auth)])
async def reindex(req: ReindexReq):
    # Rebuild FAISS from DB chunks
    with DB(DB_PATH) as db:
        cur = db.conn.execute("SELECT id, text FROM chunk ORDER BY id ASC")
        chunk_rows = cur.fetchall()
        store.index = None
        store.ids = []
        encoded_rows = [{"embedding_ref_id": i, "chunk_id": r["id"], "text": r["text"]} 
                        for i, r in enumerate(chunk_rows)]
        if not encoded_rows:
            return {"ok": True, "message": "No chunks to index"}
        store.build(encoded_rows)
        faiss_ids = list(range(len(encoded_rows)))
        db.create_embedding_refs([r["chunk_id"] for r in encoded_rows], req.index_name, 
                                 vector_dim=store.model.get_sentence_embedding_dimension(),
                                 faiss_ids=faiss_ids)
    return {"ok": True, "vectors": len(encoded_rows)}

@app.post("/reindex-multi-layer", dependencies=[Depends(auth)])
async def reindex_multi_layer():
    """
    Build layer-specific FAISS indexes from existing multi-layer chunks.
    Identifies layers by chunk length ranges.
    """
    from .vectorstore import FaissStore
    
    # Layer definitions based on chunk length analysis
    layer_configs = {
        'precision': {'min_len': 300, 'max_len': 500, 'index': 'data/indexes/precision.faiss', 'ids': 'data/indexes/precision.pkl'},
        'balanced': {'min_len': 800, 'max_len': 1400, 'index': 'data/indexes/balanced.faiss', 'ids': 'data/indexes/balanced.pkl'},
        'context': {'min_len': 3200, 'max_len': 5200, 'index': 'data/indexes/context.faiss', 'ids': 'data/indexes/context.pkl'}
    }
    
    results = {}
    
    with DB(DB_PATH) as db:
        for layer_name, config in layer_configs.items():
            # Get chunks for this layer from multi-layer documents
            query = """
                SELECT c.id, c.text 
                FROM chunk c 
                JOIN document d ON c.document_id = d.id 
                WHERE d.doc_type = 'chatgpt_export_multi' 
                AND LENGTH(c.text) BETWEEN ? AND ?
                ORDER BY c.id ASC
            """
            cur = db.conn.execute(query, (config['min_len'], config['max_len']))
            chunk_rows = cur.fetchall()
            
            if not chunk_rows:
                results[layer_name] = {"vectors": 0, "message": "No chunks found"}
                continue
            
            # Create vectorstore for this layer
            layer_store = FaissStore(Path(config['index']), Path(config['ids']), EMBED_MODEL)
            
            # Build embeddings
            encoded_rows = [{"embedding_ref_id": i, "chunk_id": r["id"], "text": r["text"]} 
                           for i, r in enumerate(chunk_rows)]
            
            layer_store.build(encoded_rows)
            faiss_ids = list(range(len(encoded_rows)))
            
            # Create embedding references in database
            db.create_embedding_refs(
                [r["chunk_id"] for r in encoded_rows], 
                layer_name, 
                vector_dim=layer_store.model.get_sentence_embedding_dimension(),
                faiss_ids=faiss_ids
            )
            
            results[layer_name] = {"vectors": len(encoded_rows)}
    
    # Reload fusion search stores to pick up new indexes
    fusion_search.__init__(DB_PATH, EMBED_MODEL)
    
    return {
        "ok": True,
        "layer_results": results,
        "total_vectors": sum(r["vectors"] for r in results.values() if "vectors" in r)
    }