import os
import time
import logging
import functools
import json
import numpy as np
import uuid
from pathlib import Path
from fastapi import FastAPI, Depends, HTTPException, Header
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, conint, confloat
from typing import Optional, List, Dict
from dotenv import load_dotenv
from sse_starlette.sse import EventSourceResponse

from .db import DB
from .vectorstore import FaissStore
from .ingest_chatgpt import ingest_export
from .ingest_multi_layer import ingest_export_multi_layer
from .search_fusion import MultiLayerSearchFusion
from .mmr import mmr
from .rag_pipeline import RAGPipeline
from .chat_agent import ChatConfig

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

API_KEY = os.getenv("API_KEY", "change-me")
DB_PATH = os.getenv("DB_PATH", "sidecar.db")
INDEX_PATH = Path(os.getenv("INDEX_PATH", "data/indexes/main.faiss"))
IDS_PATH = Path(os.getenv("IDS_PATH", "data/indexes/main.pkl"))
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

class SearchConfig:
    DEFAULT_K: int = 8
    DEFAULT_CANDIDATES: int = 50
    MAX_CANDIDATES: int = 200
    MIN_LAMBDA: float = 0.0
    MAX_LAMBDA: float = 1.0

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

# Chat configuration (lazy loading)
chat_config = ChatConfig(
    model_name=os.getenv("CHAT_MODEL", "EleutherAI/gpt-j-6B"),
    max_context_length=int(os.getenv("CHAT_MAX_CONTEXT", "2048")),
    max_new_tokens=int(os.getenv("CHAT_MAX_TOKENS", "512")),
    temperature=float(os.getenv("CHAT_TEMPERATURE", "0.7")),
    use_8bit=os.getenv("CHAT_USE_8BIT", "true").lower() == "true",
    device=os.getenv("CHAT_DEVICE", None)  # If None, ChatConfig will auto-detect
)

# Global RAG pipeline instance (initialized on first use)
rag_pipeline = None

def get_rag_pipeline():
    """Get or initialize RAG pipeline (lazy loading)"""
    global rag_pipeline
    if rag_pipeline is None:
        rag_pipeline = RAGPipeline(DB_PATH, EMBED_MODEL, chat_config)
    return rag_pipeline

# Caching: query embedding only (small, string-key)
@functools.lru_cache(maxsize=256)
def cached_query_vec(text: str) -> np.ndarray:
    v = store.encode([text])[0]  # np.ndarray
    return v  # cached by text content

def present_row(r: dict, rank: int, score: float) -> dict:
    snippet = r["text"]
    preview = (snippet[:320] + "…") if len(snippet) > 320 else snippet
    return {
        "rank": rank,
        "score": float(score),
        "source": r["title"],
        "preview": preview,
        "loc": {"doc_id": r["doc_id"], "chunk_id": r["chunk_id"], "start": r["start_char"], "end": r["end_char"]},
        "context_json": {
            "context": [{
                "doc": r["title"],
                "loc": {"doc_id": r["doc_id"], "chunk_id": r["chunk_id"], "start": r["start_char"], "end": r["end_char"]},
                "quote": snippet
            }]
        }
    }

# Request models
class SearchReq(BaseModel):
    query: str = Field(min_length=1)
    k: conint(ge=1, le=30) = Field(default=SearchConfig.DEFAULT_K)
    index_name: str = "main"

class AdvancedSearchReq(BaseModel):
    query: str = Field(min_length=1)
    k: conint(ge=1, le=30) = Field(default=SearchConfig.DEFAULT_K)
    candidates: conint(ge=10, le=SearchConfig.MAX_CANDIDATES) = Field(default=SearchConfig.DEFAULT_CANDIDATES)
    lambda_: confloat(ge=SearchConfig.MIN_LAMBDA, le=SearchConfig.MAX_LAMBDA) = Field(default=0.5, alias="lambda")
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

class ChatReq(BaseModel):
    query: str = Field(min_length=1)
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    k: conint(ge=1, le=20) = Field(default=5)
    search_mode: str = Field(default="adaptive", pattern="^(adaptive|multi_layer|basic)$")

class ChatStreamReq(BaseModel):
    query: str = Field(min_length=1)
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    k: conint(ge=1, le=20) = Field(default=5)
    search_mode: str = Field(default="adaptive", pattern="^(adaptive|multi_layer|basic)$")

class AnalyzeReq(BaseModel):
    doc_ids: Optional[List[int]] = None
    limit: conint(ge=10, le=500) = Field(default=100)

class SuggestionsReq(BaseModel):
    query: str
    results: List[dict] = Field(default_factory=list)

class ConversationReq(BaseModel):
    session_id: str

@app.get("/")
async def root():
    """Serve the search UI"""
    html_path = Path(__file__).resolve().parent.parent / "static" / "index.html"
    with open(html_path, "r") as f:
        content = f.read()
    return HTMLResponse(content=content)

@app.get("/chat")
async def chat_ui():
    """Serve the chat UI"""
    html_path = Path(__file__).resolve().parent.parent / "static" / "chat.html"
    with open(html_path, "r") as f:
        content = f.read()
    return HTMLResponse(content=content)

@app.get("/healthz")
async def healthz():
    return {"ok": True}

@app.get("/healthz/chat")
async def chat_health():
    """Health check specifically for chat model availability"""
    try:
        # Check if RAG pipeline can be initialized
        pipeline = get_rag_pipeline()
        
        # Try to get the chat agent (this will trigger lazy loading if needed)
        chat_agent = await pipeline._get_chat_agent()
        
        # Basic model info
        model_info = {
            "model_loaded": chat_agent.model is not None,
            "tokenizer_loaded": chat_agent.tokenizer is not None,
            "model_name": chat_agent.config.model_name,
            "device": chat_agent.config.device,
        }
        
        return {
            "ok": True,
            "chat_available": True,
            "model_info": model_info,
            "openmp_settings": {
                "OMP_NUM_THREADS": os.getenv("OMP_NUM_THREADS"),
                "MKL_NUM_THREADS": os.getenv("MKL_NUM_THREADS"),
                "TOKENIZERS_PARALLELISM": os.getenv("TOKENIZERS_PARALLELISM")
            }
        }
        
    except Exception as e:
        logging.error(f"Chat health check failed: {e}")
        return {
            "ok": False,
            "chat_available": False,
            "error": str(e),
            "openmp_settings": {
                "OMP_NUM_THREADS": os.getenv("OMP_NUM_THREADS"),
                "MKL_NUM_THREADS": os.getenv("MKL_NUM_THREADS"),
                "TOKENIZERS_PARALLELISM": os.getenv("TOKENIZERS_PARALLELISM")
            }
        }

@app.post("/search", dependencies=[Depends(auth)])
async def search(req: SearchReq):
    t0 = time.time()
    try:
        results = store.search(req.query, k=req.k)
        if not results:
            return {"query": req.query, "hits": []}
        faiss_ids = [faiss_idx for faiss_idx, _ in results]
        with DB(DB_PATH) as db:
            rows_by_fid = db.fetch_chunks_by_faiss_indices(faiss_ids, req.index_name)
        hits = []
        for rank, (faiss_idx, score) in enumerate(results, start=1):
            r = rows_by_fid.get(faiss_idx)
            if r:
                hits.append(present_row(r, rank, score))
        return {"query": req.query, "hits": hits}
    finally:
        logging.info("search k=%s hits=%s dur=%.3fs", req.k, len(results or []), time.time() - t0)

@app.post("/search/advanced", dependencies=[Depends(auth)])
async def search_advanced(req: AdvancedSearchReq):
    t0 = time.time()
    try:
        base = store.search(req.query, k=req.candidates)
        if not base:
            return {"query": req.query, "hits": []}

        faiss_ids = [faiss_idx for faiss_idx, _ in base]
        with DB(DB_PATH) as db:
            rows_by_fid = db.fetch_chunks_by_faiss_indices(faiss_ids, req.index_name)

        ordered = []
        for (faiss_idx, score) in base:
            r = rows_by_fid.get(faiss_idx)
            if r:
                ordered.append((faiss_idx, score, r))
        if not ordered:
            return {"query": req.query, "hits": []}

        texts = [r["text"] for _, _, r in ordered]

        # encode with cache for query only
        qv = cached_query_vec(req.query)
        cand_vecs = store.encode(texts)

        sel = mmr(qv, cand_vecs, lamb=float(req.lambda_), k=min(req.k, len(ordered)))
        hits = []
        for rank, i in enumerate(sel, start=1):
            _, base_score, row = ordered[i]
            hits.append(present_row(row, rank, base_score))
        return {"query": req.query, "hits": hits}
    except Exception as e:
        logging.exception("advanced search error: %s", e)
        raise HTTPException(status_code=500, detail="Search error")
    finally:
        logging.info("search_adv k=%s cand=%s dur=%.3fs", req.k, req.candidates, time.time() - t0)

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

@app.post("/chat", dependencies=[Depends(auth)])
async def chat(req: ChatReq):
    """Chat with your ChatGPT export data using GPT-J"""
    t0 = time.time()
    try:
        result = await get_rag_pipeline().chat_async(
            query=req.query,
            session_id=req.session_id,
            k=req.k,
            search_mode=req.search_mode,
            stream=False
        )
        
        # Add follow-up suggestions
        suggestions = get_rag_pipeline().suggest_follow_up_questions(req.query, result["context"])
        result["suggestions"] = suggestions
        
        return result
    
    except RuntimeError as e:
        if "model loading" in str(e).lower():
            logging.error("Model loading failed completely: %s", e)
            raise HTTPException(
                status_code=503, 
                detail="Chat model is temporarily unavailable. Please try again later or contact support."
            )
        else:
            logging.exception("Runtime error in chat: %s", e)
            raise HTTPException(status_code=500, detail="Chat processing error")
    
    except Exception as e:
        logging.exception("Chat error: %s", e)
        # Check if it's an OpenMP/threading related error
        if any(keyword in str(e).lower() for keyword in ['openmp', 'omp', 'thread', 'segmentation']):
            raise HTTPException(
                status_code=503,
                detail="Chat service is experiencing threading issues. Server restart may be required."
            )
        else:
            raise HTTPException(status_code=500, detail="Chat processing error")
    finally:
        logging.info("chat dur=%.3fs", time.time() - t0)

@app.post("/chat/stream", dependencies=[Depends(auth)])
async def chat_stream(req: ChatStreamReq):
    """Stream chat responses for real-time interaction"""
    try:
        result = await get_rag_pipeline().chat_async(
            query=req.query,
            session_id=req.session_id,
            k=req.k,
            search_mode=req.search_mode,
            stream=True
        )
        
        async def generate():
            try:
                # First send context
                yield f"data: {json.dumps({'type': 'context', 'data': result['context']})}\n\n"
                yield f"data: {json.dumps({'type': 'session_id', 'data': result['session_id']})}\n\n"
                
                # Then stream the response
                for token in result["response_generator"]:
                    yield f"data: {json.dumps({'type': 'token', 'data': token})}\n\n"
                
                # Finally send suggestions
                suggestions = get_rag_pipeline().suggest_follow_up_questions(req.query, result["context"])
                yield f"data: {json.dumps({'type': 'suggestions', 'data': suggestions})}\n\n"
                yield f"data: {json.dumps({'type': 'done'})}\n\n"
                
            except Exception as e:
                logging.error(f"Error during streaming: {e}")
                error_msg = {"type": "error", "data": "Streaming interrupted due to model error"}
                yield f"data: {json.dumps(error_msg)}\n\n"
        
        return EventSourceResponse(generate())
    
    except RuntimeError as e:
        if "model loading" in str(e).lower():
            logging.error("Model loading failed in streaming: %s", e)
            raise HTTPException(
                status_code=503, 
                detail="Chat model is temporarily unavailable for streaming."
            )
        else:
            logging.exception("Runtime error in chat streaming: %s", e)
            raise HTTPException(status_code=500, detail="Chat streaming error")
    
    except Exception as e:
        logging.exception("Chat stream error: %s", e)
        if any(keyword in str(e).lower() for keyword in ['openmp', 'omp', 'thread', 'segmentation']):
            raise HTTPException(
                status_code=503,
                detail="Chat streaming service is experiencing threading issues."
            )
        else:
            raise HTTPException(status_code=500, detail="Chat streaming error")

@app.post("/analyze", dependencies=[Depends(auth)])
async def analyze_topics(req: AnalyzeReq):
    """Analyze topics across conversations"""
    try:
        result = get_rag_pipeline().analyze_conversation_topics(req.doc_ids, req.limit)
        return result
    except Exception as e:
        logging.exception("Analysis error: %s", e)
        raise HTTPException(status_code=500, detail="Analysis error")

@app.post("/suggest", dependencies=[Depends(auth)])
async def suggest_questions(req: SuggestionsReq):
    """Get follow-up question suggestions"""
    try:
        suggestions = get_rag_pipeline().suggest_follow_up_questions(req.query, req.results)
        return {"suggestions": suggestions}
    except Exception as e:
        logging.exception("Suggestions error: %s", e)
        raise HTTPException(status_code=500, detail="Suggestions error")

@app.get("/chat/history/{session_id}", dependencies=[Depends(auth)])
async def get_chat_history(session_id: str):
    """Get conversation history for a session"""
    try:
        history = get_rag_pipeline().get_conversation_history(session_id)
        return {"session_id": session_id, "history": history}
    except Exception as e:
        logging.exception("History error: %s", e)
        raise HTTPException(status_code=500, detail="History retrieval error")

@app.delete("/chat/history/{session_id}", dependencies=[Depends(auth)])
async def clear_chat_history(session_id: str):
    """Clear conversation history for a session"""
    try:
        get_rag_pipeline().clear_conversation_history(session_id)
        return {"message": "History cleared", "session_id": session_id}
    except Exception as e:
        logging.exception("History clear error: %s", e)
        raise HTTPException(status_code=500, detail="History clear error")

@app.get("/summarize/{doc_id}", dependencies=[Depends(auth)])
async def summarize_conversation(doc_id: int):
    """Get AI-generated summary of a specific conversation"""
    try:
        summary = get_rag_pipeline().get_conversation_summary(doc_id)
        return {"doc_id": doc_id, "summary": summary}
    except Exception as e:
        logging.exception("Summarization error: %s", e)
        raise HTTPException(status_code=500, detail="Summarization error")