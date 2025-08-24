import os
import time
import logging
import functools
import json
import numpy as np
import uuid
import secrets
from datetime import datetime
from pathlib import Path
from fastapi import FastAPI, Depends, HTTPException, Header, Request, Response
from fastapi.responses import HTMLResponse, StreamingResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, conint, confloat
from typing import Optional, List, Dict, Any
from dotenv import load_dotenv
from sse_starlette.sse import EventSourceResponse

from .db import DB
from .vectorstore import FaissStore
from .ingest_chatgpt import ingest_export
from .ingest_multi_layer import ingest_export_multi_layer
from .search_fusion import MultiLayerSearchFusion
from .mmr import mmr
from .rag_pipeline import RAGPipeline
# ChatConfig import removed - using local ChatConfigWithDB instead
from .llm_providers import LLMProviderFactory
from .auth import oauth_handler, session_manager, get_current_user, get_optional_user, OAUTH_ENABLED
from .api_auth import require_read_access, require_admin_access, require_super_access, auth
from .sync_service import ChatGPTSync, sync_user_chatgpt_data
from .auth_endpoints import router as auth_router

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

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

# Configure CORS middleware with strict allow-list
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "").split(",")
CORS_ORIGINS = [origin.strip() for origin in CORS_ORIGINS if origin.strip()]

if CORS_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=CORS_ORIGINS,  # Strict allow-list from environment
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["X-API-Key", "Authorization", "Content-Type", "Accept"],
        max_age=86400,  # Cache preflight requests for 24 hours
    )
    logging.info(f"CORS enabled for origins: {CORS_ORIGINS}")
else:
    logging.warning("CORS_ORIGINS not configured. Cross-origin requests will be blocked.")

# Include authentication router
app.include_router(auth_router)

# Mount static files
static_path = Path(__file__).resolve().parent.parent / "static"
app.mount("/static", StaticFiles(directory=str(static_path)), name="static")

# Legacy auth function moved to api_auth.py for backward compatibility

# ensure DB schema
with DB(DB_PATH) as db:
    db.init_schema(Path(__file__).resolve().parent.parent / "schema.sql")
    # Ensure default user exists for API key authentication
    try:
        db.upsert_user("api@localhost", "API User")
    except Exception as e:
        logging.warning(f"Could not create default user: {e}")

# global FAISS store
store = FaissStore(INDEX_PATH, IDS_PATH, EMBED_MODEL)
store.load()

# Multi-layer search fusion
fusion_search = MultiLayerSearchFusion(DB_PATH, EMBED_MODEL)

# Chat configuration (lazy loading)
from dataclasses import dataclass

@dataclass
class ChatConfigWithDB:
    model_name: str = "EleutherAI/gpt-j-6B"
    max_context_length: int = 2048
    max_new_tokens: int = 512
    temperature: float = 0.7
    use_8bit: bool = True
    device: Optional[str] = None
    db_path: str = "sidecar.db"

chat_config = ChatConfigWithDB(
    model_name=os.getenv("CHAT_MODEL", "EleutherAI/gpt-j-6B"),
    max_context_length=int(os.getenv("CHAT_MAX_CONTEXT", "2048")),
    max_new_tokens=int(os.getenv("CHAT_MAX_TOKENS", "512")),
    temperature=float(os.getenv("CHAT_TEMPERATURE", "0.7")),
    use_8bit=os.getenv("CHAT_USE_8BIT", "true").lower() == "true",
    device=os.getenv("CHAT_DEVICE", None),  # If None, ChatConfig will auto-detect
    db_path=DB_PATH  # Pass database path for training data collection
)

# Global RAG pipeline instance (initialized on first use)
rag_pipeline = None

def get_rag_pipeline():
    """Get or initialize RAG pipeline (lazy loading)"""
    global rag_pipeline
    if rag_pipeline is None:
        # Check if we should use the new LLM provider system
        use_provider = os.getenv("LLM_PROVIDER", "").lower() in ["openai", "anthropic", "ollama"]
        
        if use_provider:
            try:
                llm_provider = LLMProviderFactory.from_env()
                rag_pipeline = RAGPipeline(DB_PATH, EMBED_MODEL, chat_config, llm_provider)
                logging.info(f"RAG pipeline initialized with LLM provider: {llm_provider.provider_type.value}")
            except Exception as e:
                logging.warning(f"Failed to initialize LLM provider, falling back to legacy: {e}")
                rag_pipeline = RAGPipeline(DB_PATH, EMBED_MODEL, chat_config)
        else:
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

class TrainingFeedbackReq(BaseModel):
    training_data_id: int
    rating: Optional[int] = Field(None, ge=1, le=5)  # 1-5 star rating
    correction: Optional[str] = None

class StartTrainingReq(BaseModel):
    base_model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    lora_r: int = Field(16, ge=1, le=64)
    lora_alpha: int = Field(32, ge=1, le=128)
    num_epochs: int = Field(3, ge=1, le=10)
    learning_rate: float = Field(2e-4, gt=0, le=1e-2)

class EvaluationRequest(BaseModel):
    model_version_id: int
    model_configuration: Optional[Dict] = None

class MonthlyReportRequest(BaseModel):
    year: int = Field(ge=2020, le=3000)
    month: int = Field(ge=1, le=12)

class UserIdentityRequest(BaseModel):
    full_name: str = Field(min_length=1)
    role: Optional[str] = None
    email: Optional[str] = None
    preferences: Optional[Dict] = None

class UpdateUserIdentityRequest(BaseModel):
    full_name: Optional[str] = None
    role: Optional[str] = None
    preferences: Optional[Dict] = None

class SyncConfigRequest(BaseModel):
    sync_enabled: bool
    chatgpt_export_url: Optional[str] = None
    sync_frequency_hours: int = 24

class ManualSyncRequest(BaseModel):
    export_url: Optional[str] = None

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

# OAuth Authentication Endpoints
@app.get("/auth/login")
async def login(request: Request):
    """Initiate Google OAuth login flow."""
    if not OAUTH_ENABLED:
        raise HTTPException(status_code=501, detail="OAuth authentication not configured")
    
    state = secrets.token_urlsafe(32)
    
    # Store state in session (you might want to use Redis in production)
    # For now, we'll include it in the redirect and validate it later
    authorization_url = oauth_handler.get_authorization_url(state)
    
    response = RedirectResponse(url=authorization_url)
    response.set_cookie("oauth_state", state, httponly=True, secure=True, samesite="lax")
    return response

@app.get("/auth/callback")
async def oauth_callback(request: Request, code: str = None, state: str = None, error: str = None):
    """Handle Google OAuth callback."""
    if not OAUTH_ENABLED:
        raise HTTPException(status_code=501, detail="OAuth authentication not configured")
    
    if error:
        raise HTTPException(status_code=400, detail=f"OAuth error: {error}")
    
    if not code or not state:
        raise HTTPException(status_code=400, detail="Missing authorization code or state")
    
    # Validate state (basic implementation)
    stored_state = request.cookies.get("oauth_state")
    if stored_state != state:
        raise HTTPException(status_code=400, detail="Invalid state parameter")
    
    try:
        # Exchange code for user info
        user_info = oauth_handler.exchange_code_for_token(code, state)
        
        # Create session
        session_token = session_manager.create_session(user_info)
        
        # Redirect to main app with session
        response = RedirectResponse(url="/")
        response.set_cookie("session_token", session_token, httponly=True, secure=True, samesite="lax")
        response.delete_cookie("oauth_state")
        
        return response
        
    except Exception as e:
        logging.error(f"OAuth callback error: {e}")
        raise HTTPException(status_code=400, detail="Authentication failed")

@app.post("/auth/logout")
async def logout(request: Request):
    """Logout user and invalidate session."""
    if not OAUTH_ENABLED:
        raise HTTPException(status_code=501, detail="OAuth authentication not configured")
    
    session_token = request.cookies.get("session_token")
    
    if session_token:
        session_manager.invalidate_session(session_token)
    
    response = Response(content=json.dumps({"message": "Logged out successfully"}))
    response.delete_cookie("session_token")
    return response

@app.get("/auth/user")
async def get_user_info(current_user: Dict = Depends(get_optional_user)):
    """Get current user information."""
    if not current_user:
        return {"authenticated": False}
    
    return {
        "authenticated": True,
        "user": {
            "email": current_user["email"],
            "display_name": current_user["display_name"],
            "picture": current_user.get("picture"),
            "verified_email": current_user.get("verified_email", False)
        }
    }

@app.post("/search")
async def search(req: SearchReq, auth_info: dict = Depends(require_read_access)):
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

@app.post("/search/advanced")
async def search_advanced(req: AdvancedSearchReq, auth_info: dict = Depends(require_read_access)):
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

@app.post("/search-multi-layer")
async def search_multi_layer(req: MultiLayerSearchReq, auth_info: dict = Depends(require_read_access)):
    return fusion_search.search_multi_layer(req.query, req.k)

@app.post("/ingest/chatgpt-export")
async def ingest_chatgpt(req: IngestReq, auth_info: dict = Depends(require_admin_access)):
    count = ingest_export(req.root_path, req.project_id, req.chunk_chars)
    return {"imported_conversations": count}

@app.post("/ingest/chatgpt-export-multi-layer")
async def ingest_chatgpt_multi_layer(req: IngestReq, auth_info: dict = Depends(require_admin_access)):
    result = ingest_export_multi_layer(req.root_path, req.project_id)
    return result

@app.post("/reindex")
async def reindex(req: ReindexReq, auth_info: dict = Depends(require_admin_access)):
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

@app.post("/reindex-multi-layer")
async def reindex_multi_layer(auth_info: dict = Depends(require_admin_access)):
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

@app.post("/chat")
async def chat(req: ChatReq, auth_info: dict = Depends(require_read_access)):
    """Chat with your ChatGPT export data using GPT-J"""
    t0 = time.time()
    try:
        # Get user identity for this session
        user_identity = get_user_identity_for_session(req.session_id)
        
        # Ensure session is linked to user identity if available
        if user_identity and user_identity.get('id'):
            with DB(DB_PATH) as db:
                db.create_user_session(req.session_id, user_identity['id'])
        
        pipeline = get_rag_pipeline()
        
        # Use provider-based chat if LLM provider is configured
        if hasattr(pipeline, 'llm_provider') and pipeline.llm_provider:
            result = await pipeline.chat_with_provider(
                query=req.query,
                session_id=req.session_id,
                k=req.k,
                search_mode=req.search_mode,
                stream=False,
                user_identity=user_identity
            )
        else:
            result = await pipeline.chat_async(
                query=req.query,
                session_id=req.session_id,
                k=req.k,
                search_mode=req.search_mode,
                stream=False,
                user_identity=user_identity
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

@app.post("/chat/stream")
async def chat_stream(req: ChatStreamReq, auth_info: dict = Depends(require_read_access)):
    """Stream chat responses for real-time interaction"""
    try:
        # Get user identity for this session
        user_identity = get_user_identity_for_session(req.session_id)
        
        # Ensure session is linked to user identity if available
        if user_identity and user_identity.get('id'):
            with DB(DB_PATH) as db:
                db.create_user_session(req.session_id, user_identity['id'])
        
        pipeline = get_rag_pipeline()
        
        # Use provider-based chat if LLM provider is configured
        if hasattr(pipeline, 'llm_provider') and pipeline.llm_provider:
            result = await pipeline.chat_with_provider(
                query=req.query,
                session_id=req.session_id,
                k=req.k,
                search_mode=req.search_mode,
                stream=True,
                user_identity=user_identity
            )
        else:
            result = await pipeline.chat_async(
                query=req.query,
                session_id=req.session_id,
                k=req.k,
                search_mode=req.search_mode,
                stream=True,
                user_identity=user_identity
            )
        
        async def generate():
            try:
                # First send context
                yield json.dumps({'type': 'context', 'data': result['context']})
                yield json.dumps({'type': 'session_id', 'data': result['session_id']})
                
                # Then stream the response
                for token in result["response_generator"]:
                    yield json.dumps({'type': 'token', 'data': token})
                
                # Finally send suggestions
                suggestions = get_rag_pipeline().suggest_follow_up_questions(req.query, result["context"])
                yield json.dumps({'type': 'suggestions', 'data': suggestions})
                yield json.dumps({'type': 'done'})
                
            except Exception as e:
                logging.error(f"Error during streaming: {e}")
                error_msg = {"type": "error", "data": "Streaming interrupted due to model error"}
                yield json.dumps(error_msg)
        
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

@app.post("/analyze")
async def analyze_topics(req: AnalyzeReq, auth_info: dict = Depends(require_read_access)):
    """Analyze topics across conversations"""
    try:
        result = get_rag_pipeline().analyze_conversation_topics(req.doc_ids, req.limit)
        return result
    except Exception as e:
        logging.exception("Analysis error: %s", e)
        raise HTTPException(status_code=500, detail="Analysis error")

@app.post("/suggest")
async def suggest_questions(req: SuggestionsReq, auth_info: dict = Depends(require_read_access)):
    """Get follow-up question suggestions"""
    try:
        suggestions = get_rag_pipeline().suggest_follow_up_questions(req.query, req.results)
        return {"suggestions": suggestions}
    except Exception as e:
        logging.exception("Suggestions error: %s", e)
        raise HTTPException(status_code=500, detail="Suggestions error")

@app.get("/chat/history/{session_id}")
async def get_chat_history(session_id: str, auth_info: dict = Depends(require_read_access)):
    """Get conversation history for a session"""
    try:
        history = get_rag_pipeline().get_conversation_history(session_id)
        return {"session_id": session_id, "history": history}
    except Exception as e:
        logging.exception("History error: %s", e)
        raise HTTPException(status_code=500, detail="History retrieval error")

@app.delete("/chat/history/{session_id}")
async def clear_chat_history(session_id: str, auth_info: dict = Depends(require_read_access)):
    """Clear conversation history for a session"""
    try:
        get_rag_pipeline().clear_conversation_history(session_id)
        return {"message": "History cleared", "session_id": session_id}
    except Exception as e:
        logging.exception("History clear error: %s", e)
        raise HTTPException(status_code=500, detail="History clear error")

@app.get("/summarize/{doc_id}")
async def summarize_conversation(doc_id: int, auth_info: dict = Depends(require_read_access)):
    """Get AI-generated summary of a specific conversation"""
    try:
        summary = get_rag_pipeline().get_conversation_summary(doc_id)
        return {"doc_id": doc_id, "summary": summary}
    except Exception as e:
        logging.exception("Summarization error: %s", e)
        raise HTTPException(status_code=500, detail="Summarization error")

@app.post("/training/feedback")
async def submit_training_feedback(req: TrainingFeedbackReq, auth_info: dict = Depends(require_admin_access)):
    """Submit feedback on a training data response"""
    try:
        with DB(DB_PATH) as db:
            db.update_training_feedback(
                training_data_id=req.training_data_id,
                rating=req.rating,
                correction=req.correction
            )
        return {"message": "Feedback recorded", "training_data_id": req.training_data_id}
    except Exception as e:
        logging.exception("Training feedback error: %s", e)
        raise HTTPException(status_code=500, detail="Failed to record feedback")

@app.get("/training/data")
async def get_training_data(limit: int = 100, min_rating: Optional[int] = None, 
                           model_name: Optional[str] = None, auth_info: dict = Depends(require_admin_access)):
    """Get training data for review and analysis"""
    try:
        with DB(DB_PATH) as db:
            data = db.get_training_data(limit=limit, min_rating=min_rating, model_name=model_name)
            return {
                "training_data": [dict(row) for row in data],
                "count": len(data)
            }
    except Exception as e:
        logging.exception("Training data retrieval error: %s", e)
        raise HTTPException(status_code=500, detail="Failed to retrieve training data")

@app.get("/training/stats")
async def get_training_stats(auth_info: dict = Depends(require_admin_access)):
    """Get training data statistics"""
    try:
        with DB(DB_PATH) as db:
            stats = {}
            
            # Total training examples
            total = db.conn.execute("SELECT COUNT(*) as count FROM training_data").fetchone()
            stats["total_examples"] = total["count"]
            
            # Examples with feedback
            with_feedback = db.conn.execute(
                "SELECT COUNT(*) as count FROM training_data WHERE feedback_rating IS NOT NULL"
            ).fetchone()
            stats["examples_with_feedback"] = with_feedback["count"]
            
            # Average rating
            avg_rating = db.conn.execute(
                "SELECT AVG(feedback_rating) as avg FROM training_data WHERE feedback_rating IS NOT NULL"
            ).fetchone()
            stats["average_rating"] = float(avg_rating["avg"]) if avg_rating["avg"] else None
            
            # Examples by model
            by_model = db.conn.execute(
                "SELECT model_name, COUNT(*) as count FROM training_data GROUP BY model_name"
            ).fetchall()
            stats["by_model"] = {row["model_name"]: row["count"] for row in by_model}
            
            return stats
    except Exception as e:
        logging.exception("Training stats error: %s", e)
        raise HTTPException(status_code=500, detail="Failed to retrieve training stats")

@app.post("/training/start")
async def start_model_training(req: StartTrainingReq, auth_info: dict = Depends(require_admin_access)):
    """Start LoRA fine-tuning on collected training data"""
    try:
        from .training import TrainingManager, TrainingConfig
        
        # Create training config
        config = TrainingConfig(
            base_model_name=req.base_model_name,
            lora_r=req.lora_r,
            lora_alpha=req.lora_alpha,
            num_train_epochs=req.num_epochs,
            learning_rate=req.learning_rate
        )
        
        # Start training (this will run in the background)
        manager = TrainingManager(DB_PATH)
        result = manager.start_training(config)
        
        return result
        
    except ImportError as e:
        raise HTTPException(
            status_code=501, 
            detail="Training dependencies not installed. Install with: pip install peft transformers datasets accelerate bitsandbytes"
        )
    except Exception as e:
        logging.exception("Training start error: %s", e)
        raise HTTPException(status_code=500, detail="Failed to start training")

@app.get("/training/status")
async def get_model_training_status(auth_info: dict = Depends(require_admin_access)):
    """Get training status and available fine-tuned models"""
    try:
        from .training import TrainingManager
        
        manager = TrainingManager(DB_PATH)
        status = manager.get_training_status()
        
        return status
        
    except ImportError:
        return {
            "training_available": False,
            "message": "Training dependencies not installed"
        }
    except Exception as e:
        logging.exception("Training status error: %s", e)
        raise HTTPException(status_code=500, detail="Failed to get training status")

@app.post("/training/activate-model")
async def activate_fine_tuned_model(model_version_id: int, auth_info: dict = Depends(require_admin_access)):
    """Activate a fine-tuned model for use in chat"""
    try:
        with DB(DB_PATH) as db:
            # Verify model exists
            model = db.conn.execute(
                "SELECT * FROM model_version WHERE id = ?", 
                (model_version_id,)
            ).fetchone()
            
            if not model:
                raise HTTPException(status_code=404, detail="Model version not found")
            
            # Set as active
            db.set_active_model(model_version_id)
            
            # Note: In a production system, you'd want to hot-reload the model here
            # For now, the change will take effect on next server restart
            
            return {
                "message": "Model activated successfully",
                "model_version_id": model_version_id,
                "note": "Restart server to load the new model"
            }
            
    except HTTPException:
        raise
    except Exception as e:
        logging.exception("Model activation error: %s", e)
        raise HTTPException(status_code=500, detail="Failed to activate model")

@app.get("/document/{doc_id}")
async def get_document(doc_id: int, auth_info: dict = Depends(require_read_access)):
    """Get full document metadata and content"""
    try:
        with DB(DB_PATH) as db:
            document = db.get_document_by_id(doc_id)
            if not document:
                raise HTTPException(status_code=404, detail="Document not found")
            
            chunks = db.get_document_chunks(doc_id)
            
            # Convert sqlite.Row to dict
            doc_dict = dict(document)
            chunks_list = [dict(chunk) for chunk in chunks]
            
            return {
                "document": doc_dict,
                "chunks": chunks_list,
                "total_chunks": len(chunks_list)
            }
    except HTTPException:
        raise
    except Exception as e:
        logging.exception("Document retrieval error: %s", e)
        raise HTTPException(status_code=500, detail="Document retrieval error")

@app.get("/document/{doc_id}/chunk/{chunk_id}")
async def get_chunk_with_context(doc_id: int, chunk_id: int, context: int = 2, auth_info: dict = Depends(require_read_access)):
    """Get specific chunk with surrounding context"""
    try:
        with DB(DB_PATH) as db:
            result = db.get_chunk_with_context(doc_id, chunk_id, context)
            if not result:
                raise HTTPException(status_code=404, detail="Chunk not found")
            
            # Convert sqlite.Row to dict
            result["target_chunk"] = dict(result["target_chunk"])
            result["context_chunks"] = [dict(chunk) for chunk in result["context_chunks"]]
            
            return result
    except HTTPException:
        raise
    except Exception as e:
        logging.exception("Chunk retrieval error: %s", e)
        raise HTTPException(status_code=500, detail="Chunk retrieval error")

# Model Evaluation and Monthly Reporting Endpoints

@app.post("/evaluation/run")
async def run_model_evaluation(req: EvaluationRequest, auth_info: dict = Depends(require_admin_access)):
    """Run comprehensive evaluation on a model version"""
    try:
        from .model_evaluation import ModelEvaluator
        from .fast_llm_agent import FastLLMConfig
        
        # Create evaluator
        evaluator = ModelEvaluator(DB_PATH)
        
        # Get model configuration or use default
        if req.model_configuration:
            model_config = FastLLMConfig(**req.model_configuration)
        else:
            # Use default config for evaluation
            model_config = FastLLMConfig(
                preset="ultra_fast",  # Fast preset for evaluation
                db_path=DB_PATH,
                collect_training_data=False  # Don't collect training data during evaluation
            )
        
        # Run evaluation
        result = evaluator.run_full_evaluation(model_config, req.model_version_id)
        
        # Convert datetime objects to strings for JSON serialization
        if "evaluation_results" in result:
            for eval_result in result["evaluation_results"]:
                if hasattr(eval_result, 'evaluated_at'):
                    eval_result.evaluated_at = eval_result.evaluated_at.isoformat()
        
        if "snapshot" in result and hasattr(result["snapshot"], 'created_at'):
            result["snapshot"].created_at = result["snapshot"].created_at.isoformat()
        
        return {
            "status": "completed",
            "model_version_id": req.model_version_id,
            "evaluation_summary": {
                "prompts_evaluated": len(result.get("evaluation_results", [])),
                "aggregate_metrics": result.get("aggregate_metrics", {}),
                "evaluated_at": result.get("evaluated_at")
            },
            "message": "Model evaluation completed successfully"
        }
        
    except ImportError as e:
        raise HTTPException(
            status_code=501,
            detail="Evaluation dependencies not available. Install with: pip install sentence-transformers torch"
        )
    except Exception as e:
        logging.exception("Model evaluation error: %s", e)
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")

@app.get("/evaluation/results")
async def get_evaluation_results(model_version_id: Optional[int] = None, limit: int = 100, auth_info: dict = Depends(require_read_access)):
    """Get evaluation results with optional filtering"""
    try:
        with DB(DB_PATH) as db:
            results = db.get_evaluation_results(
                model_version_id=model_version_id,
                limit=limit
            )
            
            return {
                "evaluation_results": results,
                "count": len(results)
            }
            
    except Exception as e:
        logging.exception("Evaluation results retrieval error: %s", e)
        raise HTTPException(status_code=500, detail="Failed to retrieve evaluation results")

@app.get("/evaluation/benchmarks")
async def get_benchmark_prompts(category: Optional[str] = None, auth_info: dict = Depends(require_read_access)):
    """Get benchmark prompts, optionally filtered by category"""
    try:
        with DB(DB_PATH) as db:
            prompts = db.get_benchmark_prompts(category=category)
            return {
                "benchmark_prompts": prompts,
                "count": len(prompts)
            }
            
    except Exception as e:
        logging.exception("Benchmark prompts retrieval error: %s", e)
        raise HTTPException(status_code=500, detail="Failed to retrieve benchmark prompts")

@app.get("/evaluation/snapshots")
async def get_model_snapshots(model_version_id: Optional[int] = None, limit: int = 50, auth_info: dict = Depends(require_read_access)):
    """Get model snapshots with optional filtering"""
    try:
        with DB(DB_PATH) as db:
            snapshots = db.get_model_snapshots(
                model_version_id=model_version_id,
                limit=limit
            )
            
            return {
                "model_snapshots": snapshots,
                "count": len(snapshots)
            }
            
    except Exception as e:
        logging.exception("Model snapshots retrieval error: %s", e)
        raise HTTPException(status_code=500, detail="Failed to retrieve model snapshots")

@app.post("/evaluation/monthly-report")
async def generate_monthly_report(req: MonthlyReportRequest, auth_info: dict = Depends(require_admin_access)):
    """Generate monthly check-in report for model performance tracking"""
    try:
        from .model_evaluation import MonthlyReporter
        from datetime import datetime
        
        # Create target date for the specified month
        target_date = datetime(req.year, req.month, 15)  # Use middle of month
        
        # Generate report
        reporter = MonthlyReporter(DB_PATH)
        report = reporter.generate_monthly_report(target_date)
        
        return {
            "status": "completed",
            "report": report,
            "message": f"Monthly report generated for {req.year}-{req.month:02d}"
        }
        
    except ImportError as e:
        raise HTTPException(
            status_code=501,
            detail="Monthly reporting dependencies not available"
        )
    except Exception as e:
        logging.exception("Monthly report generation error: %s", e)
        raise HTTPException(status_code=500, detail=f"Monthly report generation failed: {str(e)}")

@app.get("/evaluation/monthly-reports")
async def get_monthly_reports(limit: int = 12, auth_info: dict = Depends(require_read_access)):
    """Get recent monthly check-in reports"""
    try:
        with DB(DB_PATH) as db:
            reports = db.get_monthly_checkins(limit=limit)
            
            return {
                "monthly_reports": reports,
                "count": len(reports)
            }
            
    except Exception as e:
        logging.exception("Monthly reports retrieval error: %s", e)
        raise HTTPException(status_code=500, detail="Failed to retrieve monthly reports")

@app.get("/evaluation/monthly-report/{period}")
async def get_monthly_report(period: str, auth_info: dict = Depends(require_read_access)):
    """Get specific monthly report by period (YYYY-MM format)"""
    try:
        # Validate period format
        import re
        if not re.match(r'^\d{4}-\d{2}$', period):
            raise HTTPException(status_code=400, detail="Period must be in YYYY-MM format")
        
        with DB(DB_PATH) as db:
            report = db.get_monthly_checkin(period)
            
            if not report:
                raise HTTPException(status_code=404, detail=f"Monthly report not found for period {period}")
            
            return {
                "period": period,
                "report": report
            }
            
    except HTTPException:
        raise
    except Exception as e:
        logging.exception("Monthly report retrieval error: %s", e)
        raise HTTPException(status_code=500, detail="Failed to retrieve monthly report")

# User Identity Management Endpoints

@app.post("/identity/create")
async def create_user_identity(req: UserIdentityRequest, auth_info: dict = Depends(require_admin_access)):
    """Create or update user identity for personalized responses"""
    try:
        with DB(DB_PATH) as db:
            # Create default user if none exists
            default_email = req.email or "user@localhost"
            user_id = db.upsert_user(default_email, req.full_name)
            
            # Create user identity
            identity_id = db.create_user_identity(
                user_id=user_id,
                full_name=req.full_name,
                role=req.role,
                email=req.email,
                preferences=req.preferences
            )
            
            return {
                "status": "success",
                "identity_id": identity_id,
                "message": f"User identity created for {req.full_name}"
            }
            
    except Exception as e:
        logging.exception("User identity creation error: %s", e)
        raise HTTPException(status_code=500, detail=f"Failed to create user identity: {str(e)}")

@app.get("/identity/current")
async def get_current_identity(auth_info: dict = Depends(require_read_access)):
    """Get the current active user identity"""
    try:
        with DB(DB_PATH) as db:
            identity = db.get_active_user_identity()
            
            if not identity:
                return {"identity": None, "message": "No active user identity found"}
            
            # Convert to dict and format for response
            identity_dict = dict(identity)
            if identity_dict.get('preferences_json'):
                identity_dict['preferences'] = json.loads(identity_dict['preferences_json'])
                del identity_dict['preferences_json']
            
            return {
                "identity": identity_dict,
                "message": "Current user identity retrieved"
            }
            
    except Exception as e:
        logging.exception("User identity retrieval error: %s", e)
        raise HTTPException(status_code=500, detail="Failed to retrieve user identity")

@app.put("/identity/{identity_id}")
async def update_user_identity(identity_id: int, req: UpdateUserIdentityRequest, auth_info: dict = Depends(require_admin_access)):
    """Update user identity information"""
    try:
        with DB(DB_PATH) as db:
            db.update_user_identity(
                identity_id=identity_id,
                full_name=req.full_name,
                role=req.role,
                preferences=req.preferences
            )
            
            return {
                "status": "success",
                "identity_id": identity_id,
                "message": "User identity updated successfully"
            }
            
    except Exception as e:
        logging.exception("User identity update error: %s", e)
        raise HTTPException(status_code=500, detail=f"Failed to update user identity: {str(e)}")

@app.post("/identity/activate/{identity_id}")
async def activate_user_identity(identity_id: int, auth_info: dict = Depends(require_admin_access)):
    """Activate a specific user identity"""
    try:
        with DB(DB_PATH) as db:
            # Check if identity exists
            identity = db.conn.execute(
                "SELECT * FROM user_identity WHERE id = ?",
                (identity_id,)
            ).fetchone()
            
            if not identity:
                raise HTTPException(status_code=404, detail="User identity not found")
            
            # Activate this identity (deactivates others for the same user)
            user_id = identity["user_id"]
            db.conn.execute(
                "UPDATE user_identity SET is_active = 0 WHERE user_id = ?",
                (user_id,)
            )
            db.conn.execute(
                "UPDATE user_identity SET is_active = 1, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                (identity_id,)
            )
            
            return {
                "status": "success",
                "identity_id": identity_id,
                "full_name": identity["full_name"],
                "message": f"User identity activated: {identity['full_name']}"
            }
            
    except HTTPException:
        raise
    except Exception as e:
        logging.exception("User identity activation error: %s", e)
        raise HTTPException(status_code=500, detail="Failed to activate user identity")

# Sync Service Endpoints
@app.post("/sync/config")
async def configure_sync(req: SyncConfigRequest, auth_info: dict = Depends(require_admin_access)):
    """Configure ChatGPT sync settings for the current user."""
    try:
        # For API key auth, use default user
        user_id = 1  # Default user ID for API key authentication
        
        with DB(DB_PATH) as db:
            # Insert or update sync configuration
            db.conn.execute("""
                INSERT OR REPLACE INTO user_sync_config 
                (user_id, sync_enabled, chatgpt_export_url, sync_frequency_hours, updated_at)
                VALUES (?, ?, ?, ?, ?)
            """, (
                user_id,
                req.sync_enabled,
                req.chatgpt_export_url,
                req.sync_frequency_hours,
                datetime.now().isoformat()
            ))
        
        return {
            "status": "success",
            "message": "Sync configuration updated",
            "config": {
                "sync_enabled": req.sync_enabled,
                "sync_frequency_hours": req.sync_frequency_hours,
                "has_export_url": bool(req.chatgpt_export_url)
            }
        }
        
    except Exception as e:
        logging.error(f"Error configuring sync: {e}")
        raise HTTPException(status_code=500, detail="Failed to configure sync")

@app.get("/sync/config")
async def get_sync_config(auth_info: dict = Depends(require_admin_access)):
    """Get current sync configuration for the user."""
    try:
        # For API key auth, use default user
        user_id = 1  # Default user ID for API key authentication
        
        with DB(DB_PATH) as db:
            cur = db.conn.execute("""
                SELECT sync_enabled, chatgpt_export_url, sync_frequency_hours, 
                       last_sync_at, created_at, updated_at
                FROM user_sync_config 
                WHERE user_id = ?
            """, (user_id,))
            
            config = cur.fetchone()
            
            if not config:
                return {
                    "sync_enabled": False,
                    "sync_frequency_hours": 24,
                    "has_export_url": False,
                    "last_sync_at": None
                }
            
            return {
                "sync_enabled": bool(config["sync_enabled"]),
                "sync_frequency_hours": config["sync_frequency_hours"],
                "has_export_url": bool(config["chatgpt_export_url"]),
                "last_sync_at": config["last_sync_at"],
                "created_at": config["created_at"],
                "updated_at": config["updated_at"]
            }
            
    except Exception as e:
        logging.error(f"Error getting sync config: {e}")
        raise HTTPException(status_code=500, detail="Failed to get sync configuration")

@app.post("/sync/manual")
async def manual_sync(req: ManualSyncRequest, auth_info: dict = Depends(require_admin_access)):
    """Trigger a manual sync for the current user."""
    try:
        # For API key auth, use default user
        user_id = 1  # Default user ID for API key authentication
        
        # Check if user has sync configured or provided URL
        export_url = req.export_url
        
        if not export_url:
            # Get URL from user config
            with DB(DB_PATH) as db:
                cur = db.conn.execute("""
                    SELECT chatgpt_export_url FROM user_sync_config 
                    WHERE user_id = ?
                """, (user_id,))
                
                config = cur.fetchone()
                if config and config["chatgpt_export_url"]:
                    export_url = config["chatgpt_export_url"]
        
        if not export_url:
            raise HTTPException(
                status_code=400, 
                detail="No export URL provided. Configure sync or provide URL in request."
            )
        
        # Queue sync task
        task = sync_user_chatgpt_data.delay(user_id, export_url)
        
        return {
            "status": "success",
            "message": "Manual sync started",
            "task_id": task.id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error starting manual sync: {e}")
        raise HTTPException(status_code=500, detail="Failed to start manual sync")

@app.get("/sync/history")
async def get_sync_history(limit: int = 10, auth_info: dict = Depends(require_admin_access)):
    """Get sync history for the current user."""
    try:
        # For API key auth, use default user
        user_id = 1  # Default user ID for API key authentication
        
        with DB(DB_PATH) as db:
            cur = db.conn.execute("""
                SELECT sync_id, sync_type, status, started_at, completed_at, 
                       error_message, files_processed, conversations_added, conversations_updated
                FROM sync_history 
                WHERE user_id = ? 
                ORDER BY started_at DESC 
                LIMIT ?
            """, (user_id, limit))
            
            history = []
            for row in cur.fetchall():
                history.append({
                    "sync_id": row["sync_id"],
                    "sync_type": row["sync_type"],
                    "status": row["status"],
                    "started_at": row["started_at"],
                    "completed_at": row["completed_at"],
                    "error_message": row["error_message"],
                    "files_processed": row["files_processed"],
                    "conversations_added": row["conversations_added"],
                    "conversations_updated": row["conversations_updated"]
                })
            
            return {
                "history": history,
                "total_syncs": len(history)
            }
            
    except Exception as e:
        logging.error(f"Error getting sync history: {e}")
        raise HTTPException(status_code=500, detail="Failed to get sync history")

@app.get("/sync/status/{task_id}")
async def get_sync_task_status(task_id: str):
    """Get status of a specific sync task."""
    try:
        from .sync_service import celery_app
        
        result = celery_app.AsyncResult(task_id)
        
        return {
            "task_id": task_id,
            "status": result.status,
            "result": result.result if result.ready() else None,
            "info": result.info
        }
        
    except Exception as e:
        logging.error(f"Error getting task status: {e}")
        raise HTTPException(status_code=500, detail="Failed to get task status")

# LLM Provider Management Endpoints
@app.get("/providers/available")
async def get_available_providers(auth_info: dict = Depends(require_read_access)):
    """Get list of available LLM providers and their status."""
    try:
        providers = LLMProviderFactory.list_available_providers()
        
        # Add current provider info
        current_provider = None
        try:
            pipeline = get_rag_pipeline()
            if hasattr(pipeline, 'llm_provider') and pipeline.llm_provider:
                current_provider = {
                    "provider": pipeline.llm_provider.provider_type.value,
                    "model": pipeline.llm_provider.model,
                    "model_info": pipeline.llm_provider.get_model_info()
                }
        except Exception as e:
            logger.warning(f"Could not get current provider info: {e}")
        
        return {
            "available_providers": providers,
            "current_provider": current_provider,
            "environment_config": {
                "LLM_PROVIDER": os.getenv("LLM_PROVIDER", "gptj"),
                "LLM_MODEL": os.getenv("LLM_MODEL", "default"),
                "provider_enabled": bool(os.getenv("LLM_PROVIDER", "").lower() in ["openai", "anthropic", "ollama"])
            }
        }
        
    except Exception as e:
        logging.error(f"Error getting provider info: {e}")
        raise HTTPException(status_code=500, detail="Failed to get provider information")

@app.get("/providers/current")  
async def get_current_provider(auth_info: dict = Depends(require_read_access)):
    """Get detailed information about the currently active LLM provider."""
    try:
        pipeline = get_rag_pipeline()
        
        if hasattr(pipeline, 'llm_provider') and pipeline.llm_provider:
            provider_info = pipeline.llm_provider.get_model_info()
            provider_info["available"] = pipeline.llm_provider.is_available()
            
            return {
                "provider_active": True,
                "provider_info": provider_info
            }
        else:
            # Legacy chat agent info
            try:
                chat_agent = await pipeline._get_chat_agent()
                return {
                    "provider_active": False,
                    "legacy_info": {
                        "model_loaded": chat_agent.model is not None if chat_agent else False,
                        "model_name": getattr(chat_agent.config, 'model_name', 'unknown') if chat_agent else 'unknown',
                        "device": getattr(chat_agent.config, 'device', 'unknown') if chat_agent else 'unknown'
                    }
                }
            except Exception as e:
                return {
                    "provider_active": False,
                    "legacy_info": {"error": str(e)}
                }
                
    except Exception as e:
        logging.error(f"Error getting current provider: {e}")
        raise HTTPException(status_code=500, detail="Failed to get current provider")

@app.post("/providers/test")
async def test_provider_connection(auth_info: dict = Depends(require_admin_access)):
    """Test the connection to the currently configured LLM provider."""
    try:
        pipeline = get_rag_pipeline()
        
        if hasattr(pipeline, 'llm_provider') and pipeline.llm_provider:
            # Test with a simple prompt
            test_prompt = "Say 'Hello' to test the connection."
            
            try:
                response = pipeline.llm_provider.generate(test_prompt)
                return {
                    "success": True,
                    "provider": pipeline.llm_provider.provider_type.value,
                    "model": pipeline.llm_provider.model,
                    "test_response": response[:100] + "..." if len(response) > 100 else response
                }
            except Exception as e:
                return {
                    "success": False,
                    "provider": pipeline.llm_provider.provider_type.value,
                    "error": str(e)
                }
        else:
            return {
                "success": False,
                "error": "No LLM provider configured. Using legacy chat agent."
            }
            
    except Exception as e:
        logging.error(f"Error testing provider: {e}")
        raise HTTPException(status_code=500, detail="Failed to test provider connection")

# Helper function to get user identity for chat sessions
def get_user_identity_for_session(session_id: str) -> Optional[Dict]:
    """Get user identity for a chat session"""
    try:
        with DB(DB_PATH) as db:
            # Check if session has an associated user identity
            session_info = db.get_user_session(session_id)
            if session_info:
                identity_dict = {
                    'id': session_info['user_identity_id'],
                    'full_name': session_info['full_name'],
                    'role': session_info['role'],
                    'email': session_info['email']
                }
                if session_info.get('preferences_json'):
                    identity_dict['preferences'] = json.loads(session_info['preferences_json'])
                return identity_dict
            
            # If no session-specific identity, get the active default identity
            identity = db.get_active_user_identity()
            if identity:
                identity_dict = dict(identity)
                if identity_dict.get('preferences_json'):
                    identity_dict['preferences'] = json.loads(identity_dict['preferences_json'])
                    del identity_dict['preferences_json']
                return identity_dict
            
    except Exception as e:
        logging.error(f"Error getting user identity for session {session_id}: {e}")
    
    return None