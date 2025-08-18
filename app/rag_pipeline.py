import logging
import uuid
from typing import Dict, List, Optional, Any
from .db import DB
from .vectorstore import FaissStore
from .mmr import mmr
from .chat_agent import ChatAgent, ChatConfig
import numpy as np

logger = logging.getLogger(__name__)

class RAGPipeline:
    def __init__(self, db_path: str, embed_model: str, chat_config: Optional[ChatConfig] = None):
        self.db_path = db_path
        self.embed_model = embed_model
        self.chat_agent = ChatAgent(chat_config)
        
        self.main_store = None
        self.precision_store = None
        self.balanced_store = None
        self.context_store = None
        
        self._load_stores()
    
    def _load_stores(self):
        try:
            from pathlib import Path
            
            index_dir = Path("data/indexes")
            
            self.main_store = FaissStore(
                index_dir / "main.faiss",
                index_dir / "main.pkl",
                self.embed_model
            )
            self.main_store.load()
            
            if (index_dir / "precision.faiss").exists():
                self.precision_store = FaissStore(
                    index_dir / "precision.faiss",
                    index_dir / "precision.pkl",
                    self.embed_model
                )
                self.precision_store.load()
            
            if (index_dir / "balanced.faiss").exists():
                self.balanced_store = FaissStore(
                    index_dir / "balanced.faiss",
                    index_dir / "balanced.pkl",
                    self.embed_model
                )
                self.balanced_store.load()
            
            if (index_dir / "context.faiss").exists():
                self.context_store = FaissStore(
                    index_dir / "context.faiss",
                    index_dir / "context.pkl",
                    self.embed_model
                )
                self.context_store.load()
                
        except Exception as e:
            logger.warning(f"Could not load all vector stores: {e}")
    
    def search_with_context(
        self, 
        query: str, 
        k: int = 8,
        use_mmr: bool = True,
        lambda_param: float = 0.6,
        search_mode: str = "adaptive"
    ) -> List[Dict]:
        
        results = []
        
        if search_mode == "adaptive":
            results = self._adaptive_search(query, k)
        elif search_mode == "multi_layer":
            results = self._multi_layer_search(query, k)
        else:
            results = self._basic_search(query, k, use_mmr, lambda_param)
        
        return self._enrich_results(results)
    
    def _basic_search(self, query: str, k: int, use_mmr: bool, lambda_param: float) -> List[Dict]:
        if not self.main_store or self.main_store.index is None:
            return []
        
        candidates = k * 5 if use_mmr else k
        base_results = self.main_store.search(query, candidates)
        
        if not base_results:
            return []
        
        faiss_ids = [faiss_idx for faiss_idx, _ in base_results]
        
        with DB(self.db_path) as db:
            rows_by_fid = db.fetch_chunks_by_faiss_indices(faiss_ids, "main")
        
        ordered = []
        for (faiss_idx, score) in base_results:
            r = rows_by_fid.get(faiss_idx)
            if r:
                ordered.append((faiss_idx, score, r))
        
        if not ordered:
            return []
        
        if use_mmr and len(ordered) > k:
            texts = [r["text"] for _, _, r in ordered]
            
            qv = self.main_store.encode([query])[0]
            cand_vecs = self.main_store.encode(texts)
            
            selected_indices = mmr(qv, cand_vecs, lamb=lambda_param, k=k)
            ordered = [ordered[i] for i in selected_indices]
        
        results = []
        for rank, (_, score, row) in enumerate(ordered[:k], start=1):
            results.append({
                "rank": rank,
                "score": float(score),
                "source": row["title"],
                "preview": self._create_preview(row["text"]),
                "text": row["text"],
                "doc_id": row["doc_id"],
                "chunk_id": row["chunk_id"],
                "start_char": row["start_char"],
                "end_char": row["end_char"]
            })
        
        return results
    
    def _adaptive_search(self, query: str, k: int) -> List[Dict]:
        query_length = len(query.split())
        
        if query_length <= 5:
            return self._search_store_with_fallback(self.precision_store, "precision", query, k)
        elif query_length <= 15:
            return self._search_store_with_fallback(self.balanced_store, "balanced", query, k)
        else:
            return self._search_store_with_fallback(self.context_store, "context", query, k)
    
    def _multi_layer_search(self, query: str, k: int) -> List[Dict]:
        all_results = []
        k_per_layer = max(1, k // 3)
        
        stores = [
            (self.precision_store, "precision"),
            (self.balanced_store, "balanced"),
            (self.context_store, "context")
        ]
        
        for store, index_name in stores:
            if store and store.index is not None:
                layer_results = self._search_store_with_fallback(store, index_name, query, k_per_layer)
                all_results.extend(layer_results)
        
        if not all_results:
            return self._basic_search(query, k, True, 0.6)
        
        if len(all_results) > k:
            texts = [r["text"] for r in all_results]
            qv = self.main_store.encode([query])[0] if self.main_store else None
            
            if qv is not None:
                cand_vecs = self.main_store.encode(texts)
                selected_indices = mmr(qv, cand_vecs, lamb=0.6, k=k)
                all_results = [all_results[i] for i in selected_indices]
        
        for rank, result in enumerate(all_results[:k], start=1):
            result["rank"] = rank
        
        return all_results[:k]
    
    def _search_store_with_fallback(self, store: FaissStore, index_name: str, query: str, k: int) -> List[Dict]:
        if not store or store.index is None:
            return self._basic_search(query, k, True, 0.6)
        
        base_results = store.search(query, k * 2)
        if not base_results:
            return []
        
        faiss_ids = [faiss_idx for faiss_idx, _ in base_results]
        
        with DB(self.db_path) as db:
            rows_by_fid = db.fetch_chunks_by_faiss_indices(faiss_ids, index_name)
        
        results = []
        for rank, (faiss_idx, score) in enumerate(base_results[:k], start=1):
            r = rows_by_fid.get(faiss_idx)
            if r:
                results.append({
                    "rank": rank,
                    "score": float(score),
                    "source": r["title"],
                    "preview": self._create_preview(r["text"]),
                    "text": r["text"],
                    "doc_id": r["doc_id"],
                    "chunk_id": r["chunk_id"],
                    "start_char": r["start_char"],
                    "end_char": r["end_char"]
                })
        
        return results
    
    def _create_preview(self, text: str, max_length: int = 320) -> str:
        if len(text) <= max_length:
            return text
        return text[:max_length] + "…"
    
    def _enrich_results(self, results: List[Dict]) -> List[Dict]:
        for result in results:
            result["context_json"] = {
                "context": [{
                    "doc": result["source"],
                    "loc": {
                        "doc_id": result["doc_id"],
                        "chunk_id": result["chunk_id"],
                        "start": result["start_char"],
                        "end": result["end_char"]
                    },
                    "quote": result["text"]
                }]
            }
        return results
    
    def chat(
        self, 
        query: str, 
        session_id: Optional[str] = None,
        k: int = 5,
        search_mode: str = "adaptive",
        stream: bool = False
    ) -> str | Dict[str, Any]:
        
        if session_id is None:
            session_id = str(uuid.uuid4())
        
        context_results = self.search_with_context(
            query, 
            k=k, 
            search_mode=search_mode
        )
        
        if stream:
            return {
                "session_id": session_id,
                "context": context_results,
                "response_generator": self.chat_agent.generate_response(
                    query, context_results, session_id, stream=True
                )
            }
        else:
            response = self.chat_agent.generate_response(
                query, context_results, session_id, stream=False
            )
            
            return {
                "session_id": session_id,
                "response": response,
                "context": context_results,
                "query": query
            }
    
    def analyze_conversation_topics(self, doc_ids: Optional[List[int]] = None, limit: int = 100) -> Dict[str, Any]:
        with DB(self.db_path) as db:
            if doc_ids:
                cur = db.conn.execute(
                    "SELECT text FROM chunk WHERE document_id IN ({}) LIMIT ?".format(
                        ",".join("?" * len(doc_ids))
                    ),
                    doc_ids + [limit]
                )
            else:
                cur = db.conn.execute("SELECT text FROM chunk LIMIT ?", (limit,))
            
            chunks = [{"preview": row["text"]} for row in cur.fetchall()]
        
        if not chunks:
            return {"error": "No chunks found for analysis"}
        
        return self.chat_agent.analyze_topics(chunks)
    
    def suggest_follow_up_questions(self, query: str, results: List[Dict]) -> List[str]:
        return self.chat_agent.suggest_questions(query, results)
    
    def get_conversation_history(self, session_id: str) -> List[Dict]:
        return self.chat_agent.get_history(session_id)
    
    def clear_conversation_history(self, session_id: str):
        self.chat_agent.clear_history(session_id)
    
    def get_conversation_summary(self, doc_id: int) -> str:
        with DB(self.db_path) as db:
            cur = db.conn.execute(
                "SELECT title, text FROM chunk WHERE document_id = ? ORDER BY start_char ASC",
                (doc_id,)
            )
            chunks = cur.fetchall()
        
        if not chunks:
            return "No conversation found"
        
        title = chunks[0]["title"] if chunks else "Unknown"
        full_text = "\n".join([chunk["text"] for chunk in chunks])
        
        summary_query = f"Summarize this ChatGPT conversation titled '{title}'"
        
        return self.chat_agent.generate_response(
            summary_query,
            [{"preview": full_text[:2000]}],
            stream=False
        )