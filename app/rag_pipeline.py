import logging
import uuid
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Any
from .db import DB
from .vectorstore import FaissStore
from .mmr import mmr
from .chat_agent import ChatAgent, ChatConfig
from .llm_providers import LLMProviderFactory, LLMProvider, LLMConfig, ProviderType
import numpy as np

logger = logging.getLogger(__name__)

class RAGPipeline:
    def __init__(self, db_path: str, embed_model: str, chat_config: Optional[ChatConfig] = None, 
                 llm_provider: Optional[LLMProvider] = None):
        self.db_path = db_path
        self.embed_model = embed_model
        self.chat_config = chat_config
        self.llm_provider = llm_provider  # New provider-based LLM
        self.chat_agent = None  # Legacy chat agent (for backward compatibility)
        self._chat_agent_loading = False
        self._chat_agent_lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="chat_model")
        
        # Initialize LLM provider if not provided
        if not self.llm_provider:
            try:
                self.llm_provider = LLMProviderFactory.from_env()
                logger.info(f"Initialized LLM provider: {self.llm_provider.provider_type.value}")
            except Exception as e:
                logger.warning(f"Failed to initialize LLM provider: {e}, falling back to legacy chat agent")
        
        self.main_store = None
        self.precision_store = None
        self.balanced_store = None
        self.context_store = None
        
        self._load_stores()
    
    def _generate_with_provider(self, prompt: str, stream: bool = False):
        """Generate response using the configured LLM provider."""
        if self.llm_provider and self.llm_provider.is_available():
            try:
                if stream:
                    return self.llm_provider.generate_stream(prompt)
                else:
                    return self.llm_provider.generate(prompt)
            except Exception as e:
                logger.error(f"LLM provider error: {e}, falling back to legacy chat agent")
                return self._generate_with_legacy_agent(prompt, stream)
        else:
            return self._generate_with_legacy_agent(prompt, stream)
    
    def _generate_with_legacy_agent(self, prompt: str, stream: bool = False):
        """Generate response using the legacy chat agent."""
        # This would need to be implemented based on how the legacy agent works
        # For now, raise an error to indicate fallback is needed
        raise RuntimeError("Legacy chat agent fallback not implemented in provider mode")
    
    def _build_rag_prompt(self, query: str, context_results: List[Dict]) -> str:
        """Build RAG prompt from query and context."""
        context_text = ""
        for i, result in enumerate(context_results[:5], 1):  # Limit context
            source = result.get('source', 'Unknown')
            preview = result.get('preview', '')[:500]  # Limit length
            context_text += f"\n[{i}] From {source}:\n{preview}\n"
        
        prompt = f"""You are a helpful AI assistant. Use the following context to answer the user's question. If the context doesn't contain enough information, say so.

Context:
{context_text}

User Question: {query}

Please provide a helpful and accurate response based on the context above."""
        
        return prompt
    
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
    
    def _load_chat_agent_sync(self) -> ChatAgent:
        """Synchronous chat agent loading (runs in thread pool)"""
        logger.info("Loading chat agent in background thread...")
        return ChatAgent(self.chat_config)
    
    async def _get_chat_agent(self) -> ChatAgent:
        """Async lazy load the chat agent to prevent server blocking"""
        if self.chat_agent is not None:
            return self.chat_agent
            
        with self._chat_agent_lock:
            # Double-check pattern
            if self.chat_agent is not None:
                return self.chat_agent
                
            if self._chat_agent_loading:
                # Wait for another thread to finish loading
                while self._chat_agent_loading and self.chat_agent is None:
                    await asyncio.sleep(0.1)
                if self.chat_agent is not None:
                    return self.chat_agent
                    
            self._chat_agent_loading = True
            
        try:
            logger.info("Initializing chat agent asynchronously...")
            # Load model in thread pool with timeout
            loop = asyncio.get_event_loop()
            self.chat_agent = await asyncio.wait_for(
                loop.run_in_executor(self._executor, self._load_chat_agent_sync),
                timeout=60.0  # 60 second timeout
            )
            logger.info("Chat agent initialized successfully")
            return self.chat_agent
            
        except asyncio.TimeoutError:
            logger.error("Chat agent loading timed out after 60 seconds")
            raise Exception("Chat agent loading timed out")
        except Exception as e:
            logger.error(f"Failed to initialize chat agent: {e}")
            raise
        finally:
            self._chat_agent_loading = False
    
    def _get_chat_agent_sync(self) -> ChatAgent:
        """Synchronous version for backwards compatibility"""
        if self.chat_agent is None:
            logger.info("Initializing chat agent synchronously...")
            try:
                self.chat_agent = ChatAgent(self.chat_config)
                logger.info("Chat agent initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize chat agent: {e}")
                raise
        return self.chat_agent
    
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
                "response_generator": self._get_chat_agent_sync().generate_response(
                    query, context_results, session_id, stream=True
                )
            }
        else:
            response = self._get_chat_agent_sync().generate_response(
                query, context_results, session_id, stream=False
            )
            
            return {
                "session_id": session_id,
                "response": response,
                "context": context_results,
                "query": query
            }
    
    async def chat_with_provider(
        self, 
        query: str, 
        session_id: Optional[str] = None,
        k: int = 5,
        search_mode: str = "adaptive",
        stream: bool = False,
        user_identity: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Chat method using the new LLM provider system."""
        
        if session_id is None:
            session_id = str(uuid.uuid4())
        
        # Get context using search
        context_results = self.search_with_context(
            query, 
            k=k, 
            search_mode=search_mode
        )
        
        # Build RAG prompt
        prompt = self._build_rag_prompt(query, context_results)
        
        # Store the conversation in database
        with DB(self.db_path) as db:
            try:
                # Get user ID from identity if available
                user_id = user_identity.get('id', 1) if user_identity else 1
                
                # Store user query
                db.store_chat_message(session_id, 'user', query, user_id=user_id)
            except Exception as e:
                logger.warning(f"Failed to store user message: {e}")
        
        try:
            if stream:
                # For streaming, we need to collect the response to store it
                def response_generator():
                    full_response = ""
                    try:
                        for token in self._generate_with_provider(prompt, stream=True):
                            full_response += token
                            yield token
                    finally:
                        # Store the complete response
                        with DB(self.db_path) as db:
                            try:
                                user_id = user_identity.get('id', 1) if user_identity else 1
                                db.store_chat_message(session_id, 'assistant', full_response, user_id=user_id)
                                
                                # Store training data if enabled
                                model_name = getattr(self.llm_provider, 'model', 'unknown') if self.llm_provider else 'unknown'
                                db.store_training_data(
                                    session_id=session_id,
                                    user_query=query,
                                    context_json=context_results,
                                    model_response=full_response,
                                    model_name=model_name
                                )
                            except Exception as e:
                                logger.warning(f"Failed to store assistant message: {e}")
                
                return {
                    "session_id": session_id,
                    "context": context_results,
                    "response_generator": response_generator()
                }
            
            else:
                # Non-streaming response
                response = self._generate_with_provider(prompt, stream=False)
                
                # Store the response
                with DB(self.db_path) as db:
                    try:
                        user_id = user_identity.get('id', 1) if user_identity else 1
                        db.store_chat_message(session_id, 'assistant', response, user_id=user_id)
                        
                        # Store training data if enabled
                        model_name = getattr(self.llm_provider, 'model', 'unknown') if self.llm_provider else 'unknown'
                        db.store_training_data(
                            session_id=session_id,
                            user_query=query,
                            context_json=context_results,
                            model_response=response,
                            model_name=model_name
                        )
                    except Exception as e:
                        logger.warning(f"Failed to store assistant message: {e}")
                
                return {
                    "session_id": session_id,
                    "response": response,
                    "context": context_results,
                    "query": query
                }
                
        except Exception as e:
            logger.error(f"Chat generation error: {e}")
            # Fallback to legacy method if provider fails
            return await self.chat_async(query, session_id, k, search_mode, stream, user_identity)
    
    async def chat_async(
        self, 
        query: str, 
        session_id: Optional[str] = None,
        k: int = 5,
        search_mode: str = "adaptive",
        stream: bool = False,
        user_identity: Optional[Dict] = None
    ) -> str | Dict[str, Any]:
        """Async version of chat method (legacy)"""
        
        if session_id is None:
            session_id = str(uuid.uuid4())
        
        context_results = self.search_with_context(
            query, 
            k=k, 
            search_mode=search_mode
        )
        
        chat_agent = await self._get_chat_agent()
        
        # Set user identity for this request
        if user_identity and hasattr(chat_agent, 'config'):
            chat_agent.config.user_identity = user_identity
        
        if stream:
            return {
                "session_id": session_id,
                "context": context_results,
                "response_generator": chat_agent.generate_response(
                    query, context_results, session_id, stream=True
                )
            }
        else:
            response = chat_agent.generate_response(
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
        
        return self._get_chat_agent_sync().analyze_topics(chunks)
    
    def suggest_follow_up_questions(self, query: str, results: List[Dict]) -> List[str]:
        return self._get_chat_agent_sync().suggest_questions(query, results)
    
    def get_conversation_history(self, session_id: str) -> List[Dict]:
        return self._get_chat_agent_sync().get_history(session_id)
    
    def clear_conversation_history(self, session_id: str):
        self._get_chat_agent_sync().clear_history(session_id)
    
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
        
        return self._get_chat_agent_sync().generate_response(
            summary_query,
            [{"preview": full_text[:2000]}],
            stream=False
        )