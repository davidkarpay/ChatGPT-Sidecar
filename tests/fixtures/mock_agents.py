"""
Mock agents and components for testing
"""
import time
from typing import Dict, List, Optional, Any, Generator
from unittest.mock import Mock, MagicMock
import uuid

from app.chat_agent import ChatConfig
from .mock_data import MockResponses, MockConversations, MockEmbeddings


class MockChatAgent:
    """Lightweight mock chat agent for testing"""
    
    def __init__(self, config: Optional[ChatConfig] = None):
        self.config = config or ChatConfig()
        self.conversation_history = {}
        self.model = None
        self.tokenizer = None
        self._responses = MockResponses.get_sample_chat_responses()
        self._suggestions = MockResponses.get_sample_suggestions()
        self._analysis_results = MockResponses.get_sample_analysis_results()
    
    def generate_response(
        self, 
        query: str, 
        context: List[Dict], 
        session_id: Optional[str] = None,
        stream: bool = False
    ) -> str | Generator[str, None, None]:
        """Generate mock response"""
        if stream:
            return self._stream_response(query, session_id)
        else:
            response = self._get_response_for_query(query)
            
            if session_id:
                if session_id not in self.conversation_history:
                    self.conversation_history[session_id] = []
                self.conversation_history[session_id].append({
                    "user": query,
                    "assistant": response
                })
            
            return response
    
    def _stream_response(self, query: str, session_id: Optional[str] = None) -> Generator[str, None, None]:
        """Generate streaming mock response"""
        response = self._get_response_for_query(query)
        words = response.split()
        
        for word in words:
            yield word + " "
            time.sleep(0.01)  # Small delay to simulate streaming
        
        # Add to history after streaming is complete
        if session_id:
            if session_id not in self.conversation_history:
                self.conversation_history[session_id] = []
            self.conversation_history[session_id].append({
                "user": query,
                "assistant": response
            })
    
    def _get_response_for_query(self, query: str) -> str:
        """Get appropriate response based on query content"""
        query_lower = query.lower()
        
        if "machine learning" in query_lower or "ml" in query_lower:
            return self._responses[0]
        elif "python" in query_lower or "programming" in query_lower:
            return self._responses[1]
        elif "deep learning" in query_lower or "neural" in query_lower:
            return self._responses[2]
        elif "conversation" in query_lower or "history" in query_lower:
            return self._responses[3]
        else:
            return self._responses[4]  # Default response
    
    def suggest_questions(self, query: str, results: List[Dict]) -> List[str]:
        """Generate mock question suggestions"""
        query_lower = query.lower()
        
        if "machine learning" in query_lower:
            return self._suggestions[0]
        elif "python" in query_lower:
            return self._suggestions[1]
        elif "deep learning" in query_lower:
            return self._suggestions[2]
        else:
            return self._suggestions[0]  # Default suggestions
    
    def analyze_topics(self, chunks: List[Dict]) -> Dict[str, Any]:
        """Generate mock topic analysis"""
        if not chunks:
            return {"topics": [], "summary": "No content to analyze"}
        
        # Simple heuristic based on content
        text_content = " ".join([chunk.get("preview", "") for chunk in chunks])
        
        if "machine learning" in text_content.lower():
            return self._analysis_results[0]
        elif "python" in text_content.lower():
            return self._analysis_results[1]
        elif "deep learning" in text_content.lower():
            return self._analysis_results[2]
        else:
            return {
                "topics": ["General Discussion"],
                "summary": "Mixed topic conversation covering various subjects"
            }
    
    def get_history(self, session_id: str) -> List[Dict]:
        """Get conversation history"""
        return self.conversation_history.get(session_id, [])
    
    def clear_history(self, session_id: str):
        """Clear conversation history"""
        if session_id in self.conversation_history:
            del self.conversation_history[session_id]


class MockFaissStore:
    """Mock FAISS vector store for testing"""
    
    def __init__(self, index_path=None, ids_path=None, model_name=None):
        self.index_path = index_path
        self.ids_path = ids_path
        self.model_name = model_name
        self.index = Mock()
        self.ids = [(i, i+10) for i in range(10)]  # Mock (embedding_ref_id, chunk_id) pairs
        self.embeddings = MockEmbeddings.get_sample_embeddings()
        
        # Mock model for encoding
        self.model = Mock()
        self.model.encode.side_effect = self._mock_encode
        self.model.get_sentence_embedding_dimension.return_value = 384
    
    def _mock_encode(self, texts, **kwargs):
        """Mock encoding function"""
        if isinstance(texts, str):
            texts = [texts]
        
        # Return mock embeddings based on text content
        embeddings = []
        for text in texts:
            # Simple hash-based mock embedding
            hash_val = hash(text) % 1000
            embedding = MockEmbeddings.get_sample_embeddings(1, 384)[0]
            # Modify embedding slightly based on text hash for consistency
            embedding[0] = hash_val / 1000.0
            embeddings.append(embedding)
        
        return embeddings
    
    def load(self):
        """Mock load operation"""
        pass
    
    def save(self):
        """Mock save operation"""
        pass
    
    def search(self, query: str, k: int = 8) -> List[tuple]:
        """Mock search operation"""
        # Simulate search results with decreasing scores
        results = []
        for i in range(min(k, len(self.ids))):
            faiss_idx = i
            score = 0.95 - (i * 0.05)  # Decreasing relevance scores
            results.append((faiss_idx, score))
        
        return results
    
    def add(self, rows):
        """Mock add operation"""
        return list(range(len(rows)))
    
    def build(self, rows):
        """Mock build operation"""
        self.ids = [(i, row['chunk_id']) for i, row in enumerate(rows)]


class MockRAGPipeline:
    """Mock RAG pipeline for testing"""
    
    def __init__(self, db_path: str, embed_model: str, chat_config=None):
        self.db_path = db_path
        self.embed_model = embed_model
        self.chat_agent = MockChatAgent(chat_config)
        
        # Mock stores
        self.main_store = MockFaissStore()
        self.precision_store = MockFaissStore()
        self.balanced_store = MockFaissStore()
        self.context_store = MockFaissStore()
        
        self._sample_contexts = MockConversations.get_sample_search_results()
    
    def search_with_context(
        self, 
        query: str, 
        k: int = 8,
        use_mmr: bool = True,
        lambda_param: float = 0.6,
        search_mode: str = "adaptive"
    ) -> List[Dict]:
        """Mock context search"""
        # Return mock search results based on query
        results = self._sample_contexts[:k]
        
        # Modify results based on query content for more realistic testing
        query_lower = query.lower()
        if "python" in query_lower:
            # Simulate Python-related results
            for result in results:
                result["source"] = "Python Programming Tips"
                result["preview"] = "Python best practices and optimization techniques..."
        elif "deep learning" in query_lower:
            # Simulate deep learning results
            for result in results:
                result["source"] = "Deep Learning Fundamentals"
                result["preview"] = "Neural networks and deep learning concepts..."
        
        return results
    
    def chat(
        self, 
        query: str, 
        session_id: Optional[str] = None,
        k: int = 5,
        search_mode: str = "adaptive",
        stream: bool = False
    ) -> Dict[str, Any] | str:
        """Mock chat functionality"""
        if session_id is None:
            session_id = str(uuid.uuid4())
        
        # Get mock context
        context_results = self.search_with_context(query, k=k, search_mode=search_mode)
        
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
    
    def suggest_follow_up_questions(self, query: str, results: List[Dict]) -> List[str]:
        """Mock suggestion generation"""
        return self.chat_agent.suggest_questions(query, results)
    
    def analyze_conversation_topics(self, doc_ids: Optional[List[int]] = None, limit: int = 100) -> Dict[str, Any]:
        """Mock topic analysis"""
        # Simulate analysis based on doc_ids
        if doc_ids:
            chunks = [{"preview": f"Content from document {doc_id}"} for doc_id in doc_ids[:5]]
        else:
            chunks = [{"preview": "General conversation content"}]
        
        return self.chat_agent.analyze_topics(chunks)
    
    def get_conversation_history(self, session_id: str) -> List[Dict]:
        """Mock history retrieval"""
        return self.chat_agent.get_history(session_id)
    
    def clear_conversation_history(self, session_id: str):
        """Mock history clearing"""
        self.chat_agent.clear_history(session_id)
    
    def get_conversation_summary(self, doc_id: int) -> str:
        """Mock conversation summarization"""
        # Simulate summary based on doc_id
        summaries = [
            "This conversation explores machine learning fundamentals and practical applications.",
            "A comprehensive discussion about Python programming best practices and optimization.",
            "An in-depth exploration of deep learning architectures and frameworks.",
            "General programming discussion covering various languages and techniques.",
            "Data science workflow and analysis methodologies."
        ]
        
        return summaries[doc_id % len(summaries)]


class MockDatabase:
    """Mock database for testing"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conversations = MockConversations.get_sample_conversations()
        self.chunks = MockConversations.get_sample_chunks()
        self.conn = Mock()  # Mock connection
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
    
    def fetch_chunks_by_faiss_indices(self, faiss_ids: List[int], index_name: str) -> Dict[int, Dict]:
        """Mock chunk fetching by FAISS indices"""
        result = {}
        for i, faiss_id in enumerate(faiss_ids):
            if i < len(self.chunks):
                chunk = self.chunks[i].copy()
                result[faiss_id] = chunk
        return result
    
    def init_schema(self, schema_path):
        """Mock schema initialization"""
        pass
    
    def create_embedding_refs(self, chunk_ids, index_name, vector_dim, faiss_ids):
        """Mock embedding reference creation"""
        pass


def create_mock_environment():
    """Create a complete mock environment for testing"""
    return {
        "chat_agent": MockChatAgent(),
        "rag_pipeline": MockRAGPipeline("mock.db", "mock-model"),
        "faiss_store": MockFaissStore(),
        "database": MockDatabase("mock.db")
    }