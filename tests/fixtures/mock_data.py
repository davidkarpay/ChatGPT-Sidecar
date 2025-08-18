"""
Mock data for testing chat functionality
"""
import uuid
from typing import List, Dict, Any
import numpy as np


class MockConversations:
    """Mock ChatGPT conversation data"""
    
    @staticmethod
    def get_sample_conversations() -> List[Dict[str, Any]]:
        """Get sample conversation data"""
        return [
            {
                "id": 1,
                "title": "Introduction to Machine Learning",
                "messages": [
                    {
                        "role": "user",
                        "content": "What is machine learning?"
                    },
                    {
                        "role": "assistant", 
                        "content": "Machine learning is a subset of artificial intelligence (AI) that provides systems the ability to automatically learn and improve from experience without being explicitly programmed. It focuses on the development of computer programs that can access data and use it to learn for themselves."
                    },
                    {
                        "role": "user",
                        "content": "What are the main types of machine learning?"
                    },
                    {
                        "role": "assistant",
                        "content": "The main types of machine learning are: 1) Supervised Learning - uses labeled data to train models, 2) Unsupervised Learning - finds patterns in unlabeled data, 3) Reinforcement Learning - learns through interaction with an environment, and 4) Semi-supervised Learning - uses both labeled and unlabeled data."
                    }
                ]
            },
            {
                "id": 2,
                "title": "Python Programming Tips",
                "messages": [
                    {
                        "role": "user",
                        "content": "What are some best practices for Python programming?"
                    },
                    {
                        "role": "assistant",
                        "content": "Here are key Python best practices: 1) Follow PEP 8 style guidelines, 2) Use meaningful variable names, 3) Write docstrings for functions and classes, 4) Handle exceptions properly, 5) Use list comprehensions when appropriate, 6) Keep functions small and focused, 7) Use virtual environments for dependency management."
                    },
                    {
                        "role": "user",
                        "content": "How do I optimize Python code for performance?"
                    },
                    {
                        "role": "assistant",
                        "content": "To optimize Python performance: 1) Use built-in functions and libraries (NumPy, Pandas), 2) Profile your code to identify bottlenecks, 3) Use generators for memory efficiency, 4) Cache expensive computations, 5) Consider Cython for CPU-intensive tasks, 6) Use multiprocessing for parallel tasks, 7) Optimize data structures (use sets for membership testing)."
                    }
                ]
            },
            {
                "id": 3,
                "title": "Deep Learning Fundamentals",
                "messages": [
                    {
                        "role": "user",
                        "content": "Explain neural networks and deep learning"
                    },
                    {
                        "role": "assistant",
                        "content": "Neural networks are computing systems inspired by biological neural networks. They consist of interconnected nodes (neurons) organized in layers. Deep learning uses neural networks with multiple hidden layers (hence 'deep') to learn complex patterns in data. Each layer transforms the input data, allowing the network to learn hierarchical representations of features."
                    },
                    {
                        "role": "user",
                        "content": "What are some popular deep learning frameworks?"
                    },
                    {
                        "role": "assistant",
                        "content": "Popular deep learning frameworks include: 1) TensorFlow (Google) - comprehensive and production-ready, 2) PyTorch (Facebook) - dynamic and research-friendly, 3) Keras - high-level API for TensorFlow, 4) JAX - research-oriented with strong performance, 5) MXNet (Apache) - flexible and efficient, 6) Caffe - focused on computer vision tasks."
                    }
                ]
            }
        ]
    
    @staticmethod
    def get_sample_chunks() -> List[Dict[str, Any]]:
        """Get sample text chunks for search testing"""
        return [
            {
                "id": 1,
                "document_id": 1,
                "text": "Machine learning is a subset of artificial intelligence (AI) that provides systems the ability to automatically learn and improve from experience without being explicitly programmed.",
                "start_char": 0,
                "end_char": 150,
                "title": "Introduction to Machine Learning"
            },
            {
                "id": 2,
                "document_id": 1,
                "text": "The main types of machine learning are: 1) Supervised Learning - uses labeled data to train models, 2) Unsupervised Learning - finds patterns in unlabeled data, 3) Reinforcement Learning - learns through interaction with an environment.",
                "start_char": 151,
                "end_char": 350,
                "title": "Introduction to Machine Learning"
            },
            {
                "id": 3,
                "document_id": 2,
                "text": "Here are key Python best practices: 1) Follow PEP 8 style guidelines, 2) Use meaningful variable names, 3) Write docstrings for functions and classes, 4) Handle exceptions properly.",
                "start_char": 0,
                "end_char": 180,
                "title": "Python Programming Tips"
            },
            {
                "id": 4,
                "document_id": 3,
                "text": "Neural networks are computing systems inspired by biological neural networks. They consist of interconnected nodes (neurons) organized in layers. Deep learning uses neural networks with multiple hidden layers.",
                "start_char": 0,
                "end_char": 200,
                "title": "Deep Learning Fundamentals"
            }
        ]
    
    @staticmethod
    def get_sample_search_results() -> List[Dict[str, Any]]:
        """Get sample search results"""
        return [
            {
                "rank": 1,
                "score": 0.95,
                "source": "Introduction to Machine Learning",
                "preview": "Machine learning is a subset of artificial intelligence...",
                "text": "Machine learning is a subset of artificial intelligence (AI) that provides systems the ability to automatically learn and improve from experience without being explicitly programmed.",
                "doc_id": 1,
                "chunk_id": 1,
                "start_char": 0,
                "end_char": 150,
                "context_json": {
                    "context": [{
                        "doc": "Introduction to Machine Learning",
                        "loc": {"doc_id": 1, "chunk_id": 1, "start": 0, "end": 150},
                        "quote": "Machine learning is a subset of artificial intelligence..."
                    }]
                }
            },
            {
                "rank": 2,
                "score": 0.87,
                "source": "Deep Learning Fundamentals", 
                "preview": "Neural networks are computing systems inspired by biological neural networks...",
                "text": "Neural networks are computing systems inspired by biological neural networks. They consist of interconnected nodes (neurons) organized in layers.",
                "doc_id": 3,
                "chunk_id": 4,
                "start_char": 0,
                "end_char": 140,
                "context_json": {
                    "context": [{
                        "doc": "Deep Learning Fundamentals",
                        "loc": {"doc_id": 3, "chunk_id": 4, "start": 0, "end": 140},
                        "quote": "Neural networks are computing systems inspired by biological neural networks..."
                    }]
                }
            }
        ]


class MockEmbeddings:
    """Mock embeddings for testing"""
    
    @staticmethod
    def get_sample_embeddings(num_embeddings: int = 10, dim: int = 384) -> np.ndarray:
        """Generate sample embeddings"""
        np.random.seed(42)  # For reproducible tests
        return np.random.rand(num_embeddings, dim).astype(np.float32)
    
    @staticmethod
    def get_query_embedding(dim: int = 384) -> np.ndarray:
        """Generate a sample query embedding"""
        np.random.seed(123)
        return np.random.rand(dim).astype(np.float32)


class MockResponses:
    """Mock AI responses for testing"""
    
    @staticmethod
    def get_sample_chat_responses() -> List[str]:
        """Get sample chat responses"""
        return [
            "Based on your conversation history, machine learning is a powerful technology that enables computers to learn from data without explicit programming.",
            "I found several discussions about Python programming in your conversations. The key themes include best practices, performance optimization, and framework recommendations.",
            "Your conversations cover various aspects of deep learning, including neural network architectures, popular frameworks like TensorFlow and PyTorch, and practical applications.",
            "From your ChatGPT history, I can see you've explored topics ranging from basic programming concepts to advanced machine learning techniques.",
            "The conversations show a progression from fundamental concepts to more advanced topics, demonstrating your learning journey in AI and programming."
        ]
    
    @staticmethod
    def get_sample_suggestions() -> List[List[str]]:
        """Get sample follow-up suggestions"""
        return [
            [
                "What are the practical applications of machine learning?",
                "How do I get started with implementing ML algorithms?",
                "What are the career opportunities in machine learning?"
            ],
            [
                "Can you show me Python code examples for data analysis?",
                "What are the most important Python libraries to learn?",
                "How do I debug Python performance issues?"
            ],
            [
                "What's the difference between deep learning and traditional ML?",
                "How do I choose the right neural network architecture?",
                "What are the latest developments in deep learning?"
            ]
        ]
    
    @staticmethod
    def get_sample_analysis_results() -> List[Dict[str, Any]]:
        """Get sample topic analysis results"""
        return [
            {
                "topics": ["Machine Learning", "Artificial Intelligence", "Data Science"],
                "summary": "Comprehensive discussion about machine learning fundamentals, including supervised and unsupervised learning approaches."
            },
            {
                "topics": ["Python Programming", "Best Practices", "Performance Optimization"],
                "summary": "In-depth exploration of Python programming techniques, focusing on code quality and performance improvements."
            },
            {
                "topics": ["Deep Learning", "Neural Networks", "TensorFlow", "PyTorch"],
                "summary": "Technical discussions about deep learning frameworks and neural network architectures for various applications."
            }
        ]


class MockDatabase:
    """Mock database operations for testing"""
    
    def __init__(self):
        self.conversations = MockConversations.get_sample_conversations()
        self.chunks = MockConversations.get_sample_chunks()
        self.embeddings = MockEmbeddings.get_sample_embeddings()
    
    def get_chunks_by_ids(self, chunk_ids: List[int]) -> List[Dict[str, Any]]:
        """Get chunks by their IDs"""
        return [chunk for chunk in self.chunks if chunk["id"] in chunk_ids]
    
    def get_conversation_by_id(self, doc_id: int) -> Dict[str, Any]:
        """Get conversation by document ID"""
        for conv in self.conversations:
            if conv["id"] == doc_id:
                return conv
        return {}
    
    def search_chunks(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Mock search functionality"""
        # Simple keyword matching for testing
        query_words = query.lower().split()
        results = []
        
        for chunk in self.chunks:
            text_words = chunk["text"].lower().split()
            matches = sum(1 for word in query_words if word in text_words)
            if matches > 0:
                score = matches / len(query_words)
                results.append({
                    **chunk,
                    "score": score,
                    "rank": len(results) + 1
                })
        
        # Sort by score and limit results
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:limit]


class MockSessionManager:
    """Mock session management for testing"""
    
    def __init__(self):
        self.sessions = {}
    
    def get_or_create_session(self, session_id: str = None) -> str:
        """Get existing session or create new one"""
        if session_id is None:
            session_id = str(uuid.uuid4())
        
        if session_id not in self.sessions:
            self.sessions[session_id] = {
                "history": [],
                "created_at": "2024-01-01T00:00:00Z",
                "last_activity": "2024-01-01T00:00:00Z"
            }
        
        return session_id
    
    def add_to_history(self, session_id: str, user_msg: str, assistant_msg: str):
        """Add exchange to session history"""
        if session_id in self.sessions:
            self.sessions[session_id]["history"].append({
                "user": user_msg,
                "assistant": assistant_msg
            })
    
    def get_history(self, session_id: str) -> List[Dict[str, str]]:
        """Get session history"""
        return self.sessions.get(session_id, {}).get("history", [])
    
    def clear_history(self, session_id: str):
        """Clear session history"""
        if session_id in self.sessions:
            self.sessions[session_id]["history"] = []


# Convenience functions for creating test data
def create_test_chat_request(
    query: str = "What is machine learning?",
    session_id: str = None,
    k: int = 5,
    search_mode: str = "adaptive"
) -> Dict[str, Any]:
    """Create a test chat request"""
    return {
        "query": query,
        "session_id": session_id or str(uuid.uuid4()),
        "k": k,
        "search_mode": search_mode
    }


def create_test_chat_response(
    query: str = "What is machine learning?",
    session_id: str = None,
    include_context: bool = True,
    include_suggestions: bool = True
) -> Dict[str, Any]:
    """Create a test chat response"""
    response = {
        "session_id": session_id or str(uuid.uuid4()),
        "response": MockResponses.get_sample_chat_responses()[0],
        "query": query
    }
    
    if include_context:
        response["context"] = MockConversations.get_sample_search_results()
    else:
        response["context"] = []
    
    if include_suggestions:
        response["suggestions"] = MockResponses.get_sample_suggestions()[0]
    else:
        response["suggestions"] = []
    
    return response


def create_test_streaming_response(session_id: str = None):
    """Create a test streaming response generator"""
    def token_generator():
        tokens = ["Hello", " ", "there", "!", " ", "This", " ", "is", " ", "a", " ", "test", " ", "response", "."]
        for token in tokens:
            yield token
    
    return {
        "session_id": session_id or str(uuid.uuid4()),
        "context": MockConversations.get_sample_search_results(),
        "response_generator": token_generator()
    }