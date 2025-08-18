"""
Pytest configuration and global fixtures
"""
import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, Mock
from fastapi.testclient import TestClient

from app.main import app
from tests.fixtures.mock_agents import create_mock_environment, MockRAGPipeline, MockChatAgent
from tests.fixtures.mock_data import MockDatabase, MockSessionManager


@pytest.fixture(scope="session")
def test_app():
    """Create test FastAPI application"""
    return app


@pytest.fixture
def client(test_app):
    """Create test client"""
    return TestClient(test_app)


@pytest.fixture
def auth_headers():
    """Standard authentication headers"""
    return {"X-API-Key": "change-me"}


@pytest.fixture
def invalid_auth_headers():
    """Invalid authentication headers for testing auth failures"""
    return {"X-API-Key": "invalid-key"}


@pytest.fixture
def mock_environment():
    """Create complete mock environment"""
    return create_mock_environment()


@pytest.fixture
def mock_rag_pipeline():
    """Mock RAG pipeline"""
    return MockRAGPipeline("test.db", "test-model")


@pytest.fixture
def mock_chat_agent():
    """Mock chat agent"""
    return MockChatAgent()


@pytest.fixture
def mock_session_manager():
    """Mock session manager"""
    return MockSessionManager()


@pytest.fixture
def temp_db_path():
    """Create temporary database path"""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        yield tmp.name
    os.unlink(tmp.name)


@pytest.fixture
def temp_index_dir():
    """Create temporary index directory"""
    with tempfile.TemporaryDirectory() as tmpdir:
        index_dir = Path(tmpdir) / "indexes"
        index_dir.mkdir(exist_ok=True)
        yield index_dir


@pytest.fixture(autouse=True)
def mock_rag_pipeline_getter():
    """Auto-mock the RAG pipeline getter for all tests"""
    with patch('app.main.get_rag_pipeline') as mock_getter:
        mock_pipeline = MockRAGPipeline("test.db", "test-model")
        mock_getter.return_value = mock_pipeline
        yield mock_pipeline


@pytest.fixture
def sample_chat_request():
    """Standard chat request for testing"""
    return {
        "query": "What is machine learning?",
        "session_id": "test-session-123",
        "k": 5,
        "search_mode": "adaptive"
    }


@pytest.fixture
def sample_streaming_request():
    """Standard streaming request for testing"""
    return {
        "query": "Explain deep learning",
        "session_id": "stream-session-456", 
        "k": 8,
        "search_mode": "multi_layer"
    }


@pytest.fixture
def sample_analyze_request():
    """Standard analyze request for testing"""
    return {
        "doc_ids": [1, 2, 3],
        "limit": 50
    }


@pytest.fixture
def sample_suggestions_request():
    """Standard suggestions request for testing"""
    return {
        "query": "What is artificial intelligence?",
        "results": [
            {"preview": "AI is a field of computer science", "source": "AI Basics"},
            {"preview": "Machine learning is a subset of AI", "source": "ML Guide"}
        ]
    }


@pytest.fixture(scope="session")
def sample_conversation_data():
    """Sample conversation data for testing"""
    return {
        "conversations": [
            {
                "id": 1,
                "title": "ML Discussion",
                "messages": [
                    {"role": "user", "content": "What is ML?"},
                    {"role": "assistant", "content": "Machine learning is..."}
                ]
            }
        ],
        "chunks": [
            {
                "id": 1,
                "text": "Machine learning is a subset of AI",
                "doc_id": 1,
                "start_char": 0,
                "end_char": 35
            }
        ]
    }


@pytest.fixture
def mock_model_loading():
    """Mock model loading to avoid actual model downloads"""
    with patch('app.chat_agent.AutoTokenizer.from_pretrained') as mock_tokenizer, \
         patch('app.chat_agent.AutoModelForCausalLM.from_pretrained') as mock_model:
        
        # Setup tokenizer mock
        mock_tokenizer_instance = Mock()
        mock_tokenizer_instance.pad_token = None
        mock_tokenizer_instance.eos_token = "<|endoftext|>"
        mock_tokenizer_instance.pad_token_id = 0
        mock_tokenizer_instance.eos_token_id = 1
        mock_tokenizer.return_value = mock_tokenizer_instance
        
        # Setup model mock
        mock_model_instance = Mock()
        mock_model.return_value = mock_model_instance
        
        yield {
            "tokenizer": mock_tokenizer_instance,
            "model": mock_model_instance
        }


@pytest.fixture
def mock_vector_store():
    """Mock vector store operations"""
    with patch('app.vectorstore.FaissStore') as mock_store_class:
        mock_store = Mock()
        mock_store.search.return_value = [(0, 0.95), (1, 0.87), (2, 0.82)]
        mock_store.encode.return_value = [[0.1] * 384]  # Mock embedding
        mock_store.model.get_sentence_embedding_dimension.return_value = 384
        mock_store_class.return_value = mock_store
        yield mock_store


@pytest.fixture
def mock_database():
    """Mock database operations"""
    with patch('app.db.DB') as mock_db_class:
        mock_db = MockDatabase("test.db")
        mock_db_class.return_value = mock_db
        yield mock_db


# Test markers for different test categories
def pytest_configure(config):
    """Configure custom pytest markers"""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )
    config.addinivalue_line(
        "markers", "performance: marks tests as performance tests"
    )
    config.addinivalue_line(
        "markers", "edge_case: marks tests as edge case tests"
    )


# Test collection and execution hooks
def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test file names"""
    for item in items:
        # Add markers based on file names
        if "performance" in str(item.fspath):
            item.add_marker(pytest.mark.performance)
        elif "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        elif "edge_case" in str(item.fspath):
            item.add_marker(pytest.mark.edge_case)
        elif "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)


@pytest.fixture(autouse=True)
def setup_test_environment():
    """Setup test environment for each test"""
    # Set test environment variables
    os.environ["CHAT_MODEL"] = "distilgpt2"  # Small model for testing
    os.environ["CHAT_MAX_CONTEXT"] = "512"
    os.environ["CHAT_MAX_TOKENS"] = "50"
    os.environ["CHAT_USE_8BIT"] = "false"
    
    yield
    
    # Cleanup after test
    test_env_vars = [
        "CHAT_MODEL", "CHAT_MAX_CONTEXT", 
        "CHAT_MAX_TOKENS", "CHAT_USE_8BIT"
    ]
    for var in test_env_vars:
        if var in os.environ:
            del os.environ[var]


# Performance testing utilities
@pytest.fixture
def performance_timer():
    """Timer utility for performance tests"""
    import time
    
    class Timer:
        def __init__(self):
            self.start_time = None
            self.end_time = None
        
        def start(self):
            self.start_time = time.time()
        
        def stop(self):
            self.end_time = time.time()
        
        @property
        def elapsed(self):
            if self.start_time and self.end_time:
                return self.end_time - self.start_time
            return None
    
    return Timer()


# Memory monitoring for performance tests
@pytest.fixture
def memory_monitor():
    """Memory monitoring utility"""
    import psutil
    
    class MemoryMonitor:
        def __init__(self):
            self.process = psutil.Process()
            self.baseline = None
            self.measurements = []
        
        def start(self):
            self.baseline = self.process.memory_info().rss / 1024 / 1024  # MB
        
        def measure(self):
            current = self.process.memory_info().rss / 1024 / 1024  # MB
            self.measurements.append(current)
            return current
        
        @property
        def peak(self):
            return max(self.measurements) if self.measurements else None
        
        @property
        def increase(self):
            return self.peak - self.baseline if self.peak and self.baseline else None
    
    return MemoryMonitor()