"""Integration tests for API endpoints."""

import pytest
import json
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import numpy as np

from fastapi.testclient import TestClient


class TestAPIFixtures:
    """Test fixtures for API testing."""
    
    @pytest.fixture
    def temp_db_with_data(self):
        """Create temporary database with test data."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        
        # Initialize schema and add test data
        from app.db import DB
        with DB(db_path) as db:
            schema_path = Path(__file__).parent.parent / 'schema.sql'
            if schema_path.exists():
                db.init_schema(str(schema_path))
            
            # Create test user
            user_id = db.upsert_user("test@example.com", "Test User")
            
            # Create test documents
            doc1_id = db.upsert_document(
                user_id=user_id,
                title="Python Tutorial",
                doc_type="chatgpt_export",
                fingerprint="abc123",
                metadata={"topic": "programming"}
            )
            
            doc2_id = db.upsert_document(
                user_id=user_id,
                title="Machine Learning Basics",
                doc_type="chatgpt_export",
                fingerprint="def456",
                metadata={"topic": "ai"}
            )
            
            # Create test chunks
            chunks1 = [
                {"text": "Python is a programming language", "start_char": 0, "end_char": 32},
                {"text": "It's great for data science and web development", "start_char": 33, "end_char": 80}
            ]
            chunks2 = [
                {"text": "Machine learning is a subset of AI", "start_char": 0, "end_char": 34},
                {"text": "It uses algorithms to find patterns in data", "start_char": 35, "end_char": 78}
            ]
            
            chunk_ids1 = db.insert_chunks(doc1_id, chunks1)
            chunk_ids2 = db.insert_chunks(doc2_id, chunks2)
            
            # Create embedding references (mock FAISS IDs)
            all_chunk_ids = chunk_ids1 + chunk_ids2
            faiss_ids = list(range(len(all_chunk_ids)))
            db.create_embedding_refs(all_chunk_ids, "main", 384, faiss_ids)
        
        yield db_path, all_chunk_ids
        
        # Cleanup
        if os.path.exists(db_path):
            os.unlink(db_path)
    
    @pytest.fixture
    def mock_vector_store(self):
        """Mock FAISS vector store."""
        mock_store = MagicMock()
        
        # Mock search results
        def mock_search(query, k=8):
            # Return mock scores and indices
            scores = np.array([0.9, 0.8, 0.7, 0.6])[:k]
            indices = np.array([0, 1, 2, 3])[:k]
            return scores, indices
        
        mock_store.search = mock_search
        mock_store.load = MagicMock()
        
        return mock_store
    
    @pytest.fixture
    def api_client(self, temp_db_with_data, mock_vector_store):
        """Create test client with mocked dependencies."""
        db_path, chunk_ids = temp_db_with_data
        
        with patch.dict(os.environ, {
            'API_KEY': 'test-api-key',
            'DB_PATH': db_path
        }):
            # Mock the vector store
            with patch('app.main.store', mock_vector_store):
                from app.main import app
                return TestClient(app), chunk_ids


class TestHealthEndpoint:
    """Test health check endpoint."""
    
    def test_health_check(self, api_client):
        """Test health check endpoint."""
        client, _ = api_client
        
        response = client.get("/healthz")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
    
    def test_health_check_no_auth_required(self, api_client):
        """Test that health check doesn't require authentication."""
        client, _ = api_client
        
        # No API key header
        response = client.get("/healthz")
        assert response.status_code == 200


class TestSearchEndpoints:
    """Test search-related endpoints."""
    
    def test_basic_search_success(self, api_client):
        """Test successful basic search."""
        client, chunk_ids = api_client
        
        response = client.post(
            "/search",
            json={"query": "Python programming", "k": 2},
            headers={"X-API-Key": "test-api-key"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        assert len(data["results"]) <= 2
        
        # Check result structure
        if data["results"]:
            result = data["results"][0]
            assert "text" in result
            assert "score" in result
            assert "document_title" in result
    
    def test_search_without_api_key(self, api_client):
        """Test search without API key returns 401."""
        client, _ = api_client
        
        response = client.post(
            "/search",
            json={"query": "test"}
        )
        
        assert response.status_code == 401
    
    def test_search_invalid_parameters(self, api_client):
        """Test search with invalid parameters."""
        client, _ = api_client
        
        # Missing query
        response = client.post(
            "/search",
            json={"k": 5},
            headers={"X-API-Key": "test-api-key"}
        )
        assert response.status_code == 422  # Validation error
        
        # Invalid k value
        response = client.post(
            "/search",
            json={"query": "test", "k": -1},
            headers={"X-API-Key": "test-api-key"}
        )
        assert response.status_code == 422
    
    def test_advanced_search_with_mmr(self, api_client):
        """Test advanced search with MMR parameters."""
        client, _ = api_client
        
        response = client.post(
            "/search/advanced",
            json={
                "query": "machine learning",
                "k": 3,
                "candidates": 20,
                "lambda": 0.7
            },
            headers={"X-API-Key": "test-api-key"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        assert "search_params" in data
        assert data["search_params"]["lambda"] == 0.7
    
    def test_search_empty_query(self, api_client):
        """Test search with empty query."""
        client, _ = api_client
        
        response = client.post(
            "/search",
            json={"query": "", "k": 5},
            headers={"X-API-Key": "test-api-key"}
        )
        
        # Should handle gracefully
        assert response.status_code in [200, 400]
    
    def test_search_large_k_value(self, api_client):
        """Test search with very large k value."""
        client, _ = api_client
        
        response = client.post(
            "/search",
            json={"query": "test", "k": 1000},
            headers={"X-API-Key": "test-api-key"}
        )
        
        assert response.status_code == 200
        data = response.json()
        # Should limit results to available documents
        assert len(data["results"]) <= 4  # We have 4 chunks in test data


class TestIngestionEndpoints:
    """Test data ingestion endpoints."""
    
    def test_reindex_endpoint(self, api_client):
        """Test FAISS index rebuild endpoint."""
        client, _ = api_client
        
        with patch('app.main.store') as mock_store:
            mock_store.build = MagicMock()
            
            response = client.post(
                "/reindex",
                headers={"X-API-Key": "test-api-key"}
            )
            
            assert response.status_code == 200
            data = response.json()
            assert "message" in data
            assert "indexed" in data["message"] or "rebuilt" in data["message"]
            
            # Verify build was called
            mock_store.build.assert_called_once()
    
    def test_chatgpt_import_endpoint_no_file(self, api_client):
        """Test ChatGPT import endpoint without file."""
        client, _ = api_client
        
        response = client.post(
            "/ingest/chatgpt-export",
            headers={"X-API-Key": "test-api-key"}
        )
        
        # Should return error for missing file
        assert response.status_code == 422
    
    @pytest.mark.integration
    def test_chatgpt_import_endpoint_with_file(self, api_client):
        """Test ChatGPT import with actual file upload."""
        client, _ = api_client
        
        # Create a minimal valid export
        test_export = {
            "conversations": [
                {
                    "title": "Test Import",
                    "mapping": {
                        "root": {
                            "id": "root",
                            "message": {
                                "author": {"role": "user"},
                                "content": {"parts": ["Test message"]}
                            }
                        }
                    }
                }
            ]
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(test_export, f)
            temp_file = f.name
        
        try:
            with patch('app.main.ingest_export') as mock_ingest:
                mock_ingest.return_value = 1
                
                with open(temp_file, 'rb') as upload_file:
                    response = client.post(
                        "/ingest/chatgpt-export",
                        files={"file": ("export.json", upload_file, "application/json")},
                        headers={"X-API-Key": "test-api-key"}
                    )
                
                assert response.status_code == 200
                data = response.json()
                assert "ingested" in data["message"]
        
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)


class TestMultiLayerSearch:
    """Test multi-layer search functionality."""
    
    def test_multi_layer_search_endpoint(self, api_client):
        """Test multi-layer search endpoint."""
        client, _ = api_client
        
        with patch('app.main.fusion_search') as mock_fusion:
            mock_fusion.search_all_layers.return_value = [
                {"text": "Result 1", "score": 0.9, "layer": "layer1"},
                {"text": "Result 2", "score": 0.8, "layer": "layer2"}
            ]
            
            response = client.post(
                "/search/multi-layer",
                json={"query": "test query", "k": 5},
                headers={"X-API-Key": "test-api-key"}
            )
            
            assert response.status_code == 200
            data = response.json()
            assert "results" in data
            assert len(data["results"]) == 2
    
    def test_specific_layer_search(self, api_client):
        """Test searching specific layer."""
        client, _ = api_client
        
        with patch('app.main.fusion_search') as mock_fusion:
            mock_fusion.search_layer.return_value = [
                {"text": "Layer result", "score": 0.95, "layer": "specific_layer"}
            ]
            
            response = client.post(
                "/search/layer/specific_layer",
                json={"query": "layer specific query", "k": 3},
                headers={"X-API-Key": "test-api-key"}
            )
            
            assert response.status_code == 200
            data = response.json()
            assert "results" in data
            assert data["results"][0]["layer"] == "specific_layer"


class TestChatEndpoints:
    """Test chat-related endpoints."""
    
    @pytest.fixture
    def mock_rag_pipeline(self):
        """Mock RAG pipeline for chat tests."""
        mock_pipeline = MagicMock()
        
        def mock_chat(query, **kwargs):
            return f"Response to: {query}"
        
        def mock_chat_stream(query, **kwargs):
            response_parts = [f"Streaming response to: {query}"]
            for part in response_parts:
                yield f"data: {json.dumps({'content': part})}\n\n"
        
        mock_pipeline.chat = mock_chat
        mock_pipeline.chat_stream = mock_chat_stream
        
        return mock_pipeline
    
    def test_chat_endpoint(self, api_client, mock_rag_pipeline):
        """Test chat endpoint."""
        client, _ = api_client
        
        with patch('app.main.get_rag_pipeline', return_value=mock_rag_pipeline):
            response = client.post(
                "/chat",
                json={
                    "message": "What is Python?",
                    "search_mode": "basic"
                },
                headers={"X-API-Key": "test-api-key"}
            )
            
            assert response.status_code == 200
            data = response.json()
            assert "response" in data
            assert "What is Python?" in data["response"]
    
    def test_chat_stream_endpoint(self, api_client, mock_rag_pipeline):
        """Test streaming chat endpoint."""
        client, _ = api_client
        
        with patch('app.main.get_rag_pipeline', return_value=mock_rag_pipeline):
            response = client.post(
                "/chat/stream",
                json={
                    "message": "Explain machine learning",
                    "search_mode": "advanced"
                },
                headers={"X-API-Key": "test-api-key"}
            )
            
            assert response.status_code == 200
            assert response.headers["content-type"].startswith("text/plain")
    
    def test_chat_without_message(self, api_client):
        """Test chat endpoint without message."""
        client, _ = api_client
        
        response = client.post(
            "/chat",
            json={"search_mode": "basic"},
            headers={"X-API-Key": "test-api-key"}
        )
        
        assert response.status_code == 422  # Validation error
    
    def test_chat_with_context(self, api_client, mock_rag_pipeline):
        """Test chat with conversation context."""
        client, _ = api_client
        
        with patch('app.main.get_rag_pipeline', return_value=mock_rag_pipeline):
            response = client.post(
                "/chat",
                json={
                    "message": "Tell me more about that",
                    "context": [
                        {"role": "user", "content": "What is Python?"},
                        {"role": "assistant", "content": "Python is a programming language"}
                    ],
                    "search_mode": "basic"
                },
                headers={"X-API-Key": "test-api-key"}
            )
            
            assert response.status_code == 200
            data = response.json()
            assert "response" in data


class TestStaticFiles:
    """Test static file serving."""
    
    def test_static_file_access(self, api_client):
        """Test accessing static files."""
        client, _ = api_client
        
        # Test accessing the main HTML file
        response = client.get("/static/index_enhanced.html")
        
        # Should either return the file or 404 if it doesn't exist
        assert response.status_code in [200, 404]
        
        if response.status_code == 200:
            assert "html" in response.text.lower()
    
    def test_root_redirect(self, api_client):
        """Test root path redirect."""
        client, _ = api_client
        
        response = client.get("/", allow_redirects=False)
        
        # Should redirect to static content
        assert response.status_code in [200, 302, 307]


class TestErrorHandling:
    """Test API error handling."""
    
    def test_404_for_nonexistent_endpoint(self, api_client):
        """Test 404 response for non-existent endpoints."""
        client, _ = api_client
        
        response = client.get("/nonexistent/endpoint")
        assert response.status_code == 404
    
    def test_method_not_allowed(self, api_client):
        """Test 405 for wrong HTTP methods."""
        client, _ = api_client
        
        # Try GET on POST endpoint
        response = client.get("/search")
        assert response.status_code == 405
    
    def test_large_payload_handling(self, api_client):
        """Test handling of large request payload."""
        client, _ = api_client
        
        # Create a large query
        large_query = "x" * 10000
        
        response = client.post(
            "/search",
            json={"query": large_query, "k": 5},
            headers={"X-API-Key": "test-api-key"}
        )
        
        # Should handle gracefully (either process or reject cleanly)
        assert response.status_code in [200, 400, 413, 422]
    
    def test_invalid_json_handling(self, api_client):
        """Test handling of invalid JSON in requests."""
        client, _ = api_client
        
        response = client.post(
            "/search",
            data="invalid json{",
            headers={
                "X-API-Key": "test-api-key",
                "Content-Type": "application/json"
            }
        )
        
        assert response.status_code == 422


class TestConcurrency:
    """Test API concurrency handling."""
    
    def test_concurrent_search_requests(self, api_client):
        """Test handling concurrent search requests."""
        import threading
        import time
        
        client, _ = api_client
        results = []
        
        def make_request():
            response = client.post(
                "/search",
                json={"query": f"test query {threading.current_thread().ident}", "k": 2},
                headers={"X-API-Key": "test-api-key"}
            )
            results.append(response.status_code)
        
        # Create multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=make_request)
            threads.append(thread)
        
        # Start all threads
        start_time = time.time()
        for thread in threads:
            thread.start()
        
        # Wait for all to complete
        for thread in threads:
            thread.join()
        
        end_time = time.time()
        
        # All requests should succeed
        assert all(status == 200 for status in results)
        assert len(results) == 5
        
        # Should complete reasonably quickly (not serialized)
        assert end_time - start_time < 10  # Generous timeout


if __name__ == "__main__":
    pytest.main([__file__, "-v"])