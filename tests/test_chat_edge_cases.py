"""
Edge case tests for chat functionality
"""
import pytest
import json
import uuid
from unittest.mock import patch, Mock, MagicMock
from fastapi.testclient import TestClient

from app.main import app
from app.chat_agent import ChatConfig, ChatAgent


class TestChatEdgeCases:
    """Test edge cases and failure scenarios"""
    
    @pytest.fixture
    def client(self):
        return TestClient(app)
    
    @pytest.fixture
    def auth_headers(self):
        return {"X-API-Key": "change-me"}
    
    @pytest.fixture
    def mock_rag_pipeline(self):
        with patch('app.main.get_rag_pipeline') as mock_get_pipeline:
            mock_pipeline = Mock()
            mock_get_pipeline.return_value = mock_pipeline
            yield mock_pipeline
    
    def test_extremely_long_query(self, client, auth_headers, mock_rag_pipeline):
        """Test handling of extremely long queries"""
        # Create a very long query (10KB)
        extremely_long_query = "Explain machine learning " * 1000
        
        mock_rag_pipeline.chat.return_value = {
            "session_id": "long-query-test",
            "response": "Handled long query successfully",
            "context": [],
            "query": extremely_long_query
        }
        mock_rag_pipeline.suggest_follow_up_questions.return_value = []
        
        response = client.post(
            "/chat",
            json={"query": extremely_long_query},
            headers=auth_headers
        )
        
        # Should either succeed or fail gracefully
        assert response.status_code in [200, 413, 422]  # OK, Request Entity Too Large, or Validation Error
        
        if response.status_code == 200:
            data = response.json()
            assert "response" in data
    
    def test_unicode_and_special_characters(self, client, auth_headers, mock_rag_pipeline):
        """Test handling of unicode and special characters"""
        unicode_queries = [
            "你好，机器学习是什么？",  # Chinese
            "¿Qué es el aprendizaje automático?",  # Spanish with accents
            "🤖🔥💻 What is AI? 🚀✨",  # Emojis
            "SELECT * FROM users; DROP TABLE users;",  # SQL injection attempt
            "<script>alert('xss')</script>",  # XSS attempt
            "Machine learning\n\n\r\t with weird\x00 characters",  # Control characters
            "C:\\Windows\\System32\\cmd.exe",  # Windows path
            "../../../etc/passwd",  # Path traversal
            "{'json': 'injection', 'attempt': true}",  # JSON injection
            "A" * 1000,  # Very long string
        ]
        
        for query in unicode_queries:
            mock_rag_pipeline.chat.return_value = {
                "session_id": "unicode-test",
                "response": f"Processed unicode query: {query[:50]}...",
                "context": [],
                "query": query
            }
            mock_rag_pipeline.suggest_follow_up_questions.return_value = []
            
            response = client.post(
                "/chat",
                json={"query": query},
                headers=auth_headers
            )
            
            # Should handle gracefully without crashing
            assert response.status_code in [200, 400, 422]
            
            if response.status_code == 200:
                data = response.json()
                assert "response" in data
                # Ensure no code injection occurred
                assert "<script>" not in data["response"]
    
    def test_malformed_json_requests(self, client, auth_headers):
        """Test handling of malformed JSON requests"""
        malformed_requests = [
            '{"query": "test"',  # Incomplete JSON
            '{"query": "test",}',  # Trailing comma
            '{"query": }',  # Missing value
            '',  # Empty body
            'not json at all',  # Plain text
            '{"query": null}',  # Null query
            '{"query": []}',  # Array instead of string
            '{"query": {"nested": "object"}}',  # Object instead of string
        ]
        
        for malformed_json in malformed_requests:
            response = client.post(
                "/chat",
                content=malformed_json,
                headers={**auth_headers, "Content-Type": "application/json"}
            )
            
            # Should return 422 (Validation Error) or 400 (Bad Request)
            assert response.status_code in [400, 422]
    
    def test_missing_required_fields(self, client, auth_headers):
        """Test requests with missing required fields"""
        incomplete_requests = [
            {},  # Empty request
            {"session_id": "test"},  # Missing query
            {"k": 5},  # Missing query
            {"search_mode": "adaptive"},  # Missing query
        ]
        
        for request_data in incomplete_requests:
            response = client.post(
                "/chat",
                json=request_data,
                headers=auth_headers
            )
            
            assert response.status_code == 422  # Validation Error
            error_data = response.json()
            assert "detail" in error_data
    
    def test_invalid_authentication(self, client):
        """Test various authentication failure scenarios"""
        auth_scenarios = [
            {},  # No auth header
            {"X-API-Key": ""},  # Empty key
            {"X-API-Key": "wrong-key"},  # Wrong key
            {"Authorization": "Bearer token"},  # Wrong header format
            {"X-API-Key": None},  # None value (won't be sent)
            {"x-api-key": "change-me"},  # Wrong case
        ]
        
        for headers in auth_scenarios:
            response = client.post(
                "/chat",
                json={"query": "test"},
                headers=headers
            )
            
            assert response.status_code == 401
            assert response.json()["detail"] == "Unauthorized"
    
    def test_model_loading_failures(self, mock_rag_pipeline):
        """Test chat agent behavior when model loading fails"""
        with patch('app.chat_agent.AutoTokenizer.from_pretrained') as mock_tokenizer, \
             patch('app.chat_agent.AutoModelForCausalLM.from_pretrained') as mock_model:
            
            # Test tokenizer loading failure
            mock_tokenizer.side_effect = Exception("Tokenizer loading failed")
            
            with pytest.raises(Exception, match="Tokenizer loading failed"):
                ChatAgent(ChatConfig(model_name="nonexistent-model"))
            
            # Test model loading failure
            mock_tokenizer.side_effect = None
            mock_tokenizer.return_value = Mock()
            mock_model.side_effect = Exception("Model loading failed")
            
            with pytest.raises(Exception, match="Model loading failed"):
                ChatAgent(ChatConfig(model_name="nonexistent-model"))
    
    def test_generation_failures(self, mock_rag_pipeline):
        """Test handling of model generation failures"""
        with patch('app.chat_agent.ChatAgent._load_model'):
            agent = ChatAgent()
            
            # Mock tokenizer and model
            mock_tokenizer = Mock()
            mock_tokenizer.return_value = {"input_ids": torch.tensor([[1, 2, 3]])}
            mock_tokenizer.pad_token_id = 0
            mock_tokenizer.eos_token_id = 1
            agent.tokenizer = mock_tokenizer
            
            mock_model = Mock()
            mock_model.generate.side_effect = Exception("Generation failed")
            agent.model = mock_model
            
            # Should handle generation failure gracefully
            with pytest.raises(Exception, match="Generation failed"):
                agent.generate_response("test query", [])
    
    def test_empty_search_results(self, client, auth_headers, mock_rag_pipeline):
        """Test behavior when search returns no results"""
        # Mock empty search results
        mock_rag_pipeline.chat.return_value = {
            "session_id": "empty-results-test",
            "response": "I couldn't find any relevant information.",
            "context": [],  # Empty context
            "query": "obscure query with no results"
        }
        mock_rag_pipeline.suggest_follow_up_questions.return_value = []
        
        response = client.post(
            "/chat",
            json={"query": "xyz123nonexistentquery456"},
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["context"] == []
        assert "response" in data
    
    def test_extremely_large_context(self, client, auth_headers, mock_rag_pipeline):
        """Test handling of extremely large context results"""
        # Create very large context (simulating many search results)
        large_context = []
        for i in range(100):  # 100 context items
            large_context.append({
                "rank": i + 1,
                "score": 0.9 - (i * 0.001),
                "source": f"Document {i}",
                "preview": "Very long preview text " * 100,  # Large preview
                "text": "Extremely long full text " * 500,  # Very large text
                "doc_id": i,
                "chunk_id": i * 10
            })
        
        mock_rag_pipeline.chat.return_value = {
            "session_id": "large-context-test",
            "response": "Response with massive context",
            "context": large_context,
            "query": "query requiring lots of context"
        }
        mock_rag_pipeline.suggest_follow_up_questions.return_value = []
        
        response = client.post(
            "/chat",
            json={"query": "Find everything about machine learning", "k": 20},
            headers=auth_headers
        )
        
        # Should either succeed or fail gracefully due to size limits
        assert response.status_code in [200, 413, 500]
    
    def test_rapid_fire_requests(self, client, auth_headers, mock_rag_pipeline):
        """Test handling of rapid consecutive requests"""
        mock_rag_pipeline.chat.return_value = {
            "session_id": "rapid-fire",
            "response": "Rapid response",
            "context": [],
            "query": "rapid query"
        }
        mock_rag_pipeline.suggest_follow_up_questions.return_value = []
        
        # Send many requests in quick succession
        responses = []
        for i in range(20):
            response = client.post(
                "/chat",
                json={"query": f"Rapid query {i}"},
                headers=auth_headers
            )
            responses.append(response)
        
        # All requests should be handled (may be queued)
        success_count = sum(1 for r in responses if r.status_code == 200)
        error_count = sum(1 for r in responses if r.status_code >= 400)
        
        # Should handle most requests successfully
        assert success_count >= 15  # At least 75% success rate
        print(f"Rapid fire results: {success_count} successes, {error_count} errors")
    
    def test_session_id_edge_cases(self, client, auth_headers, mock_rag_pipeline):
        """Test edge cases with session IDs"""
        edge_case_sessions = [
            "",  # Empty string
            " ",  # Whitespace
            "a" * 1000,  # Very long session ID
            "session-with-unicode-你好",  # Unicode in session ID
            "session/with/slashes",  # Special characters
            "session with spaces",  # Spaces
            "session\nwith\nnewlines",  # Newlines
            str(uuid.uuid4()) * 10,  # Very long UUID-based ID
        ]
        
        for session_id in edge_case_sessions:
            mock_rag_pipeline.chat.return_value = {
                "session_id": session_id,
                "response": "Response for edge case session",
                "context": [],
                "query": "test"
            }
            mock_rag_pipeline.suggest_follow_up_questions.return_value = []
            
            response = client.post(
                "/chat",
                json={"query": "test", "session_id": session_id},
                headers=auth_headers
            )
            
            # Should handle gracefully
            assert response.status_code in [200, 400, 422]
    
    def test_invalid_parameter_ranges(self, client, auth_headers):
        """Test invalid parameter values"""
        invalid_requests = [
            {"query": "test", "k": 0},  # k too small
            {"query": "test", "k": 25},  # k too large
            {"query": "test", "k": -1},  # negative k
            {"query": "test", "k": "invalid"},  # non-numeric k
            {"query": "test", "search_mode": "invalid_mode"},  # invalid search mode
            {"query": "test", "search_mode": ""},  # empty search mode
            {"query": "test", "search_mode": 123},  # numeric search mode
        ]
        
        for request_data in invalid_requests:
            response = client.post(
                "/chat",
                json=request_data,
                headers=auth_headers
            )
            
            assert response.status_code == 422  # Validation error
    
    def test_rag_pipeline_initialization_failure(self, client, auth_headers):
        """Test behavior when RAG pipeline fails to initialize"""
        with patch('app.main.get_rag_pipeline') as mock_get_pipeline:
            mock_get_pipeline.side_effect = Exception("RAG pipeline initialization failed")
            
            response = client.post(
                "/chat",
                json={"query": "test"},
                headers=auth_headers
            )
            
            assert response.status_code == 500
            assert "error" in response.json()["detail"].lower()
    
    def test_database_connection_failure(self, client, auth_headers, mock_rag_pipeline):
        """Test behavior when database operations fail"""
        # Mock database failure in RAG pipeline
        mock_rag_pipeline.chat.side_effect = Exception("Database connection failed")
        
        response = client.post(
            "/chat",
            json={"query": "test"},
            headers=auth_headers
        )
        
        assert response.status_code == 500
        assert "Chat processing error" in response.json()["detail"]
    
    def test_streaming_edge_cases(self, client, auth_headers, mock_rag_pipeline):
        """Test edge cases in streaming responses"""
        # Test streaming with empty generator
        def empty_generator():
            return
            yield  # Unreachable, but makes it a generator
        
        mock_rag_pipeline.chat.return_value = {
            "session_id": "stream-edge-test",
            "context": [],
            "response_generator": empty_generator()
        }
        mock_rag_pipeline.suggest_follow_up_questions.return_value = []
        
        response = client.post(
            "/chat/stream",
            json={"query": "test streaming"},
            headers=auth_headers
        )
        
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        
        # Should handle empty stream gracefully
        content = response.content.decode()
        assert "data:" in content  # Should at least send context/session data
    
    def test_memory_exhaustion_simulation(self, client, auth_headers, mock_rag_pipeline):
        """Test behavior under simulated memory pressure"""
        def memory_intensive_response(*args, **kwargs):
            # Simulate memory-intensive operation
            large_data = "x" * 10000  # 10KB string
            context = [{"large_field": large_data} for _ in range(100)]  # ~1MB context
            
            return {
                "session_id": "memory-test",
                "response": large_data,
                "context": context,
                "query": kwargs.get("query", "test")
            }
        
        mock_rag_pipeline.chat.side_effect = memory_intensive_response
        mock_rag_pipeline.suggest_follow_up_questions.return_value = ["question"] * 100
        
        response = client.post(
            "/chat",
            json={"query": "memory intensive query"},
            headers=auth_headers
        )
        
        # Should either succeed or fail gracefully
        assert response.status_code in [200, 413, 500]  # OK, Too Large, or Internal Error


if __name__ == "__main__":
    import torch  # Import here to avoid issues if torch not available
    pytest.main([__file__, "-v"])