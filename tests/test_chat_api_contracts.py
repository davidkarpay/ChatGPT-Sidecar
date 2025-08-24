"""
API contract tests for chat endpoints - validating request/response schemas
"""
import pytest
import uuid
from typing import Dict, Any
from unittest.mock import patch, Mock
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.main import app, ChatReq, ChatStreamReq, AnalyzeReq, SuggestionsReq, ConversationReq


class TestChatAPIContracts:
    """Test API contracts for request/response validation"""
    
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
    
    def test_chat_request_schema_validation(self):
        """Test ChatReq schema validation"""
        # Valid request
        valid_data = {
            "query": "What is machine learning?",
            "session_id": str(uuid.uuid4()),
            "k": 5,
            "search_mode": "adaptive"
        }
        req = ChatReq(**valid_data)
        assert req.query == valid_data["query"]
        assert req.k == 5
        assert req.search_mode == "adaptive"
        
        # Test defaults
        minimal_data = {"query": "test"}
        req_minimal = ChatReq(**minimal_data)
        assert req_minimal.query == "test"
        assert req_minimal.k == 5  # default
        assert req_minimal.search_mode == "adaptive"  # default
        assert isinstance(req_minimal.session_id, str)  # auto-generated
        
        # Invalid query (empty string)
        with pytest.raises(ValidationError) as exc_info:
            ChatReq(query="")
        assert "min_length" in str(exc_info.value)
        
        # Invalid k (out of range)
        with pytest.raises(ValidationError):
            ChatReq(query="test", k=0)
        
        with pytest.raises(ValidationError):
            ChatReq(query="test", k=25)
        
        # Invalid search mode
        with pytest.raises(ValidationError):
            ChatReq(query="test", search_mode="invalid_mode")
    
    def test_chat_stream_request_schema_validation(self):
        """Test ChatStreamReq schema validation"""
        # Valid request
        valid_data = {
            "query": "Stream this response",
            "session_id": "stream-session",
            "k": 10,
            "search_mode": "multi_layer"
        }
        req = ChatStreamReq(**valid_data)
        assert req.query == valid_data["query"]
        assert req.k == 10
        assert req.search_mode == "multi_layer"
        
        # Test schema is identical to ChatReq
        assert ChatStreamReq.__annotations__ == ChatReq.__annotations__
    
    def test_analyze_request_schema_validation(self):
        """Test AnalyzeReq schema validation"""
        # Valid request with doc_ids
        req = AnalyzeReq(doc_ids=[1, 2, 3], limit=50)
        assert req.doc_ids == [1, 2, 3]
        assert req.limit == 50
        
        # Valid request without doc_ids
        req_no_docs = AnalyzeReq(limit=100)
        assert req_no_docs.doc_ids is None
        assert req_no_docs.limit == 100
        
        # Test defaults
        req_defaults = AnalyzeReq()
        assert req_defaults.doc_ids is None
        assert req_defaults.limit == 100
        
        # Invalid limit (out of range)
        with pytest.raises(ValidationError):
            AnalyzeReq(limit=5)  # Below minimum of 10
        
        with pytest.raises(ValidationError):
            AnalyzeReq(limit=1000)  # Above maximum of 500
    
    def test_suggestions_request_schema_validation(self):
        """Test SuggestionsReq schema validation"""
        # Valid request
        results = [
            {"preview": "This is a preview", "source": "Document 1"},
            {"preview": "Another preview", "source": "Document 2"}
        ]
        req = SuggestionsReq(query="What is AI?", results=results)
        assert req.query == "What is AI?"
        assert req.results == results
        
        # Valid request with empty results
        req_empty = SuggestionsReq(query="Test query", results=[])
        assert req_empty.results == []
        
        # Test defaults
        req_defaults = SuggestionsReq(query="Default test")
        assert req_defaults.results == []
    
    def test_conversation_request_schema_validation(self):
        """Test ConversationReq schema validation"""
        # Valid request
        req = ConversationReq(session_id="test-session-123")
        assert req.session_id == "test-session-123"
        
        # Test various session ID formats
        valid_session_ids = [
            str(uuid.uuid4()),
            "simple-session",
            "session_with_underscores",
            "session-with-dashes",
            "123456789",
            "a" * 100  # Long session ID
        ]
        
        for session_id in valid_session_ids:
            req = ConversationReq(session_id=session_id)
            assert req.session_id == session_id
    
    def test_chat_response_schema(self, client, auth_headers, mock_rag_pipeline):
        """Test chat endpoint response schema"""
        expected_response = {
            "session_id": "test-session",
            "response": "This is a test response",
            "context": [
                {
                    "rank": 1,
                    "score": 0.95,
                    "source": "Test Document",
                    "preview": "This is a preview of the content",
                    "loc": {
                        "doc_id": 1,
                        "chunk_id": 10,
                        "start": 0,
                        "end": 100
                    },
                    "context_json": {
                        "context": [{
                            "doc": "Test Document",
                            "loc": {"doc_id": 1, "chunk_id": 10, "start": 0, "end": 100},
                            "quote": "Full text content"
                        }]
                    }
                }
            ],
            "query": "Test query",
            "suggestions": [
                "What else would you like to know?",
                "Can you elaborate on that topic?"
            ]
        }
        
        mock_rag_pipeline.chat.return_value = {
            "session_id": expected_response["session_id"],
            "response": expected_response["response"],
            "context": expected_response["context"],
            "query": expected_response["query"]
        }
        mock_rag_pipeline.suggest_follow_up_questions.return_value = expected_response["suggestions"]
        
        response = client.post(
            "/chat",
            json={"query": "Test query", "session_id": "test-session"},
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Validate response structure
        assert "session_id" in data
        assert "response" in data
        assert "context" in data
        assert "query" in data
        assert "suggestions" in data
        
        # Validate types
        assert isinstance(data["session_id"], str)
        assert isinstance(data["response"], str)
        assert isinstance(data["context"], list)
        assert isinstance(data["query"], str)
        assert isinstance(data["suggestions"], list)
        
        # Validate context structure
        if data["context"]:
            context_item = data["context"][0]
            required_context_fields = ["rank", "score", "source", "preview", "loc", "context_json"]
            for field in required_context_fields:
                assert field in context_item
    
    def test_analyze_response_schema(self, client, auth_headers, mock_rag_pipeline):
        """Test analyze endpoint response schema"""
        expected_response = {
            "topics": ["AI", "Machine Learning", "Deep Learning"],
            "summary": "This analysis covers various AI and ML topics",
            "additional_field": "Extra data"  # Test additional fields are preserved
        }
        
        mock_rag_pipeline.analyze_conversation_topics.return_value = expected_response
        
        response = client.post(
            "/analyze",
            json={"doc_ids": [1, 2, 3], "limit": 50},
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Should preserve the structure returned by the analysis
        assert data["topics"] == expected_response["topics"]
        assert data["summary"] == expected_response["summary"]
        assert data["additional_field"] == expected_response["additional_field"]
    
    def test_suggestions_response_schema(self, client, auth_headers, mock_rag_pipeline):
        """Test suggestions endpoint response schema"""
        expected_suggestions = [
            "What are the applications of machine learning?",
            "How does AI impact society?",
            "What are the latest developments in deep learning?"
        ]
        
        mock_rag_pipeline.suggest_follow_up_questions.return_value = expected_suggestions
        
        response = client.post(
            "/suggest",
            json={
                "query": "What is AI?",
                "results": [{"preview": "AI is artificial intelligence"}]
            },
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Validate response structure
        assert "suggestions" in data
        assert isinstance(data["suggestions"], list)
        assert data["suggestions"] == expected_suggestions
        
        # All suggestions should be strings
        for suggestion in data["suggestions"]:
            assert isinstance(suggestion, str)
            assert len(suggestion) > 0
    
    def test_chat_history_response_schema(self, client, auth_headers, mock_rag_pipeline):
        """Test chat history endpoint response schema"""
        expected_history = [
            {"user": "Hello", "assistant": "Hi there!"},
            {"user": "What is AI?", "assistant": "AI is artificial intelligence."}
        ]
        
        mock_rag_pipeline.get_conversation_history.return_value = expected_history
        
        session_id = "test-session-456"
        response = client.get(
            f"/chat/history/{session_id}",
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Validate response structure
        assert "session_id" in data
        assert "history" in data
        assert data["session_id"] == session_id
        assert isinstance(data["history"], list)
        
        # Validate history structure
        for turn in data["history"]:
            assert "user" in turn
            assert "assistant" in turn
            assert isinstance(turn["user"], str)
            assert isinstance(turn["assistant"], str)
    
    def test_summarize_response_schema(self, client, auth_headers, mock_rag_pipeline):
        """Test summarize endpoint response schema"""
        expected_summary = "This conversation discusses machine learning fundamentals and applications."
        
        mock_rag_pipeline.get_conversation_summary.return_value = expected_summary
        
        doc_id = 42
        response = client.get(
            f"/summarize/{doc_id}",
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Validate response structure
        assert "doc_id" in data
        assert "summary" in data
        assert data["doc_id"] == doc_id
        assert isinstance(data["summary"], str)
        assert data["summary"] == expected_summary
    
    def test_error_response_schemas(self, client, auth_headers):
        """Test error response schemas"""
        # Test validation error (422)
        response = client.post(
            "/chat",
            json={"query": ""},  # Empty query triggers validation error
            headers=auth_headers
        )
        
        assert response.status_code == 422
        error_data = response.json()
        
        # FastAPI validation error structure
        assert "detail" in error_data
        assert isinstance(error_data["detail"], list)
        
        error_detail = error_data["detail"][0]
        assert "loc" in error_detail
        assert "msg" in error_detail
        assert "type" in error_detail
        
        # Test authentication error (401)
        response = client.post(
            "/chat",
            json={"query": "test"},
            # No auth headers
        )
        
        assert response.status_code == 401
        error_data = response.json()
        assert "detail" in error_data
        assert error_data["detail"] == "Unauthorized"
    
    def test_streaming_response_format(self, client, auth_headers, mock_rag_pipeline):
        """Test streaming response format"""
        def mock_token_generator():
            tokens = ["Hello", " ", "world", "!"]
            for token in tokens:
                yield token
        
        mock_rag_pipeline.chat.return_value = {
            "session_id": "stream-test",
            "context": [{"source": "Test", "preview": "Test content"}],
            "response_generator": mock_token_generator()
        }
        mock_rag_pipeline.suggest_follow_up_questions.return_value = ["Follow-up?"]
        
        response = client.post(
            "/chat/stream",
            json={"query": "Test streaming"},
            headers=auth_headers
        )
        
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        
        # Parse SSE format
        content = response.content.decode()
        lines = content.strip().split('\n')
        
        sse_events = []
        for line in lines:
            if line.startswith('data: '):
                try:
                    import json
                    event_data = json.loads(line[6:])
                    sse_events.append(event_data)
                except json.JSONDecodeError:
                    pass
        
        # Should contain different event types
        event_types = [event.get("type") for event in sse_events]
        expected_types = ["context", "session_id", "token", "suggestions", "done"]
        
        # Check that we have the expected event types
        for expected_type in expected_types:
            if expected_type == "token":
                # Should have multiple token events
                assert event_types.count(expected_type) >= 1
            else:
                assert expected_type in event_types
    
    def test_content_type_validation(self, client, auth_headers):
        """Test content type validation"""
        # Test with correct content type
        response = client.post(
            "/chat",
            json={"query": "test"},
            headers=auth_headers
        )
        assert response.status_code in [200, 500]  # Should not fail due to content type
        
        # Test with missing content type (FastAPI should handle this)
        response = client.post(
            "/chat",
            content='{"query": "test"}',
            headers={**auth_headers, "Content-Type": "text/plain"}
        )
        # FastAPI typically handles this gracefully or returns 415/422
        assert response.status_code in [415, 422, 400]
    
    def test_parameter_type_coercion(self, client, auth_headers, mock_rag_pipeline):
        """Test parameter type coercion and validation"""
        mock_rag_pipeline.chat.return_value = {
            "session_id": "type-test",
            "response": "Type coercion test",
            "context": [],
            "query": "test"
        }
        mock_rag_pipeline.suggest_follow_up_questions.return_value = []
        
        # Test string k value (should be coerced to int)
        response = client.post(
            "/chat",
            json={"query": "test", "k": "5"},
            headers=auth_headers
        )
        assert response.status_code == 200
        
        # Test invalid k value (non-numeric string)
        response = client.post(
            "/chat",
            json={"query": "test", "k": "invalid"},
            headers=auth_headers
        )
        assert response.status_code == 422


if __name__ == "__main__":
    pytest.main([__file__, "-v"])