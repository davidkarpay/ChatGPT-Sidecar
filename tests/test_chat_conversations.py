"""
Tests for conversation management and session handling
"""
import pytest
import uuid
from unittest.mock import patch, Mock
from fastapi.testclient import TestClient

from app.main import app
from app.chat_agent import ChatAgent, ChatConfig
from app.rag_pipeline import RAGPipeline
from tests.fixtures.mock_agents import MockChatAgent, MockRAGPipeline
from tests.fixtures.mock_data import MockSessionManager


class TestConversationHistory:
    """Test conversation history management"""
    
    @pytest.fixture
    def mock_agent(self):
        return MockChatAgent()
    
    def test_empty_session_history(self, mock_agent):
        """Test getting history for new session"""
        session_id = "new-session"
        history = mock_agent.get_history(session_id)
        
        assert history == []
    
    def test_add_single_exchange(self, mock_agent):
        """Test adding single conversation exchange"""
        session_id = "test-session"
        
        # Generate response (should add to history)
        response = mock_agent.generate_response(
            "Hello", [], session_id=session_id, stream=False
        )
        
        assert isinstance(response, str)
        
        history = mock_agent.get_history(session_id)
        assert len(history) == 1
        assert history[0]["user"] == "Hello"
        assert history[0]["assistant"] == response
    
    def test_multiple_exchanges(self, mock_agent):
        """Test multiple conversation exchanges"""
        session_id = "multi-turn-session"
        
        exchanges = [
            "What is machine learning?",
            "How does it differ from traditional programming?",
            "What are some practical applications?"
        ]
        
        for query in exchanges:
            mock_agent.generate_response(query, [], session_id=session_id)
        
        history = mock_agent.get_history(session_id)
        assert len(history) == 3
        
        for i, exchange in enumerate(exchanges):
            assert history[i]["user"] == exchange
            assert isinstance(history[i]["assistant"], str)
    
    def test_clear_history(self, mock_agent):
        """Test clearing conversation history"""
        session_id = "clear-test-session"
        
        # Add some conversation
        mock_agent.generate_response("Test message", [], session_id=session_id)
        assert len(mock_agent.get_history(session_id)) == 1
        
        # Clear history
        mock_agent.clear_history(session_id)
        assert mock_agent.get_history(session_id) == []
    
    def test_isolated_sessions(self, mock_agent):
        """Test that different sessions are isolated"""
        session1 = "session-1"
        session2 = "session-2"
        
        mock_agent.generate_response("Message from session 1", [], session_id=session1)
        mock_agent.generate_response("Message from session 2", [], session_id=session2)
        
        history1 = mock_agent.get_history(session1)
        history2 = mock_agent.get_history(session2)
        
        assert len(history1) == 1
        assert len(history2) == 1
        assert history1[0]["user"] != history2[0]["user"]
    
    def test_history_with_streaming(self, mock_agent):
        """Test that streaming responses are added to history"""
        session_id = "streaming-session"
        
        # Generate streaming response
        generator = mock_agent.generate_response(
            "Stream this", [], session_id=session_id, stream=True
        )
        
        # Consume the generator
        response_parts = list(generator)
        
        # Check history was updated
        history = mock_agent.get_history(session_id)
        assert len(history) == 1
        assert history[0]["user"] == "Stream this"
        # Response should be the complete text, not the individual tokens
        assert isinstance(history[0]["assistant"], str)


class TestSessionManagement:
    """Test session management functionality"""
    
    @pytest.fixture
    def client(self):
        return TestClient(app)
    
    @pytest.fixture
    def auth_headers(self):
        return {"X-API-Key": "change-me"}
    
    @pytest.fixture
    def mock_rag_pipeline(self):
        return MockRAGPipeline("test.db", "test-model")
    
    def test_session_id_generation(self, client, auth_headers, mock_rag_pipeline):
        """Test automatic session ID generation"""
        with patch('app.main.get_rag_pipeline', return_value=mock_rag_pipeline):
            # Request without session_id
            response = client.post(
                "/chat",
                json={"query": "Test message"},
                headers=auth_headers
            )
            
            assert response.status_code == 200
            data = response.json()
            
            # Should have generated a session_id
            assert "session_id" in data
            assert isinstance(data["session_id"], str)
            assert len(data["session_id"]) > 0
    
    def test_session_id_persistence(self, client, auth_headers, mock_rag_pipeline):
        """Test that provided session ID is preserved"""
        with patch('app.main.get_rag_pipeline', return_value=mock_rag_pipeline):
            session_id = "custom-session-123"
            
            response = client.post(
                "/chat",
                json={"query": "Test message", "session_id": session_id},
                headers=auth_headers
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data["session_id"] == session_id
    
    def test_get_conversation_history_endpoint(self, client, auth_headers, mock_rag_pipeline):
        """Test getting conversation history via API"""
        with patch('app.main.get_rag_pipeline', return_value=mock_rag_pipeline):
            session_id = "history-test-session"
            
            # Add some conversation history
            mock_rag_pipeline.chat_agent.conversation_history[session_id] = [
                {"user": "Hello", "assistant": "Hi there!"},
                {"user": "How are you?", "assistant": "I'm doing well!"}
            ]
            
            response = client.get(
                f"/chat/history/{session_id}",
                headers=auth_headers
            )
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["session_id"] == session_id
            assert len(data["history"]) == 2
            assert data["history"][0]["user"] == "Hello"
            assert data["history"][1]["assistant"] == "I'm doing well!"
    
    def test_clear_conversation_history_endpoint(self, client, auth_headers, mock_rag_pipeline):
        """Test clearing conversation history via API"""
        with patch('app.main.get_rag_pipeline', return_value=mock_rag_pipeline):
            session_id = "clear-test-session"
            
            # Add some conversation history
            mock_rag_pipeline.chat_agent.conversation_history[session_id] = [
                {"user": "Test", "assistant": "Response"}
            ]
            
            # Clear history
            response = client.delete(
                f"/chat/history/{session_id}",
                headers=auth_headers
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data["session_id"] == session_id
            assert "cleared" in data["message"].lower()
            
            # Verify history was actually cleared
            assert mock_rag_pipeline.chat_agent.get_history(session_id) == []
    
    def test_nonexistent_session_history(self, client, auth_headers, mock_rag_pipeline):
        """Test getting history for nonexistent session"""
        with patch('app.main.get_rag_pipeline', return_value=mock_rag_pipeline):
            session_id = "nonexistent-session"
            
            response = client.get(
                f"/chat/history/{session_id}",
                headers=auth_headers
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data["session_id"] == session_id
            assert data["history"] == []


class TestContextBuilding:
    """Test conversation context building for prompts"""
    
    @pytest.fixture
    def mock_agent(self):
        with patch('app.chat_agent.ChatAgent._load_model'):
            return ChatAgent()
    
    def test_prompt_building_no_context_no_history(self, mock_agent):
        """Test prompt building with no context or history"""
        prompt = mock_agent._build_prompt("What is AI?", [])
        
        assert "What is AI?" in prompt
        assert "You are an AI assistant" in prompt
        assert "Assistant:" in prompt
        assert "Relevant Context" not in prompt
        assert "Recent Conversation" not in prompt
    
    def test_prompt_building_with_context_only(self, mock_agent):
        """Test prompt building with context but no history"""
        context = [
            {"source": "AI Basics", "preview": "AI is artificial intelligence"},
            {"source": "ML Guide", "preview": "Machine learning is a subset of AI"}
        ]
        
        prompt = mock_agent._build_prompt("Explain AI", context)
        
        assert "Explain AI" in prompt
        assert "Relevant Context" in prompt
        assert "AI Basics" in prompt
        assert "artificial intelligence" in prompt
        assert "Machine learning" in prompt
        assert "Recent Conversation" not in prompt
    
    def test_prompt_building_with_history_only(self, mock_agent):
        """Test prompt building with history but no context"""
        session_id = "history-test"
        mock_agent.conversation_history[session_id] = [
            {"user": "Hello", "assistant": "Hi!"},
            {"user": "What's 2+2?", "assistant": "4"}
        ]
        
        prompt = mock_agent._build_prompt("What's 3+3?", [], session_id)
        
        assert "What's 3+3?" in prompt
        assert "Recent Conversation" in prompt
        assert "Hello" in prompt
        assert "What's 2+2?" in prompt
        assert "Relevant Context" not in prompt
    
    def test_prompt_building_with_context_and_history(self, mock_agent):
        """Test prompt building with both context and history"""
        session_id = "full-test"
        mock_agent.conversation_history[session_id] = [
            {"user": "Tell me about AI", "assistant": "AI is fascinating"}
        ]
        
        context = [
            {"source": "AI Research", "preview": "Recent advances in AI"}
        ]
        
        prompt = mock_agent._build_prompt("What are the latest developments?", context, session_id)
        
        assert "What are the latest developments?" in prompt
        assert "Relevant Context" in prompt
        assert "Recent Conversation" in prompt
        assert "AI Research" in prompt
        assert "Tell me about AI" in prompt
    
    def test_prompt_building_history_limit(self, mock_agent):
        """Test that prompt building limits conversation history"""
        session_id = "long-history"
        
        # Add many conversation turns
        long_history = []
        for i in range(10):
            long_history.append({
                "user": f"User message {i}",
                "assistant": f"Assistant response {i}"
            })
        
        mock_agent.conversation_history[session_id] = long_history
        
        prompt = mock_agent._build_prompt("Current question", [], session_id)
        
        # Should only include recent history (last 3 turns based on implementation)
        assert "User message 9" in prompt  # Most recent
        assert "User message 8" in prompt  # Second most recent
        assert "User message 7" in prompt  # Third most recent
        assert "User message 0" not in prompt  # Oldest should be excluded
    
    def test_prompt_building_context_limit(self, mock_agent):
        """Test that prompt building limits context items"""
        # Create many context items
        large_context = []
        for i in range(10):
            large_context.append({
                "source": f"Document {i}",
                "preview": f"Content from document {i}"
            })
        
        prompt = mock_agent._build_prompt("Test question", large_context)
        
        # Should only include first 5 context items (based on implementation)
        assert "Document 0" in prompt
        assert "Document 4" in prompt
        assert "Document 5" not in prompt  # Should be excluded


class TestConversationFlow:
    """Test end-to-end conversation flows"""
    
    @pytest.fixture
    def client(self):
        return TestClient(app)
    
    @pytest.fixture
    def auth_headers(self):
        return {"X-API-Key": "change-me"}
    
    def test_multi_turn_conversation(self, client, auth_headers):
        """Test a multi-turn conversation flow"""
        with patch('app.main.get_rag_pipeline') as mock_get_pipeline:
            mock_pipeline = MockRAGPipeline("test.db", "test-model")
            mock_get_pipeline.return_value = mock_pipeline
            
            session_id = str(uuid.uuid4())
            
            # First turn
            response1 = client.post(
                "/chat",
                json={
                    "query": "What is machine learning?",
                    "session_id": session_id
                },
                headers=auth_headers
            )
            
            assert response1.status_code == 200
            data1 = response1.json()
            assert data1["session_id"] == session_id
            
            # Second turn (should have access to first turn's context)
            response2 = client.post(
                "/chat",
                json={
                    "query": "Can you give me an example?",
                    "session_id": session_id
                },
                headers=auth_headers
            )
            
            assert response2.status_code == 200
            data2 = response2.json()
            assert data2["session_id"] == session_id
            
            # Check conversation history
            history_response = client.get(
                f"/chat/history/{session_id}",
                headers=auth_headers
            )
            
            assert history_response.status_code == 200
            history_data = history_response.json()
            assert len(history_data["history"]) == 2
    
    def test_conversation_with_topic_analysis(self, client, auth_headers):
        """Test conversation combined with topic analysis"""
        with patch('app.main.get_rag_pipeline') as mock_get_pipeline:
            mock_pipeline = MockRAGPipeline("test.db", "test-model")
            mock_get_pipeline.return_value = mock_pipeline
            
            # Have a conversation about ML
            session_id = str(uuid.uuid4())
            
            client.post(
                "/chat",
                json={
                    "query": "Tell me about neural networks",
                    "session_id": session_id
                },
                headers=auth_headers
            )
            
            # Analyze topics in related documents
            analysis_response = client.post(
                "/analyze",
                json={"doc_ids": [1, 2, 3], "limit": 50},
                headers=auth_headers
            )
            
            assert analysis_response.status_code == 200
            analysis_data = analysis_response.json()
            assert "topics" in analysis_data
            assert "summary" in analysis_data
    
    def test_conversation_with_suggestions(self, client, auth_headers):
        """Test conversation with follow-up suggestions"""
        with patch('app.main.get_rag_pipeline') as mock_get_pipeline:
            mock_pipeline = MockRAGPipeline("test.db", "test-model")
            mock_get_pipeline.return_value = mock_pipeline
            
            response = client.post(
                "/chat",
                json={"query": "What is deep learning?"},
                headers=auth_headers
            )
            
            assert response.status_code == 200
            data = response.json()
            
            # Should include suggestions
            assert "suggestions" in data
            assert isinstance(data["suggestions"], list)
            
            if data["suggestions"]:  # If suggestions are provided
                # Test using a suggestion as the next query
                next_query = data["suggestions"][0]
                
                response2 = client.post(
                    "/chat",
                    json={
                        "query": next_query,
                        "session_id": data["session_id"]
                    },
                    headers=auth_headers
                )
                
                assert response2.status_code == 200
    
    def test_conversation_memory_consistency(self, client, auth_headers):
        """Test that conversation memory remains consistent"""
        with patch('app.main.get_rag_pipeline') as mock_get_pipeline:
            mock_pipeline = MockRAGPipeline("test.db", "test-model")
            mock_get_pipeline.return_value = mock_pipeline
            
            session_id = "memory-test-session"
            
            # First message
            response1 = client.post(
                "/chat",
                json={
                    "query": "My name is Alice",
                    "session_id": session_id
                },
                headers=auth_headers
            )
            assert response1.status_code == 200
            
            # Second message referring to previous context
            response2 = client.post(
                "/chat",
                json={
                    "query": "What did I just tell you?",
                    "session_id": session_id
                },
                headers=auth_headers
            )
            assert response2.status_code == 200
            
            # Check that both messages are in history
            history_response = client.get(
                f"/chat/history/{session_id}",
                headers=auth_headers
            )
            
            history_data = history_response.json()
            assert len(history_data["history"]) == 2
            assert "Alice" in history_data["history"][0]["user"]


class TestConversationSummarization:
    """Test conversation summarization functionality"""
    
    @pytest.fixture
    def client(self):
        return TestClient(app)
    
    @pytest.fixture
    def auth_headers(self):
        return {"X-API-Key": "change-me"}
    
    def test_summarize_conversation(self, client, auth_headers):
        """Test conversation summarization endpoint"""
        with patch('app.main.get_rag_pipeline') as mock_get_pipeline:
            mock_pipeline = MockRAGPipeline("test.db", "test-model")
            mock_get_pipeline.return_value = mock_pipeline
            
            doc_id = 42
            response = client.get(
                f"/summarize/{doc_id}",
                headers=auth_headers
            )
            
            assert response.status_code == 200
            data = response.json()
            
            assert "doc_id" in data
            assert "summary" in data
            assert data["doc_id"] == doc_id
            assert isinstance(data["summary"], str)
            assert len(data["summary"]) > 0
    
    def test_summarize_multiple_conversations(self, client, auth_headers):
        """Test summarizing multiple conversations"""
        with patch('app.main.get_rag_pipeline') as mock_get_pipeline:
            mock_pipeline = MockRAGPipeline("test.db", "test-model")
            mock_get_pipeline.return_value = mock_pipeline
            
            doc_ids = [1, 2, 3, 4, 5]
            summaries = []
            
            for doc_id in doc_ids:
                response = client.get(
                    f"/summarize/{doc_id}",
                    headers=auth_headers
                )
                
                assert response.status_code == 200
                data = response.json()
                summaries.append(data["summary"])
            
            # Each summary should be different (based on mock implementation)
            assert len(set(summaries)) > 1  # At least some variation


if __name__ == "__main__":
    pytest.main([__file__, "-v"])