"""
Unit tests for chat functionality core components
"""
import pytest
import torch
from unittest.mock import Mock, patch, MagicMock
from dataclasses import dataclass
import json
import numpy as np

from app.chat_agent import ChatConfig, ChatAgent
from app.rag_pipeline import RAGPipeline


class TestChatConfig:
    """Test ChatConfig dataclass functionality"""
    
    def test_default_config(self):
        """Test default configuration values"""
        config = ChatConfig()
        
        assert config.model_name == "EleutherAI/gpt-j-6B"
        assert config.max_context_length == 2048
        assert config.max_new_tokens == 512
        assert config.temperature == 0.7
        assert config.top_p == 0.9
        assert config.top_k == 50
        assert config.repetition_penalty == 1.1
        assert config.use_8bit == True
        
    def test_custom_config(self):
        """Test custom configuration values"""
        config = ChatConfig(
            model_name="distilgpt2",
            max_context_length=1024,
            max_new_tokens=256,
            temperature=0.5,
            use_8bit=False
        )
        
        assert config.model_name == "distilgpt2"
        assert config.max_context_length == 1024
        assert config.max_new_tokens == 256
        assert config.temperature == 0.5
        assert config.use_8bit == False
        
    def test_device_detection(self):
        """Test device detection logic"""
        # Test explicit device setting
        config_cuda = ChatConfig(device="cuda")
        assert config_cuda.device == "cuda"
        
        config_cpu = ChatConfig(device="cpu")
        assert config_cpu.device == "cpu"
        
        config_mps = ChatConfig(device="mps")
        assert config_mps.device == "mps"
            
    def test_config_validation_ranges(self):
        """Test configuration parameter ranges"""
        # Valid ranges
        config = ChatConfig(
            temperature=0.0,
            top_p=0.1,
            top_k=1,
            repetition_penalty=1.0
        )
        assert config.temperature == 0.0
        assert config.top_p == 0.1
        assert config.top_k == 1
        assert config.repetition_penalty == 1.0


class TestChatAgentUnit:
    """Unit tests for ChatAgent without model loading"""
    
    @patch('app.chat_agent.ChatAgent._load_model')
    def test_init_with_default_config(self, mock_load_model):
        """Test ChatAgent initialization with default config"""
        agent = ChatAgent()
        
        assert isinstance(agent.config, ChatConfig)
        assert agent.config.model_name == "EleutherAI/gpt-j-6B"
        assert agent.conversation_history == {}
        mock_load_model.assert_called_once()
        
    @patch('app.chat_agent.ChatAgent._load_model')
    def test_init_with_custom_config(self, mock_load_model):
        """Test ChatAgent initialization with custom config"""
        config = ChatConfig(model_name="distilgpt2", temperature=0.5)
        agent = ChatAgent(config)
        
        assert agent.config == config
        assert agent.config.model_name == "distilgpt2"
        assert agent.config.temperature == 0.5
        mock_load_model.assert_called_once()
        
    @patch('app.chat_agent.ChatAgent._load_model')
    def test_conversation_history_management(self, mock_load_model):
        """Test conversation history operations"""
        agent = ChatAgent()
        session_id = "test_session"
        
        # Test empty history
        assert agent.get_history(session_id) == []
        
        # Test adding to history
        agent.conversation_history[session_id] = [
            {"user": "Hello", "assistant": "Hi there!"},
            {"user": "How are you?", "assistant": "I'm doing well!"}
        ]
        
        history = agent.get_history(session_id)
        assert len(history) == 2
        assert history[0]["user"] == "Hello"
        assert history[1]["assistant"] == "I'm doing well!"
        
        # Test clearing history
        agent.clear_history(session_id)
        assert agent.get_history(session_id) == []
        
    @patch('app.chat_agent.ChatAgent._load_model')
    def test_build_prompt_no_context(self, mock_load_model):
        """Test prompt building without context or history"""
        agent = ChatAgent()
        
        prompt = agent._build_prompt("What is AI?", [])
        
        assert "What is AI?" in prompt
        assert "Assistant:" in prompt
        assert "You are an AI assistant" in prompt
        assert "Relevant Context" not in prompt
        
    @patch('app.chat_agent.ChatAgent._load_model')
    def test_build_prompt_with_context(self, mock_load_model):
        """Test prompt building with context"""
        agent = ChatAgent()
        context = [
            {"source": "Conversation 1", "preview": "AI is artificial intelligence"},
            {"source": "Conversation 2", "preview": "Machine learning is a subset of AI"}
        ]
        
        prompt = agent._build_prompt("What is AI?", context)
        
        assert "What is AI?" in prompt
        assert "Relevant Context" in prompt
        assert "Conversation 1" in prompt
        assert "artificial intelligence" in prompt
        assert "Machine learning" in prompt
        
    @patch('app.chat_agent.ChatAgent._load_model')
    def test_build_prompt_with_history(self, mock_load_model):
        """Test prompt building with conversation history"""
        agent = ChatAgent()
        session_id = "test_session"
        agent.conversation_history[session_id] = [
            {"user": "Hello", "assistant": "Hi!"},
            {"user": "What's 2+2?", "assistant": "2+2 equals 4."}
        ]
        
        prompt = agent._build_prompt("What's 3+3?", [], session_id)
        
        assert "Recent Conversation" in prompt
        assert "Hello" in prompt
        assert "What's 2+2?" in prompt
        assert "What's 3+3?" in prompt
        
    @patch('app.chat_agent.ChatAgent._load_model')
    def test_build_analysis_prompt(self, mock_load_model):
        """Test analysis prompt building"""
        agent = ChatAgent()
        chunks = [
            {"preview": "Discussion about machine learning algorithms"},
            {"preview": "Conversation about neural networks"}
        ]
        
        prompt = agent._build_analysis_prompt(chunks)
        
        assert "Analyze the following" in prompt
        assert "JSON object" in prompt
        assert "machine learning" in prompt
        assert "neural networks" in prompt
        
    @patch('app.chat_agent.ChatAgent._load_model')
    def test_build_suggestion_prompt(self, mock_load_model):
        """Test suggestion prompt building"""
        agent = ChatAgent()
        results = [
            {"preview": "AI is transforming industries"},
            {"preview": "Machine learning requires data"}
        ]
        
        prompt = agent._build_suggestion_prompt("What is AI?", results)
        
        assert "What is AI?" in prompt
        assert "follow-up questions" in prompt
        assert "transforming industries" in prompt
        assert "requires data" in prompt


class TestChatAgentMocked:
    """Tests for ChatAgent with mocked model operations"""
    
    @patch('app.chat_agent.AutoModelForCausalLM.from_pretrained')
    @patch('app.chat_agent.AutoTokenizer.from_pretrained')
    def test_model_loading_success(self, mock_tokenizer, mock_model):
        """Test successful model loading"""
        # Setup mocks
        mock_tokenizer_instance = Mock()
        mock_tokenizer_instance.pad_token = None
        mock_tokenizer_instance.eos_token = "<|endoftext|>"
        mock_tokenizer.return_value = mock_tokenizer_instance
        
        mock_model_instance = Mock()
        mock_model.return_value = mock_model_instance
        
        config = ChatConfig(model_name="distilgpt2", use_8bit=False)
        agent = ChatAgent(config)
        
        # Verify model loading was called
        mock_tokenizer.assert_called_once_with("distilgpt2")
        mock_model.assert_called_once()
        assert agent.tokenizer == mock_tokenizer_instance
        assert agent.model == mock_model_instance
        
    @patch('app.chat_agent.AutoModelForCausalLM.from_pretrained')
    @patch('app.chat_agent.AutoTokenizer.from_pretrained')
    def test_model_loading_with_quantization(self, mock_tokenizer, mock_model):
        """Test model loading with 8-bit quantization"""
        mock_tokenizer_instance = Mock()
        mock_tokenizer.return_value = mock_tokenizer_instance
        mock_model_instance = Mock()
        mock_model.return_value = mock_model_instance
        
        with patch('torch.cuda.is_available', return_value=True):
            config = ChatConfig(model_name="distilgpt2", use_8bit=True, device="cuda")
            agent = ChatAgent(config)
            
        # Check that quantization config was used
        call_kwargs = mock_model.call_args[1]
        assert 'quantization_config' in call_kwargs
        assert 'device_map' in call_kwargs
        
    @patch('app.chat_agent.ChatAgent._load_model')
    def test_generate_response_mock(self, mock_load_model):
        """Test response generation with mocked model"""
        agent = ChatAgent()
        
        # Mock tokenizer and model
        mock_tokenizer = Mock()
        mock_tokenizer.return_value = {"input_ids": torch.tensor([[1, 2, 3]])}
        mock_tokenizer.decode.return_value = "This is a test response"
        mock_tokenizer.pad_token_id = 0
        mock_tokenizer.eos_token_id = 1
        agent.tokenizer = mock_tokenizer
        
        mock_model = Mock()
        mock_outputs = Mock()
        mock_outputs.shape = torch.Size([1, 10])
        mock_model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5, 6, 7]])
        agent.model = mock_model
        
        response = agent.generate_response("Test query", [], stream=False)
        
        assert response == "This is a test response"
        mock_model.generate.assert_called_once()
        
    @patch('app.chat_agent.ChatAgent._load_model')
    def test_suggest_questions_parsing(self, mock_load_model):
        """Test question suggestion parsing"""
        agent = ChatAgent()
        
        # Mock tokenizer and model
        mock_tokenizer = Mock()
        mock_tokenizer.return_value = {"input_ids": torch.tensor([[1, 2, 3]])}
        mock_tokenizer.decode.return_value = """
        1. What are the main applications of machine learning?
        2. How does deep learning differ from traditional ML?
        - What are the ethical implications of AI?
        • Which programming languages are best for AI?
        """
        agent.tokenizer = mock_tokenizer
        
        mock_model = Mock()
        mock_model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5]])
        agent.model = mock_model
        
        suggestions = agent.suggest_questions("What is AI?", [])
        
        assert len(suggestions) >= 3
        assert "What are the main applications of machine learning?" in suggestions
        assert "How does deep learning differ from traditional ML?" in suggestions
        assert "What are the ethical implications of AI?" in suggestions
            
    @patch('app.chat_agent.ChatAgent._load_model')
    def test_analyze_topics_json_parsing(self, mock_load_model):
        """Test topic analysis JSON parsing"""
        agent = ChatAgent()
        
        # Mock tokenizer and model
        mock_tokenizer = Mock()
        mock_tokenizer.return_value = {"input_ids": torch.tensor([[1, 2, 3]])}
        agent.tokenizer = mock_tokenizer
        
        mock_model = Mock()
        mock_model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5]])
        agent.model = mock_model
        
        valid_json = '{"topics": ["AI", "Machine Learning"], "summary": "Discussion about AI"}'
        invalid_json = "This is not valid JSON"
        
        # Test valid JSON parsing
        mock_tokenizer.decode.return_value = valid_json
        result = agent.analyze_topics([{"preview": "test"}])
        
        assert "topics" in result
        assert "summary" in result
        assert result["topics"] == ["AI", "Machine Learning"]
        
        # Test invalid JSON handling
        mock_tokenizer.decode.return_value = invalid_json
        result = agent.analyze_topics([{"preview": "test"}])
        
        assert "error" in result
        assert result["summary"] == invalid_json


class TestRAGPipelineUnit:
    """Unit tests for RAGPipeline components"""
    
    @patch('app.rag_pipeline.ChatAgent')
    @patch('app.rag_pipeline.RAGPipeline._load_stores')
    def test_init(self, mock_load_stores, mock_chat_agent):
        """Test RAGPipeline initialization"""
        pipeline = RAGPipeline("test.db", "test-model")
        
        assert pipeline.db_path == "test.db"
        assert pipeline.embed_model == "test-model"
        mock_chat_agent.assert_called_once()
        mock_load_stores.assert_called_once()
        
    def test_create_preview(self):
        """Test preview text creation"""
        with patch('app.rag_pipeline.ChatAgent'), \
             patch('app.rag_pipeline.RAGPipeline._load_stores'):
            pipeline = RAGPipeline("test.db", "test-model")
            
            # Test short text
            short_text = "This is short"
            preview = pipeline._create_preview(short_text)
            assert preview == short_text
            
            # Test long text
            long_text = "This is a very long text " * 20  # > 320 chars
            preview = pipeline._create_preview(long_text)
            assert len(preview) <= 321  # 320 + "…"
            assert preview.endswith("…")
            
    def test_enrich_results(self):
        """Test result enrichment with context"""
        with patch('app.rag_pipeline.ChatAgent'), \
             patch('app.rag_pipeline.RAGPipeline._load_stores'):
            pipeline = RAGPipeline("test.db", "test-model")
            
            results = [
                {
                    "source": "Test Doc",
                    "doc_id": 1,
                    "chunk_id": 10,
                    "start_char": 0,
                    "end_char": 100,
                    "text": "Test content"
                }
            ]
            
            enriched = pipeline._enrich_results(results)
            
            assert len(enriched) == 1
            result = enriched[0]
            assert "context_json" in result
            assert "context" in result["context_json"]
            
            context = result["context_json"]["context"][0]
            assert context["doc"] == "Test Doc"
            assert context["loc"]["doc_id"] == 1
            assert context["quote"] == "Test content"


if __name__ == "__main__":
    pytest.main([__file__])