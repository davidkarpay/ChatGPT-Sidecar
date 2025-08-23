"""Tests for LLM provider abstraction system."""

import pytest
import os
from unittest.mock import Mock, patch, MagicMock

from app.llm_providers import (
    LLMConfig, ProviderType, LLMProviderFactory, 
    OpenAIProvider, AnthropicProvider, OllamaProvider, GPTJProvider
)


class TestLLMConfig:
    """Test LLM configuration."""
    
    def test_basic_config(self):
        """Test basic configuration creation."""
        config = LLMConfig(
            provider=ProviderType.OPENAI,
            model="gpt-3.5-turbo"
        )
        
        assert config.provider == ProviderType.OPENAI
        assert config.model == "gpt-3.5-turbo"
        assert config.max_tokens == 512  # default
        assert config.temperature == 0.7  # default


class TestLLMProviderFactory:
    """Test provider factory functionality."""
    
    def test_create_openai_provider(self):
        """Test OpenAI provider creation."""
        config = LLMConfig(
            provider=ProviderType.OPENAI,
            model="gpt-3.5-turbo",
            api_key="test-key"
        )
        
        with patch('app.llm_providers.openai'):
            provider = LLMProviderFactory.create_provider(config)
            assert isinstance(provider, OpenAIProvider)
            assert provider.model == "gpt-3.5-turbo"
    
    def test_create_anthropic_provider(self):
        """Test Anthropic provider creation."""
        config = LLMConfig(
            provider=ProviderType.ANTHROPIC,
            model="claude-3-sonnet-20240229",
            api_key="test-key"
        )
        
        with patch('app.llm_providers.anthropic'):
            provider = LLMProviderFactory.create_provider(config)
            assert isinstance(provider, AnthropicProvider)
            assert provider.model == "claude-3-sonnet-20240229"
    
    def test_create_ollama_provider(self):
        """Test Ollama provider creation."""
        config = LLMConfig(
            provider=ProviderType.OLLAMA,
            model="llama2"
        )
        
        provider = LLMProviderFactory.create_provider(config)
        assert isinstance(provider, OllamaProvider)
        assert provider.model == "llama2"
    
    def test_create_gptj_provider(self):
        """Test GPT-J provider creation."""
        config = LLMConfig(
            provider=ProviderType.GPTJ,
            model="EleutherAI/gpt-j-6B"
        )
        
        with patch('app.llm_providers.ChatAgent'):
            with patch('app.llm_providers.ChatConfig'):
                provider = LLMProviderFactory.create_provider(config)
                assert isinstance(provider, GPTJProvider)
                assert provider.model == "EleutherAI/gpt-j-6B"
    
    def test_from_env_openai(self):
        """Test creating provider from environment variables."""
        with patch.dict(os.environ, {
            "LLM_PROVIDER": "openai",
            "LLM_MODEL": "gpt-4",
            "OPENAI_API_KEY": "test-key"
        }):
            with patch('app.llm_providers.openai'):
                provider = LLMProviderFactory.from_env()
                assert provider.provider_type == ProviderType.OPENAI
                assert provider.model == "gpt-4"
    
    def test_from_env_fallback_to_gptj(self):
        """Test fallback to GPT-J when provider not specified."""
        with patch.dict(os.environ, {}, clear=True):
            with patch('app.llm_providers.ChatAgent'):
                with patch('app.llm_providers.ChatConfig'):
                    provider = LLMProviderFactory.from_env()
                    assert provider.provider_type == ProviderType.GPTJ
    
    def test_unsupported_provider(self):
        """Test error handling for unsupported provider."""
        config = LLMConfig(
            provider="unsupported",  # This should cause an error
            model="test"
        )
        
        with pytest.raises(ValueError, match="Unsupported provider"):
            LLMProviderFactory.create_provider(config)


class TestOpenAIProvider:
    """Test OpenAI provider implementation."""
    
    @pytest.fixture
    def openai_config(self):
        """OpenAI test configuration."""
        return LLMConfig(
            provider=ProviderType.OPENAI,
            model="gpt-3.5-turbo",
            api_key="test-key"
        )
    
    def test_initialization(self, openai_config):
        """Test OpenAI provider initialization."""
        with patch('app.llm_providers.openai') as mock_openai:
            mock_client = Mock()
            mock_openai.OpenAI.return_value = mock_client
            
            provider = OpenAIProvider(openai_config)
            
            assert provider.model == "gpt-3.5-turbo"
            assert provider.client == mock_client
            mock_openai.OpenAI.assert_called_once()
    
    def test_generate_response(self, openai_config):
        """Test response generation."""
        with patch('app.llm_providers.openai') as mock_openai:
            # Mock the response structure
            mock_response = Mock()
            mock_response.choices = [Mock()]
            mock_response.choices[0].message.content = "Test response"
            
            mock_client = Mock()
            mock_client.chat.completions.create.return_value = mock_response
            mock_openai.OpenAI.return_value = mock_client
            
            provider = OpenAIProvider(openai_config)
            result = provider.generate("Test prompt")
            
            assert result == "Test response"
            mock_client.chat.completions.create.assert_called_once()
    
    def test_generate_stream(self, openai_config):
        """Test streaming response generation."""
        with patch('app.llm_providers.openai') as mock_openai:
            # Mock streaming response
            mock_chunk1 = Mock()
            mock_chunk1.choices = [Mock()]
            mock_chunk1.choices[0].delta.content = "Hello"
            
            mock_chunk2 = Mock()
            mock_chunk2.choices = [Mock()]
            mock_chunk2.choices[0].delta.content = " World"
            
            mock_chunk3 = Mock()
            mock_chunk3.choices = [Mock()]
            mock_chunk3.choices[0].delta.content = None  # End marker
            
            mock_client = Mock()
            mock_client.chat.completions.create.return_value = [mock_chunk1, mock_chunk2, mock_chunk3]
            mock_openai.OpenAI.return_value = mock_client
            
            provider = OpenAIProvider(openai_config)
            result = list(provider.generate_stream("Test prompt"))
            
            assert result == ["Hello", " World"]


class TestAnthropicProvider:
    """Test Anthropic provider implementation."""
    
    @pytest.fixture
    def anthropic_config(self):
        """Anthropic test configuration."""
        return LLMConfig(
            provider=ProviderType.ANTHROPIC,
            model="claude-3-sonnet-20240229",
            api_key="test-key"
        )
    
    def test_initialization(self, anthropic_config):
        """Test Anthropic provider initialization."""
        with patch('app.llm_providers.anthropic') as mock_anthropic:
            mock_client = Mock()
            mock_anthropic.Anthropic.return_value = mock_client
            
            provider = AnthropicProvider(anthropic_config)
            
            assert provider.model == "claude-3-sonnet-20240229"
            assert provider.client == mock_client
            mock_anthropic.Anthropic.assert_called_once()
    
    def test_generate_response(self, anthropic_config):
        """Test response generation."""
        with patch('app.llm_providers.anthropic') as mock_anthropic:
            # Mock the response structure
            mock_response = Mock()
            mock_response.content = [Mock()]
            mock_response.content[0].text = "Test response from Claude"
            
            mock_client = Mock()
            mock_client.messages.create.return_value = mock_response
            mock_anthropic.Anthropic.return_value = mock_client
            
            provider = AnthropicProvider(anthropic_config)
            result = provider.generate("Test prompt")
            
            assert result == "Test response from Claude"
            mock_client.messages.create.assert_called_once()


class TestOllamaProvider:
    """Test Ollama provider implementation."""
    
    @pytest.fixture
    def ollama_config(self):
        """Ollama test configuration."""
        return LLMConfig(
            provider=ProviderType.OLLAMA,
            model="llama2"
        )
    
    def test_initialization(self, ollama_config):
        """Test Ollama provider initialization."""
        provider = OllamaProvider(ollama_config)
        
        assert provider.model == "llama2"
        assert provider.base_url == "http://localhost:11434"
    
    def test_generate_response(self, ollama_config):
        """Test response generation."""
        with patch('app.llm_providers.requests') as mock_requests:
            # Mock successful response
            mock_response = Mock()
            mock_response.json.return_value = {"response": "Ollama test response"}
            mock_response.raise_for_status.return_value = None
            mock_requests.post.return_value = mock_response
            
            provider = OllamaProvider(ollama_config)
            result = provider.generate("Test prompt")
            
            assert result == "Ollama test response"
            mock_requests.post.assert_called_once()
    
    def test_is_available(self, ollama_config):
        """Test availability check."""
        with patch('app.llm_providers.requests') as mock_requests:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_requests.get.return_value = mock_response
            
            provider = OllamaProvider(ollama_config)
            assert provider.is_available() is True
            
            # Test unavailable
            mock_response.status_code = 500
            assert provider.is_available() is False


class TestGPTJProvider:
    """Test GPT-J provider implementation."""
    
    @pytest.fixture
    def gptj_config(self):
        """GPT-J test configuration."""
        return LLMConfig(
            provider=ProviderType.GPTJ,
            model="EleutherAI/gpt-j-6B"
        )
    
    def test_initialization(self, gptj_config):
        """Test GPT-J provider initialization."""
        with patch('app.llm_providers.ChatAgent') as mock_agent:
            with patch('app.llm_providers.ChatConfig') as mock_config:
                mock_chat_agent = Mock()
                mock_agent.return_value = mock_chat_agent
                
                provider = GPTJProvider(gptj_config)
                
                assert provider.model == "EleutherAI/gpt-j-6B"
                assert provider.chat_agent == mock_chat_agent
                mock_agent.assert_called_once()
    
    def test_generate_response(self, gptj_config):
        """Test response generation."""
        with patch('app.llm_providers.ChatAgent') as mock_agent:
            with patch('app.llm_providers.ChatConfig'):
                mock_chat_agent = Mock()
                mock_chat_agent.generate_sync.return_value = "GPT-J response"
                mock_agent.return_value = mock_chat_agent
                
                provider = GPTJProvider(gptj_config)
                result = provider.generate("Test prompt")
                
                assert result == "GPT-J response"
                mock_chat_agent.generate_sync.assert_called_once_with("Test prompt")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])