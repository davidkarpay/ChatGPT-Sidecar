"""LLM Provider abstraction for plug-and-play model switching."""

import os
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Generator, Union
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class ProviderType(Enum):
    """Supported LLM provider types."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OLLAMA = "ollama"
    GPTJ = "gptj"
    LLAMACPP = "llamacpp"

@dataclass
class LLMConfig:
    """Configuration for LLM providers."""
    provider: ProviderType
    model: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    max_tokens: int = 512
    temperature: float = 0.7
    timeout: int = 30
    streaming: bool = True
    # Provider-specific options
    extra_params: Optional[Dict[str, Any]] = None

class LLMProvider(ABC):
    """Base abstract class for all LLM providers."""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self.provider_type = config.provider
        self.model = config.model
        
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate a single response from the model."""
        pass
    
    @abstractmethod 
    def generate_stream(self, prompt: str, **kwargs) -> Generator[str, None, None]:
        """Generate a streaming response from the model."""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the provider is available and properly configured."""
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model."""
        pass
    
    def validate_config(self) -> None:
        """Validate provider configuration. Override if needed."""
        if not self.model:
            raise ValueError(f"Model name is required for {self.provider_type.value} provider")

class OpenAIProvider(LLMProvider):
    """OpenAI API provider (GPT-3.5, GPT-4, etc.)."""
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self.client = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize OpenAI client."""
        try:
            import openai
            
            api_key = self.config.api_key or os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY or provide api_key in config.")
            
            base_url = self.config.base_url or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
            
            self.client = openai.OpenAI(
                api_key=api_key,
                base_url=base_url,
                timeout=self.config.timeout
            )
            
            logger.info(f"OpenAI provider initialized with model: {self.model}")
            
        except ImportError:
            raise ImportError("OpenAI library not installed. Install with: pip install openai")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI provider: {e}")
            raise
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate response using OpenAI API."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=kwargs.get("max_tokens", self.config.max_tokens),
                temperature=kwargs.get("temperature", self.config.temperature),
                stream=False,
                **kwargs.get("extra_params", self.config.extra_params or {})
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"OpenAI generation error: {e}")
            raise RuntimeError(f"OpenAI API error: {e}")
    
    def generate_stream(self, prompt: str, **kwargs) -> Generator[str, None, None]:
        """Generate streaming response using OpenAI API."""
        try:
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=kwargs.get("max_tokens", self.config.max_tokens),
                temperature=kwargs.get("temperature", self.config.temperature),
                stream=True,
                **kwargs.get("extra_params", self.config.extra_params or {})
            )
            
            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            logger.error(f"OpenAI streaming error: {e}")
            raise RuntimeError(f"OpenAI API streaming error: {e}")
    
    def is_available(self) -> bool:
        """Check if OpenAI provider is available."""
        try:
            # Try a simple API call to verify connectivity
            self.client.models.list()
            return True
        except Exception as e:
            logger.warning(f"OpenAI provider not available: {e}")
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get OpenAI model information."""
        return {
            "provider": "OpenAI",
            "model": self.model,
            "base_url": self.client.base_url if self.client else None,
            "streaming_supported": True,
            "max_tokens": self.config.max_tokens
        }

class AnthropicProvider(LLMProvider):
    """Anthropic Claude API provider."""
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self.client = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize Anthropic client."""
        try:
            import anthropic
            
            api_key = self.config.api_key or os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("Anthropic API key is required. Set ANTHROPIC_API_KEY or provide api_key in config.")
            
            self.client = anthropic.Anthropic(
                api_key=api_key,
                timeout=self.config.timeout
            )
            
            logger.info(f"Anthropic provider initialized with model: {self.model}")
            
        except ImportError:
            raise ImportError("Anthropic library not installed. Install with: pip install anthropic")
        except Exception as e:
            logger.error(f"Failed to initialize Anthropic provider: {e}")
            raise
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate response using Anthropic API."""
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=kwargs.get("max_tokens", self.config.max_tokens),
                temperature=kwargs.get("temperature", self.config.temperature),
                messages=[{"role": "user", "content": prompt}],
                **kwargs.get("extra_params", self.config.extra_params or {})
            )
            
            return response.content[0].text
            
        except Exception as e:
            logger.error(f"Anthropic generation error: {e}")
            raise RuntimeError(f"Anthropic API error: {e}")
    
    def generate_stream(self, prompt: str, **kwargs) -> Generator[str, None, None]:
        """Generate streaming response using Anthropic API."""
        try:
            stream = self.client.messages.create(
                model=self.model,
                max_tokens=kwargs.get("max_tokens", self.config.max_tokens),
                temperature=kwargs.get("temperature", self.config.temperature),
                messages=[{"role": "user", "content": prompt}],
                stream=True,
                **kwargs.get("extra_params", self.config.extra_params or {})
            )
            
            for chunk in stream:
                if chunk.type == "content_block_delta":
                    yield chunk.delta.text
                    
        except Exception as e:
            logger.error(f"Anthropic streaming error: {e}")
            raise RuntimeError(f"Anthropic API streaming error: {e}")
    
    def is_available(self) -> bool:
        """Check if Anthropic provider is available."""
        try:
            # Try a simple API call to verify connectivity
            self.client.messages.create(
                model=self.model,
                max_tokens=1,
                messages=[{"role": "user", "content": "test"}]
            )
            return True
        except Exception as e:
            logger.warning(f"Anthropic provider not available: {e}")
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get Anthropic model information."""
        return {
            "provider": "Anthropic",
            "model": self.model,
            "streaming_supported": True,
            "max_tokens": self.config.max_tokens
        }

class OllamaProvider(LLMProvider):
    """Ollama local model provider."""
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self.base_url = config.base_url or "http://localhost:11434"
        
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate response using Ollama API."""
        try:
            import requests
            
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": kwargs.get("temperature", self.config.temperature),
                        "num_predict": kwargs.get("max_tokens", self.config.max_tokens),
                        **kwargs.get("extra_params", self.config.extra_params or {})
                    }
                },
                timeout=self.config.timeout
            )
            
            response.raise_for_status()
            return response.json()["response"]
            
        except ImportError:
            raise ImportError("Requests library not installed. Install with: pip install requests")
        except Exception as e:
            logger.error(f"Ollama generation error: {e}")
            raise RuntimeError(f"Ollama API error: {e}")
    
    def generate_stream(self, prompt: str, **kwargs) -> Generator[str, None, None]:
        """Generate streaming response using Ollama API."""
        try:
            import requests
            import json
            
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": True,
                    "options": {
                        "temperature": kwargs.get("temperature", self.config.temperature),
                        "num_predict": kwargs.get("max_tokens", self.config.max_tokens),
                        **kwargs.get("extra_params", self.config.extra_params or {})
                    }
                },
                stream=True,
                timeout=self.config.timeout
            )
            
            response.raise_for_status()
            
            for line in response.iter_lines():
                if line:
                    chunk = json.loads(line.decode('utf-8'))
                    if 'response' in chunk:
                        yield chunk['response']
                        
        except Exception as e:
            logger.error(f"Ollama streaming error: {e}")
            raise RuntimeError(f"Ollama API streaming error: {e}")
    
    def is_available(self) -> bool:
        """Check if Ollama provider is available."""
        try:
            import requests
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get Ollama model information."""
        return {
            "provider": "Ollama",
            "model": self.model,
            "base_url": self.base_url,
            "streaming_supported": True,
            "max_tokens": self.config.max_tokens
        }

class GPTJProvider(LLMProvider):
    """Legacy GPT-J provider (wraps existing chat_agent)."""
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self.chat_agent = None
        self._initialize_agent()
    
    def _initialize_agent(self):
        """Initialize GPT-J chat agent."""
        try:
            from .chat_agent import ChatAgent, ChatConfig
            
            # Use existing chat configuration
            chat_config = ChatConfig(
                model_name=self.model,
                max_new_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                use_8bit=self.config.extra_params.get("use_8bit", True) if self.config.extra_params else True,
                device=self.config.extra_params.get("device") if self.config.extra_params else None,
                db_path=self.config.extra_params.get("db_path", "sidecar.db") if self.config.extra_params else "sidecar.db"
            )
            
            self.chat_agent = ChatAgent(chat_config)
            logger.info(f"GPT-J provider initialized with model: {self.model}")
            
        except Exception as e:
            logger.error(f"Failed to initialize GPT-J provider: {e}")
            raise
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate response using GPT-J."""
        try:
            # Use the chat agent's synchronous generate method
            response = self.chat_agent.generate_sync(prompt)
            return response
        except Exception as e:
            logger.error(f"GPT-J generation error: {e}")
            raise RuntimeError(f"GPT-J error: {e}")
    
    def generate_stream(self, prompt: str, **kwargs) -> Generator[str, None, None]:
        """Generate streaming response using GPT-J."""
        try:
            # Use the chat agent's streaming method
            for token in self.chat_agent.generate_stream(prompt):
                yield token
        except Exception as e:
            logger.error(f"GPT-J streaming error: {e}")
            raise RuntimeError(f"GPT-J streaming error: {e}")
    
    def is_available(self) -> bool:
        """Check if GPT-J provider is available."""
        try:
            return self.chat_agent is not None and hasattr(self.chat_agent, 'model')
        except:
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get GPT-J model information."""
        return {
            "provider": "GPT-J",
            "model": self.model,
            "streaming_supported": True,
            "max_tokens": self.config.max_tokens,
            "device": getattr(self.chat_agent.config, 'device', 'unknown') if self.chat_agent else 'unknown'
        }

class LLMProviderFactory:
    """Factory for creating LLM providers."""
    
    PROVIDERS = {
        ProviderType.OPENAI: OpenAIProvider,
        ProviderType.ANTHROPIC: AnthropicProvider,
        ProviderType.OLLAMA: OllamaProvider,
        ProviderType.GPTJ: GPTJProvider,
        # LLAMACPP will be added later
    }
    
    @classmethod
    def create_provider(cls, config: LLMConfig) -> LLMProvider:
        """Create an LLM provider instance."""
        provider_class = cls.PROVIDERS.get(config.provider)
        
        if not provider_class:
            raise ValueError(f"Unsupported provider: {config.provider}")
        
        try:
            provider = provider_class(config)
            provider.validate_config()
            return provider
        except Exception as e:
            logger.error(f"Failed to create {config.provider.value} provider: {e}")
            raise
    
    @classmethod
    def from_env(cls) -> LLMProvider:
        """Create provider from environment variables."""
        provider_name = os.getenv("LLM_PROVIDER", "gptj").lower()
        
        try:
            provider_type = ProviderType(provider_name)
        except ValueError:
            logger.warning(f"Unknown provider '{provider_name}', falling back to GPT-J")
            provider_type = ProviderType.GPTJ
        
        # Model selection based on provider
        default_models = {
            ProviderType.OPENAI: "gpt-3.5-turbo",
            ProviderType.ANTHROPIC: "claude-3-sonnet-20240229", 
            ProviderType.OLLAMA: "llama2",
            ProviderType.GPTJ: os.getenv("CHAT_MODEL", "EleutherAI/gpt-j-6B"),
        }
        
        config = LLMConfig(
            provider=provider_type,
            model=os.getenv("LLM_MODEL", default_models[provider_type]),
            api_key=os.getenv(f"{provider_type.value.upper()}_API_KEY"),
            base_url=os.getenv("LLM_BASE_URL"),
            max_tokens=int(os.getenv("LLM_MAX_TOKENS", "512")),
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.7")),
            timeout=int(os.getenv("LLM_TIMEOUT", "30")),
            streaming=os.getenv("LLM_STREAMING", "true").lower() == "true",
            extra_params={
                "use_8bit": os.getenv("CHAT_USE_8BIT", "true").lower() == "true",
                "device": os.getenv("CHAT_DEVICE"),
                "db_path": os.getenv("DB_PATH", "sidecar.db")
            }
        )
        
        return cls.create_provider(config)
    
    @classmethod
    def list_available_providers(cls) -> List[Dict[str, Any]]:
        """List all available providers and their status."""
        providers = []
        
        for provider_type in ProviderType:
            try:
                # Create a test config to check availability
                config = LLMConfig(provider=provider_type, model="test")
                provider_class = cls.PROVIDERS.get(provider_type)
                
                if provider_class:
                    # Try to determine if the provider could work
                    available = False
                    error = None
                    
                    try:
                        if provider_type == ProviderType.OPENAI:
                            available = bool(os.getenv("OPENAI_API_KEY"))
                        elif provider_type == ProviderType.ANTHROPIC:
                            available = bool(os.getenv("ANTHROPIC_API_KEY"))
                        elif provider_type == ProviderType.OLLAMA:
                            # Check if Ollama is running (simplified)
                            available = True  # Assume available for now
                        elif provider_type == ProviderType.GPTJ:
                            available = True  # Always available (local)
                        
                    except Exception as e:
                        error = str(e)
                    
                    providers.append({
                        "provider": provider_type.value,
                        "available": available,
                        "error": error
                    })
                    
            except Exception as e:
                providers.append({
                    "provider": provider_type.value,
                    "available": False,
                    "error": str(e)
                })
        
        return providers