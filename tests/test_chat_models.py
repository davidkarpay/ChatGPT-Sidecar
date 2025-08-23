"""
Model integration tests for chat functionality
"""
import pytest
import torch
import numpy as np
from unittest.mock import patch, Mock, MagicMock
from pathlib import Path
import tempfile
import os

from app.chat_agent import ChatAgent, ChatConfig
from app.vectorstore import FaissStore


class TestModelConfiguration:
    """Test model configuration and device handling"""
    
    def test_config_device_detection_cuda(self):
        """Test device detection when CUDA is available"""
        with patch('torch.cuda.is_available', return_value=True):
            config = ChatConfig()
            assert config.device == "cuda"
    
    def test_config_device_detection_cpu(self):
        """Test device detection when CUDA is not available"""
        with patch('torch.cuda.is_available', return_value=False):
            config = ChatConfig()
            assert config.device == "cpu"
    
    def test_config_manual_device_override(self):
        """Test manual device configuration"""
        config = ChatConfig(device="mps")
        assert config.device == "mps"
    
    def test_config_quantization_settings(self):
        """Test quantization configuration"""
        # 8-bit enabled
        config_8bit = ChatConfig(use_8bit=True)
        assert config_8bit.use_8bit == True
        
        # 8-bit disabled
        config_full = ChatConfig(use_8bit=False)
        assert config_full.use_8bit == False
    
    def test_config_generation_parameters(self):
        """Test generation parameter configuration"""
        config = ChatConfig(
            temperature=0.8,
            top_p=0.95,
            top_k=40,
            repetition_penalty=1.2
        )
        
        assert config.temperature == 0.8
        assert config.top_p == 0.95
        assert config.top_k == 40
        assert config.repetition_penalty == 1.2


class TestModelLoading:
    """Test model loading with different configurations"""
    
    @patch('app.chat_agent.AutoModelForCausalLM.from_pretrained')
    @patch('app.chat_agent.AutoTokenizer.from_pretrained')
    def test_model_loading_cpu(self, mock_tokenizer, mock_model):
        """Test model loading for CPU"""
        # Setup mocks
        mock_tokenizer_instance = Mock()
        mock_tokenizer_instance.pad_token = None
        mock_tokenizer_instance.eos_token = "<|endoftext|>"
        mock_tokenizer.return_value = mock_tokenizer_instance
        
        mock_model_instance = Mock()
        mock_model.return_value = mock_model_instance
        
        config = ChatConfig(model_name="distilgpt2", use_8bit=False, device="cpu")
        agent = ChatAgent(config)
        
        # Verify model was loaded with correct parameters
        mock_model.assert_called_once()
        call_kwargs = mock_model.call_args[1]
        assert call_kwargs['torch_dtype'] == torch.float32
        assert 'quantization_config' not in call_kwargs
        assert 'device_map' not in call_kwargs
    
    @patch('torch.cuda.is_available', return_value=True)
    @patch('app.chat_agent.AutoModelForCausalLM.from_pretrained')
    @patch('app.chat_agent.AutoTokenizer.from_pretrained')
    def test_model_loading_cuda_8bit(self, mock_tokenizer, mock_model, mock_cuda):
        """Test model loading for CUDA with 8-bit quantization"""
        mock_tokenizer_instance = Mock()
        mock_tokenizer.return_value = mock_tokenizer_instance
        mock_model_instance = Mock()
        mock_model.return_value = mock_model_instance
        
        config = ChatConfig(model_name="distilgpt2", use_8bit=True, device="cuda")
        agent = ChatAgent(config)
        
        # Verify quantization config was used
        call_kwargs = mock_model.call_args[1]
        assert 'quantization_config' in call_kwargs
        assert 'device_map' in call_kwargs
        assert call_kwargs['device_map'] == "auto"
    
    @patch('torch.cuda.is_available', return_value=True)
    @patch('app.chat_agent.AutoModelForCausalLM.from_pretrained')
    @patch('app.chat_agent.AutoTokenizer.from_pretrained')
    def test_model_loading_cuda_full_precision(self, mock_tokenizer, mock_model, mock_cuda):
        """Test model loading for CUDA without quantization"""
        mock_tokenizer_instance = Mock()
        mock_tokenizer.return_value = mock_tokenizer_instance
        mock_model_instance = Mock()
        mock_model.return_value = mock_model_instance
        
        config = ChatConfig(model_name="distilgpt2", use_8bit=False, device="cuda")
        agent = ChatAgent(config)
        
        # Verify no quantization config
        call_kwargs = mock_model.call_args[1]
        assert 'quantization_config' not in call_kwargs
        assert call_kwargs['torch_dtype'] == torch.float16
        
        # Should call .to(device) for non-quantized models
        mock_model_instance.to.assert_called_once_with("cuda")
    
    @patch('app.chat_agent.AutoModelForCausalLM.from_pretrained')
    @patch('app.chat_agent.AutoTokenizer.from_pretrained')
    def test_tokenizer_pad_token_setup(self, mock_tokenizer, mock_model):
        """Test tokenizer pad token configuration"""
        # Case 1: No existing pad token
        mock_tokenizer_instance = Mock()
        mock_tokenizer_instance.pad_token = None
        mock_tokenizer_instance.eos_token = "<|endoftext|>"
        mock_tokenizer.return_value = mock_tokenizer_instance
        mock_model.return_value = Mock()
        
        agent = ChatAgent(ChatConfig(model_name="distilgpt2"))
        
        # Should set pad_token to eos_token
        assert agent.tokenizer.pad_token == "<|endoftext|>"
        
        # Case 2: Existing pad token
        mock_tokenizer_instance.pad_token = "<|pad|>"
        agent2 = ChatAgent(ChatConfig(model_name="distilgpt2"))
        
        # Should keep existing pad token
        assert agent2.tokenizer.pad_token == "<|pad|>"
    
    def test_model_loading_failure_handling(self):
        """Test handling of model loading failures"""
        with patch('app.chat_agent.AutoTokenizer.from_pretrained') as mock_tokenizer:
            mock_tokenizer.side_effect = Exception("Model not found")
            
            with pytest.raises(Exception, match="Model not found"):
                ChatAgent(ChatConfig(model_name="nonexistent-model"))


class TestModelGeneration:
    """Test model generation functionality"""
    
    @patch('app.chat_agent.ChatAgent._load_model')
    def test_generation_parameters_application(self, mock_load_model):
        """Test that generation parameters are correctly applied"""
        config = ChatConfig(
            temperature=0.8,
            top_p=0.9,
            top_k=50,
            repetition_penalty=1.1,
            max_new_tokens=100
        )
        agent = ChatAgent(config)
        
        # Mock tokenizer and model
        mock_tokenizer = Mock()
        mock_tokenizer.return_value = {"input_ids": torch.tensor([[1, 2, 3]])}
        mock_tokenizer.decode.return_value = "Generated response"
        mock_tokenizer.pad_token_id = 0
        mock_tokenizer.eos_token_id = 1
        agent.tokenizer = mock_tokenizer
        
        mock_model = Mock()
        mock_model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5]])
        agent.model = mock_model
        
        response = agent.generate_response("test query", [])
        
        # Verify generation was called with correct parameters
        mock_model.generate.assert_called_once()
        call_kwargs = mock_model.generate.call_args[1]
        
        assert call_kwargs["temperature"] == 0.8
        assert call_kwargs["top_p"] == 0.9
        assert call_kwargs["top_k"] == 50
        assert call_kwargs["repetition_penalty"] == 1.1
        assert call_kwargs["max_new_tokens"] == 100
    
    @patch('app.chat_agent.ChatAgent._load_model')
    def test_streaming_generation(self, mock_load_model):
        """Test streaming response generation"""
        agent = ChatAgent()
        
        # Mock tokenizer
        mock_tokenizer = Mock()
        mock_tokenizer.return_value = {"input_ids": torch.tensor([[1, 2, 3]])}
        mock_tokenizer.decode.side_effect = ["Hello", " ", "world", "!"]
        mock_tokenizer.pad_token_id = 0
        mock_tokenizer.eos_token_id = 1
        agent.tokenizer = mock_tokenizer
        
        # Mock model for streaming
        mock_model = Mock()
        # Mock model outputs for each generation step
        mock_outputs = Mock()
        mock_outputs.logits = torch.tensor([[[0.1, 0.9, 0.1]]])  # Mock logits
        mock_model.return_value = mock_outputs
        agent.model = mock_model
        
        # Mock torch operations
        with patch('torch.nn.functional.softmax') as mock_softmax, \
             patch('torch.sort') as mock_sort, \
             patch('torch.cumsum') as mock_cumsum, \
             patch('torch.multinomial') as mock_multinomial, \
             patch('torch.cat') as mock_cat, \
             patch('torch.ones') as mock_ones:
            
            # Setup mock returns
            mock_softmax.return_value = torch.tensor([0.1, 0.8, 0.1])
            mock_sort.return_value = (torch.tensor([0.8, 0.1, 0.1]), torch.tensor([1, 0, 2]))
            mock_cumsum.return_value = torch.tensor([0.8, 0.9, 1.0])
            mock_multinomial.return_value = torch.tensor([0])
            mock_cat.return_value = torch.tensor([[1, 2, 3, 1]])
            mock_ones.return_value = torch.tensor([[1]])
            
            # Test streaming
            generator = agent._stream_response(
                {"input_ids": torch.tensor([[1, 2, 3]])},
                {
                    "max_new_tokens": 4,
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "top_k": 50,
                    "do_sample": True
                },
                "test query",
                "test_session"
            )
            
            tokens = list(generator)
            assert len(tokens) > 0
    
    @patch('app.chat_agent.ChatAgent._load_model')
    def test_context_length_handling(self, mock_load_model):
        """Test handling of context length limits"""
        config = ChatConfig(max_context_length=512)
        agent = ChatAgent(config)
        
        # Create very long prompt
        long_query = "very long query " * 100  # Create long text
        long_context = [{"preview": "long context " * 50} for _ in range(10)]
        
        # Mock tokenizer
        mock_tokenizer = Mock()
        # Mock tokenization to return long sequence
        mock_tokenizer.return_value = {
            "input_ids": torch.tensor([[1] * 1000])  # Longer than max_context_length
        }
        mock_tokenizer.decode.return_value = "Response"
        mock_tokenizer.pad_token_id = 0
        mock_tokenizer.eos_token_id = 1
        agent.tokenizer = mock_tokenizer
        
        mock_model = Mock()
        mock_model.generate.return_value = torch.tensor([[1, 2, 3, 4]])
        agent.model = mock_model
        
        response = agent.generate_response(long_query, long_context)
        
        # Verify tokenizer was called with truncation and max_length
        call_kwargs = mock_tokenizer.call_args[1]
        assert call_kwargs["truncation"] == True
        assert call_kwargs["max_length"] == 512


class TestVectorStoreIntegration:
    """Test vector store model integration"""
    
    @pytest.fixture
    def temp_paths(self):
        """Create temporary paths for testing"""
        with tempfile.TemporaryDirectory() as tmpdir:
            index_path = Path(tmpdir) / "test.faiss"
            ids_path = Path(tmpdir) / "test.pkl"
            yield index_path, ids_path
    
    def test_vector_store_model_loading(self, temp_paths):
        """Test vector store model loading"""
        index_path, ids_path = temp_paths
        
        with patch('sentence_transformers.SentenceTransformer') as mock_st:
            mock_model = Mock()
            mock_model.get_sentence_embedding_dimension.return_value = 384
            mock_st.return_value = mock_model
            
            store = FaissStore(index_path, ids_path, "test-model")
            
            mock_st.assert_called_once_with("test-model")
            assert store.model == mock_model
    
    def test_vector_store_encoding(self, temp_paths):
        """Test vector store text encoding"""
        index_path, ids_path = temp_paths
        
        with patch('sentence_transformers.SentenceTransformer') as mock_st:
            mock_model = Mock()
            # Mock encoding to return normalized embeddings
            mock_embeddings = np.array([[0.6, 0.8], [0.8, 0.6]], dtype=np.float32)
            mock_model.encode.return_value = mock_embeddings
            mock_st.return_value = mock_model
            
            store = FaissStore(index_path, ids_path, "test-model")
            
            texts = ["text 1", "text 2"]
            embeddings = store.encode(texts)
            
            # Verify encoding parameters
            mock_model.encode.assert_called_once_with(
                texts,
                batch_size=64,
                show_progress_bar=False,
                normalize_embeddings=True
            )
            
            # Check output format
            assert isinstance(embeddings, np.ndarray)
            assert embeddings.dtype == np.float32
            assert embeddings.shape == (2, 2)
    
    def test_vector_store_faiss_operations(self, temp_paths):
        """Test FAISS index operations"""
        index_path, ids_path = temp_paths
        
        with patch('sentence_transformers.SentenceTransformer') as mock_st, \
             patch('faiss.IndexFlatIP') as mock_index_class, \
             patch('faiss.write_index') as mock_write, \
             patch('faiss.read_index') as mock_read:
            
            # Setup mocks
            mock_model = Mock()
            mock_model.get_sentence_embedding_dimension.return_value = 384
            mock_model.encode.return_value = np.array([[0.6, 0.8], [0.8, 0.6]], dtype=np.float32)
            mock_st.return_value = mock_model
            
            mock_index = Mock()
            mock_index.search.return_value = (
                np.array([[0.95, 0.87]]),  # scores
                np.array([[0, 1]])         # indices
            )
            mock_index_class.return_value = mock_index
            
            store = FaissStore(index_path, ids_path, "test-model")
            
            # Test building index
            rows = [
                {"embedding_ref_id": 0, "chunk_id": 10, "text": "text 1"},
                {"embedding_ref_id": 1, "chunk_id": 11, "text": "text 2"}
            ]
            
            store.build(rows)
            
            # Verify FAISS operations
            mock_index_class.assert_called_with(384)  # Correct dimension
            mock_index.add.assert_called_once()
            mock_write.assert_called_once()
            
            # Test search
            results = store.search("query text", k=2)
            
            assert len(results) == 2
            assert results[0] == (0, 0.95)
            assert results[1] == (1, 0.87)


class TestModelMemoryManagement:
    """Test model memory management and optimization"""
    
    @pytest.mark.slow
    def test_memory_usage_8bit_vs_full(self):
        """Test memory usage difference between 8-bit and full precision"""
        # This test is marked as slow and would require actual model loading
        # In practice, you'd measure actual memory usage
        pytest.skip("Requires actual model loading for memory measurement")
    
    def test_model_caching_behavior(self):
        """Test model caching and reuse"""
        with patch('app.chat_agent.AutoTokenizer.from_pretrained') as mock_tokenizer, \
             patch('app.chat_agent.AutoModelForCausalLM.from_pretrained') as mock_model:
            
            mock_tokenizer.return_value = Mock()
            mock_model.return_value = Mock()
            
            # Create multiple agents with same model
            config = ChatConfig(model_name="distilgpt2")
            agent1 = ChatAgent(config)
            agent2 = ChatAgent(config)
            
            # Each agent should load the model independently (no caching implemented)
            assert mock_tokenizer.call_count == 2
            assert mock_model.call_count == 2
    
    @patch('app.chat_agent.ChatAgent._load_model')
    def test_generation_memory_cleanup(self, mock_load_model):
        """Test that generation operations don't leak memory"""
        agent = ChatAgent()
        
        # Mock model operations
        mock_tokenizer = Mock()
        mock_tokenizer.return_value = {"input_ids": torch.tensor([[1, 2, 3]])}
        mock_tokenizer.decode.return_value = "Response"
        mock_tokenizer.pad_token_id = 0
        mock_tokenizer.eos_token_id = 1
        agent.tokenizer = mock_tokenizer
        
        mock_model = Mock()
        mock_model.generate.return_value = torch.tensor([[1, 2, 3, 4]])
        agent.model = mock_model
        
        # Generate multiple responses
        for i in range(10):
            response = agent.generate_response(f"query {i}", [])
            assert isinstance(response, str)
        
        # In a real test, you'd measure memory usage here
        # This test verifies the code doesn't crash with multiple generations


class TestModelCompatibility:
    """Test compatibility with different model types"""
    
    def test_different_model_architectures(self):
        """Test compatibility with different model architectures"""
        model_configs = [
            {"model_name": "distilgpt2", "expected_compatible": True},
            {"model_name": "gpt2", "expected_compatible": True},
            {"model_name": "EleutherAI/gpt-j-6B", "expected_compatible": True},
            {"model_name": "microsoft/DialoGPT-medium", "expected_compatible": True},
        ]
        
        for config_data in model_configs:
            with patch('app.chat_agent.AutoTokenizer.from_pretrained') as mock_tokenizer, \
                 patch('app.chat_agent.AutoModelForCausalLM.from_pretrained') as mock_model:
                
                if config_data["expected_compatible"]:
                    mock_tokenizer.return_value = Mock()
                    mock_model.return_value = Mock()
                    
                    # Should not raise exception
                    config = ChatConfig(model_name=config_data["model_name"])
                    agent = ChatAgent(config)
                    assert agent is not None
                else:
                    # Test would handle incompatible models
                    pass
    
    def test_model_size_configurations(self):
        """Test different model size configurations"""
        size_configs = [
            {"model": "distilgpt2", "context": 1024, "tokens": 256},      # Small
            {"model": "gpt2", "context": 1024, "tokens": 512},           # Medium  
            {"model": "gpt2-large", "context": 2048, "tokens": 1024},    # Large
        ]
        
        for size_config in size_configs:
            config = ChatConfig(
                model_name=size_config["model"],
                max_context_length=size_config["context"],
                max_new_tokens=size_config["tokens"]
            )
            
            assert config.model_name == size_config["model"]
            assert config.max_context_length == size_config["context"]
            assert config.max_new_tokens == size_config["tokens"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])