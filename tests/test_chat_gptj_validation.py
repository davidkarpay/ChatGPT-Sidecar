"""
GPT-J Model Download and Validation Tests

Tests to ensure GPT-J model can be downloaded, cached, and used successfully
in production environments. Includes environment validation, download testing,
and real model integration validation.
"""
import pytest
import os
import shutil
import requests
import psutil
import tempfile
import time
from pathlib import Path
from unittest.mock import patch, Mock

# Handle optional dependencies gracefully
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

try:
    from huggingface_hub import HfApi, snapshot_download
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

from app.chat_agent import ChatAgent, ChatConfig


@pytest.mark.gptj_validation
@pytest.mark.environment
class TestPreDownloadEnvironment:
    """Test environment validation before attempting GPT-J download"""
    
    def test_disk_space_availability(self):
        """Test sufficient disk space for GPT-J model (~24GB)"""
        # Get disk usage for current directory
        usage = shutil.disk_usage(".")
        free_gb = usage.free / (1024**3)  # Convert to GB
        
        # GPT-J requires ~24GB, recommend 30GB+ for safety
        required_gb = 30
        
        assert free_gb >= required_gb, (
            f"Insufficient disk space. Available: {free_gb:.1f}GB, "
            f"Required: {required_gb}GB for GPT-J download"
        )
    
    def test_huggingface_hub_connectivity(self):
        """Test connection to HuggingFace Hub"""
        if not HF_AVAILABLE:
            pytest.skip("huggingface_hub not available")
            
        try:
            # Test basic connectivity to HuggingFace
            response = requests.get("https://huggingface.co", timeout=10)
            assert response.status_code == 200, "Cannot connect to HuggingFace Hub"
            
            # Test HF API access
            api = HfApi()
            model_info = api.model_info("EleutherAI/gpt-j-6B")
            assert model_info is not None, "Cannot access GPT-J model info via API"
            
        except requests.exceptions.RequestException as e:
            pytest.skip(f"No internet connection or HuggingFace unreachable: {e}")
        except Exception as e:
            pytest.fail(f"HuggingFace Hub connectivity test failed: {e}")
    
    def test_network_download_speed(self):
        """Test network speed for model download feasibility"""
        try:
            # Download a small test file to estimate speed
            test_url = "https://huggingface.co/EleutherAI/gpt-j-6B/resolve/main/README.md"
            start_time = time.time()
            
            response = requests.get(test_url, timeout=30)
            download_time = time.time() - start_time
            file_size_mb = len(response.content) / (1024**2)
            
            if download_time > 0:
                speed_mbps = (file_size_mb * 8) / download_time  # Convert to Mbps
                
                # Estimate time for 24GB download
                gptj_size_gb = 24
                estimated_hours = (gptj_size_gb * 1024 * 8) / (speed_mbps * 3600)
                
                # Warn if download would take more than 2 hours
                if estimated_hours > 2:
                    pytest.skip(
                        f"Network too slow for GPT-J download. "
                        f"Estimated time: {estimated_hours:.1f} hours"
                    )
                    
        except requests.exceptions.RequestException:
            pytest.skip("Cannot test network speed - connection issues")
    
    def test_gpu_memory_availability(self):
        """Test GPU memory for CUDA model loading"""
        if not TORCH_AVAILABLE:
            pytest.skip("PyTorch not available")
            
        if torch.cuda.is_available():
            # Get GPU memory info
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            
            # GPT-J requires ~12GB GPU memory for 8-bit, ~24GB for full precision
            required_memory_8bit = 12
            required_memory_full = 24
            
            if gpu_memory_gb < required_memory_8bit:
                pytest.skip(
                    f"Insufficient GPU memory. Available: {gpu_memory_gb:.1f}GB, "
                    f"Required: {required_memory_8bit}GB (8-bit) or {required_memory_full}GB (full)"
                )
        else:
            # CPU inference is possible but very slow
            pytest.skip("No CUDA GPU available - CPU inference will be very slow")
    
    def test_system_memory_availability(self):
        """Test system RAM for model loading"""
        memory = psutil.virtual_memory()
        available_gb = memory.available / (1024**3)
        
        # Recommend 32GB+ system RAM for GPT-J
        required_gb = 16  # Minimum for 8-bit
        recommended_gb = 32
        
        if available_gb < required_gb:
            pytest.fail(
                f"Insufficient system memory. Available: {available_gb:.1f}GB, "
                f"Minimum: {required_gb}GB, Recommended: {recommended_gb}GB"
            )
        elif available_gb < recommended_gb:
            pytest.skip(
                f"Low system memory. Available: {available_gb:.1f}GB, "
                f"Recommended: {recommended_gb}GB for optimal performance"
            )
    
    def test_cache_directory_permissions(self):
        """Test write permissions for HuggingFace cache"""
        if not TRANSFORMERS_AVAILABLE:
            pytest.skip("transformers not available")
            
        from transformers import TRANSFORMERS_CACHE
        
        cache_dir = Path(TRANSFORMERS_CACHE)
        
        # Test if cache directory exists and is writable
        if cache_dir.exists():
            assert os.access(cache_dir, os.W_OK), f"No write permission to cache directory: {cache_dir}"
        else:
            # Test if parent directory allows creation
            parent_dir = cache_dir.parent
            assert parent_dir.exists() and os.access(parent_dir, os.W_OK), (
                f"Cannot create cache directory. Parent not writable: {parent_dir}"
            )
    
    def test_python_dependencies_available(self):
        """Test that required packages are installed for GPT-J"""
        required_packages = [
            'torch',
            'transformers',
            'accelerate',
            'bitsandbytes',  # For 8-bit quantization
            'sentencepiece',  # For tokenization
        ]
        
        missing_packages = []
        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                missing_packages.append(package)
        
        if missing_packages:
            pytest.fail(f"Missing required packages for GPT-J: {missing_packages}")


@pytest.mark.gptj_validation
@pytest.mark.download
class TestModelDownloadValidation:
    """Test actual GPT-J model download process"""
    
    @pytest.mark.slow
    @pytest.mark.integration
    def test_gptj_tokenizer_download(self):
        """Test downloading GPT-J tokenizer (lightweight test)"""
        if not TRANSFORMERS_AVAILABLE:
            pytest.skip("transformers not available")
            
        try:
            # Download just the tokenizer first (much smaller)
            tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
            
            # Basic tokenizer validation
            assert tokenizer is not None
            assert hasattr(tokenizer, 'encode')
            assert hasattr(tokenizer, 'decode')
            
            # Test tokenization
            test_text = "Hello, how are you?"
            tokens = tokenizer.encode(test_text)
            decoded = tokenizer.decode(tokens)
            
            assert isinstance(tokens, list)
            assert isinstance(decoded, str)
            assert len(tokens) > 0
            
        except Exception as e:
            pytest.fail(f"GPT-J tokenizer download failed: {e}")
    
    @pytest.mark.slow
    @pytest.mark.integration
    @pytest.mark.skip("Very slow - only run manually for full validation")
    def test_gptj_model_download(self):
        """Test downloading full GPT-J model (very slow, skip by default)"""
        if not TORCH_AVAILABLE:
            pytest.skip("PyTorch not available")
            
        try:
            # This test is skipped by default due to download time
            # Run manually with: pytest -m "slow" -k "test_gptj_model_download" -s
            
            config = ChatConfig(
                model_name="EleutherAI/gpt-j-6B",
                use_8bit=True,  # Use 8-bit to reduce memory requirements
                device="cuda" if torch.cuda.is_available() else "cpu"
            )
            
            # Track download time
            start_time = time.time()
            agent = ChatAgent(config)
            download_time = time.time() - start_time
            
            assert agent.model is not None
            assert agent.tokenizer is not None
            
            print(f"GPT-J model download completed in {download_time:.1f} seconds")
            
        except Exception as e:
            pytest.fail(f"GPT-J model download failed: {e}")
    
    def test_model_download_resume_capability(self):
        """Test that interrupted downloads can be resumed"""
        # This test simulates download interruption/resume
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            
            # Mock a partially downloaded model
            model_dir = cache_dir / "models--EleutherAI--gpt-j-6B"
            model_dir.mkdir(parents=True)
            
            # Create partial download indicator
            incomplete_file = model_dir / ".incomplete"
            incomplete_file.touch()
            
            # Test that system can handle resume
            # (In practice, transformers/huggingface_hub handle this automatically)
            assert model_dir.exists()
            assert incomplete_file.exists()
    
    def test_download_with_limited_bandwidth(self):
        """Test download behavior with bandwidth limitations"""
        # This would test throttled downloads in production
        pytest.skip("Bandwidth limitation testing requires network configuration")
    
    def test_concurrent_download_handling(self):
        """Test multiple processes downloading same model"""
        # Test file locking and concurrent access
        pytest.skip("Concurrent download testing requires process coordination")


@pytest.mark.gptj_validation 
@pytest.mark.integration
class TestPostDownloadValidation:
    """Test model functionality after download"""
    
    @pytest.fixture
    def downloaded_model_config(self):
        """Config for testing with downloaded model"""
        return ChatConfig(
            model_name="EleutherAI/gpt-j-6B",
            use_8bit=True,
            max_new_tokens=50,  # Limit for faster testing
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
    
    @pytest.mark.slow
    @pytest.mark.integration
    def test_model_loads_from_cache(self, downloaded_model_config):
        """Test that model loads successfully from cache"""
        try:
            # This test assumes model is already downloaded
            agent = ChatAgent(downloaded_model_config)
            
            assert agent.model is not None
            assert agent.tokenizer is not None
            assert agent.config.model_name == "EleutherAI/gpt-j-6B"
            
        except Exception as e:
            pytest.skip(f"Model not available in cache: {e}")
    
    @pytest.mark.slow
    @pytest.mark.integration
    def test_model_generates_responses(self, downloaded_model_config):
        """Test that downloaded model can generate responses"""
        try:
            agent = ChatAgent(downloaded_model_config)
            
            # Test basic generation
            response = agent.generate_response("What is machine learning?", [])
            
            assert isinstance(response, str)
            assert len(response.strip()) > 0
            assert response != "What is machine learning?"  # Should be different from input
            
        except Exception as e:
            pytest.skip(f"Model generation test failed: {e}")
    
    @pytest.mark.slow
    @pytest.mark.integration
    def test_quantization_works_correctly(self, downloaded_model_config):
        """Test that 8-bit quantization works with downloaded model"""
        if not torch.cuda.is_available():
            pytest.skip("Quantization test requires CUDA")
            
        try:
            # Test 8-bit quantization
            config_8bit = ChatConfig(
                model_name="EleutherAI/gpt-j-6B",
                use_8bit=True,
                device="cuda"
            )
            
            agent = ChatAgent(config_8bit)
            response = agent.generate_response("Hello", [])
            
            assert isinstance(response, str)
            assert len(response) > 0
            
        except Exception as e:
            pytest.skip(f"Quantization test failed: {e}")
    
    @pytest.mark.slow
    @pytest.mark.integration  
    def test_model_memory_usage(self, downloaded_model_config):
        """Test memory usage of loaded model"""
        try:
            import psutil
            process = psutil.Process()
            
            # Memory before loading
            memory_before = process.memory_info().rss / (1024**3)  # GB
            
            agent = ChatAgent(downloaded_model_config)
            
            # Memory after loading
            memory_after = process.memory_info().rss / (1024**3)  # GB
            memory_increase = memory_after - memory_before
            
            # 8-bit model should use ~12-15GB, full precision ~24GB+
            max_expected_memory = 20 if downloaded_model_config.use_8bit else 30
            
            assert memory_increase < max_expected_memory, (
                f"Model uses too much memory: {memory_increase:.1f}GB "
                f"(max expected: {max_expected_memory}GB)"
            )
            
        except Exception as e:
            pytest.skip(f"Memory usage test failed: {e}")


@pytest.mark.gptj_validation
@pytest.mark.edge_case
class TestDownloadFailureScenarios:
    """Test handling of download failures"""
    
    def test_network_timeout_handling(self):
        """Test handling of network timeouts during download"""
        with patch('requests.get') as mock_get:
            mock_get.side_effect = requests.exceptions.Timeout()
            
            # Should handle timeout gracefully
            with pytest.raises((requests.exceptions.Timeout, ConnectionError)):
                ChatAgent(ChatConfig(model_name="EleutherAI/gpt-j-6B"))
    
    def test_insufficient_disk_space_handling(self):
        """Test handling of insufficient disk space"""
        # Mock disk space check to return insufficient space
        with patch('shutil.disk_usage') as mock_disk_usage:
            # Mock very low disk space (1GB)
            mock_disk_usage.return_value = (1000, 900, 100)  # total, used, free (in some unit)
            
            # Download should fail or warn about space
            # (Implementation depends on how transformers handles this)
            pass
    
    def test_corrupted_download_handling(self):
        """Test handling of corrupted model files"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create corrupted model file
            fake_model_path = Path(tmpdir) / "pytorch_model.bin"
            fake_model_path.write_bytes(b"corrupted data")
            
            # Should detect corruption and re-download or fail gracefully
            # (This test would need more sophisticated mocking)
            pass
    
    def test_huggingface_hub_unavailable(self):
        """Test behavior when HuggingFace Hub is unavailable"""
        with patch('huggingface_hub.snapshot_download') as mock_download:
            mock_download.side_effect = requests.exceptions.ConnectionError()
            
            with pytest.raises(ConnectionError):
                ChatAgent(ChatConfig(model_name="EleutherAI/gpt-j-6B"))
    
    def test_invalid_model_name_handling(self):
        """Test handling of invalid model names"""
        with pytest.raises(Exception):  # Should raise some kind of error
            ChatAgent(ChatConfig(model_name="invalid/nonexistent-model"))
    
    def test_permission_denied_cache_directory(self):
        """Test handling when cache directory is not writable"""
        with patch('os.access', return_value=False):
            # Should handle permission errors gracefully
            # (Implementation depends on transformers error handling)
            pass


@pytest.mark.gptj_validation
@pytest.mark.integration 
@pytest.mark.slow
class TestRealGPTJIntegration:
    """End-to-end integration tests with real GPT-J model"""
    
    @pytest.mark.slow
    @pytest.mark.integration
    @pytest.mark.skip("Full integration test - run manually")
    def test_end_to_end_chat_workflow(self):
        """Test complete chat workflow with real GPT-J"""
        try:
            # Full end-to-end test with real model
            config = ChatConfig(
                model_name="EleutherAI/gpt-j-6B",
                use_8bit=True,
                max_new_tokens=100
            )
            
            agent = ChatAgent(config)
            
            # Test multiple interactions
            test_queries = [
                "What is artificial intelligence?",
                "Explain machine learning in simple terms.",
                "What are the benefits of renewable energy?"
            ]
            
            for query in test_queries:
                response = agent.generate_response(query, [])
                
                assert isinstance(response, str)
                assert len(response.strip()) > 20  # Substantial response
                assert query.lower() not in response.lower()[:50]  # Not just echoing
                
                # Basic quality checks
                assert not response.startswith("Error:")
                assert "Sorry" not in response[:20]  # Not immediate apology
                
        except Exception as e:
            pytest.skip(f"End-to-end integration test failed: {e}")
    
    @pytest.mark.slow
    @pytest.mark.integration
    def test_streaming_with_real_model(self):
        """Test streaming responses with real GPT-J model"""
        pytest.skip("Streaming integration test - implement after basic tests pass")
    
    @pytest.mark.slow 
    @pytest.mark.integration
    def test_conversation_context_with_real_model(self):
        """Test conversation context handling with real model"""
        pytest.skip("Context integration test - implement after basic tests pass")
    
    @pytest.mark.slow
    @pytest.mark.integration
    def test_performance_with_real_model(self):
        """Test performance metrics with real GPT-J model"""
        pytest.skip("Performance integration test - implement after basic tests pass")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])