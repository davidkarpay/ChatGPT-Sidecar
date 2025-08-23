import os
import logging
import torch
import threading
import time
import gc
from typing import Optional, Dict, List, Generator, Any
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from dataclasses import dataclass
import json

logger = logging.getLogger(__name__)

def configure_openmp_for_apple_silicon():
    """Configure OpenMP settings to prevent threading conflicts on Apple Silicon"""
    openmp_settings = {
        'OMP_NUM_THREADS': '1',
        'MKL_NUM_THREADS': '1', 
        'NUMEXPR_NUM_THREADS': '1',
        'TOKENIZERS_PARALLELISM': 'false',
        'OMP_WAIT_POLICY': 'PASSIVE',  # Reduce CPU spinning
        'OMP_MAX_ACTIVE_LEVELS': '1'   # Limit nested parallelism
    }
    
    for key, value in openmp_settings.items():
        if key not in os.environ:
            os.environ[key] = value
            logger.info(f"Set {key}={value} for OpenMP stability")
        else:
            logger.info(f"Using existing {key}={os.environ[key]}")

def configure_memory_for_apple_silicon():
    """Configure memory settings for BLAS stability on Apple Silicon"""
    memory_settings = {
        'PYTORCH_MPS_HIGH_WATERMARK_RATIO': '0.0',  # Disable MPS memory pooling
        'PYTORCH_ENABLE_MPS_FALLBACK': '1',         # Enable CPU fallback for MPS
        'OPENBLAS_NUM_THREADS': '1',                # Single thread for OpenBLAS
        'VECLIB_MAXIMUM_THREADS': '1',              # Single thread for Accelerate
        'BLIS_NUM_THREADS': '1',                    # Single thread for BLIS
    }
    
    for key, value in memory_settings.items():
        if key not in os.environ:
            os.environ[key] = value
            logger.info(f"Set {key}={value} for memory stability")
        else:
            logger.info(f"Using existing {key}={os.environ[key]}")

def configure_pytorch_allocator():
    """Configure PyTorch memory allocator for Apple Silicon stability"""
    try:
        # Configure PyTorch memory management
        if torch.backends.mps.is_available():
            # Set MPS allocator configuration for stability
            torch.mps.set_per_process_memory_fraction(0.8)  # Limit MPS memory usage
            logger.info("Configured MPS memory fraction to 0.8")
        
        # Set memory management for better BLAS compatibility
        torch.set_num_threads(1)  # Limit PyTorch threading
        torch.set_num_interop_threads(1)  # Limit interop threading
        
        # Enable deterministic operations for stability
        if hasattr(torch, 'use_deterministic_algorithms'):
            torch.use_deterministic_algorithms(False)  # Disable for compatibility
        
        logger.info("PyTorch allocator configured for Apple Silicon")
        
    except Exception as e:
        logger.warning(f"Could not configure PyTorch allocator: {e}")

# Configure both OpenMP and memory settings
configure_openmp_for_apple_silicon()
configure_memory_for_apple_silicon()
configure_pytorch_allocator()

def ensure_tensor_memory_safety(inputs: Dict[str, torch.Tensor], device: str) -> Dict[str, torch.Tensor]:
    """Ensure tensors are memory-safe for BLAS operations on Apple Silicon"""
    safe_inputs = {}
    
    for key, tensor in inputs.items():
        try:
            # Ensure tensor is contiguous in memory (prevents alignment issues)
            if not tensor.is_contiguous():
                logger.debug(f"Making tensor '{key}' contiguous for memory safety")
                tensor = tensor.contiguous()
            
            # For Apple Silicon, ensure proper data layout
            if device == "mps":
                # Move to CPU first, ensure contiguous, then back to MPS
                if tensor.device.type == "mps":
                    cpu_tensor = tensor.cpu().contiguous()
                    safe_inputs[key] = cpu_tensor.to(device)
                else:
                    safe_inputs[key] = tensor.contiguous().to(device)
            else:
                # For CPU and CUDA, just ensure contiguous
                safe_inputs[key] = tensor.contiguous().to(device)
                
        except RuntimeError as e:
            logger.warning(f"Memory safety operation failed for tensor '{key}': {e}")
            # Fallback to CPU if MPS operations fail
            safe_inputs[key] = tensor.cpu().contiguous()
    
    return safe_inputs

def cleanup_memory():
    """Aggressive memory cleanup for Apple Silicon stability"""
    try:
        # Force garbage collection
        gc.collect()
        
        # Clear PyTorch caches
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
            torch.mps.synchronize()
        
        # Additional cleanup
        gc.collect()
        
        logger.debug("Memory cleanup completed")
        
    except Exception as e:
        logger.warning(f"Memory cleanup failed: {e}")

def monitor_memory_usage():
    """Monitor memory usage for debugging"""
    try:
        import psutil
        process = psutil.Process()
        memory_info = process.memory_info()
        memory_mb = memory_info.rss / 1024 / 1024
        
        logger.debug(f"Current memory usage: {memory_mb:.1f} MB")
        
        if memory_mb > 2000:  # Over 2GB
            logger.warning(f"High memory usage detected: {memory_mb:.1f} MB")
            cleanup_memory()
        
        return memory_mb
        
    except ImportError:
        # psutil not available, skip monitoring
        return None
    except Exception as e:
        logger.warning(f"Memory monitoring failed: {e}")
        return None

@dataclass
class ChatConfig:
    model_name: str = "EleutherAI/gpt-j-6B"
    max_context_length: int = 2048
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1
    use_8bit: bool = True
    device: Optional[str] = None
    
    def __post_init__(self):
        """Auto-detect device if not specified"""
        if self.device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available() and self.model_name not in ["distilgpt2", "gpt2"]:
                # For small models, prefer CPU to avoid MPS issues
                self.device = "mps"
            else:
                self.device = "cpu"

class ChatAgent:
    def __init__(self, config: Optional[ChatConfig] = None):
        self.config = config or ChatConfig()
        self.tokenizer = None
        self.model = None
        self.conversation_history = {}
        self._load_model()
    
    def _load_model(self):
        """Load model with comprehensive error handling and fallback strategies"""
        logger.info(f"Loading model {self.config.model_name} on device: {self.config.device}")
        
        # Clean up before loading
        cleanup_memory()
        
        max_retries = 3
        retry_delay = 2  # seconds
        
        for attempt in range(max_retries):
            try:
                return self._attempt_model_load(attempt)
            except Exception as e:
                logger.error(f"Model loading attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    logger.info(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    logger.error("All model loading attempts failed, trying CPU fallback")
                    return self._cpu_fallback_load()
    
    def _attempt_model_load(self, attempt: int):
        """Attempt to load model with current configuration"""
        # Monitor memory for this attempt
        memory_before = monitor_memory_usage()
        
        # Load tokenizer first (usually more stable)
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name,
                use_fast=True,  # Use fast tokenizer when available
                trust_remote_code=True
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
        except Exception as e:
            logger.error(f"Failed to load tokenizer: {e}")
            raise
        
        # Configure model loading parameters
        model_kwargs = {
            "low_cpu_mem_usage": True,  # Reduce memory usage during loading
            "trust_remote_code": True
        }
        
        # Device-specific configuration
        original_device = self.config.device
        
        # On Apple Silicon, prefer CPU for problematic models after first attempt
        if attempt > 0 and self.config.device == "mps":
            logger.warning("Falling back to CPU after MPS failure")
            self.config.device = "cpu"
        
        # 8-bit quantization only works with CUDA
        if self.config.use_8bit and self.config.device == "cuda":
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_8bit_compute_dtype=torch.float16
            )
            model_kwargs["quantization_config"] = quantization_config
            model_kwargs["device_map"] = "auto"
            logger.info("Using 8-bit quantization for CUDA")
        elif self.config.use_8bit and self.config.device != "cuda":
            logger.warning(f"8-bit quantization not supported on {self.config.device}, using full precision")
        
        # Set appropriate data types for each device
        if self.config.device == "cuda":
            torch_dtype = torch.float16
        elif self.config.device == "mps":
            torch_dtype = torch.float32  # MPS works better with float32
        else:  # cpu
            torch_dtype = torch.float32
        
        model_kwargs["torch_dtype"] = torch_dtype
        
        # Load model with timeout protection
        try:
            logger.info(f"Attempting to load model with torch_dtype={torch_dtype}")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                **model_kwargs
            )
            
            # Move model to device if not using quantization
            if not (self.config.use_8bit and self.config.device == "cuda"):
                # For Apple Silicon, ensure model parameters are memory-aligned
                if self.config.device == "mps":
                    logger.info("Preparing model for MPS with memory safety")
                    # Move to CPU first, then to MPS for better memory layout
                    self.model = self.model.cpu()
                    # Ensure all parameters are contiguous
                    for param in self.model.parameters():
                        if not param.is_contiguous():
                            param.data = param.data.contiguous()
                    
                    # Now move to MPS
                    try:
                        self.model = self.model.to(self.config.device)
                        logger.info(f"Model moved to {self.config.device} with memory safety")
                    except RuntimeError as e:
                        logger.warning(f"MPS loading failed: {e}, falling back to CPU")
                        self.config.device = "cpu"
                        self.model = self.model.to("cpu")
                else:
                    self.model = self.model.to(self.config.device)
                    logger.info(f"Model moved to {self.config.device}")
            
            logger.info("Model loaded successfully")
            
            # Monitor memory after loading
            memory_after = monitor_memory_usage()
            if memory_before and memory_after:
                memory_increase = memory_after - memory_before
                logger.info(f"Model loading increased memory by {memory_increase:.1f} MB")
            
            return True
            
        except Exception as e:
            # Restore original device setting for next attempt
            self.config.device = original_device
            raise e
    
    def _cpu_fallback_load(self):
        """Emergency CPU-only fallback for when all other loading attempts fail"""
        logger.warning("Attempting emergency CPU-only fallback")
        
        try:
            # Force CPU and minimal configuration
            self.config.device = "cpu"
            
            # Load with most conservative settings
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True,
                trust_remote_code=True
            )
            
            self.model = self.model.to("cpu")
            logger.info("Emergency CPU fallback successful")
            return True
            
        except Exception as e:
            logger.error(f"Emergency CPU fallback also failed: {e}")
            # Try alternative smaller model as last resort
            return self._try_alternative_model()
    
    def _try_alternative_model(self):
        """Try loading a smaller, more stable model as absolute last resort"""
        alternative_models = [
            "distilgpt2",
            "gpt2", 
            "microsoft/DialoGPT-small"
        ]
        
        original_model = self.config.model_name
        
        for alt_model in alternative_models:
            if alt_model == original_model:
                continue
                
            try:
                logger.warning(f"Trying alternative model: {alt_model}")
                self.config.model_name = alt_model
                self.config.device = "cpu"
                
                self.tokenizer = AutoTokenizer.from_pretrained(alt_model)
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                
                self.model = AutoModelForCausalLM.from_pretrained(
                    alt_model,
                    torch_dtype=torch.float32,
                    low_cpu_mem_usage=True
                )
                
                self.model = self.model.to("cpu")
                logger.info(f"Successfully loaded alternative model: {alt_model}")
                return True
                
            except Exception as e:
                logger.error(f"Alternative model {alt_model} also failed: {e}")
                continue
        
        # If we get here, everything failed
        self.config.model_name = original_model
        raise RuntimeError("All model loading attempts failed, including fallbacks")
    
    def _build_prompt(self, query: str, context: List[Dict], session_id: Optional[str] = None) -> str:
        prompt_parts = []
        
        prompt_parts.append("You are an AI assistant helping users explore and understand their ChatGPT conversation history. ")
        prompt_parts.append("Use the provided context from their conversations to answer questions accurately.\n\n")
        
        if context:
            prompt_parts.append("### Relevant Context from User's Conversations:\n")
            for i, ctx in enumerate(context[:5], 1):
                source = ctx.get("source", "Unknown")
                preview = ctx.get("preview", "")
                prompt_parts.append(f"\n[{i}] From '{source}':\n{preview}\n")
            prompt_parts.append("\n")
        
        if session_id and session_id in self.conversation_history:
            history = self.conversation_history[session_id][-3:]
            if history:
                prompt_parts.append("### Recent Conversation:\n")
                for turn in history:
                    prompt_parts.append(f"User: {turn['user']}\n")
                    prompt_parts.append(f"Assistant: {turn['assistant']}\n\n")
        
        prompt_parts.append(f"### Current Question:\nUser: {query}\n\n")
        prompt_parts.append("### Response:\nAssistant:")
        
        return "".join(prompt_parts)
    
    def generate_response(
        self, 
        query: str, 
        context: List[Dict], 
        session_id: Optional[str] = None,
        stream: bool = False
    ) -> Generator[str, None, None] | str:
        
        prompt = self._build_prompt(query, context, session_id)
        
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt", 
            truncation=True, 
            max_length=self.config.max_context_length
        )
        
        # Prepare inputs with memory safety for BLAS operations
        if not self.config.use_8bit:
            try:
                # Use memory-safe tensor handling
                inputs = ensure_tensor_memory_safety(inputs, self.config.device)
                logger.debug(f"Inputs prepared safely for {self.config.device}")
            except RuntimeError as e:
                if "MPS" in str(e) or "placeholder" in str(e).lower() or "alignment" in str(e).lower():
                    logger.warning(f"Memory safety error during input preparation, falling back to CPU: {e}")
                    # Move model and inputs to CPU
                    self.model = self.model.to("cpu")
                    self.config.device = "cpu"
                    inputs = ensure_tensor_memory_safety(inputs, "cpu")
                else:
                    raise
        
        generation_config = {
            "max_new_tokens": self.config.max_new_tokens,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "top_k": self.config.top_k,
            "repetition_penalty": self.config.repetition_penalty,
            "do_sample": True,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        
        if stream:
            return self._stream_response(inputs, generation_config, query, session_id)
        else:
            try:
                # Monitor memory before generation
                memory_before = monitor_memory_usage()
                cleanup_memory()  # Aggressive cleanup before generation
                
                with torch.no_grad():
                    # Ensure model is in eval mode for inference stability
                    self.model.eval()
                    outputs = self.model.generate(**inputs, **generation_config)
                
                response = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
                
                # Monitor memory after generation
                memory_after = monitor_memory_usage()
                if memory_before and memory_after:
                    memory_increase = memory_after - memory_before
                    if memory_increase > 100:  # Over 100MB increase
                        logger.warning(f"Generation increased memory by {memory_increase:.1f} MB")
                        cleanup_memory()  # Clean up if memory increased significantly
                
            except RuntimeError as e:
                if any(keyword in str(e).lower() for keyword in ["mps", "placeholder", "alignment", "bus error", "segmentation"]):
                    logger.warning(f"Memory/BLAS error during generation, falling back to CPU: {e}")
                    # Move everything to CPU and retry with memory safety
                    self.model = self.model.to("cpu")
                    self.config.device = "cpu"
                    inputs = ensure_tensor_memory_safety(inputs, "cpu")
                    
                    with torch.no_grad():
                        self.model.eval()
                        outputs = self.model.generate(**inputs, **generation_config)
                    
                    response = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
                else:
                    raise
            
            if session_id:
                if session_id not in self.conversation_history:
                    self.conversation_history[session_id] = []
                self.conversation_history[session_id].append({
                    "user": query,
                    "assistant": response
                })
            
            return response
    
    def _stream_response(
        self, 
        inputs: Dict, 
        generation_config: Dict,
        query: str,
        session_id: Optional[str] = None
    ) -> Generator[str, None, None]:
        
        full_response = []
        
        try:
            with torch.no_grad():
                for _ in range(generation_config["max_new_tokens"]):
                    outputs = self.model(**inputs)
                    logits = outputs.logits[0, -1, :]
                    
                    if generation_config["do_sample"]:
                        probs = torch.nn.functional.softmax(logits / generation_config["temperature"], dim=-1)
                        
                        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                        cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
                        
                        mask = cumsum_probs <= generation_config["top_p"]
                        mask[min(generation_config["top_k"], mask.sum().item())] = True
                        
                        filtered_probs = sorted_probs * mask
                        filtered_probs = filtered_probs / filtered_probs.sum()
                        
                        next_token = sorted_indices[torch.multinomial(filtered_probs, 1)]
                    else:
                        next_token = torch.argmax(logits, dim=-1, keepdim=True)
                    
                    if next_token.item() == self.tokenizer.eos_token_id:
                        break
                    
                    token_text = self.tokenizer.decode(next_token, skip_special_tokens=True)
                    full_response.append(token_text)
                    yield token_text
                    
                    inputs["input_ids"] = torch.cat([inputs["input_ids"], next_token.unsqueeze(0)], dim=1)
                    if "attention_mask" in inputs:
                        inputs["attention_mask"] = torch.cat([
                            inputs["attention_mask"], 
                            torch.ones((1, 1), dtype=inputs["attention_mask"].dtype)
                        ], dim=1)
        
        except RuntimeError as e:
            if any(keyword in str(e).lower() for keyword in ["mps", "placeholder", "alignment", "bus error", "segmentation"]):
                logger.warning(f"Memory/BLAS error during streaming, falling back to CPU: {e}")
                # Move everything to CPU and retry non-streaming with memory safety
                self.model = self.model.to("cpu")
                self.config.device = "cpu"
                inputs = ensure_tensor_memory_safety(inputs, "cpu")
                
                # Fall back to non-streaming generation
                with torch.no_grad():
                    self.model.eval()
                    outputs = self.model.generate(**inputs, **generation_config)
                
                response = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
                full_response = [response]
                yield response
            else:
                raise
        
        if session_id:
            if session_id not in self.conversation_history:
                self.conversation_history[session_id] = []
            self.conversation_history[session_id].append({
                "user": query,
                "assistant": "".join(full_response)
            })
    
    def analyze_topics(self, chunks: List[Dict]) -> Dict[str, Any]:
        prompt = self._build_analysis_prompt(chunks)
        
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt", 
            truncation=True, 
            max_length=self.config.max_context_length
        )
        
        if self.config.device == "cuda" and not self.config.use_8bit:
            inputs = {k: v.to(self.config.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.3,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        response = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return {
                "topics": [],
                "summary": response,
                "error": "Could not parse structured response"
            }
    
    def _build_analysis_prompt(self, chunks: List[Dict]) -> str:
        prompt_parts = [
            "Analyze the following conversation excerpts and identify the main topics discussed.\n",
            "Return a JSON object with 'topics' (list of topic names) and 'summary' (brief overview).\n\n",
            "### Conversation Excerpts:\n"
        ]
        
        for i, chunk in enumerate(chunks[:10], 1):
            preview = chunk.get("preview", "")
            prompt_parts.append(f"[{i}] {preview}\n\n")
        
        prompt_parts.append("### Analysis (JSON format):\n")
        
        return "".join(prompt_parts)
    
    def suggest_questions(self, query: str, results: List[Dict]) -> List[str]:
        prompt = self._build_suggestion_prompt(query, results)
        
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt", 
            truncation=True, 
            max_length=self.config.max_context_length
        )
        
        if self.config.device == "cuda" and not self.config.use_8bit:
            inputs = {k: v.to(self.config.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=150,
                temperature=0.8,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        response = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        
        questions = []
        for line in response.split("\n"):
            line = line.strip()
            if line and (line.startswith("-") or line.startswith("•") or line[0].isdigit()):
                question = line.lstrip("-•").strip()
                if question.find(".") > 0:
                    question = question[question.find(".") + 1:].strip()
                if question and len(question) > 10:
                    questions.append(question)
        
        return questions[:5]
    
    def _build_suggestion_prompt(self, query: str, results: List[Dict]) -> str:
        prompt_parts = [
            f"Based on the user's question '{query}' and the search results, ",
            "suggest 3-5 follow-up questions they might want to explore.\n\n",
            "### Search Results Summary:\n"
        ]
        
        for i, result in enumerate(results[:3], 1):
            preview = result.get("preview", "")
            prompt_parts.append(f"[{i}] {preview}\n\n")
        
        prompt_parts.append("### Suggested Follow-up Questions:\n")
        
        return "".join(prompt_parts)
    
    def clear_history(self, session_id: str):
        if session_id in self.conversation_history:
            del self.conversation_history[session_id]
    
    def get_history(self, session_id: str) -> List[Dict]:
        return self.conversation_history.get(session_id, [])