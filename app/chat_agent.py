import os
import logging
import torch
from typing import Optional, Dict, List, Generator, Any
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from dataclasses import dataclass
import json

logger = logging.getLogger(__name__)

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
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

class ChatAgent:
    def __init__(self, config: Optional[ChatConfig] = None):
        self.config = config or ChatConfig()
        self.tokenizer = None
        self.model = None
        self.conversation_history = {}
        self._load_model()
    
    def _load_model(self):
        logger.info(f"Loading model {self.config.model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        model_kwargs = {}
        if self.config.use_8bit and self.config.device == "cuda":
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_8bit_compute_dtype=torch.float16
            )
            model_kwargs["quantization_config"] = quantization_config
            model_kwargs["device_map"] = "auto"
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=torch.float16 if self.config.device == "cuda" else torch.float32,
            low_cpu_mem_usage=True,
            **model_kwargs
        )
        
        if not self.config.use_8bit and self.config.device == "cuda":
            self.model = self.model.to(self.config.device)
        
        logger.info("Model loaded successfully")
    
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
        
        if self.config.device == "cuda" and not self.config.use_8bit:
            inputs = {k: v.to(self.config.device) for k, v in inputs.items()}
        
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
            with torch.no_grad():
                outputs = self.model.generate(**inputs, **generation_config)
            
            response = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            
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