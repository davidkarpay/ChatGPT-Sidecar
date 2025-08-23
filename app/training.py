"""
Local model training module with LoRA/QLoRA fine-tuning support
Optimized for quantized models with efficient adapter training
"""

import os
import json
import logging
import torch
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

# Training dependencies (install with: pip install peft transformers datasets accelerate bitsandbytes)
try:
    from transformers import (
        AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer,
        DataCollatorForLanguageModeling, BitsAndBytesConfig
    )
    from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
    from datasets import Dataset
    import bitsandbytes as bnb
    TRAINING_AVAILABLE = True
except ImportError as e:
    TRAINING_AVAILABLE = False
    Dataset = None  # Define Dataset as None when imports fail
    print(f"Training dependencies not available: {e}")

from .db import DB

logger = logging.getLogger(__name__)

@dataclass
class TrainingConfig:
    """Configuration for LoRA/QLoRA fine-tuning"""
    # Model settings
    base_model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    output_dir: str = "./models/fine-tuned"
    
    # LoRA settings
    lora_r: int = 16  # LoRA rank
    lora_alpha: int = 32  # LoRA alpha
    lora_dropout: float = 0.1
    target_modules: List[str] = None  # Auto-detect if None
    
    # Quantization settings
    use_4bit: bool = True
    bnb_4bit_compute_dtype: str = "float16"
    bnb_4bit_quant_type: str = "nf4"
    use_nested_quant: bool = False
    
    # Training settings
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    max_grad_norm: float = 0.3
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "constant"
    
    # Data settings
    max_length: int = 512
    min_examples: int = 10  # Minimum training examples required
    
    def __post_init__(self):
        if self.target_modules is None:
            # Default LoRA targets for TinyLlama
            self.target_modules = ["q_proj", "v_proj"]

class ModelTrainer:
    """Handles LoRA/QLoRA fine-tuning of language models"""
    
    def __init__(self, config: TrainingConfig, db_path: str):
        self.config = config
        self.db_path = db_path
        self.tokenizer = None
        self.model = None
        
        if not TRAINING_AVAILABLE:
            raise ImportError("Training dependencies not available. Install with: pip install peft transformers datasets accelerate bitsandbytes")
    
    def prepare_training_data(self, min_rating: int = 3):
        """Prepare training data from database feedback"""
        logger.info("Preparing training data from database...")
        
        with DB(self.db_path) as db:
            # Get training data with good ratings
            training_rows = db.get_training_data(
                limit=10000,  # Large limit to get all data
                min_rating=min_rating
            )
        
        if len(training_rows) < self.config.min_examples:
            raise ValueError(f"Insufficient training data: {len(training_rows)} examples (minimum: {self.config.min_examples})")
        
        # Format data for instruction tuning
        formatted_data = []
        for row in training_rows:
            # Parse context if available
            context_info = ""
            if row["context_json"]:
                try:
                    context = json.loads(row["context_json"])
                    if context:
                        context_info = "\\n\\nContext from your conversations:\\n" + "\\n".join([
                            f"- {ctx['source']}: {ctx['preview'][:100]}..."
                            for ctx in context[:2]  # Limit context for training
                        ])
                except:
                    pass
            
            # Create instruction-following format
            instruction = f"Based on your conversation history, answer this question: {row['user_query']}{context_info}"
            
            # Use corrected response if available, otherwise use original
            response = row["feedback_correction"] if row["feedback_correction"] else row["model_response"]
            
            formatted_data.append({
                "instruction": instruction,
                "response": response
            })
        
        logger.info(f"Prepared {len(formatted_data)} training examples")
        return Dataset.from_list(formatted_data)
    
    def load_model_and_tokenizer(self):
        """Load model and tokenizer with quantization"""
        logger.info(f"Loading model: {self.config.base_model_name}")
        
        # Configure quantization
        if self.config.use_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=self.config.use_nested_quant,
                bnb_4bit_quant_type=self.config.bnb_4bit_quant_type,
                bnb_4bit_compute_dtype=getattr(torch, self.config.bnb_4bit_compute_dtype)
            )
        else:
            bnb_config = None
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.base_model_name,
            trust_remote_code=True
        )
        
        # Set padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.base_model_name,
            quantization_config=bnb_config,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True,
            torch_dtype=torch.float16 if self.config.use_4bit else torch.float32
        )
        
        # Prepare model for training if using quantization
        if self.config.use_4bit:
            self.model = prepare_model_for_kbit_training(self.model)
    
    def setup_lora(self):
        """Configure LoRA adapter"""
        logger.info("Setting up LoRA adapter...")
        
        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            target_modules=self.config.target_modules,
            lora_dropout=self.config.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
    
    def tokenize_data(self, dataset: Dataset) -> Dataset:
        """Tokenize training data"""
        def tokenize_function(examples):
            # Combine instruction and response
            texts = [
                f"<|system|>\\nYou are a helpful assistant that answers questions based on conversation history.\\n<|user|>\\n{inst}\\n<|assistant|>\\n{resp}<|end|>"
                for inst, resp in zip(examples["instruction"], examples["response"])
            ]
            
            # Tokenize
            tokenized = self.tokenizer(
                texts,
                truncation=True,
                padding=False,
                max_length=self.config.max_length,
                return_overflowing_tokens=False
            )
            
            # Set labels (same as input_ids for causal LM)
            tokenized["labels"] = tokenized["input_ids"].copy()
            
            return tokenized
        
        return dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)
    
    def train(self) -> Tuple[str, Dict]:
        """Execute fine-tuning training"""
        logger.info("Starting fine-tuning training...")
        
        # Prepare data
        dataset = self.prepare_training_data()
        
        # Load model and setup LoRA
        self.load_model_and_tokenizer()
        self.setup_lora()
        
        # Tokenize data
        tokenized_dataset = self.tokenize_data(dataset)
        
        # Split data (80% train, 20% eval)
        train_size = int(0.8 * len(tokenized_dataset))
        eval_size = len(tokenized_dataset) - train_size
        
        if eval_size == 0:
            train_dataset = tokenized_dataset
            eval_dataset = None
        else:
            split_dataset = tokenized_dataset.train_test_split(test_size=eval_size, seed=42)
            train_dataset = split_dataset["train"]
            eval_dataset = split_dataset["test"]
        
        # Create output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(self.config.output_dir) / f"lora_adapter_{timestamp}"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=self.config.num_train_epochs,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.per_device_train_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            optim="paged_adamw_32bit" if self.config.use_4bit else "adamw_torch",
            save_steps=100,
            logging_steps=10,
            learning_rate=self.config.learning_rate,
            weight_decay=0.001,
            fp16=False,
            bf16=False,
            max_grad_norm=self.config.max_grad_norm,
            max_steps=-1,
            warmup_ratio=self.config.warmup_ratio,
            group_by_length=True,
            lr_scheduler_type=self.config.lr_scheduler_type,
            report_to=None,  # Disable wandb
            evaluation_strategy="steps" if eval_dataset else "no",
            eval_steps=50 if eval_dataset else None,
            save_strategy="epoch",
            load_best_model_at_end=True if eval_dataset else False,
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )
        
        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
        )
        
        # Train
        trainer.train()
        
        # Save adapter
        trainer.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        # Save training config
        config_path = output_dir / "training_config.json"
        with open(config_path, "w") as f:
            json.dump(self.config.__dict__, f, indent=2, default=str)
        
        # Get training metrics
        metrics = trainer.state.log_history[-1] if trainer.state.log_history else {}
        
        logger.info(f"Training completed. Adapter saved to: {output_dir}")
        
        return str(output_dir), metrics
    
    def create_model_version(self, adapter_path: str, training_config: Dict, 
                           training_data_count: int, metrics: Dict) -> int:
        """Create model version record in database"""
        with DB(self.db_path) as db:
            # Create version name
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            version_name = f"lora_tuned_{timestamp}"
            
            # Store training config and metrics together
            full_config = {
                "training_config": training_config,
                "training_metrics": metrics,
                "created_at": datetime.now().isoformat()
            }
            
            model_version_id = db.create_model_version(
                base_model=self.config.base_model_name,
                version_name=version_name,
                adapter_path=adapter_path,
                training_config=full_config,
                training_data_count=training_data_count
            )
            
            return model_version_id

class TrainingManager:
    """High-level training management"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
    
    def start_training(self, config: TrainingConfig) -> Dict:
        """Start a training session"""
        try:
            # Create training session record
            with DB(self.db_path) as db:
                session_id = db.create_training_session(
                    model_version_id=0,  # Will update after creating model version
                    training_data_filter=f"min_rating >= 3",
                    config=config.__dict__
                )
                
                # Update status to running
                db.update_training_session(session_id, "running")
            
            # Run training
            trainer = ModelTrainer(config, self.db_path)
            adapter_path, metrics = trainer.train()
            
            # Get training data count
            with DB(self.db_path) as db:
                training_data = db.get_training_data(limit=10000, min_rating=3)
                data_count = len(training_data)
            
            # Create model version
            model_version_id = trainer.create_model_version(
                adapter_path=adapter_path,
                training_config=config.__dict__,
                training_data_count=data_count,
                metrics=metrics
            )
            
            # Update training session with completion
            with DB(self.db_path) as db:
                db.update_training_session(session_id, "completed", metrics=metrics)
            
            return {
                "status": "completed",
                "session_id": session_id,
                "model_version_id": model_version_id,
                "adapter_path": adapter_path,
                "training_data_count": data_count,
                "metrics": metrics
            }
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            
            # Update training session with failure
            try:
                with DB(self.db_path) as db:
                    db.update_training_session(session_id, "failed", error_message=str(e))
            except:
                pass
            
            return {
                "status": "failed",
                "error": str(e)
            }
    
    def get_training_status(self) -> Dict:
        """Get current training status and available models"""
        with DB(self.db_path) as db:
            # Get recent training sessions
            sessions = db.conn.execute(
                """SELECT ts.*, mv.version_name, mv.base_model 
                   FROM training_session ts 
                   LEFT JOIN model_version mv ON ts.model_version_id = mv.id 
                   ORDER BY ts.started_at DESC LIMIT 10"""
            ).fetchall()
            
            # Get available model versions
            models = db.conn.execute(
                """SELECT * FROM model_version 
                   ORDER BY created_at DESC"""
            ).fetchall()
            
            return {
                "recent_sessions": [dict(session) for session in sessions],
                "available_models": [dict(model) for model in models],
                "training_available": TRAINING_AVAILABLE
            }