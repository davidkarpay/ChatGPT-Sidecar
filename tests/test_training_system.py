"""
Comprehensive tests for the training system
"""

import pytest
import tempfile
import json
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

from app.training import TrainingConfig, ModelTrainer, TrainingManager
from app.db import DB
from app.fast_llm_agent import FastLLMConfig


class TestTrainingConfig:
    """Test training configuration functionality"""
    
    def test_default_config(self):
        """Test default training configuration"""
        config = TrainingConfig()
        
        assert config.base_model_name == "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        assert config.lora_r == 16
        assert config.lora_alpha == 32
        assert config.use_4bit == True
        assert config.num_train_epochs == 3
        assert config.min_examples == 10
        assert config.target_modules == ["q_proj", "v_proj"]
    
    def test_custom_config(self):
        """Test custom training configuration"""
        config = TrainingConfig(
            base_model_name="test/model",
            lora_r=32,
            lora_alpha=64,
            num_train_epochs=5,
            learning_rate=1e-4,
            target_modules=["q_proj", "k_proj", "v_proj"]
        )
        
        assert config.base_model_name == "test/model"
        assert config.lora_r == 32
        assert config.lora_alpha == 64
        assert config.num_train_epochs == 5
        assert config.learning_rate == 1e-4
        assert config.target_modules == ["q_proj", "k_proj", "v_proj"]
    
    def test_post_init_target_modules(self):
        """Test automatic target modules setting"""
        # Test with None target_modules
        config = TrainingConfig(target_modules=None)
        assert config.target_modules == ["q_proj", "v_proj"]
        
        # Test with custom target_modules
        custom_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
        config = TrainingConfig(target_modules=custom_modules)
        assert config.target_modules == custom_modules


class TestModelTrainer:
    """Test model trainer functionality"""
    
    def setup_method(self):
        """Setup test environment"""
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.db_path = self.temp_db.name
        self.temp_db.close()
        
        # Initialize database schema
        with DB(self.db_path) as db:
            schema_path = Path(__file__).parent.parent / "schema.sql"
            db.init_schema(str(schema_path))
        
        self.config = TrainingConfig(
            base_model_name="test/model",
            output_dir=tempfile.mkdtemp(),
            min_examples=3  # Lower threshold for testing
        )
    
    def teardown_method(self):
        """Clean up test environment"""
        Path(self.db_path).unlink(missing_ok=True)
    
    def test_trainer_initialization_without_dependencies(self):
        """Test trainer initialization when training dependencies are missing"""
        with patch('app.training.TRAINING_AVAILABLE', False):
            with pytest.raises(ImportError) as exc_info:
                ModelTrainer(self.config, self.db_path)
            
            assert "Training dependencies not available" in str(exc_info.value)
    
    @patch('app.training.TRAINING_AVAILABLE', True)
    def test_trainer_initialization_with_dependencies(self):
        """Test trainer initialization when training dependencies are available"""
        trainer = ModelTrainer(self.config, self.db_path)
        
        assert trainer.config == self.config
        assert trainer.db_path == self.db_path
        assert trainer.tokenizer is None  # Not loaded until training
        assert trainer.model is None
    
    def test_prepare_training_data_insufficient(self):
        """Test training data preparation with insufficient data"""
        trainer = ModelTrainer(self.config, self.db_path)
        
        # Add only 2 training examples (below minimum of 3)
        with DB(self.db_path) as db:
            db.store_training_data("session1", "query1", "response1", [], "model")
            db.store_training_data("session2", "query2", "response2", [], "model")
        
        with pytest.raises(ValueError) as exc_info:
            trainer.prepare_training_data(min_rating=3)
        
        assert "Insufficient training data" in str(exc_info.value)
    
    def test_prepare_training_data_sufficient(self):
        """Test training data preparation with sufficient data"""
        trainer = ModelTrainer(self.config, self.db_path)
        
        # Add sufficient training examples with good ratings
        with DB(self.db_path) as db:
            for i in range(5):
                training_id = db.store_training_data(
                    f"session{i}", f"query{i}", f"response{i}", 
                    [{"source": f"doc{i}", "preview": f"content{i}"}], "model"
                )
                db.update_training_feedback(training_id, rating=4)
        
        dataset = trainer.prepare_training_data(min_rating=3)
        
        assert len(dataset) == 5
        
        # Check dataset structure
        first_example = dataset[0]
        assert "instruction" in first_example
        assert "response" in first_example
        assert "query0" in first_example["instruction"]
        assert first_example["response"] == "response0"
    
    def test_prepare_training_data_with_corrections(self):
        """Test training data preparation with user corrections"""
        trainer = ModelTrainer(self.config, self.db_path)
        
        # Add training data with corrections
        with DB(self.db_path) as db:
            for i in range(3):
                training_id = db.store_training_data(
                    f"session{i}", f"query{i}", f"original_response{i}", [], "model"
                )
                # Add corrections
                db.update_training_feedback(training_id, rating=4, correction=f"corrected_response{i}")
        
        dataset = trainer.prepare_training_data(min_rating=3)
        
        # Should use corrected responses
        for i, example in enumerate(dataset):
            assert example["response"] == f"corrected_response{i}"
    
    def test_prepare_training_data_with_context(self):
        """Test training data preparation includes context properly"""
        trainer = ModelTrainer(self.config, self.db_path)
        
        context_sources = [
            {"source": "Legal Document", "preview": "This case involves assault charges"},
            {"source": "Court Filing", "preview": "Motion filed under Rule 3.191"}
        ]
        
        with DB(self.db_path) as db:
            for i in range(3):
                training_id = db.store_training_data(
                    f"session{i}", f"legal query {i}", f"legal response {i}", 
                    context_sources, "model"
                )
                db.update_training_feedback(training_id, rating=5)
        
        dataset = trainer.prepare_training_data(min_rating=3)
        
        # Check that context is included in instructions
        for example in dataset:
            instruction = example["instruction"]
            assert "Context from your conversations:" in instruction
            assert "Legal Document" in instruction
            assert "Court Filing" in instruction
    
    @patch('app.training.TRAINING_AVAILABLE', True)
    @patch('app.training.AutoTokenizer')
    @patch('app.training.AutoModelForCausalLM')
    @patch('app.training.BitsAndBytesConfig')
    def test_load_model_and_tokenizer(self, mock_bnb_config, mock_model_class, mock_tokenizer_class):
        """Test model and tokenizer loading"""
        # Setup mocks
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "</s>"
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        
        mock_model = Mock()
        mock_model_class.from_pretrained.return_value = mock_model
        
        mock_bnb_config.return_value = Mock()
        
        trainer = ModelTrainer(self.config, self.db_path)
        trainer.load_model_and_tokenizer()
        
        # Verify model and tokenizer were loaded
        assert trainer.tokenizer == mock_tokenizer
        assert trainer.model == mock_model
        
        # Verify tokenizer configuration
        assert trainer.tokenizer.pad_token == "</s>"
        
        # Verify model loading called with correct parameters
        mock_model_class.from_pretrained.assert_called_once()
        call_kwargs = mock_model_class.from_pretrained.call_args[1]
        assert call_kwargs["trust_remote_code"] == True
        assert "device_map" in call_kwargs
    
    @patch('app.training.TRAINING_AVAILABLE', True)
    @patch('app.training.LoraConfig')
    @patch('app.training.get_peft_model')
    def test_setup_lora(self, mock_get_peft_model, mock_lora_config):
        """Test LoRA adapter setup"""
        trainer = ModelTrainer(self.config, self.db_path)
        trainer.model = Mock()  # Mock model
        
        mock_lora = Mock()
        mock_lora_config.return_value = mock_lora
        
        mock_peft_model = Mock()
        mock_peft_model.print_trainable_parameters = Mock()
        mock_get_peft_model.return_value = mock_peft_model
        
        trainer.setup_lora()
        
        # Verify LoRA configuration
        mock_lora_config.assert_called_once()
        call_kwargs = mock_lora_config.call_args[1]
        assert call_kwargs["r"] == self.config.lora_r
        assert call_kwargs["lora_alpha"] == self.config.lora_alpha
        assert call_kwargs["target_modules"] == self.config.target_modules
        
        # Verify PEFT model creation
        mock_get_peft_model.assert_called_once_with(trainer.model, mock_lora)
        assert trainer.model == mock_peft_model
        mock_peft_model.print_trainable_parameters.assert_called_once()
    
    def test_create_model_version(self):
        """Test model version creation in database"""
        trainer = ModelTrainer(self.config, self.db_path)
        
        adapter_path = "/path/to/adapter"
        training_config = {"epochs": 3, "lr": 2e-4}
        training_data_count = 50
        metrics = {"loss": 0.5, "accuracy": 0.85}
        
        model_version_id = trainer.create_model_version(
            adapter_path, training_config, training_data_count, metrics
        )
        
        assert isinstance(model_version_id, int)
        
        # Verify database storage
        with DB(self.db_path) as db:
            # Check model version was created
            versions = db.conn.execute("SELECT * FROM model_version WHERE id = ?", (model_version_id,)).fetchall()
            assert len(versions) == 1
            
            version = versions[0]
            assert version["base_model"] == self.config.base_model_name
            assert version["adapter_path"] == adapter_path
            assert version["training_data_count"] == training_data_count
            
            # Check stored configuration includes metrics
            stored_config = json.loads(version["training_config_json"])
            assert "training_config" in stored_config
            assert "training_metrics" in stored_config
            assert stored_config["training_metrics"]["loss"] == 0.5


class TestTrainingManager:
    """Test training manager functionality"""
    
    def setup_method(self):
        """Setup test environment"""
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.db_path = self.temp_db.name
        self.temp_db.close()
        
        # Initialize database schema
        with DB(self.db_path) as db:
            schema_path = Path(__file__).parent.parent / "schema.sql"
            db.init_schema(str(schema_path))
        
        self.manager = TrainingManager(self.db_path)
    
    def teardown_method(self):
        """Clean up test environment"""
        Path(self.db_path).unlink(missing_ok=True)
    
    def test_training_manager_initialization(self):
        """Test training manager initialization"""
        assert self.manager.db_path == self.db_path
    
    @patch('app.training.ModelTrainer')
    def test_start_training_success(self, mock_trainer_class):
        """Test successful training session"""
        # Setup training data
        with DB(self.db_path) as db:
            for i in range(15):  # Sufficient training data
                training_id = db.store_training_data(
                    f"session{i}", f"query{i}", f"response{i}", [], "model"
                )
                db.update_training_feedback(training_id, rating=4)
        
        # Mock trainer
        mock_trainer = Mock()
        mock_trainer.train.return_value = ("/path/to/adapter", {"loss": 0.5})
        mock_trainer.create_model_version.return_value = 123
        mock_trainer_class.return_value = mock_trainer
        
        config = TrainingConfig(min_examples=10)
        result = self.manager.start_training(config)
        
        # Verify successful result
        assert result["status"] == "completed"
        assert "session_id" in result
        assert "model_version_id" in result
        assert result["model_version_id"] == 123
        assert "adapter_path" in result
        assert "training_data_count" in result
        assert "metrics" in result
        
        # Verify trainer was called
        mock_trainer_class.assert_called_once()
        mock_trainer.train.assert_called_once()
        mock_trainer.create_model_version.assert_called_once()
    
    @patch('app.training.ModelTrainer')
    def test_start_training_failure(self, mock_trainer_class):
        """Test failed training session"""
        # Mock trainer that raises exception
        mock_trainer = Mock()
        mock_trainer.train.side_effect = Exception("Training failed")
        mock_trainer_class.return_value = mock_trainer
        
        config = TrainingConfig()
        result = self.manager.start_training(config)
        
        # Verify failure result
        assert result["status"] == "failed"
        assert "error" in result
        assert "Training failed" in result["error"]
    
    def test_get_training_status_empty(self):
        """Test getting training status with no data"""
        status = self.manager.get_training_status()
        
        assert "recent_sessions" in status
        assert "available_models" in status
        assert "training_available" in status
        
        assert len(status["recent_sessions"]) == 0
        assert len(status["available_models"]) == 0
    
    def test_get_training_status_with_data(self):
        """Test getting training status with existing data"""
        # Create test model version
        with DB(self.db_path) as db:
            model_version_id = db.create_model_version(
                base_model="test/model",
                version_name="test_v1",
                adapter_path="/path/to/adapter",
                training_config={"epochs": 3},
                training_data_count=50
            )
            
            # Create test training session
            session_id = db.create_training_session(
                model_version_id=model_version_id,
                training_data_filter="rating >= 3",
                config={"lr": 2e-4}
            )
            db.update_training_session(session_id, "completed", metrics={"loss": 0.5})
        
        status = self.manager.get_training_status()
        
        # Verify data is returned
        assert len(status["recent_sessions"]) == 1
        assert len(status["available_models"]) == 1
        
        # Check session data
        session = status["recent_sessions"][0]
        assert session["id"] == session_id
        assert session["status"] == "completed"
        assert session["version_name"] == "test_v1"
        
        # Check model data
        model = status["available_models"][0]
        assert model["id"] == model_version_id
        assert model["version_name"] == "test_v1"
        assert model["base_model"] == "test/model"


class TestTrainingDataCollection:
    """Test training data collection functionality"""
    
    def setup_method(self):
        """Setup test environment"""
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.db_path = self.temp_db.name
        self.temp_db.close()
        
        # Initialize database schema
        with DB(self.db_path) as db:
            schema_path = Path(__file__).parent.parent / "schema.sql"
            db.init_schema(str(schema_path))
    
    def teardown_method(self):
        """Clean up test environment"""
        Path(self.db_path).unlink(missing_ok=True)
    
    def test_store_training_data(self):
        """Test storing training data"""
        with DB(self.db_path) as db:
            training_id = db.store_training_data(
                session_id="test_session",
                user_query="What is the legal issue?",
                model_response="Based on the case files...",
                context_sources=[{"source": "Case File", "preview": "Legal case content"}],
                model_name="TinyLlama-1.1B"
            )
            
            assert isinstance(training_id, int)
            
            # Verify data was stored
            training_data = db.get_training_data(limit=10)
            assert len(training_data) == 1
            
            data = training_data[0]
            assert data["session_id"] == "test_session"
            assert data["user_query"] == "What is the legal issue?"
            assert data["model_response"] == "Based on the case files..."
            assert data["model_name"] == "TinyLlama-1.1B"
            
            # Check context was stored as JSON
            context = json.loads(data["context_json"])
            assert len(context) == 1
            assert context[0]["source"] == "Case File"
    
    def test_update_training_feedback(self):
        """Test updating training data with feedback"""
        with DB(self.db_path) as db:
            # Store initial training data
            training_id = db.store_training_data(
                "session", "query", "response", [], "model"
            )
            
            # Update with rating only
            db.update_training_feedback(training_id, rating=4)
            
            data = db.get_training_data(limit=1)[0]
            assert data["feedback_rating"] == 4
            assert data["feedback_correction"] is None
            
            # Update with correction
            db.update_training_feedback(training_id, correction="Better response")
            
            data = db.get_training_data(limit=1)[0]
            assert data["feedback_rating"] == 4
            assert data["feedback_correction"] == "Better response"
            
            # Update with both rating and correction
            db.update_training_feedback(training_id, rating=5, correction="Much better response")
            
            data = db.get_training_data(limit=1)[0]
            assert data["feedback_rating"] == 5
            assert data["feedback_correction"] == "Much better response"
    
    def test_get_training_data_filtering(self):
        """Test filtering training data retrieval"""
        with DB(self.db_path) as db:
            # Store training data with different ratings and models
            for i in range(5):
                training_id = db.store_training_data(
                    f"session{i}", f"query{i}", f"response{i}", [], "TinyLlama"
                )
                db.update_training_feedback(training_id, rating=i + 1)  # Ratings 1-5
            
            # Store data for different model
            training_id = db.store_training_data(
                "session_other", "query_other", "response_other", [], "OtherModel"
            )
            db.update_training_feedback(training_id, rating=5)
            
            # Test filtering by minimum rating
            good_data = db.get_training_data(min_rating=4)
            assert len(good_data) == 3  # Ratings 4, 5, and 5 from other model
            
            # Test filtering by model name
            tinyllama_data = db.get_training_data(model_name="TinyLlama")
            assert len(tinyllama_data) == 5
            
            # Test combined filtering
            good_tinyllama_data = db.get_training_data(min_rating=4, model_name="TinyLlama")
            assert len(good_tinyllama_data) == 2  # Only ratings 4 and 5 from TinyLlama
    
    def test_training_data_limit(self):
        """Test training data retrieval limit"""
        with DB(self.db_path) as db:
            # Store more data than default limit
            for i in range(15):
                db.store_training_data(f"session{i}", f"query{i}", f"response{i}", [], "model")
            
            # Test default limit
            data = db.get_training_data()
            assert len(data) <= 1000  # Default limit
            
            # Test custom limit
            limited_data = db.get_training_data(limit=5)
            assert len(limited_data) == 5
            
            # Test higher limit
            all_data = db.get_training_data(limit=20)
            assert len(all_data) == 15  # Only 15 records exist


class TestTrainingIntegration:
    """Integration tests for the complete training workflow"""
    
    def setup_method(self):
        """Setup integration test environment"""
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.db_path = self.temp_db.name
        self.temp_db.close()
        
        # Initialize database schema
        with DB(self.db_path) as db:
            schema_path = Path(__file__).parent.parent / "schema.sql"
            db.init_schema(str(schema_path))
    
    def teardown_method(self):
        """Clean up test environment"""
        Path(self.db_path).unlink(missing_ok=True)
    
    def test_end_to_end_training_workflow(self):
        """Test complete training workflow from data collection to model creation"""
        # Step 1: Collect training data (simulating chat interactions)
        with DB(self.db_path) as db:
            session_interactions = [
                ("What are the charges against Robert Miller?", "Based on the case files, Robert Miller was charged with aggravated assault."),
                ("What procedural defenses are available?", "The case files show a speedy trial motion was filed under Rule 3.191."),
                ("Summarize the key legal issues.", "The main issues are the speedy trial requirements and evidence admissibility."),
                ("What happened in the Miller case?", "Robert Miller was charged with assault and filed procedural motions."),
                ("Explain the legal timeline.", "The case involves charges filed in 2025 with subsequent motions.")
            ]
            
            training_ids = []
            for i, (query, response) in enumerate(session_interactions):
                context = [{"source": f"Legal File {i}", "preview": f"Case content {i}"}]
                training_id = db.store_training_data(
                    session_id=f"session_{i}",
                    user_query=query,
                    model_response=response,
                    context_sources=context,
                    model_name="TinyLlama-1.1B"
                )
                training_ids.append(training_id)
            
            # Step 2: Add user feedback
            for i, training_id in enumerate(training_ids):
                # Give good ratings to most interactions
                rating = 4 if i < 4 else 5
                correction = "Improved response" if i == 0 else None
                db.update_training_feedback(training_id, rating=rating, correction=correction)
        
        # Step 3: Verify training data is ready
        with DB(self.db_path) as db:
            training_data = db.get_training_data(min_rating=3)
            assert len(training_data) == 5
            
            # Check that one has a correction
            corrections = [d for d in training_data if d["feedback_correction"]]
            assert len(corrections) == 1
        
        # Step 4: Test training preparation (without actual model training)
        config = TrainingConfig(min_examples=3, base_model_name="test/model")
        trainer = ModelTrainer(config, self.db_path)
        
        # This should succeed since we have enough data
        dataset = trainer.prepare_training_data(min_rating=3)
        assert len(dataset) == 5
        
        # Verify dataset format
        for example in dataset:
            assert "instruction" in example
            assert "response" in example
            assert "Based on your conversation history" in example["instruction"]
        
        # Step 5: Test model version creation
        adapter_path = "/fake/path/to/adapter"
        training_config = config.__dict__
        metrics = {"training_loss": 0.45, "eval_loss": 0.52}
        
        model_version_id = trainer.create_model_version(
            adapter_path, training_config, len(dataset), metrics
        )
        
        # Step 6: Verify complete training record
        with DB(self.db_path) as db:
            # Check model version was created
            versions = db.conn.execute(
                "SELECT * FROM model_version WHERE id = ?", 
                (model_version_id,)
            ).fetchall()
            assert len(versions) == 1
            
            version = versions[0]
            assert version["base_model"] == "test/model"
            assert version["adapter_path"] == adapter_path
            assert version["training_data_count"] == 5
            
            # Check stored configuration
            stored_config = json.loads(version["training_config_json"])
            assert "training_config" in stored_config
            assert "training_metrics" in stored_config
            assert stored_config["training_metrics"]["training_loss"] == 0.45
    
    @patch('app.training.TRAINING_AVAILABLE', False)
    def test_training_without_dependencies(self):
        """Test training system behavior when dependencies are not available"""
        config = TrainingConfig()
        
        # Should raise ImportError when trying to create trainer
        with pytest.raises(ImportError):
            ModelTrainer(config, self.db_path)
        
        # Training manager should handle gracefully
        manager = TrainingManager(self.db_path)
        result = manager.start_training(config)
        
        assert result["status"] == "failed"
        assert "error" in result