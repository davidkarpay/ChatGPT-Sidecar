"""
Unit tests for model evaluation and benchmarking system
"""

import pytest
import tempfile
import json
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from app.model_evaluation import (
    BenchmarkSuite, ResponseScorer, ModelEvaluator, MonthlyReporter,
    BenchmarkPrompt, EvaluationResult, ModelSnapshot
)
from app.db import DB
from app.fast_llm_agent import FastLLMConfig


class TestBenchmarkSuite:
    """Test benchmark prompt suite functionality"""
    
    def test_benchmark_suite_initialization(self):
        """Test benchmark suite creates standard prompts"""
        suite = BenchmarkSuite()
        
        assert len(suite.prompts) > 0
        assert any(p.category == "anti_hallucination" for p in suite.prompts)
        assert any(p.category == "factual" for p in suite.prompts)
        assert any(p.category == "coherence" for p in suite.prompts)
        assert any(p.category == "context_usage" for p in suite.prompts)
    
    def test_get_prompts_by_category(self):
        """Test filtering prompts by category"""
        suite = BenchmarkSuite()
        
        legal_prompts = suite.get_prompts_by_category("anti_hallucination")
        assert all(p.category == "anti_hallucination" for p in legal_prompts)
        assert len(legal_prompts) > 0
        
        factual_prompts = suite.get_prompts_by_category("factual")
        assert all(p.category == "factual" for p in factual_prompts)
    
    def test_get_prompt_by_id(self):
        """Test retrieving specific prompt by ID"""
        suite = BenchmarkSuite()
        
        prompt = suite.get_prompt_by_id("legal_case_analysis")
        assert prompt is not None
        assert prompt.id == "legal_case_analysis"
        assert prompt.category == "anti_hallucination"
        
        # Test non-existent prompt
        assert suite.get_prompt_by_id("nonexistent") is None
    
    def test_benchmark_prompt_structure(self):
        """Test that benchmark prompts have required structure"""
        suite = BenchmarkSuite()
        
        for prompt in suite.prompts:
            assert prompt.id
            assert prompt.category
            assert prompt.prompt
            assert isinstance(prompt.context, list)
            assert isinstance(prompt.expected_qualities, list)
            assert isinstance(prompt.scoring_criteria, dict)


class TestResponseScorer:
    """Test response scoring functionality"""
    
    def setup_method(self):
        """Setup test scorer"""
        self.scorer = ResponseScorer()
        
        # Create test benchmark prompt
        self.test_prompt = BenchmarkPrompt(
            id="test_prompt",
            category="anti_hallucination",
            prompt="Test question about case facts",
            context=[{
                "source": "Test Case",
                "preview": "Robert Miller was charged with assault in 2025",
                "doc_id": 1,
                "chunk_id": 1
            }],
            expected_qualities=["factual accuracy"],
            scoring_criteria={"accuracy": "No invented facts"}
        )
    
    def test_basic_metrics(self):
        """Test basic response metrics calculation"""
        response = "This is a test response with multiple sentences. It has specific word counts."
        
        scores = self.scorer._basic_metrics(response)
        
        assert "response_length" in scores
        assert "word_count" in scores
        assert "sentence_count" in scores
        assert "avg_word_length" in scores
        
        assert scores["response_length"] == len(response)
        assert scores["word_count"] == len(response.split())
        assert scores["sentence_count"] > 0
        assert scores["avg_word_length"] > 0
    
    def test_anti_hallucination_scoring(self):
        """Test anti-hallucination scoring"""
        # Good response - stays within context
        good_response = "Based on the case file, Robert Miller was charged with assault in 2025."
        scores_good = self.scorer._anti_hallucination_score(good_response, self.test_prompt)
        
        assert "anti_hallucination_score" in scores_good
        assert scores_good["anti_hallucination_score"] > 0.5
        
        # Bad response - adds fabricated details
        bad_response = "According to studies, Robert Miller, age 28, was charged on January 15th, 2017 with aggravated assault involving $5000 in damages."
        scores_bad = self.scorer._anti_hallucination_score(bad_response, self.test_prompt)
        
        assert scores_bad["fabrication_indicators"] > 0
        assert scores_bad["invented_specifics"] > 0
        assert scores_bad["anti_hallucination_score"] < scores_good["anti_hallucination_score"]
    
    def test_factual_accuracy_scoring(self):
        """Test factual accuracy scoring"""
        response = "Robert Miller assault case charge"
        scores = self.scorer._factual_accuracy_score(response, self.test_prompt)
        
        assert "factual_accuracy_score" in scores
        assert "context_keyword_overlap" in scores
        assert 0 <= scores["factual_accuracy_score"] <= 1
    
    def test_coherence_scoring(self):
        """Test coherence scoring"""
        coherent_response = "This is a coherent response. It flows logically. Each sentence builds on the previous."
        scores = self.scorer._coherence_score(coherent_response, self.test_prompt)
        
        assert "coherence_score" in scores
        assert "avg_sentence_length" in scores
        assert 0 <= scores["coherence_score"] <= 1
    
    def test_context_usage_scoring(self):
        """Test context usage scoring"""
        response_using_context = "Robert Miller was charged with assault"
        scores = self.scorer._context_usage_score(response_using_context, self.test_prompt)
        
        assert "context_usage_score" in scores
        assert "contexts_referenced" in scores
        assert 0 <= scores["context_usage_score"] <= 1
        assert scores["contexts_referenced"] > 0
    
    def test_score_response_integration(self):
        """Test full response scoring integration"""
        response = "Based on the provided case file, Robert Miller was charged with assault in 2025."
        scores = self.scorer.score_response(response, self.test_prompt)
        
        # Should have scores from all relevant categories
        assert "response_length" in scores
        assert "anti_hallucination_score" in scores
        assert isinstance(scores, dict)
        assert all(isinstance(v, (int, float)) for v in scores.values())


class TestModelEvaluator:
    """Test model evaluation functionality"""
    
    def setup_method(self):
        """Setup test evaluator with temporary database"""
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.db_path = self.temp_db.name
        self.temp_db.close()
        
        # Initialize database schema
        with DB(self.db_path) as db:
            schema_path = Path(__file__).parent.parent / "schema.sql"
            db.init_schema(str(schema_path))
        
        self.evaluator = ModelEvaluator(self.db_path)
    
    def teardown_method(self):
        """Clean up temporary database"""
        Path(self.db_path).unlink(missing_ok=True)
    
    @patch('app.model_evaluation.FastLLMAgent')
    def test_run_full_evaluation(self, mock_agent_class):
        """Test full evaluation workflow"""
        # Mock the agent and its responses
        mock_agent = Mock()
        mock_agent.generate_response.return_value = "Test response from model"
        mock_agent_class.return_value = mock_agent
        
        # Create test model config
        config = FastLLMConfig(preset="ultra_fast", db_path=self.db_path)
        model_version_id = 1
        
        # Run evaluation
        result = self.evaluator.run_full_evaluation(config, model_version_id)
        
        # Verify results structure
        assert "model_version_id" in result
        assert "evaluation_results" in result
        assert "aggregate_metrics" in result
        assert "snapshot" in result
        assert result["model_version_id"] == model_version_id
        
        # Verify evaluation results were generated
        assert len(result["evaluation_results"]) > 0
        for eval_result in result["evaluation_results"]:
            assert isinstance(eval_result, EvaluationResult)
            assert eval_result.model_version == str(model_version_id)
            assert eval_result.response == "Test response from model"
            assert isinstance(eval_result.scores, dict)
        
        # Verify aggregate metrics
        assert isinstance(result["aggregate_metrics"], dict)
        
        # Verify agent was called for each benchmark prompt
        expected_calls = len(self.evaluator.benchmark_suite.prompts)
        assert mock_agent.generate_response.call_count == expected_calls
    
    def test_calculate_aggregate_metrics(self):
        """Test aggregate metrics calculation"""
        # Create test category scores
        category_scores = {
            "anti_hallucination": [
                {"anti_hallucination_score": 0.8, "fabrication_indicators": 0},
                {"anti_hallucination_score": 0.9, "fabrication_indicators": 1}
            ],
            "factual": [
                {"factual_accuracy_score": 0.7, "context_keyword_overlap": 0.6}
            ]
        }
        
        metrics = self.evaluator._calculate_aggregate_metrics(category_scores)
        
        # Should have mean values for each score type
        assert "anti_hallucination_anti_hallucination_score_mean" in metrics
        assert "anti_hallucination_fabrication_indicators_mean" in metrics
        assert "factual_factual_accuracy_score_mean" in metrics
        
        # Check calculated values
        assert metrics["anti_hallucination_anti_hallucination_score_mean"] == 0.85
        assert metrics["anti_hallucination_fabrication_indicators_mean"] == 0.5
        assert metrics["factual_factual_accuracy_score_mean"] == 0.7
        
        # Should have overall performance metrics
        assert "overall_performance_mean" in metrics
    
    def test_store_evaluation_results(self):
        """Test storing evaluation results in database"""
        # Create test evaluation results
        results = [
            EvaluationResult(
                prompt_id="test_prompt_1",
                model_version="1", 
                response="Test response 1",
                scores={"score1": 0.8, "score2": 0.7},
                metadata={"category": "test"},
                evaluated_at=datetime.now()
            ),
            EvaluationResult(
                prompt_id="test_prompt_2",
                model_version="1",
                response="Test response 2", 
                scores={"score1": 0.9, "score2": 0.6},
                metadata={"category": "test"},
                evaluated_at=datetime.now()
            )
        ]
        
        aggregate_metrics = {"overall_score": 0.75}
        
        # Store results
        self.evaluator._store_evaluation_results(1, results, aggregate_metrics)
        
        # Verify storage in database
        with DB(self.db_path) as db:
            # Check evaluation results
            stored_results = db.get_evaluation_results(model_version_id=1)
            assert len(stored_results) == 2
            
            # Check model snapshot
            snapshots = db.get_model_snapshots(model_version_id=1)
            assert len(snapshots) == 1
            
            snapshot = snapshots[0]
            performance_metrics = json.loads(snapshot["performance_metrics_json"])
            assert performance_metrics["overall_score"] == 0.75


class TestMonthlyReporter:
    """Test monthly reporting functionality"""
    
    def setup_method(self):
        """Setup test reporter with temporary database"""
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.db_path = self.temp_db.name
        self.temp_db.close()
        
        # Initialize database schema
        with DB(self.db_path) as db:
            schema_path = Path(__file__).parent.parent / "schema.sql"
            db.init_schema(str(schema_path))
        
        self.reporter = MonthlyReporter(self.db_path)
    
    def teardown_method(self):
        """Clean up temporary database"""
        Path(self.db_path).unlink(missing_ok=True)
    
    def test_generate_monthly_report_structure(self):
        """Test monthly report generation structure"""
        target_date = datetime(2025, 8, 15)
        
        report = self.reporter.generate_monthly_report(target_date)
        
        # Verify report structure
        assert "report_date" in report
        assert "period" in report
        assert "training_statistics" in report
        assert "performance_comparison" in report
        assert "recommendations" in report
        assert "generated_at" in report
        
        assert report["period"] == "2025-08"
        assert isinstance(report["recommendations"], list)
    
    def test_get_monthly_training_stats(self):
        """Test monthly training statistics collection"""
        target_date = datetime(2025, 8, 15)
        
        # Add some test training data
        with DB(self.db_path) as db:
            # Add training data for the month
            db.store_training_data(
                session_id="test_session_1",
                user_query="Test query 1",
                model_response="Test response 1",
                context_sources=[],
                model_name="test_model"
            )
            
            # Add training data with feedback
            training_id = db.store_training_data(
                session_id="test_session_2", 
                user_query="Test query 2",
                model_response="Test response 2",
                context_sources=[],
                model_name="test_model"
            )
            db.update_training_feedback(training_id, rating=4)
        
        stats = self.reporter._get_monthly_training_stats(target_date)
        
        # Verify statistics structure
        assert "period_start" in stats
        assert "period_end" in stats
        assert "total_interactions" in stats
        assert "interactions_with_feedback" in stats
        assert "average_rating" in stats
        assert "feedback_distribution" in stats
        
        # Verify calculated values
        assert stats["total_interactions"] == 2
        assert stats["interactions_with_feedback"] == 1
        assert stats["average_rating"] == 4.0
    
    def test_generate_recommendations(self):
        """Test recommendation generation logic"""
        # Test low interaction scenario
        training_stats = {
            "total_interactions": 10,
            "interactions_with_feedback": 1,
            "average_rating": 2.0
        }
        performance_comparison = {}
        
        recommendations = self.reporter._generate_recommendations(training_stats, performance_comparison)
        
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        
        # Should recommend more interactions and feedback
        recommendation_text = " ".join(recommendations).lower()
        assert "interaction" in recommendation_text or "usage" in recommendation_text
        assert "feedback" in recommendation_text
        assert "rating" in recommendation_text or "improve" in recommendation_text
    
    def test_store_monthly_report(self):
        """Test storing monthly report in database"""
        target_date = datetime(2025, 8, 15)
        test_report = {
            "report_date": target_date.isoformat(),
            "period": "2025-08",
            "training_statistics": {"total_interactions": 50},
            "performance_comparison": {"improvement": 10},
            "recommendations": ["Keep up good work"],
            "generated_at": datetime.now().isoformat()
        }
        
        self.reporter._store_monthly_report(target_date, test_report)
        
        # Verify storage
        with DB(self.db_path) as db:
            stored_report = db.get_monthly_checkin("2025-08")
            
            assert stored_report is not None
            assert stored_report["report_period"] == "2025-08"
            
            # Check stored data
            training_stats = json.loads(stored_report["training_stats_json"])
            assert training_stats["total_interactions"] == 50
            
            recommendations = json.loads(stored_report["recommendations_json"])
            assert recommendations == ["Keep up good work"]


class TestDatabaseIntegration:
    """Test database integration for evaluation system"""
    
    def setup_method(self):
        """Setup test database"""
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.db_path = self.temp_db.name
        self.temp_db.close()
        
        # Initialize database schema
        with DB(self.db_path) as db:
            schema_path = Path(__file__).parent.parent / "schema.sql"
            db.init_schema(str(schema_path))
    
    def teardown_method(self):
        """Clean up temporary database"""
        Path(self.db_path).unlink(missing_ok=True)
    
    def test_benchmark_prompt_storage(self):
        """Test storing and retrieving benchmark prompts"""
        with DB(self.db_path) as db:
            # Store test prompt
            db.store_benchmark_prompt(
                prompt_id="test_prompt",
                category="test",
                prompt="Test question",
                context=[{"source": "test", "preview": "test content"}],
                expected_qualities=["accuracy"],
                scoring_criteria={"accuracy": "Must be accurate"}
            )
            
            # Retrieve prompts
            prompts = db.get_benchmark_prompts()
            assert len(prompts) == 1
            
            prompt = prompts[0]
            assert prompt["id"] == "test_prompt"
            assert prompt["category"] == "test"
            assert prompt["prompt"] == "Test question"
            
            # Test category filtering
            test_prompts = db.get_benchmark_prompts(category="test")
            assert len(test_prompts) == 1
            
            nonexistent_prompts = db.get_benchmark_prompts(category="nonexistent")
            assert len(nonexistent_prompts) == 0
    
    def test_evaluation_result_storage(self):
        """Test storing and retrieving evaluation results"""
        with DB(self.db_path) as db:
            # First create a benchmark prompt
            db.store_benchmark_prompt(
                prompt_id="test_prompt",
                category="test",
                prompt="Test question",
                context=[],
                expected_qualities=[],
                scoring_criteria={}
            )
            
            # Store evaluation result
            result_id = db.store_evaluation_result(
                prompt_id="test_prompt",
                model_version_id=1,
                response="Test response",
                scores={"score1": 0.8, "score2": 0.7},
                metadata={"category": "test"}
            )
            
            assert isinstance(result_id, int)
            
            # Retrieve results
            results = db.get_evaluation_results(model_version_id=1)
            assert len(results) == 1
            
            result = results[0]
            assert result["prompt_id"] == "test_prompt"
            assert result["model_version_id"] == 1
            assert result["response"] == "Test response"
            
            scores = json.loads(result["scores_json"])
            assert scores["score1"] == 0.8
            assert scores["score2"] == 0.7
    
    def test_model_snapshot_storage(self):
        """Test storing and retrieving model snapshots"""
        with DB(self.db_path) as db:
            # Store snapshot
            snapshot_id = db.store_model_snapshot(
                model_version_id=1,
                parameter_stats={"param1": 0.5, "param2": 1.2},
                embedding_analysis={"dim": 384, "mean": 0.1},
                performance_metrics={"overall": 0.8, "accuracy": 0.85}
            )
            
            assert isinstance(snapshot_id, int)
            
            # Retrieve snapshots
            snapshots = db.get_model_snapshots(model_version_id=1)
            assert len(snapshots) == 1
            
            snapshot = snapshots[0]
            assert snapshot["model_version_id"] == 1
            
            # Check stored data
            parameter_stats = json.loads(snapshot["parameter_stats_json"])
            assert parameter_stats["param1"] == 0.5
            
            performance_metrics = json.loads(snapshot["performance_metrics_json"])
            assert performance_metrics["overall"] == 0.8