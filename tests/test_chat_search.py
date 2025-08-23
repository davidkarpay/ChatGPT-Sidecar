"""
Tests for chat search functionality and RAG pipeline components
"""
import pytest
import numpy as np
from unittest.mock import patch, Mock, MagicMock
from typing import List, Dict, Any

from app.rag_pipeline import RAGPipeline
from app.vectorstore import FaissStore
from app.mmr import mmr
from tests.fixtures.mock_data import MockConversations, MockEmbeddings
from tests.fixtures.mock_agents import MockFaissStore, MockDatabase


class TestSearchModes:
    """Test different search modes in RAG pipeline"""
    
    @pytest.fixture
    def mock_pipeline(self):
        """Create RAG pipeline with mocked components"""
        with patch('app.rag_pipeline.ChatAgent'), \
             patch('app.rag_pipeline.RAGPipeline._load_stores'):
            pipeline = RAGPipeline("test.db", "test-model")
            
            # Setup mock stores
            pipeline.main_store = MockFaissStore()
            pipeline.precision_store = MockFaissStore()
            pipeline.balanced_store = MockFaissStore()
            pipeline.context_store = MockFaissStore()
            
            return pipeline
    
    def test_adaptive_search_short_query(self, mock_pipeline):
        """Test adaptive search with short query (should use precision store)"""
        short_query = "ML basics"  # 2 words, should trigger precision search
        
        with patch.object(mock_pipeline, '_search_store_with_fallback') as mock_search:
            mock_search.return_value = MockConversations.get_sample_search_results()
            
            results = mock_pipeline._adaptive_search(short_query, k=5)
            
            # Should call precision store
            mock_search.assert_called_once_with(
                mock_pipeline.precision_store, "precision", short_query, 5
            )
            assert len(results) > 0
    
    def test_adaptive_search_medium_query(self, mock_pipeline):
        """Test adaptive search with medium query (should use balanced store)"""
        medium_query = "What are the main concepts in machine learning"  # 9 words
        
        with patch.object(mock_pipeline, '_search_store_with_fallback') as mock_search:
            mock_search.return_value = MockConversations.get_sample_search_results()
            
            results = mock_pipeline._adaptive_search(medium_query, k=5)
            
            # Should call balanced store
            mock_search.assert_called_once_with(
                mock_pipeline.balanced_store, "balanced", medium_query, 5
            )
            assert len(results) > 0
    
    def test_adaptive_search_long_query(self, mock_pipeline):
        """Test adaptive search with long query (should use context store)"""
        long_query = "Please explain in detail the fundamental concepts and practical applications of machine learning in modern software development"  # 18 words
        
        with patch.object(mock_pipeline, '_search_store_with_fallback') as mock_search:
            mock_search.return_value = MockConversations.get_sample_search_results()
            
            results = mock_pipeline._adaptive_search(long_query, k=5)
            
            # Should call context store
            mock_search.assert_called_once_with(
                mock_pipeline.context_store, "context", long_query, 5
            )
            assert len(results) > 0
    
    def test_multi_layer_search(self, mock_pipeline):
        """Test multi-layer search combining all stores"""
        query = "machine learning applications"
        
        with patch.object(mock_pipeline, '_search_store_with_fallback') as mock_search:
            # Mock each store returning different results
            mock_search.side_effect = [
                [{"rank": 1, "text": "precision result"}],
                [{"rank": 2, "text": "balanced result"}], 
                [{"rank": 3, "text": "context result"}]
            ]
            
            results = mock_pipeline._multi_layer_search(query, k=6)
            
            # Should call all three stores
            assert mock_search.call_count == 3
            
            # Verify calls to each store
            calls = mock_search.call_args_list
            assert calls[0][0][1] == "precision"  # First call to precision store
            assert calls[1][0][1] == "balanced"   # Second call to balanced store  
            assert calls[2][0][1] == "context"    # Third call to context store
            
            assert len(results) <= 6  # Should respect k limit
    
    def test_basic_search_without_mmr(self, mock_pipeline):
        """Test basic search without MMR diversity"""
        query = "test query"
        
        # Mock the main store search
        mock_pipeline.main_store.search.return_value = [
            (0, 0.95), (1, 0.87), (2, 0.82)
        ]
        
        with patch('app.rag_pipeline.DB') as mock_db_class:
            mock_db = Mock()
            mock_db.__enter__.return_value = mock_db
            mock_db.fetch_chunks_by_faiss_indices.return_value = {
                0: {"text": "result 1", "title": "Doc 1", "doc_id": 1, "chunk_id": 1, "start_char": 0, "end_char": 100},
                1: {"text": "result 2", "title": "Doc 2", "doc_id": 2, "chunk_id": 2, "start_char": 0, "end_char": 100},
                2: {"text": "result 3", "title": "Doc 3", "doc_id": 3, "chunk_id": 3, "start_char": 0, "end_char": 100}
            }
            mock_db_class.return_value = mock_db
            
            results = mock_pipeline._basic_search(query, k=3, use_mmr=False, lambda_param=0.5)
            
            assert len(results) == 3
            assert results[0]["rank"] == 1
            assert results[0]["score"] == 0.95
    
    def test_basic_search_with_mmr(self, mock_pipeline):
        """Test basic search with MMR diversity"""
        query = "test query"
        
        # Mock the main store search with more candidates
        mock_pipeline.main_store.search.return_value = [
            (i, 0.95 - i*0.02) for i in range(10)  # 10 candidates
        ]
        
        with patch('app.rag_pipeline.DB') as mock_db_class, \
             patch('app.rag_pipeline.mmr') as mock_mmr:
            
            mock_db = Mock()
            mock_db.__enter__.return_value = mock_db
            mock_db.fetch_chunks_by_faiss_indices.return_value = {
                i: {"text": f"result {i}", "title": f"Doc {i}", "doc_id": i, 
                    "chunk_id": i, "start_char": 0, "end_char": 100}
                for i in range(10)
            }
            mock_db_class.return_value = mock_db
            
            # Mock MMR to return first 3 indices
            mock_mmr.return_value = [0, 2, 4]
            
            # Mock encoding
            mock_pipeline.main_store.encode.return_value = np.random.rand(10, 384)
            
            results = mock_pipeline._basic_search(query, k=3, use_mmr=True, lambda_param=0.6)
            
            assert len(results) == 3
            mock_mmr.assert_called_once()
            
            # Verify MMR was called with correct parameters
            call_args = mock_mmr.call_args
            assert call_args[1]["lamb"] == 0.6
            assert call_args[1]["k"] == 3


class TestSearchFallbacks:
    """Test search fallback behavior when stores are unavailable"""
    
    @pytest.fixture
    def pipeline_with_missing_stores(self):
        """Create pipeline with some missing stores"""
        with patch('app.rag_pipeline.ChatAgent'), \
             patch('app.rag_pipeline.RAGPipeline._load_stores'):
            pipeline = RAGPipeline("test.db", "test-model")
            
            # Only main store available
            pipeline.main_store = MockFaissStore()
            pipeline.precision_store = None
            pipeline.balanced_store = None
            pipeline.context_store = None
            
            return pipeline
    
    def test_adaptive_search_fallback_to_basic(self, pipeline_with_missing_stores):
        """Test adaptive search fallback when specialized stores missing"""
        query = "short query"  # Should use precision, but it's None
        
        with patch.object(pipeline_with_missing_stores, '_basic_search') as mock_basic:
            mock_basic.return_value = MockConversations.get_sample_search_results()
            
            results = pipeline_with_missing_stores._adaptive_search(query, k=5)
            
            # Should fallback to basic search
            mock_basic.assert_called_once_with(query, 5, True, 0.6)
            assert len(results) > 0
    
    def test_multi_layer_search_with_missing_stores(self, pipeline_with_missing_stores):
        """Test multi-layer search when some stores are missing"""
        query = "test query"
        
        with patch.object(pipeline_with_missing_stores, '_basic_search') as mock_basic:
            mock_basic.return_value = MockConversations.get_sample_search_results()
            
            results = pipeline_with_missing_stores._multi_layer_search(query, k=6)
            
            # Should fallback to basic search since no specialized stores available
            mock_basic.assert_called_once()
            assert len(results) > 0
    
    def test_search_store_with_fallback_none_store(self, pipeline_with_missing_stores):
        """Test search with None store falls back to basic search"""
        with patch.object(pipeline_with_missing_stores, '_basic_search') as mock_basic:
            mock_basic.return_value = [{"text": "fallback result"}]
            
            results = pipeline_with_missing_stores._search_store_with_fallback(
                None, "missing_store", "test query", 5
            )
            
            mock_basic.assert_called_once_with("test query", 5, True, 0.6)
            assert len(results) == 1


class TestContextRetrieval:
    """Test context retrieval and enrichment"""
    
    @pytest.fixture
    def mock_pipeline(self):
        with patch('app.rag_pipeline.ChatAgent'), \
             patch('app.rag_pipeline.RAGPipeline._load_stores'):
            pipeline = RAGPipeline("test.db", "test-model")
            pipeline.main_store = MockFaissStore()
            return pipeline
    
    def test_search_with_context_basic_mode(self, mock_pipeline):
        """Test search with context in basic mode"""
        query = "machine learning"
        
        with patch.object(mock_pipeline, '_basic_search') as mock_basic, \
             patch.object(mock_pipeline, '_enrich_results') as mock_enrich:
            
            mock_basic.return_value = MockConversations.get_sample_search_results()
            mock_enrich.return_value = MockConversations.get_sample_search_results()
            
            results = mock_pipeline.search_with_context(
                query, k=5, search_mode="basic"
            )
            
            mock_basic.assert_called_once_with(query, 5, True, 0.6)
            mock_enrich.assert_called_once()
            assert len(results) > 0
    
    def test_search_with_context_adaptive_mode(self, mock_pipeline):
        """Test search with context in adaptive mode"""
        query = "what is machine learning"
        
        with patch.object(mock_pipeline, '_adaptive_search') as mock_adaptive, \
             patch.object(mock_pipeline, '_enrich_results') as mock_enrich:
            
            mock_adaptive.return_value = MockConversations.get_sample_search_results()
            mock_enrich.return_value = MockConversations.get_sample_search_results()
            
            results = mock_pipeline.search_with_context(
                query, k=5, search_mode="adaptive"
            )
            
            mock_adaptive.assert_called_once_with(query, 5)
            mock_enrich.assert_called_once()
            assert len(results) > 0
    
    def test_search_with_context_multi_layer_mode(self, mock_pipeline):
        """Test search with context in multi-layer mode"""
        query = "explain neural networks"
        
        with patch.object(mock_pipeline, '_multi_layer_search') as mock_multi, \
             patch.object(mock_pipeline, '_enrich_results') as mock_enrich:
            
            mock_multi.return_value = MockConversations.get_sample_search_results()
            mock_enrich.return_value = MockConversations.get_sample_search_results()
            
            results = mock_pipeline.search_with_context(
                query, k=5, search_mode="multi_layer"
            )
            
            mock_multi.assert_called_once_with(query, 5)
            mock_enrich.assert_called_once()
            assert len(results) > 0
    
    def test_enrich_results(self, mock_pipeline):
        """Test result enrichment with context JSON"""
        raw_results = [
            {
                "source": "Test Document",
                "doc_id": 1,
                "chunk_id": 10,
                "start_char": 0,
                "end_char": 100,
                "text": "This is test content"
            }
        ]
        
        enriched = mock_pipeline._enrich_results(raw_results)
        
        assert len(enriched) == 1
        result = enriched[0]
        
        # Check context_json was added
        assert "context_json" in result
        assert "context" in result["context_json"]
        
        context = result["context_json"]["context"][0]
        assert context["doc"] == "Test Document"
        assert context["loc"]["doc_id"] == 1
        assert context["loc"]["chunk_id"] == 10
        assert context["quote"] == "This is test content"
    
    def test_create_preview(self, mock_pipeline):
        """Test preview text creation"""
        # Short text
        short_text = "This is short"
        preview = mock_pipeline._create_preview(short_text)
        assert preview == short_text
        
        # Long text
        long_text = "This is a very long text " * 20  # > 320 chars
        preview = mock_pipeline._create_preview(long_text)
        assert len(preview) <= 321  # 320 + "…"
        assert preview.endswith("…")
        
        # Custom max length
        custom_preview = mock_pipeline._create_preview(long_text, max_length=50)
        assert len(custom_preview) <= 51  # 50 + "…"


class TestMMRIntegration:
    """Test MMR (Maximum Marginal Relevance) integration"""
    
    def test_mmr_basic_functionality(self):
        """Test basic MMR functionality"""
        # Create simple test vectors
        query_vec = np.array([1.0, 0.0], dtype=np.float32)
        candidate_vecs = np.array([
            [1.0, 0.0],    # Most similar to query
            [0.9, 0.1],    # Also similar
            [0.0, 1.0],    # Orthogonal (diverse)
            [0.8, 0.2]     # Similar but less than first two
        ], dtype=np.float32)
        
        # Test with balanced lambda
        selected = mmr(query_vec, candidate_vecs, lamb=0.5, k=2)
        
        assert len(selected) == 2
        assert 0 in selected  # Should include most relevant
        assert 2 in selected  # Should include diverse option
    
    def test_mmr_relevance_only(self):
        """Test MMR with lambda=1 (relevance only)"""
        query_vec = np.array([1.0, 0.0], dtype=np.float32)
        candidate_vecs = np.array([
            [0.9, 0.1],    # Second most similar
            [1.0, 0.0],    # Most similar
            [0.0, 1.0],    # Least similar
            [0.8, 0.2]     # Third most similar
        ], dtype=np.float32)
        
        selected = mmr(query_vec, candidate_vecs, lamb=1.0, k=3)
        
        # Should select by relevance only: indices 1, 0, 3
        assert len(selected) == 3
        assert selected[0] == 1  # Most relevant
        assert selected[1] == 0  # Second most relevant
        assert selected[2] == 3  # Third most relevant
    
    def test_mmr_diversity_only(self):
        """Test MMR with lambda=0 (diversity only)"""
        query_vec = np.array([1.0, 0.0], dtype=np.float32)
        candidate_vecs = np.array([
            [1.0, 0.0],    # Same as query
            [0.9, 0.1],    # Similar to query
            [0.0, 1.0],    # Orthogonal
            [-1.0, 0.0]    # Opposite direction
        ], dtype=np.float32)
        
        selected = mmr(query_vec, candidate_vecs, lamb=0.0, k=2)
        
        assert len(selected) == 2
        # Should pick diverse options (exact behavior depends on MMR implementation)
        # At minimum, should not pick all similar vectors
        similarities = [np.dot(query_vec, candidate_vecs[i]) for i in selected]
        assert not all(sim > 0.8 for sim in similarities)
    
    def test_mmr_empty_candidates(self):
        """Test MMR with empty candidate set"""
        query_vec = np.array([1.0, 0.0], dtype=np.float32)
        candidate_vecs = np.array([], dtype=np.float32).reshape(0, 2)
        
        selected = mmr(query_vec, candidate_vecs, lamb=0.5, k=5)
        
        assert selected == []
    
    def test_mmr_single_candidate(self):
        """Test MMR with single candidate"""
        query_vec = np.array([1.0, 0.0], dtype=np.float32)
        candidate_vecs = np.array([[0.8, 0.2]], dtype=np.float32)
        
        selected = mmr(query_vec, candidate_vecs, lamb=0.5, k=3)
        
        assert selected == [0]  # Should select the only candidate


class TestSearchErrorHandling:
    """Test error handling in search functionality"""
    
    @pytest.fixture
    def mock_pipeline(self):
        with patch('app.rag_pipeline.ChatAgent'), \
             patch('app.rag_pipeline.RAGPipeline._load_stores'):
            pipeline = RAGPipeline("test.db", "test-model")
            pipeline.main_store = MockFaissStore()
            return pipeline
    
    def test_search_with_database_error(self, mock_pipeline):
        """Test search behavior when database operations fail"""
        query = "test query"
        
        # Mock search results
        mock_pipeline.main_store.search.return_value = [(0, 0.95), (1, 0.87)]
        
        with patch('app.rag_pipeline.DB') as mock_db_class:
            mock_db = Mock()
            mock_db.__enter__.return_value = mock_db
            mock_db.fetch_chunks_by_faiss_indices.side_effect = Exception("DB Error")
            mock_db_class.return_value = mock_db
            
            # Should raise the database exception
            with pytest.raises(Exception, match="DB Error"):
                mock_pipeline._basic_search(query, k=5, use_mmr=False, lambda_param=0.5)
    
    def test_search_with_faiss_error(self, mock_pipeline):
        """Test search behavior when FAISS operations fail"""
        query = "test query"
        
        # Mock FAISS search failure
        mock_pipeline.main_store.search.side_effect = Exception("FAISS Error")
        
        with pytest.raises(Exception, match="FAISS Error"):
            mock_pipeline._basic_search(query, k=5, use_mmr=False, lambda_param=0.5)
    
    def test_search_with_no_results(self, mock_pipeline):
        """Test search behavior when no results are found"""
        query = "nonexistent query"
        
        # Mock empty search results
        mock_pipeline.main_store.search.return_value = []
        
        results = mock_pipeline._basic_search(query, k=5, use_mmr=False, lambda_param=0.5)
        
        assert results == []
    
    def test_search_with_partial_database_results(self, mock_pipeline):
        """Test search when database returns partial results"""
        query = "test query"
        
        # Mock search results
        mock_pipeline.main_store.search.return_value = [(0, 0.95), (1, 0.87), (2, 0.82)]
        
        with patch('app.rag_pipeline.DB') as mock_db_class:
            mock_db = Mock()
            mock_db.__enter__.return_value = mock_db
            # Only return results for indices 0 and 2 (missing 1)
            mock_db.fetch_chunks_by_faiss_indices.return_value = {
                0: {"text": "result 0", "title": "Doc 0", "doc_id": 0, "chunk_id": 0, "start_char": 0, "end_char": 100},
                2: {"text": "result 2", "title": "Doc 2", "doc_id": 2, "chunk_id": 2, "start_char": 0, "end_char": 100}
            }
            mock_db_class.return_value = mock_db
            
            results = mock_pipeline._basic_search(query, k=5, use_mmr=False, lambda_param=0.5)
            
            # Should only return results for available database entries
            assert len(results) == 2
            assert results[0]["text"] == "result 0"
            assert results[1]["text"] == "result 2"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])