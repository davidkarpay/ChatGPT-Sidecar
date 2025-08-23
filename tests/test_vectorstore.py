"""Tests for FAISS vector store operations."""

import pytest
import numpy as np
import tempfile
import os
import pickle
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import faiss
from sentence_transformers import SentenceTransformer

from app.vectorstore import FaissStore


class TestFaissStoreInitialization:
    """Test FaissStore initialization and configuration."""
    
    @pytest.fixture
    def temp_paths(self):
        """Create temporary paths for index and IDs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            index_path = Path(tmpdir) / "test.faiss"
            ids_path = Path(tmpdir) / "test.pkl"
            yield index_path, ids_path
    
    def test_initialization(self, temp_paths):
        """Test FaissStore initialization."""
        index_path, ids_path = temp_paths
        
        with patch('app.vectorstore.SentenceTransformer') as mock_model:
            store = FaissStore(index_path, ids_path, "test-model")
            
            assert store.index_path == index_path
            assert store.ids_path == ids_path
            assert store.index is None
            assert store.ids == []
            mock_model.assert_called_once_with("test-model")
    
    def test_load_nonexistent(self, temp_paths):
        """Test loading when index doesn't exist."""
        index_path, ids_path = temp_paths
        
        with patch('app.vectorstore.SentenceTransformer'):
            store = FaissStore(index_path, ids_path, "test-model")
            store.load()
            
            assert store.index is None
            assert store.ids == []
    
    def test_load_existing(self, temp_paths):
        """Test loading existing index and IDs."""
        index_path, ids_path = temp_paths
        
        # Create dummy index and IDs
        dim = 384
        index = faiss.IndexFlatIP(dim)
        test_vec = np.random.randn(1, dim).astype('float32')
        test_vec = test_vec / np.linalg.norm(test_vec, axis=1, keepdims=True)
        index.add(test_vec)
        faiss.write_index(index, str(index_path))
        
        test_ids = [(1, 100), (2, 200)]
        with open(ids_path, 'wb') as f:
            pickle.dump(test_ids, f)
        
        # Load them
        with patch('app.vectorstore.SentenceTransformer'):
            store = FaissStore(index_path, ids_path, "test-model")
            store.load()
            
            assert store.index is not None
            assert store.index.ntotal == 1
            assert store.ids == test_ids


class TestFaissStoreEncoding:
    """Test text encoding functionality."""
    
    @pytest.fixture
    def temp_paths(self):
        """Create temporary paths for index and IDs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            index_path = Path(tmpdir) / "test.faiss"
            ids_path = Path(tmpdir) / "test.pkl"
            yield index_path, ids_path
    
    @pytest.fixture
    def store(self, temp_paths):
        """Create a FaissStore instance with mocked model."""
        index_path, ids_path = temp_paths
        
        with patch('app.vectorstore.SentenceTransformer') as mock_model_class:
            mock_model = MagicMock()
            mock_model_class.return_value = mock_model
            
            # Mock encoding
            def mock_encode(texts, batch_size=None, show_progress_bar=None, normalize_embeddings=None):
                n = len(texts) if isinstance(texts, list) else 1
                return np.random.randn(n, 384).astype('float32')
            
            mock_model.encode = mock_encode
            mock_model.get_sentence_embedding_dimension.return_value = 384
            
            store = FaissStore(index_path, ids_path, "test-model")
            return store
    
    def test_encode_single_text(self, store):
        """Test encoding a single text."""
        text = ["This is a test sentence"]
        embeddings = store.encode(text)
        
        assert embeddings.shape == (1, 384)
        assert embeddings.dtype == np.float32
        
        # Check normalization (L2 norm should be ~1)
        # Note: Mock doesn't actually normalize, so skip this check
    
    def test_encode_multiple_texts(self, store):
        """Test encoding multiple texts."""
        texts = ["First text", "Second text", "Third text"]
        embeddings = store.encode(texts)
        
        assert embeddings.shape == (3, 384)
        assert embeddings.dtype == np.float32
    
    def test_encode_batch_processing(self, store):
        """Test that encoding uses batch processing."""
        texts = ["Text " + str(i) for i in range(100)]
        
        with patch.object(store.model, 'encode', wraps=store.model.encode) as mock_encode:
            embeddings = store.encode(texts)
            
            # Verify batch_size parameter was passed
            mock_encode.assert_called_once()
            call_kwargs = mock_encode.call_args[1]
            assert call_kwargs.get('batch_size') == 64
            assert call_kwargs.get('normalize_embeddings') is True


class TestFaissStoreBuild:
    """Test index building functionality."""
    
    @pytest.fixture
    def temp_paths(self):
        """Create temporary paths for index and IDs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            index_path = Path(tmpdir) / "test.faiss"
            ids_path = Path(tmpdir) / "test.pkl"
            yield index_path, ids_path
    
    @pytest.fixture
    def store(self, temp_paths):
        """Create a FaissStore instance."""
        index_path, ids_path = temp_paths
        
        with patch('app.vectorstore.SentenceTransformer') as mock_model_class:
            mock_model = MagicMock()
            mock_model_class.return_value = mock_model
            
            def mock_encode(texts, **kwargs):
                n = len(texts) if isinstance(texts, list) else 1
                vecs = np.random.randn(n, 384).astype('float32')
                # Normalize for cosine similarity
                return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
            
            mock_model.encode = mock_encode
            mock_model.get_sentence_embedding_dimension.return_value = 384
            
            return FaissStore(index_path, ids_path, "test-model")
    
    def test_build_index(self, store):
        """Test building FAISS index from rows."""
        rows = [
            {'text': 'First document', 'embedding_ref_id': 1, 'chunk_id': 10},
            {'text': 'Second document', 'embedding_ref_id': 2, 'chunk_id': 20},
            {'text': 'Third document', 'embedding_ref_id': 3, 'chunk_id': 30}
        ]
        
        store.build(rows)
        
        assert store.index is not None
        assert store.index.ntotal == 3
        assert len(store.ids) == 3
        assert store.ids == [(1, 10), (2, 20), (3, 30)]
    
    def test_build_saves_index(self, store):
        """Test that build saves index to disk."""
        rows = [
            {'text': 'Test document', 'embedding_ref_id': 1, 'chunk_id': 10}
        ]
        
        store.build(rows)
        
        assert store.index_path.exists()
        assert store.ids_path.exists()
        
        # Load and verify
        loaded_index = faiss.read_index(str(store.index_path))
        assert loaded_index.ntotal == 1
        
        with open(store.ids_path, 'rb') as f:
            loaded_ids = pickle.load(f)
        assert loaded_ids == [(1, 10)]


class TestFaissStoreSearch:
    """Test search functionality."""
    
    @pytest.fixture
    def temp_paths(self):
        """Create temporary paths for index and IDs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            index_path = Path(tmpdir) / "test.faiss"
            ids_path = Path(tmpdir) / "test.pkl"
            yield index_path, ids_path
    
    @pytest.fixture
    def populated_store(self, temp_paths):
        """Create a populated FaissStore."""
        index_path, ids_path = temp_paths
        
        with patch('app.vectorstore.SentenceTransformer') as mock_model_class:
            mock_model = MagicMock()
            mock_model_class.return_value = mock_model
            
            # Create deterministic embeddings for testing
            embeddings = np.array([
                [1.0, 0.0, 0.0],  # Doc 1
                [0.0, 1.0, 0.0],  # Doc 2
                [0.0, 0.0, 1.0],  # Doc 3
                [0.7, 0.7, 0.0],  # Doc 4 (similar to 1 and 2)
            ], dtype='float32')
            
            # Normalize
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
            
            encoding_index = [0]
            
            def mock_encode(texts, **kwargs):
                if isinstance(texts, str):
                    texts = [texts]
                
                result = []
                for text in texts:
                    if "query" in text.lower():
                        # Return query vector similar to first doc
                        result.append(embeddings[0])
                    else:
                        # Return sequential embeddings for documents
                        idx = encoding_index[0] % len(embeddings)
                        result.append(embeddings[idx])
                        encoding_index[0] += 1
                
                return np.array(result, dtype='float32')
            
            mock_model.encode = mock_encode
            mock_model.get_sentence_embedding_dimension.return_value = 3
            
            store = FaissStore(index_path, ids_path, "test-model")
            
            # Build index
            rows = [
                {'text': 'Document 1', 'embedding_ref_id': 1, 'chunk_id': 10},
                {'text': 'Document 2', 'embedding_ref_id': 2, 'chunk_id': 20},
                {'text': 'Document 3', 'embedding_ref_id': 3, 'chunk_id': 30},
                {'text': 'Document 4', 'embedding_ref_id': 4, 'chunk_id': 40}
            ]
            store.build(rows)
            encoding_index[0] = 0  # Reset for queries
            
            return store
    
    def test_search_basic(self, populated_store):
        """Test basic search functionality."""
        store = populated_store
        
        # Search for documents similar to "query"
        scores, indices = store.search("query", k=2)
        
        assert len(scores) == 2
        assert len(indices) == 2
        
        # First result should be most similar
        assert scores[0] > scores[1]
        
        # Map indices to chunk IDs
        chunk_ids = [store.ids[idx][1] for idx in indices]
        assert 10 in chunk_ids  # Document 1 should be in results
    
    def test_search_k_limit(self, populated_store):
        """Test that search respects k parameter."""
        store = populated_store
        
        # Request more results than available
        scores, indices = store.search("query", k=10)
        
        # Should return only available documents
        assert len(scores) == 4
        assert len(indices) == 4
    
    def test_search_empty_index(self, temp_paths):
        """Test searching an empty index."""
        index_path, ids_path = temp_paths
        
        with patch('app.vectorstore.SentenceTransformer'):
            store = FaissStore(index_path, ids_path, "test-model")
            store.index = faiss.IndexFlatIP(384)
            
            scores, indices = store.search("query", k=5)
            
            assert len(scores) == 0
            assert len(indices) == 0


class TestFaissStorePersistence:
    """Test index persistence and recovery."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)
    
    def test_save_and_load(self, temp_dir):
        """Test saving and loading index."""
        index_path = temp_dir / "test.faiss"
        ids_path = temp_dir / "test.pkl"
        
        with patch('app.vectorstore.SentenceTransformer') as mock_model_class:
            mock_model = MagicMock()
            mock_model_class.return_value = mock_model
            
            def mock_encode(texts, **kwargs):
                n = len(texts) if isinstance(texts, list) else 1
                return np.random.randn(n, 384).astype('float32')
            
            mock_model.encode = mock_encode
            mock_model.get_sentence_embedding_dimension.return_value = 384
            
            # Create and populate store
            store1 = FaissStore(index_path, ids_path, "test-model")
            rows = [
                {'text': 'Doc 1', 'embedding_ref_id': 1, 'chunk_id': 10},
                {'text': 'Doc 2', 'embedding_ref_id': 2, 'chunk_id': 20}
            ]
            store1.build(rows)
            
            # Create new store and load
            store2 = FaissStore(index_path, ids_path, "test-model")
            store2.load()
            
            assert store2.index is not None
            assert store2.index.ntotal == 2
            assert store2.ids == [(1, 10), (2, 20)]
    
    def test_save_creates_directory(self, temp_dir):
        """Test that save creates parent directory if needed."""
        nested_path = temp_dir / "nested" / "dir" / "test.faiss"
        ids_path = temp_dir / "nested" / "dir" / "test.pkl"
        
        with patch('app.vectorstore.SentenceTransformer'):
            store = FaissStore(nested_path, ids_path, "test-model")
            store.save()
            
            assert nested_path.parent.exists()


class TestFaissStoreEdgeCases:
    """Test edge cases and error handling."""
    
    @pytest.fixture
    def temp_paths(self):
        """Create temporary paths for index and IDs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            index_path = Path(tmpdir) / "test.faiss"
            ids_path = Path(tmpdir) / "test.pkl"
            yield index_path, ids_path
    
    @pytest.fixture
    def store(self, temp_paths):
        """Create a basic FaissStore."""
        index_path, ids_path = temp_paths
        with patch('app.vectorstore.SentenceTransformer'):
            return FaissStore(index_path, ids_path, "test-model")
    
    def test_search_without_index(self, store):
        """Test searching without building index first."""
        # Should handle gracefully
        with pytest.raises(AttributeError):
            store.search("query", k=5)
    
    def test_build_empty_rows(self, store):
        """Test building with empty rows."""
        with patch.object(store, '_encode') as mock_encode:
            mock_encode.return_value = np.array([], dtype='float32').reshape(0, 384)
            
            store.build([])
            
            assert store.index is not None
            assert store.index.ntotal == 0
            assert store.ids == []
    
    def test_normalize_embeddings(self, temp_paths):
        """Test that embeddings are properly normalized."""
        index_path, ids_path = temp_paths
        
        with patch('app.vectorstore.SentenceTransformer') as mock_model_class:
            mock_model = MagicMock()
            mock_model_class.return_value = mock_model
            
            # Return non-normalized embeddings
            def mock_encode(texts, normalize_embeddings=False, **kwargs):
                n = len(texts) if isinstance(texts, list) else 1
                vecs = np.random.randn(n, 384).astype('float32') * 10  # Large values
                if normalize_embeddings:
                    return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
                return vecs
            
            mock_model.encode = mock_encode
            mock_model.get_sentence_embedding_dimension.return_value = 384
            
            store = FaissStore(index_path, ids_path, "test-model")
            
            # Encode should normalize
            embeddings = store.encode(["test text"])
            
            # Check if normalized (L2 norm should be close to 1)
            norm = np.linalg.norm(embeddings[0])
            assert abs(norm - 1.0) < 0.01


if __name__ == "__main__":
    pytest.main([__file__, "-v"])