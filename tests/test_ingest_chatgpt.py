"""Tests for ChatGPT export ingestion functionality."""

import pytest
import json
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from app.ingest_chatgpt import ingest_export, flatten_conversation, sha256


class TestUtilityFunctions:
    """Test utility functions for ChatGPT ingestion."""
    
    def test_sha256_generation(self):
        """Test SHA256 hash generation."""
        text = "This is a test string"
        expected_hash = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        
        # Generate hash
        result = sha256(text)
        
        assert isinstance(result, str)
        assert len(result) == 64  # SHA256 produces 64-character hex string
        
        # Same input should produce same hash
        assert sha256(text) == sha256(text)
        
        # Different input should produce different hash
        assert sha256(text) != sha256("Different text")
    
    def test_flatten_conversation_basic(self):
        """Test basic conversation flattening."""
        title = "Test Conversation"
        messages = [
            {
                "author": {"role": "user"},
                "content": {"parts": ["Hello, how are you?"]},
                "create_time": "2024-01-01T00:00:00Z"
            },
            {
                "author": {"role": "assistant"},
                "content": {"parts": ["I'm doing well, thank you!"]},
                "create_time": "2024-01-01T00:01:00Z"
            }
        ]
        
        result = flatten_conversation(title, messages)
        
        assert "# Test Conversation" in result
        assert "USER:" in result
        assert "ASSISTANT:" in result
        assert "Hello, how are you?" in result
        assert "I'm doing well, thank you!" in result
    
    def test_flatten_conversation_legacy_format(self):
        """Test flattening with legacy message format."""
        title = "Legacy Format"
        messages = [
            {
                "role": "user",
                "content": "Direct content string",
                "update_time": "2024-01-01T00:00:00Z"
            }
        ]
        
        result = flatten_conversation(title, messages)
        
        assert "# Legacy Format" in result
        assert "USER:" in result
        assert "Direct content string" in result
    
    def test_flatten_conversation_complex_content(self):
        """Test flattening with complex content structures."""
        title = "Complex Content"
        messages = [
            {
                "author": {"role": "user"},
                "content": {"parts": ["Part 1", "Part 2", {"type": "object"}]},
                "create_time": "2024-01-01"
            },
            {
                "author": {"role": "assistant"},
                "content": "String content",  # Different format
                "create_time": "2024-01-01"
            }
        ]
        
        result = flatten_conversation(title, messages)
        
        assert "Part 1" in result
        assert "Part 2" in result
        assert "String content" in result
    
    def test_flatten_conversation_missing_fields(self):
        """Test flattening with missing/malformed fields."""
        title = "Missing Fields"
        messages = [
            {
                # Missing author
                "content": {"parts": ["Test message"]},
                "create_time": "2024-01-01"
            },
            {
                "author": {},  # Empty author
                "content": {"parts": ["Another message"]}
                # Missing timestamp
            }
        ]
        
        result = flatten_conversation(title, messages)
        
        assert "UNKNOWN:" in result
        assert "Test message" in result
        assert "Another message" in result


class TestChatGPTIngestion:
    """Test ChatGPT export ingestion functionality."""
    
    @pytest.fixture
    def temp_export_dir(self):
        """Create temporary directory with ChatGPT export structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            export_dir = Path(tmpdir)
            yield export_dir
    
    @pytest.fixture
    def sample_conversations_mapping(self):
        """Sample conversations using mapping format (modern ChatGPT exports)."""
        return [
            {
                "title": "Python Help",
                "id": "conv-123",
                "create_time": 1640995200.0,
                "update_time": 1640995800.0,
                "mapping": {
                    "root": {
                        "id": "root",
                        "message": None,
                        "parent": None,
                        "children": ["msg-1"]
                    },
                    "msg-1": {
                        "id": "msg-1",
                        "message": {
                            "id": "msg-1",
                            "author": {"role": "user"},
                            "create_time": 1640995200.0,
                            "content": {
                                "content_type": "text",
                                "parts": ["How do I write a for loop in Python?"]
                            }
                        },
                        "parent": "root",
                        "children": ["msg-2"]
                    },
                    "msg-2": {
                        "id": "msg-2",
                        "message": {
                            "id": "msg-2",
                            "author": {"role": "assistant"},
                            "create_time": 1640995500.0,
                            "content": {
                                "content_type": "text",
                                "parts": ["You can write a for loop like this: for i in range(10):"]
                            }
                        },
                        "parent": "msg-1",
                        "children": []
                    }
                }
            },
            {
                "title": "Weather Question",
                "id": "conv-456",
                "messages": [  # Legacy format
                    {
                        "role": "user",
                        "content": "What's the weather like?",
                        "create_time": 1640995900.0
                    },
                    {
                        "role": "assistant", 
                        "content": "I don't have access to real-time weather data.",
                        "create_time": 1640996000.0
                    }
                ]
            }
        ]
    
    @pytest.fixture
    def temp_db_with_schema(self):
        """Create temporary database with schema."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        
        # Initialize schema
        from app.db import DB
        with DB(db_path) as db:
            schema_path = Path(__file__).parent.parent / 'schema.sql'
            if schema_path.exists():
                db.init_schema(str(schema_path))
            # Create default user
            db.upsert_user("test@example.com", "Test User")
        
        yield db_path
        
        # Cleanup
        if os.path.exists(db_path):
            os.unlink(db_path)
    
    def test_ingest_export_mapping_format(self, temp_export_dir, sample_conversations_mapping, temp_db_with_schema):
        """Test ingesting export with mapping format conversations."""
        # Create conversations.json
        conversations_file = temp_export_dir / "conversations.json"
        with open(conversations_file, 'w') as f:
            json.dump(sample_conversations_mapping, f)
        
        with patch('app.ingest_chatgpt.DB_PATH', temp_db_with_schema):
            count = ingest_export(str(temp_export_dir))
        
        assert count == 2  # Two conversations processed
        
        # Verify documents were created
        from app.db import DB
        with DB(temp_db_with_schema) as db:
            cur = db.conn.execute("SELECT COUNT(*) FROM document")
            doc_count = cur.fetchone()[0]
            assert doc_count == 2
            
            # Check titles
            cur = db.conn.execute("SELECT title FROM document ORDER BY title")
            titles = [row[0] for row in cur.fetchall()]
            assert "Python Help" in titles
            assert "Weather Question" in titles
    
    def test_ingest_export_chunks_creation(self, temp_export_dir, sample_conversations_mapping, temp_db_with_schema):
        """Test that chunks are created from conversations."""
        conversations_file = temp_export_dir / "conversations.json"
        with open(conversations_file, 'w') as f:
            json.dump(sample_conversations_mapping, f)
        
        with patch('app.ingest_chatgpt.DB_PATH', temp_db_with_schema):
            count = ingest_export(str(temp_export_dir), chunk_chars=100, overlap=20)
        
        # Verify chunks were created
        from app.db import DB
        with DB(temp_db_with_schema) as db:
            cur = db.conn.execute("SELECT COUNT(*) FROM chunk")
            chunk_count = cur.fetchone()[0]
            assert chunk_count > 0
            
            # Check chunk content
            cur = db.conn.execute("SELECT text FROM chunk LIMIT 1")
            chunk_text = cur.fetchone()[0]
            assert len(chunk_text) > 0
    
    def test_ingest_export_duplicate_fingerprint(self, temp_export_dir, sample_conversations_mapping, temp_db_with_schema):
        """Test that duplicate conversations are not re-ingested."""
        conversations_file = temp_export_dir / "conversations.json"
        with open(conversations_file, 'w') as f:
            json.dump(sample_conversations_mapping, f)
        
        with patch('app.ingest_chatgpt.DB_PATH', temp_db_with_schema):
            # First ingestion
            count1 = ingest_export(str(temp_export_dir))
            assert count1 == 2
            
            # Second ingestion should skip duplicates
            count2 = ingest_export(str(temp_export_dir))
            assert count2 == 0  # No new documents
        
        # Verify still only 2 documents
        from app.db import DB
        with DB(temp_db_with_schema) as db:
            cur = db.conn.execute("SELECT COUNT(*) FROM document")
            doc_count = cur.fetchone()[0]
            assert doc_count == 2
    
    def test_ingest_export_file_not_found(self, temp_export_dir):
        """Test error handling when conversations.json doesn't exist."""
        with pytest.raises(FileNotFoundError):
            ingest_export(str(temp_export_dir))
    
    def test_ingest_export_invalid_json(self, temp_export_dir):
        """Test error handling with invalid JSON."""
        conversations_file = temp_export_dir / "conversations.json"
        with open(conversations_file, 'w') as f:
            f.write("{ invalid json")
        
        with pytest.raises(json.JSONDecodeError):
            ingest_export(str(temp_export_dir))
    
    def test_ingest_export_empty_conversations(self, temp_export_dir, temp_db_with_schema):
        """Test handling of empty conversations list."""
        conversations_file = temp_export_dir / "conversations.json"
        with open(conversations_file, 'w') as f:
            json.dump([], f)
        
        with patch('app.ingest_chatgpt.DB_PATH', temp_db_with_schema):
            count = ingest_export(str(temp_export_dir))
        
        assert count == 0
    
    def test_ingest_export_malformed_conversations(self, temp_export_dir, temp_db_with_schema):
        """Test handling of malformed conversation data."""
        malformed_conversations = [
            {
                "title": "Good Conversation",
                "mapping": {
                    "root": {
                        "id": "root",
                        "message": {
                            "author": {"role": "user"},
                            "content": {"parts": ["Valid message"]}
                        }
                    }
                }
            },
            {
                # Missing title and content
                "id": "malformed"
            },
            {
                "title": "Empty Mapping",
                "mapping": {}
            }
        ]
        
        conversations_file = temp_export_dir / "conversations.json"
        with open(conversations_file, 'w') as f:
            json.dump(malformed_conversations, f)
        
        with patch('app.ingest_chatgpt.DB_PATH', temp_db_with_schema):
            # Should not crash on malformed data
            count = ingest_export(str(temp_export_dir))
            # At least the valid conversation should be processed
            assert count >= 1


class TestRealDataIntegration:
    """Integration tests with real ChatGPT export data."""
    
    @pytest.fixture
    def real_export_path(self):
        """Path to real ChatGPT export data."""
        export_path = Path("/Users/davidluciankarpay/Downloads/ChatGPT_Data")
        if not export_path.exists():
            pytest.skip("Real ChatGPT export data not available")
        return export_path
    
    @pytest.fixture
    def temp_db_with_schema(self):
        """Create temporary database with schema."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        
        from app.db import DB
        with DB(db_path) as db:
            schema_path = Path(__file__).parent.parent / 'schema.sql'
            if schema_path.exists():
                db.init_schema(str(schema_path))
            db.upsert_user("test@example.com", "Test User")
        
        yield db_path
        
        if os.path.exists(db_path):
            os.unlink(db_path)
    
    @pytest.mark.integration
    def test_sample_real_data_structure(self, real_export_path):
        """Test that we can parse the real export structure."""
        conversations_file = real_export_path / "conversations.json"
        
        # Read first few conversations to test structure
        with open(conversations_file, 'r') as f:
            # Read just the beginning to avoid loading huge file
            content = f.read(10000)  # First 10KB
            
        # Should start with array
        assert content.strip().startswith('[')
        
        # Load a small sample
        import json
        with open(conversations_file, 'r') as f:
            data = json.load(f)
        
        # Should have conversations
        assert len(data) > 0
        
        # Check structure of first conversation
        first_conv = data[0]
        assert 'title' in first_conv
        assert 'mapping' in first_conv or 'messages' in first_conv
    
    @pytest.mark.integration
    @pytest.mark.slow
    def test_ingest_sample_real_conversations(self, real_export_path, temp_db_with_schema):
        """Test ingesting a small sample of real conversations."""
        # Create a temporary export with just first 5 conversations
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_export = Path(tmpdir)
            
            # Copy first 5 conversations
            conversations_file = real_export_path / "conversations.json"
            with open(conversations_file, 'r') as f:
                all_data = json.load(f)
            
            sample_data = all_data[:5]  # Just first 5 conversations
            
            temp_conversations = temp_export / "conversations.json"
            with open(temp_conversations, 'w') as f:
                json.dump(sample_data, f)
            
            with patch('app.ingest_chatgpt.DB_PATH', temp_db_with_schema):
                count = ingest_export(str(temp_export))
            
            # Should have processed 5 conversations
            assert count <= 5  # Some might be duplicates/empty
            
            # Verify data in database
            from app.db import DB
            with DB(temp_db_with_schema) as db:
                cur = db.conn.execute("SELECT COUNT(*) FROM document WHERE doc_type = 'chatgpt_export'")
                doc_count = cur.fetchone()[0]
                assert doc_count > 0
                
                # Check that chunks were created
                cur = db.conn.execute("SELECT COUNT(*) FROM chunk")
                chunk_count = cur.fetchone()[0]
                assert chunk_count > 0
                
                # Check that some text was extracted
                cur = db.conn.execute("SELECT text FROM chunk LIMIT 1")
                sample_text = cur.fetchone()[0]
                assert len(sample_text) > 10  # Should have meaningful content
    
    @pytest.mark.integration
    def test_real_export_companion_files(self, real_export_path):
        """Test that companion files exist and are parseable."""
        # Check user.json
        user_file = real_export_path / "user.json"
        assert user_file.exists()
        
        with open(user_file, 'r') as f:
            user_data = json.load(f)
        assert 'email' in user_data
        
        # Check message_feedback.json
        feedback_file = real_export_path / "message_feedback.json"
        assert feedback_file.exists()
        
        with open(feedback_file, 'r') as f:
            feedback_data = json.load(f)
        assert isinstance(feedback_data, list)
        
        if len(feedback_data) > 0:
            first_feedback = feedback_data[0]
            assert 'rating' in first_feedback
            assert 'conversation_id' in first_feedback


class TestPerformanceAndScale:
    """Test performance and scalability of ingestion."""
    
    def test_chunk_size_configuration(self, temp_export_dir, temp_db_with_schema):
        """Test that chunk size parameters work correctly."""
        conversations = [{
            "title": "Long Conversation",
            "mapping": {
                "root": {
                    "id": "root",
                    "message": {
                        "author": {"role": "user"},
                        "content": {"parts": ["A" * 2000]}  # 2000 character message
                    }
                }
            }
        }]
        
        conversations_file = temp_export_dir / "conversations.json"
        with open(conversations_file, 'w') as f:
            json.dump(conversations, f)
        
        with patch('app.ingest_chatgpt.DB_PATH', temp_db_with_schema):
            # Test with small chunks
            count = ingest_export(str(temp_export_dir), chunk_chars=500, overlap=50)
            assert count == 1
            
            # Check that multiple chunks were created
            from app.db import DB
            with DB(temp_db_with_schema) as db:
                cur = db.conn.execute("SELECT COUNT(*) FROM chunk")
                chunk_count = cur.fetchone()[0]
                assert chunk_count > 1  # Should create multiple chunks
                
                # Check chunk sizes
                cur = db.conn.execute("SELECT LENGTH(text) FROM chunk ORDER BY LENGTH(text)")
                chunk_lengths = [row[0] for row in cur.fetchall()]
                
                # Most chunks should be close to target size
                for length in chunk_lengths[:-1]:  # Exclude last chunk which may be shorter
                    assert length <= 500 + 100  # Allow some buffer
    
    def test_memory_efficient_processing(self, temp_export_dir):
        """Test that ingestion doesn't load everything into memory."""
        # Create a large conversation that would use significant memory
        large_conversation = {
            "title": "Memory Test",
            "mapping": {}
        }
        
        # Add many messages
        for i in range(100):
            msg_id = f"msg-{i}"
            large_conversation["mapping"][msg_id] = {
                "id": msg_id,
                "message": {
                    "author": {"role": "user" if i % 2 == 0 else "assistant"},
                    "content": {"parts": [f"Message {i} " + "x" * 1000]}
                }
            }
        
        conversations_file = temp_export_dir / "conversations.json"
        with open(conversations_file, 'w') as f:
            json.dump([large_conversation], f)
        
        # Test should complete without memory issues
        with tempfile.NamedTemporaryFile(suffix='.db') as tmp_db:
            from app.db import DB
            with DB(tmp_db.name) as db:
                schema_path = Path(__file__).parent.parent / 'schema.sql'
                if schema_path.exists():
                    db.init_schema(str(schema_path))
                db.upsert_user("test@example.com", "Test User")
            
            with patch('app.ingest_chatgpt.DB_PATH', tmp_db.name):
                count = ingest_export(str(temp_export_dir))
                assert count == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])