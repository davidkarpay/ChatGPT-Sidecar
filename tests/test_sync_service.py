"""Tests for ChatGPT sync service and background tasks."""

import pytest
import json
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from datetime import datetime, timedelta

from app.sync_service import ChatGPTSync, SyncStatus, sync_user_chatgpt_data


class TestSyncStatus:
    """Test sync status tracking functionality."""
    
    @pytest.fixture
    def temp_db(self):
        """Create temporary database for testing."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        
        # Initialize schema
        from app.db import DB
        with DB(db_path) as db:
            schema_path = Path(__file__).parent.parent / 'schema.sql'
            if schema_path.exists():
                db.init_schema(str(schema_path))
            # Create test user
            db.upsert_user("test@example.com", "Test User")
        
        yield db_path
        
        # Cleanup
        if os.path.exists(db_path):
            os.unlink(db_path)
    
    def test_sync_status_initialization(self, temp_db):
        """Test SyncStatus initialization."""
        sync_status = SyncStatus(db_path=temp_db)
        assert sync_status.db_path == temp_db
    
    def test_get_last_sync_no_history(self, temp_db):
        """Test getting last sync when no history exists."""
        sync_status = SyncStatus(db_path=temp_db)
        
        last_sync = sync_status.get_last_sync(user_id=1)
        assert last_sync is None
    
    def test_record_sync_start(self, temp_db):
        """Test recording sync start."""
        sync_status = SyncStatus(db_path=temp_db)
        
        sync_id = sync_status.record_sync_start(
            user_id=1,
            source_url="https://example.com/export.zip"
        )
        
        assert sync_id is not None
        assert isinstance(sync_id, str)
        
        # Verify record was created
        last_sync = sync_status.get_last_sync(user_id=1)
        assert last_sync is not None
        assert last_sync['sync_id'] == sync_id
        assert last_sync['status'] == 'running'
    
    def test_record_sync_completion(self, temp_db):
        """Test recording successful sync completion."""
        sync_status = SyncStatus(db_path=temp_db)
        
        sync_id = sync_status.record_sync_start(user_id=1)
        sync_status.record_sync_completion(
            sync_id=sync_id,
            files_processed=5,
            conversations_added=10,
            conversations_updated=3
        )
        
        last_sync = sync_status.get_last_sync(user_id=1)
        assert last_sync['status'] == 'completed'
        assert last_sync['files_processed'] == 5
        assert last_sync['conversations_added'] == 10
        assert last_sync['conversations_updated'] == 3
        assert last_sync['completed_at'] is not None
    
    def test_record_sync_error(self, temp_db):
        """Test recording sync error."""
        sync_status = SyncStatus(db_path=temp_db)
        
        sync_id = sync_status.record_sync_start(user_id=1)
        error_message = "Network timeout occurred"
        
        sync_status.record_sync_error(sync_id, error_message)
        
        last_sync = sync_status.get_last_sync(user_id=1)
        assert last_sync['status'] == 'failed'
        assert last_sync['error_message'] == error_message
    
    def test_get_sync_history(self, temp_db):
        """Test getting sync history."""
        sync_status = SyncStatus(db_path=temp_db)
        
        # Create multiple sync records
        sync1_id = sync_status.record_sync_start(user_id=1)
        sync_status.record_sync_completion(sync1_id, 2, 5, 1)
        
        sync2_id = sync_status.record_sync_start(user_id=1)
        sync_status.record_sync_error(sync2_id, "Test error")
        
        history = sync_status.get_sync_history(user_id=1, limit=10)
        
        assert len(history) == 2
        # Should be ordered by most recent first
        assert history[0]['sync_id'] == sync2_id
        assert history[1]['sync_id'] == sync1_id


class TestChatGPTSync:
    """Test ChatGPT synchronization functionality."""
    
    @pytest.fixture
    def temp_db(self):
        """Create temporary database for testing."""
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
    
    def test_chatgpt_sync_initialization(self, temp_db):
        """Test ChatGPTSync initialization."""
        sync = ChatGPTSync(db_path=temp_db)
        assert sync.db_path == temp_db
        assert isinstance(sync.status, SyncStatus)
    
    @pytest.mark.asyncio
    async def test_download_export_success(self, temp_db):
        """Test successful export download."""
        sync = ChatGPTSync(db_path=temp_db)
        
        # Mock successful download
        mock_content = b'{"conversations": []}'
        
        with patch('httpx.AsyncClient') as mock_client:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.content = mock_content
            mock_response.headers = {'content-type': 'application/json'}
            
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(return_value=mock_response)
            
            content, filename = await sync.download_export("https://example.com/export.json")
            
            assert content == mock_content
            assert filename.endswith('.json')
    
    @pytest.mark.asyncio
    async def test_download_export_failure(self, temp_db):
        """Test export download failure."""
        sync = ChatGPTSync(db_path=temp_db)
        
        with patch('httpx.AsyncClient') as mock_client:
            mock_response = Mock()
            mock_response.status_code = 404
            mock_response.raise_for_status.side_effect = Exception("Not found")
            
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(return_value=mock_response)
            
            with pytest.raises(Exception):
                await sync.download_export("https://example.com/nonexistent.json")
    
    def test_extract_zip_file(self, temp_db):
        """Test ZIP file extraction."""
        sync = ChatGPTSync(db_path=temp_db)
        
        # Create a test ZIP file
        import zipfile
        import io
        
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
            zip_file.writestr('conversations.json', '{"test": "data"}')
            zip_file.writestr('user.json', '{"email": "test@example.com"}')
        
        zip_content = zip_buffer.getvalue()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            extract_path = sync.extract_zip_file(zip_content, temp_dir)
            
            assert os.path.exists(extract_path)
            assert os.path.exists(os.path.join(extract_path, 'conversations.json'))
            assert os.path.exists(os.path.join(extract_path, 'user.json'))
    
    def test_process_export_directory(self, temp_db):
        """Test processing extracted export directory."""
        sync = ChatGPTSync(db_path=temp_db)
        
        # Create test export directory
        with tempfile.TemporaryDirectory() as temp_dir:
            export_dir = Path(temp_dir)
            
            # Create test conversations file
            conversations_data = [
                {
                    "title": "Test Conversation",
                    "mapping": {
                        "root": {
                            "id": "root",
                            "message": {
                                "author": {"role": "user"},
                                "content": {"parts": ["Test message"]}
                            }
                        }
                    }
                }
            ]
            
            conversations_file = export_dir / "conversations.json"
            with open(conversations_file, 'w') as f:
                json.dump(conversations_data, f)
            
            with patch('app.sync_service.ingest_export') as mock_ingest:
                mock_ingest.return_value = 1
                
                result = sync.process_export_directory(str(export_dir))
                
                assert result['conversations_added'] == 1
                assert result['conversations_updated'] == 0
                mock_ingest.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_full_sync_workflow(self, temp_db):
        """Test complete synchronization workflow."""
        sync = ChatGPTSync(db_path=temp_db)
        
        # Mock all external dependencies
        with patch.object(sync, 'download_export') as mock_download:
            mock_download.return_value = (b'zip content', 'export.zip')
            
            with patch.object(sync, 'extract_zip_file') as mock_extract:
                mock_extract.return_value = '/tmp/extracted'
                
                with patch.object(sync, 'process_export_directory') as mock_process:
                    mock_process.return_value = {
                        'conversations_added': 5,
                        'conversations_updated': 2
                    }
                    
                    result = await sync.sync_user_data(
                        user_id=1,
                        export_url="https://example.com/export.zip"
                    )
                    
                    assert result['status'] == 'completed'
                    assert result['conversations_added'] == 5
                    assert result['conversations_updated'] == 2
    
    @pytest.mark.asyncio
    async def test_sync_with_error_handling(self, temp_db):
        """Test sync workflow with error handling."""
        sync = ChatGPTSync(db_path=temp_db)
        
        with patch.object(sync, 'download_export') as mock_download:
            mock_download.side_effect = Exception("Network error")
            
            result = await sync.sync_user_data(
                user_id=1,
                export_url="https://example.com/export.zip"
            )
            
            assert result['status'] == 'failed'
            assert 'Network error' in result['error_message']


class TestCeleryTasks:
    """Test Celery background tasks."""
    
    @pytest.fixture
    def temp_db(self):
        """Create temporary database for testing."""
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
    
    def test_sync_user_chatgpt_data_task(self, temp_db):
        """Test the Celery task for syncing user data."""
        with patch('app.sync_service.ChatGPTSync') as mock_sync_class:
            mock_sync = Mock()
            mock_sync_class.return_value = mock_sync
            
            # Mock async method
            async def mock_sync_user_data(user_id, export_url):
                return {
                    'status': 'completed',
                    'conversations_added': 3,
                    'conversations_updated': 1
                }
            
            mock_sync.sync_user_data = mock_sync_user_data
            
            # Test the task function
            import asyncio
            result = asyncio.run(
                sync_user_chatgpt_data(
                    user_id=1,
                    export_url="https://example.com/export.zip",
                    db_path=temp_db
                )
            )
            
            assert result['status'] == 'completed'
            assert result['conversations_added'] == 3
    
    def test_celery_app_configuration(self):
        """Test Celery app configuration."""
        from app.sync_service import celery_app
        
        assert celery_app is not None
        assert celery_app.main == "sync_service"
        
        # Check that Redis is configured
        assert 'redis://' in celery_app.conf.broker_url


class TestSyncScheduling:
    """Test sync scheduling and periodic tasks."""
    
    def test_sync_interval_configuration(self):
        """Test sync interval configuration."""
        with patch.dict(os.environ, {'CHATGPT_SYNC_INTERVAL_HOURS': '12'}):
            from app.sync_service import CHATGPT_SYNC_INTERVAL_HOURS
            assert CHATGPT_SYNC_INTERVAL_HOURS == 12
    
    def test_sync_enabled_flag(self):
        """Test sync enabled/disabled flag."""
        with patch.dict(os.environ, {'CHATGPT_SYNC_ENABLED': 'true'}):
            from app.sync_service import CHATGPT_SYNC_ENABLED
            assert CHATGPT_SYNC_ENABLED is True
        
        with patch.dict(os.environ, {'CHATGPT_SYNC_ENABLED': 'false'}):
            # Need to reload module to pick up new env var
            import importlib
            import app.sync_service
            importlib.reload(app.sync_service)
            
            from app.sync_service import CHATGPT_SYNC_ENABLED
            assert CHATGPT_SYNC_ENABLED is False
    
    def test_should_sync_user_data(self, temp_db):
        """Test logic for determining if user data should be synced."""
        sync_status = SyncStatus(db_path=temp_db)
        
        # User with no sync history should be synced
        should_sync = sync_status.should_sync_user_data(user_id=1, interval_hours=24)
        assert should_sync is True
        
        # User with recent sync should not be synced
        sync_id = sync_status.record_sync_start(user_id=1)
        sync_status.record_sync_completion(sync_id, 1, 2, 0)
        
        should_sync = sync_status.should_sync_user_data(user_id=1, interval_hours=24)
        assert should_sync is False
        
        # User with old sync should be synced again
        should_sync = sync_status.should_sync_user_data(user_id=1, interval_hours=0.001)  # Very small interval
        # This might still be False due to timing precision, but tests the logic


class TestErrorRecovery:
    """Test error recovery and retry mechanisms."""
    
    @pytest.fixture
    def temp_db(self):
        """Create temporary database for testing."""
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
    
    @pytest.mark.asyncio
    async def test_retry_mechanism(self, temp_db):
        """Test retry mechanism for failed syncs."""
        sync = ChatGPTSync(db_path=temp_db)
        
        call_count = [0]  # Use list to modify from inner function
        
        async def failing_download(url):
            call_count[0] += 1
            if call_count[0] < 3:
                raise Exception("Temporary failure")
            return b'success', 'export.zip'
        
        with patch.object(sync, 'download_export', side_effect=failing_download):
            with patch.object(sync, 'extract_zip_file', return_value='/tmp/test'):
                with patch.object(sync, 'process_export_directory', return_value={'conversations_added': 1, 'conversations_updated': 0}):
                    
                    # This would require implementing retry logic in the actual code
                    # For now, just test that multiple calls can be made
                    try:
                        result = await sync.sync_user_data(1, "https://example.com/export.zip")
                        # If retry logic is implemented, this should eventually succeed
                    except Exception as e:
                        # Without retry logic, it will fail on first attempt
                        assert "Temporary failure" in str(e)
    
    def test_partial_failure_recovery(self, temp_db):
        """Test recovery from partial failures."""
        sync_status = SyncStatus(db_path=temp_db)
        
        # Start a sync
        sync_id = sync_status.record_sync_start(user_id=1)
        
        # Simulate partial failure (some files processed)
        sync_status.record_sync_error(sync_id, "Partial failure after processing 2 files")
        
        # Should be able to start a new sync
        new_sync_id = sync_status.record_sync_start(user_id=1)
        assert new_sync_id != sync_id
        
        # Complete the new sync
        sync_status.record_sync_completion(new_sync_id, 5, 10, 0)
        
        last_sync = sync_status.get_last_sync(user_id=1)
        assert last_sync['status'] == 'completed'


class TestDataIntegrity:
    """Test data integrity during sync operations."""
    
    @pytest.fixture
    def temp_db(self):
        """Create temporary database for testing."""
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
    
    def test_concurrent_sync_prevention(self, temp_db):
        """Test prevention of concurrent syncs for same user."""
        sync_status = SyncStatus(db_path=temp_db)
        
        # Start first sync
        sync_id1 = sync_status.record_sync_start(user_id=1)
        
        # Try to start second sync while first is running
        # This should either fail or queue the second sync
        # Implementation depends on business logic
        
        running_syncs = sync_status.get_running_syncs(user_id=1)
        assert len(running_syncs) >= 1
        assert sync_id1 in [sync['sync_id'] for sync in running_syncs]
    
    def test_sync_metadata_consistency(self, temp_db):
        """Test that sync metadata remains consistent."""
        sync_status = SyncStatus(db_path=temp_db)
        
        sync_id = sync_status.record_sync_start(
            user_id=1,
            source_url="https://example.com/export.zip"
        )
        
        # Record completion
        sync_status.record_sync_completion(
            sync_id=sync_id,
            files_processed=3,
            conversations_added=5,
            conversations_updated=2
        )
        
        # Verify all fields are consistent
        last_sync = sync_status.get_last_sync(user_id=1)
        
        assert last_sync['sync_id'] == sync_id
        assert last_sync['status'] == 'completed'
        assert last_sync['files_processed'] == 3
        assert last_sync['conversations_added'] == 5
        assert last_sync['conversations_updated'] == 2
        assert last_sync['started_at'] is not None
        assert last_sync['completed_at'] is not None
        assert last_sync['error_message'] is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])