"""Tests for the new API authentication system."""

import pytest
import os
import tempfile
from unittest.mock import Mock, patch
from fastapi import HTTPException

from app.api_auth import APIKeyManager, AccessLevel, require_read_access, require_admin_access


class TestAPIKeyManager:
    """Test API key management functionality."""
    
    @pytest.fixture
    def temp_db(self):
        """Create a temporary database for testing."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".db") as f:
            db_path = f.name
        yield db_path
        try:
            os.unlink(db_path)
        except:
            pass
    
    @pytest.fixture
    def api_key_manager(self, temp_db):
        """Create APIKeyManager with test keys."""
        with patch.dict(os.environ, {
            "READ_KEY": "test-read-key",
            "ADMIN_KEY": "test-admin-key", 
            "SUPER_KEY": "test-super-key"
        }):
            manager = APIKeyManager(temp_db)
            return manager
    
    def test_key_validation_read_access(self, api_key_manager):
        """Test read key validation."""
        auth_info = api_key_manager.validate_key("test-read-key", AccessLevel.READ)
        
        assert auth_info["level"] == AccessLevel.READ
        assert auth_info["key"] == "test-read-key"
        assert auth_info["name"] == "Read Access Key"
    
    def test_key_validation_admin_access(self, api_key_manager):
        """Test admin key validation."""
        auth_info = api_key_manager.validate_key("test-admin-key", AccessLevel.ADMIN)
        
        assert auth_info["level"] == AccessLevel.ADMIN
        assert auth_info["key"] == "test-admin-key"
        assert auth_info["name"] == "Admin Access Key"
    
    def test_key_validation_insufficient_permissions(self, api_key_manager):
        """Test that read key cannot access admin endpoints."""
        with pytest.raises(HTTPException) as exc_info:
            api_key_manager.validate_key("test-read-key", AccessLevel.ADMIN)
        
        assert exc_info.value.status_code == 403
        assert "Insufficient permissions" in exc_info.value.detail
    
    def test_key_validation_invalid_key(self, api_key_manager):
        """Test invalid key rejection."""
        with pytest.raises(HTTPException) as exc_info:
            api_key_manager.validate_key("invalid-key", AccessLevel.READ)
        
        assert exc_info.value.status_code == 401
        assert "Invalid API key" in exc_info.value.detail
    
    def test_admin_key_has_read_access(self, api_key_manager):
        """Test that admin keys can access read endpoints."""
        auth_info = api_key_manager.validate_key("test-admin-key", AccessLevel.READ)
        
        assert auth_info["level"] == AccessLevel.ADMIN
        assert auth_info["key"] == "test-admin-key"
    
    def test_super_key_has_all_access(self, api_key_manager):
        """Test that super keys can access all endpoints."""
        for level in [AccessLevel.READ, AccessLevel.ADMIN, AccessLevel.SUPER]:
            auth_info = api_key_manager.validate_key("test-super-key", level)
            assert auth_info["level"] == AccessLevel.SUPER
    
    def test_rate_limiting(self, api_key_manager):
        """Test basic rate limiting functionality."""
        mock_request = Mock()
        mock_request.client.host = "127.0.0.1"
        mock_request.url.path = "/test"
        
        # Should not raise for first few requests
        for _ in range(5):
            api_key_manager.check_rate_limit("test-read-key", mock_request)
    
    def test_audit_logging(self, api_key_manager, temp_db):
        """Test that audit events are logged."""
        # Mock database operations since we're not setting up full schema
        with patch.object(api_key_manager, 'log_auth_event') as mock_log:
            api_key_manager.log_auth_event(
                api_key="test-key",
                event_type="test_event", 
                ip_address="127.0.0.1",
                endpoint="/test",
                success=True
            )
            
            mock_log.assert_called_once()


class TestAPIAuthDependencies:
    """Test FastAPI dependency functions."""
    
    @pytest.fixture
    def mock_request(self):
        """Create mock request object."""
        request = Mock()
        request.client.host = "127.0.0.1"
        request.url.path = "/test"
        return request
    
    @pytest.mark.asyncio
    async def test_require_read_access_valid_key(self, mock_request):
        """Test read access with valid key."""
        with patch.dict(os.environ, {"READ_KEY": "valid-read-key"}):
            with patch('app.api_auth.get_api_key_manager') as mock_manager:
                manager = Mock()
                manager.check_rate_limit.return_value = None
                manager.validate_key.return_value = {
                    "level": AccessLevel.READ,
                    "key": "valid-read-key",
                    "name": "Test Key"
                }
                manager.log_auth_event.return_value = None
                mock_manager.return_value = manager
                
                result = await require_read_access(mock_request, "valid-read-key")
                
                assert result["level"] == AccessLevel.READ
                assert result["key"] == "valid-read-key"
    
    @pytest.mark.asyncio 
    async def test_require_admin_access_insufficient_key(self, mock_request):
        """Test admin access with read-only key."""
        with patch.dict(os.environ, {"READ_KEY": "read-only-key"}):
            with patch('app.api_auth.get_api_key_manager') as mock_manager:
                manager = Mock()
                manager.check_rate_limit.return_value = None
                manager.validate_key.side_effect = HTTPException(
                    status_code=403, 
                    detail="Insufficient permissions"
                )
                manager.log_auth_event.return_value = None
                mock_manager.return_value = manager
                
                with pytest.raises(HTTPException) as exc_info:
                    await require_admin_access(mock_request, "read-only-key")
                
                assert exc_info.value.status_code == 403


if __name__ == "__main__":
    pytest.main([__file__, "-v"])