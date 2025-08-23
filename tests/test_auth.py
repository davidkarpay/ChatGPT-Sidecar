"""Tests for authentication system including OAuth and session management."""

import pytest
import os
import json
import secrets
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
from pathlib import Path
import tempfile

from fastapi import HTTPException
from fastapi.testclient import TestClient

# Test fixtures
@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        db_path = f.name
    yield db_path
    # Cleanup
    if os.path.exists(db_path):
        os.unlink(db_path)


@pytest.fixture
def mock_google_auth():
    """Mock Google OAuth configuration."""
    with patch.dict(os.environ, {
        'GOOGLE_CLIENT_ID': 'test-client-id',
        'GOOGLE_CLIENT_SECRET': 'test-client-secret',
        'REDIRECT_URI': 'http://localhost:8088/auth/callback'
    }):
        yield


class TestSessionManager:
    """Test session management functionality."""
    
    @pytest.fixture
    def session_manager(self, temp_db):
        """Create a session manager instance."""
        from app.auth import SessionManager
        with patch('app.auth.DB_PATH', temp_db):
            # Initialize database schema
            from app.db import DB
            with DB(temp_db) as db:
                schema_path = Path(__file__).parent.parent / 'schema.sql'
                if schema_path.exists():
                    db.init_schema(str(schema_path))
            return SessionManager()
    
    def test_create_session(self, session_manager):
        """Test creating a new session."""
        user_id = 1
        session_token = session_manager.create_session(user_id)
        
        assert session_token is not None
        assert len(session_token) > 32
        
        # Verify session exists in database
        session = session_manager.get_session(session_token)
        assert session is not None
        assert session['user_id'] == user_id
    
    def test_get_valid_session(self, session_manager):
        """Test retrieving a valid session."""
        user_id = 1
        session_token = session_manager.create_session(user_id)
        
        session = session_manager.get_session(session_token)
        assert session is not None
        assert session['user_id'] == user_id
        assert session['is_valid'] is True
    
    def test_get_expired_session(self, session_manager):
        """Test that expired sessions are invalid."""
        user_id = 1
        session_token = session_manager.create_session(user_id)
        
        # Manually expire the session
        from app.db import DB
        expired_time = datetime.now() - timedelta(hours=25)
        with DB(session_manager.db_path) as db:
            db.conn.execute(
                "UPDATE session SET created_at = ? WHERE token = ?",
                (expired_time.isoformat(), session_token)
            )
        
        session = session_manager.get_session(session_token)
        assert session is None or session['is_valid'] is False
    
    def test_delete_session(self, session_manager):
        """Test deleting a session."""
        user_id = 1
        session_token = session_manager.create_session(user_id)
        
        # Verify session exists
        session = session_manager.get_session(session_token)
        assert session is not None
        
        # Delete session
        session_manager.delete_session(session_token)
        
        # Verify session no longer exists
        session = session_manager.get_session(session_token)
        assert session is None


class TestOAuthHandler:
    """Test OAuth authentication flow."""
    
    @pytest.fixture
    def oauth_handler(self, mock_google_auth, temp_db):
        """Create an OAuth handler instance."""
        with patch('app.auth.DB_PATH', temp_db):
            from app.auth import OAuth
            # Initialize database
            from app.db import DB
            with DB(temp_db) as db:
                schema_path = Path(__file__).parent.parent / 'schema.sql'
                if schema_path.exists():
                    db.init_schema(str(schema_path))
            return OAuth()
    
    def test_oauth_initialization(self, oauth_handler):
        """Test OAuth handler initialization."""
        assert oauth_handler.client_config is not None
        assert oauth_handler.client_config['web']['client_id'] == 'test-client-id'
    
    def test_get_authorization_url(self, oauth_handler):
        """Test generating authorization URL."""
        auth_url, state = oauth_handler.get_authorization_url()
        
        assert auth_url is not None
        assert 'accounts.google.com' in auth_url
        assert state is not None
        assert len(state) > 0
    
    @patch('google.oauth2.id_token.verify_oauth2_token')
    def test_verify_token_success(self, mock_verify, oauth_handler):
        """Test successful token verification."""
        mock_verify.return_value = {
            'email': 'test@example.com',
            'name': 'Test User',
            'iss': 'accounts.google.com'
        }
        
        result = oauth_handler.verify_token('test-token')
        
        assert result is not None
        assert result['email'] == 'test@example.com'
        assert result['name'] == 'Test User'
    
    @patch('google.oauth2.id_token.verify_oauth2_token')
    def test_verify_token_invalid(self, mock_verify, oauth_handler):
        """Test invalid token verification."""
        mock_verify.side_effect = ValueError("Invalid token")
        
        with pytest.raises(Exception):
            oauth_handler.verify_token('invalid-token')


class TestSimpleAuth:
    """Test simple email/password authentication."""
    
    @pytest.fixture
    def simple_auth(self, temp_db):
        """Create a simple auth instance."""
        with patch('app.auth_simple.DB_PATH', temp_db):
            from app.auth_simple import SimpleAuth
            # Initialize database
            from app.db import DB
            with DB(temp_db) as db:
                schema_path = Path(__file__).parent.parent / 'schema.sql'
                if schema_path.exists():
                    db.init_schema(str(schema_path))
            return SimpleAuth()
    
    def test_register_user(self, simple_auth):
        """Test user registration."""
        email = "test@example.com"
        password = "SecurePassword123"
        
        user_id = simple_auth.register_user(email, password, "Test User")
        
        assert user_id is not None
        assert user_id > 0
    
    def test_verify_password_success(self, simple_auth):
        """Test successful password verification."""
        email = "test@example.com"
        password = "SecurePassword123"
        
        # Register user
        simple_auth.register_user(email, password, "Test User")
        
        # Verify password
        user = simple_auth.verify_password(email, password)
        assert user is not None
        assert user['email'] == email
    
    def test_verify_password_wrong(self, simple_auth):
        """Test wrong password verification."""
        email = "test@example.com"
        password = "SecurePassword123"
        wrong_password = "WrongPassword123"
        
        # Register user
        simple_auth.register_user(email, password, "Test User")
        
        # Verify with wrong password
        user = simple_auth.verify_password(email, wrong_password)
        assert user is None
    
    def test_verify_password_nonexistent_user(self, simple_auth):
        """Test password verification for non-existent user."""
        user = simple_auth.verify_password("nonexistent@example.com", "password")
        assert user is None


class TestAPIKeyAuth:
    """Test API key authentication."""
    
    @pytest.fixture
    def api_client(self, temp_db):
        """Create a test client with mocked database."""
        with patch.dict(os.environ, {
            'API_KEY': 'test-api-key-123',
            'DB_PATH': temp_db
        }):
            # Import app after patching environment
            from app.main import app
            return TestClient(app)
    
    def test_api_key_required_endpoint(self, api_client):
        """Test endpoint that requires API key."""
        # Without API key
        response = api_client.post("/search", json={"query": "test"})
        assert response.status_code == 401
        
        # With wrong API key
        response = api_client.post(
            "/search",
            json={"query": "test"},
            headers={"X-API-Key": "wrong-key"}
        )
        assert response.status_code == 401
        
        # With correct API key
        with patch('app.main.store') as mock_store:
            mock_store.search.return_value = ([], [])
            response = api_client.post(
                "/search",
                json={"query": "test"},
                headers={"X-API-Key": "test-api-key-123"}
            )
            # May fail due to other issues, but should pass auth
            assert response.status_code != 401
    
    def test_health_check_no_auth(self, api_client):
        """Test that health check doesn't require authentication."""
        response = api_client.get("/healthz")
        assert response.status_code == 200


class TestAuthEndpoints:
    """Test authentication-related API endpoints."""
    
    @pytest.fixture
    def api_client(self, temp_db, mock_google_auth):
        """Create a test client with authentication mocked."""
        with patch.dict(os.environ, {
            'DB_PATH': temp_db,
            'ADMIN_EMAIL': 'admin@example.com',
            'ADMIN_PASSWORD': 'AdminPassword123'
        }):
            from app.main import app
            return TestClient(app)
    
    def test_login_endpoint(self, api_client):
        """Test login endpoint."""
        # Test with correct credentials
        with patch('app.auth_endpoints.simple_auth') as mock_auth:
            mock_auth.verify_password.return_value = {
                'id': 1,
                'email': 'admin@example.com'
            }
            with patch('app.auth_endpoints.session_manager') as mock_session:
                mock_session.create_session.return_value = 'test-session-token'
                
                response = api_client.post(
                    "/auth/login",
                    json={
                        "email": "admin@example.com",
                        "password": "AdminPassword123"
                    }
                )
                
                assert response.status_code == 200
                data = response.json()
                assert 'session_token' in data
    
    def test_logout_endpoint(self, api_client):
        """Test logout endpoint."""
        with patch('app.auth_endpoints.get_current_user') as mock_user:
            mock_user.return_value = {'id': 1, 'email': 'test@example.com'}
            with patch('app.auth_endpoints.session_manager') as mock_session:
                response = api_client.post(
                    "/auth/logout",
                    headers={"Authorization": "Bearer test-session-token"}
                )
                
                assert response.status_code == 200
                mock_session.delete_session.assert_called()
    
    def test_me_endpoint(self, api_client):
        """Test current user info endpoint."""
        with patch('app.auth_endpoints.get_current_user') as mock_user:
            mock_user.return_value = {
                'id': 1,
                'email': 'test@example.com',
                'display_name': 'Test User'
            }
            
            response = api_client.get(
                "/auth/me",
                headers={"Authorization": "Bearer test-session-token"}
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data['email'] == 'test@example.com'


if __name__ == "__main__":
    pytest.main([__file__, "-v"])