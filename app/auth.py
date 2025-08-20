"""Google OAuth authentication and user session management."""

import os
import secrets
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
import json

from google.auth.transport.requests import Request
from google.oauth2 import id_token
from google_auth_oauthlib.flow import Flow
import google.auth.exceptions

from fastapi import HTTPException, status, Request as FastAPIRequest, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from .db import DB


# OAuth configuration
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
GOOGLE_DISCOVERY_URL = "https://accounts.google.com/.well-known/openid_configuration"

# Session configuration
SESSION_LIFETIME_HOURS = int(os.getenv("SESSION_LIFETIME_HOURS", "24"))
REDIRECT_URI = os.getenv("REDIRECT_URI", "http://localhost:8088/auth/callback")

# Security
security = HTTPBearer(auto_error=False)

class AuthenticationError(Exception):
    """Authentication related errors."""
    pass

class OAuth:
    """Google OAuth handler."""
    
    def __init__(self):
        if not GOOGLE_CLIENT_ID or not GOOGLE_CLIENT_SECRET:
            raise ValueError("Google OAuth credentials not configured. Set GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET environment variables.")
        
        self.client_config = {
            "web": {
                "client_id": GOOGLE_CLIENT_ID,
                "client_secret": GOOGLE_CLIENT_SECRET,
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
                "redirect_uris": [REDIRECT_URI]
            }
        }
    
    def get_authorization_url(self, state: str) -> str:
        """Get Google OAuth authorization URL."""
        flow = Flow.from_client_config(
            self.client_config,
            scopes=["openid", "email", "profile"],
            redirect_uri=REDIRECT_URI
        )
        flow.redirect_uri = REDIRECT_URI
        
        authorization_url, _ = flow.authorization_url(
            access_type="offline",
            include_granted_scopes="true",
            state=state
        )
        
        return authorization_url
    
    def exchange_code_for_token(self, code: str, state: str) -> Dict[str, Any]:
        """Exchange authorization code for tokens and user info."""
        flow = Flow.from_client_config(
            self.client_config,
            scopes=["openid", "email", "profile"],
            redirect_uri=REDIRECT_URI,
            state=state
        )
        
        try:
            flow.fetch_token(code=code)
            
            # Get user info from ID token
            id_info = id_token.verify_oauth2_token(
                flow.credentials.id_token,
                Request(),
                GOOGLE_CLIENT_ID
            )
            
            return {
                "email": id_info["email"],
                "name": id_info.get("name", ""),
                "picture": id_info.get("picture", ""),
                "google_id": id_info["sub"],
                "verified_email": id_info.get("email_verified", False)
            }
            
        except Exception as e:
            raise AuthenticationError(f"OAuth token exchange failed: {e}")

class SessionManager:
    """User session management."""
    
    def create_session(self, user_info: Dict[str, Any]) -> str:
        """Create a new user session."""
        session_token = secrets.token_urlsafe(32)
        
        with DB() as db:
            # Get or create user
            user_id = db.upsert_user(
                email=user_info["email"],
                display_name=user_info.get("name", "")
            )
            
            # Store additional OAuth info
            db.conn.execute("""
                INSERT OR REPLACE INTO user_oauth (
                    user_id, google_id, email, name, picture, verified_email
                ) VALUES (?, ?, ?, ?, ?, ?)
            """, (
                user_id,
                user_info["google_id"],
                user_info["email"],
                user_info.get("name", ""),
                user_info.get("picture", ""),
                user_info.get("verified_email", False)
            ))
            
            # Create session
            expires_at = datetime.now() + timedelta(hours=SESSION_LIFETIME_HOURS)
            db.conn.execute("""
                INSERT INTO user_sessions (
                    session_token, user_id, expires_at, created_at, last_accessed
                ) VALUES (?, ?, ?, ?, ?)
            """, (
                session_token,
                user_id,
                expires_at.isoformat(),
                datetime.now().isoformat(),
                datetime.now().isoformat()
            ))
        
        return session_token
    
    def get_user_from_session(self, session_token: str) -> Optional[Dict[str, Any]]:
        """Get user info from session token."""
        with DB() as db:
            cur = db.conn.execute("""
                SELECT 
                    us.user_id,
                    us.expires_at,
                    u.email,
                    u.display_name,
                    uo.google_id,
                    uo.picture,
                    uo.verified_email
                FROM user_sessions us
                JOIN user u ON us.user_id = u.id
                LEFT JOIN user_oauth uo ON u.id = uo.user_id
                WHERE us.session_token = ? AND us.expires_at > ?
            """, (session_token, datetime.now().isoformat()))
            
            row = cur.fetchone()
            if not row:
                return None
            
            # Update last accessed
            db.conn.execute("""
                UPDATE user_sessions 
                SET last_accessed = ? 
                WHERE session_token = ?
            """, (datetime.now().isoformat(), session_token))
            
            return {
                "user_id": row["user_id"],
                "email": row["email"],
                "display_name": row["display_name"],
                "google_id": row["google_id"],
                "picture": row["picture"],
                "verified_email": bool(row["verified_email"])
            }
    
    def invalidate_session(self, session_token: str) -> bool:
        """Invalidate a session token."""
        with DB() as db:
            cur = db.conn.execute("""
                DELETE FROM user_sessions WHERE session_token = ?
            """, (session_token,))
            return cur.rowcount > 0
    
    def cleanup_expired_sessions(self):
        """Remove expired sessions from database."""
        with DB() as db:
            db.conn.execute("""
                DELETE FROM user_sessions WHERE expires_at <= ?
            """, (datetime.now().isoformat(),))

# Global instances (initialized conditionally)
try:
    oauth_handler = OAuth()
    session_manager = SessionManager()
    OAUTH_ENABLED = True
except ValueError as e:
    print(f"Warning: OAuth not configured - {e}")
    oauth_handler = None
    session_manager = None
    OAUTH_ENABLED = False

# FastAPI dependencies
async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> Dict[str, Any]:
    """FastAPI dependency to get current authenticated user."""
    if not OAUTH_ENABLED:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="OAuth authentication not configured"
        )
    
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    user = session_manager.get_user_from_session(credentials.credentials)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired session",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return user

async def get_optional_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> Optional[Dict[str, Any]]:
    """FastAPI dependency to get current user if authenticated, None otherwise."""
    if not OAUTH_ENABLED or not credentials:
        return None
    
    return session_manager.get_user_from_session(credentials.credentials)

def require_auth(func):
    """Decorator to require authentication for a route."""
    def wrapper(*args, **kwargs):
        user = kwargs.get('current_user')
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication required"
            )
        return func(*args, **kwargs)
    return wrapper