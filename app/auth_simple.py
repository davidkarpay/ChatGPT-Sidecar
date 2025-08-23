"""Simple email/password authentication for private use."""

import os
import secrets
import hashlib
from typing import Optional, Dict, Any
from datetime import datetime, timedelta

from fastapi import HTTPException, status, Request as FastAPIRequest, Depends, Form
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from passlib.context import CryptContext

from .db import DB

# Configuration
SESSION_LIFETIME_HOURS = int(os.getenv("SESSION_LIFETIME_HOURS", "24"))
ADMIN_EMAIL = os.getenv("ADMIN_EMAIL", "admin@localhost")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD")  # Set this in your environment

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Security
security = HTTPBearer(auto_error=False)

class SimpleAuth:
    """Simple email/password authentication."""
    
    def __init__(self):
        if not ADMIN_PASSWORD:
            raise ValueError("Admin password not configured. Set ADMIN_PASSWORD environment variable.")
        
        # Ensure admin user exists
        self._ensure_admin_user()
    
    def _ensure_admin_user(self):
        """Ensure admin user exists in database."""
        with DB() as db:
            # Check if admin user exists
            cur = db.conn.execute("SELECT id FROM user WHERE email = ?", (ADMIN_EMAIL,))
            if not cur.fetchone():
                # Create admin user
                user_id = db.upsert_user(ADMIN_EMAIL, "Administrator")
                
                # Store password hash
                password_hash = pwd_context.hash(ADMIN_PASSWORD)
                db.conn.execute("""
                    INSERT INTO user_passwords (user_id, password_hash, created_at)
                    VALUES (?, ?, ?)
                """, (user_id, password_hash, datetime.now().isoformat()))
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify a password against its hash."""
        return pwd_context.verify(plain_password, hashed_password)
    
    def authenticate_user(self, email: str, password: str) -> Optional[Dict[str, Any]]:
        """Authenticate user with email and password."""
        with DB() as db:
            cur = db.conn.execute("""
                SELECT u.id, u.email, u.display_name, up.password_hash
                FROM user u
                JOIN user_passwords up ON u.id = up.user_id
                WHERE u.email = ?
            """, (email,))
            
            user = cur.fetchone()
            if not user:
                return None
            
            if not self.verify_password(password, user["password_hash"]):
                return None
            
            return {
                "user_id": user["id"],
                "email": user["email"],
                "display_name": user["display_name"]
            }
    
    def create_session(self, user_info: Dict[str, Any]) -> str:
        """Create a new user session."""
        session_token = secrets.token_urlsafe(32)
        
        with DB() as db:
            expires_at = datetime.now() + timedelta(hours=SESSION_LIFETIME_HOURS)
            db.conn.execute("""
                INSERT INTO user_sessions (
                    session_token, user_id, expires_at, created_at, last_accessed
                ) VALUES (?, ?, ?, ?, ?)
            """, (
                session_token,
                user_info["user_id"],
                expires_at.isoformat(),
                datetime.now().isoformat(),
                datetime.now().isoformat()
            ))
        
        return session_token
    
    def get_user_from_session(self, session_token: str) -> Optional[Dict[str, Any]]:
        """Get user info from session token."""
        with DB() as db:
            cur = db.conn.execute("""
                SELECT us.user_id, u.email, u.display_name, us.expires_at
                FROM user_sessions us
                JOIN user u ON us.user_id = u.id
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
                "display_name": row["display_name"]
            }
    
    def invalidate_session(self, session_token: str) -> bool:
        """Invalidate a session token."""
        with DB() as db:
            cur = db.conn.execute("""
                DELETE FROM user_sessions WHERE session_token = ?
            """, (session_token,))
            return cur.rowcount > 0

# Global instance
try:
    simple_auth = SimpleAuth()
    SIMPLE_AUTH_ENABLED = True
except ValueError as e:
    print(f"Warning: Simple auth not configured - {e}")
    simple_auth = None
    SIMPLE_AUTH_ENABLED = False

# FastAPI dependencies
async def get_current_user_simple(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> Dict[str, Any]:
    """FastAPI dependency to get current authenticated user."""
    if not SIMPLE_AUTH_ENABLED:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="Simple authentication not configured"
        )
    
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    user = simple_auth.get_user_from_session(credentials.credentials)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired session",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return user

async def get_optional_user_simple(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> Optional[Dict[str, Any]]:
    """FastAPI dependency to get current user if authenticated, None otherwise."""
    if not SIMPLE_AUTH_ENABLED or not credentials:
        return None
    
    return simple_auth.get_user_from_session(credentials.credentials)