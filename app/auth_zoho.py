"""Zoho OAuth authentication integration."""

import os
import secrets
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
import json

import httpx
from google_auth_oauthlib.flow import Flow

from fastapi import HTTPException, status, Request as FastAPIRequest, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from .db import DB

# Zoho OAuth configuration
ZOHO_CLIENT_ID = os.getenv("ZOHO_CLIENT_ID")
ZOHO_CLIENT_SECRET = os.getenv("ZOHO_CLIENT_SECRET")
ZOHO_DOMAIN = os.getenv("ZOHO_DOMAIN", "accounts.zoho.com")  # or accounts.zoho.eu, accounts.zoho.in
REDIRECT_URI = os.getenv("REDIRECT_URI", "http://localhost:8088/auth/callback")

# Session configuration
SESSION_LIFETIME_HOURS = int(os.getenv("SESSION_LIFETIME_HOURS", "24"))

# Security
security = HTTPBearer(auto_error=False)

class AuthenticationError(Exception):
    """Authentication related errors."""
    pass

class ZohoOAuth:
    """Zoho OAuth handler."""
    
    def __init__(self):
        if not ZOHO_CLIENT_ID or not ZOHO_CLIENT_SECRET:
            raise ValueError("Zoho OAuth credentials not configured. Set ZOHO_CLIENT_ID and ZOHO_CLIENT_SECRET environment variables.")
        
        self.client_id = ZOHO_CLIENT_ID
        self.client_secret = ZOHO_CLIENT_SECRET
        self.domain = ZOHO_DOMAIN
        self.auth_url = f"https://{self.domain}/oauth/v2/auth"
        self.token_url = f"https://{self.domain}/oauth/v2/token"
        self.user_info_url = f"https://{self.domain}/oauth/user/info"
    
    def get_authorization_url(self, state: str) -> str:
        """Get Zoho OAuth authorization URL."""
        params = {
            "scope": "AaaServer.profile.Read",
            "client_id": self.client_id,
            "response_type": "code",
            "access_type": "offline",
            "redirect_uri": REDIRECT_URI,
            "state": state
        }
        
        param_string = "&".join([f"{k}={v}" for k, v in params.items()])
        return f"{self.auth_url}?{param_string}"
    
    async def exchange_code_for_token(self, code: str, state: str) -> Dict[str, Any]:
        """Exchange authorization code for tokens and user info."""
        try:
            # Exchange code for access token
            async with httpx.AsyncClient() as client:
                token_response = await client.post(
                    self.token_url,
                    data={
                        "grant_type": "authorization_code",
                        "client_id": self.client_id,
                        "client_secret": self.client_secret,
                        "redirect_uri": REDIRECT_URI,
                        "code": code
                    },
                    headers={"Content-Type": "application/x-www-form-urlencoded"}
                )
                
                token_response.raise_for_status()
                token_data = token_response.json()
                
                if "access_token" not in token_data:
                    raise AuthenticationError(f"No access token in response: {token_data}")
                
                access_token = token_data["access_token"]
                
                # Get user info
                user_response = await client.get(
                    self.user_info_url,
                    headers={"Authorization": f"Zoho-oauthtoken {access_token}"}
                )
                
                user_response.raise_for_status()
                user_data = user_response.json()
                
                return {
                    "email": user_data.get("Email", ""),
                    "name": user_data.get("Display_Name", ""),
                    "picture": user_data.get("profile_photo", ""),
                    "zoho_id": user_data.get("ZUID", ""),
                    "verified_email": True,  # Zoho emails are verified
                    "access_token": access_token,
                    "refresh_token": token_data.get("refresh_token", "")
                }
                
        except httpx.HTTPStatusError as e:
            raise AuthenticationError(f"HTTP error during token exchange: {e.response.text}")
        except Exception as e:
            raise AuthenticationError(f"OAuth token exchange failed: {e}")

class SessionManager:
    """User session management for Zoho OAuth."""
    
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
                    user_id, zoho_id, email, name, picture, verified_email, access_token, refresh_token
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                user_id,
                user_info["zoho_id"],
                user_info["email"],
                user_info.get("name", ""),
                user_info.get("picture", ""),
                user_info.get("verified_email", False),
                user_info.get("access_token", ""),
                user_info.get("refresh_token", "")
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
                    uo.zoho_id,
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
                "zoho_id": row["zoho_id"],
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
    zoho_oauth_handler = ZohoOAuth()
    zoho_session_manager = SessionManager()
    ZOHO_OAUTH_ENABLED = True
except ValueError as e:
    print(f"Warning: Zoho OAuth not configured - {e}")
    zoho_oauth_handler = None
    zoho_session_manager = None
    ZOHO_OAUTH_ENABLED = False

# FastAPI dependencies
async def get_current_user_zoho(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> Dict[str, Any]:
    """FastAPI dependency to get current authenticated user via Zoho."""
    if not ZOHO_OAUTH_ENABLED:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="Zoho OAuth authentication not configured"
        )
    
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    user = zoho_session_manager.get_user_from_session(credentials.credentials)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired session",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return user

async def get_optional_user_zoho(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> Optional[Dict[str, Any]]:
    """FastAPI dependency to get current user if authenticated via Zoho, None otherwise."""
    if not ZOHO_OAUTH_ENABLED or not credentials:
        return None
    
    return zoho_session_manager.get_user_from_session(credentials.credentials)