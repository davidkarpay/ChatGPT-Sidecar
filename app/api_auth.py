"""Role-based API key authentication with audit logging."""

import os
import time
import logging
from datetime import datetime
from enum import Enum
from typing import Optional, Dict, Any, List
from fastapi import HTTPException, Header, Request
from functools import wraps

from .db import DB

logger = logging.getLogger(__name__)

class AccessLevel(Enum):
    """API access levels with different permissions."""
    READ = "read"      # Can search, view documents, chat
    ADMIN = "admin"    # Can ingest, reindex, manage system
    SUPER = "super"    # Can manage users, keys, system config

class RateLimitError(HTTPException):
    """Rate limit exceeded error."""
    def __init__(self, detail: str = "Rate limit exceeded"):
        super().__init__(status_code=429, detail=detail)

class APIKeyManager:
    """Manage API keys with roles and rate limiting."""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.rate_limit_cache = {}  # In-memory rate limiting cache
        self._load_keys()
    
    def _load_keys(self):
        """Load API keys from environment variables."""
        self.keys = {}
        
        # Legacy single API key (admin access)
        legacy_key = os.getenv("API_KEY")
        if legacy_key and legacy_key != "change-me":
            self.keys[legacy_key] = {
                "level": AccessLevel.ADMIN,
                "name": "Legacy API Key",
                "rate_limit": 1000  # requests per hour
            }
        
        # Role-based API keys
        read_key = os.getenv("READ_KEY")
        if read_key:
            self.keys[read_key] = {
                "level": AccessLevel.READ,
                "name": "Read Access Key",
                "rate_limit": 500  # requests per hour
            }
        
        admin_key = os.getenv("ADMIN_KEY")
        if admin_key:
            self.keys[admin_key] = {
                "level": AccessLevel.ADMIN,
                "name": "Admin Access Key", 
                "rate_limit": 1000  # requests per hour
            }
            
        super_key = os.getenv("SUPER_KEY")
        if super_key:
            self.keys[super_key] = {
                "level": AccessLevel.SUPER,
                "name": "Super Admin Key",
                "rate_limit": 2000  # requests per hour
            }
        
        if not self.keys:
            logger.warning("No API keys configured! Set READ_KEY, ADMIN_KEY, or API_KEY environment variables.")
    
    def validate_key(self, api_key: str, required_level: AccessLevel = AccessLevel.READ) -> Dict[str, Any]:
        """Validate API key and check access level."""
        if not api_key or api_key not in self.keys:
            raise HTTPException(
                status_code=401, 
                detail="Invalid API key",
                headers={"WWW-Authenticate": "ApiKey"}
            )
        
        key_info = self.keys[api_key]
        key_level = key_info["level"]
        
        # Check if key has sufficient access level
        level_hierarchy = {
            AccessLevel.READ: 1,
            AccessLevel.ADMIN: 2,
            AccessLevel.SUPER: 3
        }
        
        if level_hierarchy[key_level] < level_hierarchy[required_level]:
            raise HTTPException(
                status_code=403,
                detail=f"Insufficient permissions. Required: {required_level.value}, have: {key_level.value}"
            )
        
        return {
            "key": api_key,
            "level": key_level,
            "name": key_info["name"],
            "rate_limit": key_info["rate_limit"]
        }
    
    def check_rate_limit(self, api_key: str, request: Request) -> None:
        """Check rate limiting for API key."""
        if api_key not in self.keys:
            return  # Invalid key will be caught by validate_key
            
        key_info = self.keys[api_key]
        rate_limit = key_info["rate_limit"]
        
        current_time = time.time()
        hour_bucket = int(current_time // 3600)  # Group by hour
        cache_key = f"{api_key}:{hour_bucket}"
        
        # Clean old entries (keep only current and previous hour)
        to_remove = []
        for key in self.rate_limit_cache:
            bucket = int(key.split(":")[1])
            if bucket < hour_bucket - 1:
                to_remove.append(key)
        
        for key in to_remove:
            del self.rate_limit_cache[key]
        
        # Check current usage
        current_count = self.rate_limit_cache.get(cache_key, 0)
        
        if current_count >= rate_limit:
            # Log rate limit violation
            self.log_auth_event(
                api_key=api_key,
                event_type="rate_limit_exceeded",
                ip_address=request.client.host,
                endpoint=request.url.path,
                success=False,
                details=f"Exceeded {rate_limit} requests per hour"
            )
            raise RateLimitError(
                detail=f"Rate limit exceeded: {rate_limit} requests per hour"
            )
        
        # Increment counter
        self.rate_limit_cache[cache_key] = current_count + 1
    
    def log_auth_event(self, api_key: str, event_type: str, ip_address: str, 
                      endpoint: str, success: bool, details: Optional[str] = None):
        """Log authentication events for audit purposes."""
        try:
            # Hash the API key for logging (keep only first 8 chars)
            key_hash = api_key[:8] + "..." if api_key else "none"
            
            with DB(self.db_path) as db:
                db.conn.execute("""
                    INSERT INTO auth_audit_log (
                        timestamp, api_key_hash, event_type, ip_address, 
                        endpoint, success, details
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    datetime.now().isoformat(),
                    key_hash,
                    event_type,
                    ip_address,
                    endpoint,
                    success,
                    details
                ))
            
            logger.info(f"Auth event: {event_type} for key {key_hash} from {ip_address} -> {success}")
            
        except Exception as e:
            logger.error(f"Failed to log auth event: {e}")

# Global API key manager instance
api_key_manager = None

def get_api_key_manager() -> APIKeyManager:
    """Get or create global API key manager."""
    global api_key_manager
    if api_key_manager is None:
        db_path = os.getenv("DB_PATH", "sidecar.db")
        api_key_manager = APIKeyManager(db_path)
    return api_key_manager

# FastAPI dependency functions
async def require_read_access(
    request: Request,
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key")
) -> Dict[str, Any]:
    """Require READ level API access."""
    manager = get_api_key_manager()
    
    try:
        # Check rate limiting first
        if x_api_key:
            manager.check_rate_limit(x_api_key, request)
        
        # Validate key and access level
        auth_info = manager.validate_key(x_api_key, AccessLevel.READ)
        
        # Log successful auth
        manager.log_auth_event(
            api_key=x_api_key,
            event_type="api_access",
            ip_address=request.client.host,
            endpoint=request.url.path,
            success=True
        )
        
        return auth_info
        
    except HTTPException as e:
        # Log failed auth attempt
        manager.log_auth_event(
            api_key=x_api_key or "none",
            event_type="api_access_denied",
            ip_address=request.client.host,
            endpoint=request.url.path,
            success=False,
            details=e.detail
        )
        raise

async def require_admin_access(
    request: Request,
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key")
) -> Dict[str, Any]:
    """Require ADMIN level API access."""
    manager = get_api_key_manager()
    
    try:
        # Check rate limiting first
        if x_api_key:
            manager.check_rate_limit(x_api_key, request)
        
        # Validate key and access level
        auth_info = manager.validate_key(x_api_key, AccessLevel.ADMIN)
        
        # Log successful auth
        manager.log_auth_event(
            api_key=x_api_key,
            event_type="admin_access",
            ip_address=request.client.host,
            endpoint=request.url.path,
            success=True
        )
        
        return auth_info
        
    except HTTPException as e:
        # Log failed auth attempt
        manager.log_auth_event(
            api_key=x_api_key or "none",
            event_type="admin_access_denied",
            ip_address=request.client.host,
            endpoint=request.url.path,
            success=False,
            details=e.detail
        )
        raise

async def require_super_access(
    request: Request,
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key")
) -> Dict[str, Any]:
    """Require SUPER level API access."""
    manager = get_api_key_manager()
    
    try:
        # Check rate limiting first  
        if x_api_key:
            manager.check_rate_limit(x_api_key, request)
        
        # Validate key and access level
        auth_info = manager.validate_key(x_api_key, AccessLevel.SUPER)
        
        # Log successful auth
        manager.log_auth_event(
            api_key=x_api_key,
            event_type="super_access",
            ip_address=request.client.host,
            endpoint=request.url.path,
            success=True
        )
        
        return auth_info
        
    except HTTPException as e:
        # Log failed auth attempt
        manager.log_auth_event(
            api_key=x_api_key or "none",
            event_type="super_access_denied",
            ip_address=request.client.host,
            endpoint=request.url.path,
            success=False,
            details=e.detail
        )
        raise

# Legacy compatibility
async def auth(x_api_key: str = Header(default=None, alias="X-API-Key")):
    """Legacy auth function for backward compatibility."""
    request = Request(scope={"type": "http", "client": ["127.0.0.1", 0]})
    return await require_read_access(request, x_api_key)