"""Security configuration and path validation for ingestion."""

import os
from pathlib import Path
from typing import Optional

# Security toggles
INGEST_ALLOW_ARBITRARY_FS = os.getenv("INGEST_ALLOW_ARBITRARY_FS", "false").lower() == "true"
ALLOWED_DATA_ROOT = Path(os.getenv("ALLOWED_DATA_ROOT", "/data/imports")).resolve()
DISABLE_ADMIN_ENDPOINTS = os.getenv("DISABLE_ADMIN_ENDPOINTS", "false").lower() == "true"

# Railway/production paths
DATA_ROOT = Path(os.getenv("DATA_ROOT", "/data"))
SQLITE_DB = Path(os.getenv("SQLITE_DB", "/data/sidecar.db"))
FAISS_INDEX_DIR = Path(os.getenv("FAISS_INDEX_DIR", "/data/indexes"))
EMBED_CACHE_DIR = Path(os.getenv("EMBED_CACHE_DIR", "/data/.cache"))


def ensure_under_allowed_root(p: Path) -> Path:
    """
    Ensure that a given path is within ALLOWED_DATA_ROOT for security.
    
    Args:
        p: Path to validate
        
    Returns:
        Resolved path if valid
        
    Raises:
        PermissionError: If path is outside allowed root and arbitrary FS access is disabled
    """
    p_resolved = p.resolve()
    
    if not INGEST_ALLOW_ARBITRARY_FS:
        if not str(p_resolved).startswith(str(ALLOWED_DATA_ROOT)):
            raise PermissionError(
                f"Ingestion path {p_resolved} is outside ALLOWED_DATA_ROOT ({ALLOWED_DATA_ROOT}). "
                f"Set INGEST_ALLOW_ARBITRARY_FS=true to allow arbitrary filesystem access."
            )
    
    return p_resolved


def ensure_directories_exist():
    """Create necessary directories if they don't exist."""
    directories = [
        DATA_ROOT,
        FAISS_INDEX_DIR, 
        EMBED_CACHE_DIR,
        ALLOWED_DATA_ROOT
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


def get_config_summary() -> dict:
    """Get current configuration for debugging/logging."""
    return {
        "ingest_allow_arbitrary_fs": INGEST_ALLOW_ARBITRARY_FS,
        "allowed_data_root": str(ALLOWED_DATA_ROOT),
        "disable_admin_endpoints": DISABLE_ADMIN_ENDPOINTS,
        "data_root": str(DATA_ROOT),
        "sqlite_db": str(SQLITE_DB),
        "faiss_index_dir": str(FAISS_INDEX_DIR),
        "embed_cache_dir": str(EMBED_CACHE_DIR),
    }