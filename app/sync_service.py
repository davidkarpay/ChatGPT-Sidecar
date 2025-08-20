"""ChatGPT data synchronization service for automated import."""

import os
import json
import hashlib
import asyncio
import aiofiles
import tempfile
import zipfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging

import httpx
from celery import Celery

from .db import DB
from .ingest_chatgpt import ingest_export


# Celery configuration
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
celery_app = Celery("sync_service", broker=REDIS_URL, backend=REDIS_URL)

# ChatGPT sync configuration
CHATGPT_SYNC_ENABLED = os.getenv("CHATGPT_SYNC_ENABLED", "false").lower() == "true"
CHATGPT_EXPORT_URL = os.getenv("CHATGPT_EXPORT_URL")  # URL to download export from
CHATGPT_SYNC_INTERVAL_HOURS = int(os.getenv("CHATGPT_SYNC_INTERVAL_HOURS", "24"))

logger = logging.getLogger(__name__)

class ChatGPTSyncError(Exception):
    """Errors related to ChatGPT synchronization."""
    pass

class SyncStatus:
    """Sync status tracking."""
    
    def __init__(self, db_path: str = None):
        self.db_path = db_path or os.getenv("DB_PATH", "sidecar.db")
    
    def get_last_sync(self, user_id: int) -> Optional[Dict[str, Any]]:
        """Get the last sync record for a user."""
        with DB(self.db_path) as db:
            cur = db.conn.execute("""
                SELECT sync_id, started_at, completed_at, status, error_message, 
                       files_processed, conversations_added, conversations_updated
                FROM sync_history 
                WHERE user_id = ? 
                ORDER BY started_at DESC 
                LIMIT 1
            """, (user_id,))
            
            row = cur.fetchone()
            if not row:
                return None
            
            return {
                "sync_id": row["sync_id"],
                "started_at": row["started_at"],
                "completed_at": row["completed_at"],
                "status": row["status"],
                "error_message": row["error_message"],
                "files_processed": row["files_processed"],
                "conversations_added": row["conversations_added"],
                "conversations_updated": row["conversations_updated"]
            }
    
    def create_sync_record(self, user_id: int, sync_type: str = "scheduled") -> str:
        """Create a new sync record and return sync_id."""
        sync_id = f"sync_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{user_id}"
        
        with DB(self.db_path) as db:
            db.conn.execute("""
                INSERT INTO sync_history (
                    sync_id, user_id, sync_type, status, started_at
                ) VALUES (?, ?, ?, ?, ?)
            """, (sync_id, user_id, sync_type, "running", datetime.now().isoformat()))
        
        return sync_id
    
    def update_sync_record(self, sync_id: str, status: str, error_message: str = None, 
                          files_processed: int = 0, conversations_added: int = 0, 
                          conversations_updated: int = 0):
        """Update sync record with results."""
        with DB(self.db_path) as db:
            db.conn.execute("""
                UPDATE sync_history 
                SET status = ?, completed_at = ?, error_message = ?,
                    files_processed = ?, conversations_added = ?, conversations_updated = ?
                WHERE sync_id = ?
            """, (
                status, 
                datetime.now().isoformat(),
                error_message,
                files_processed,
                conversations_added,
                conversations_updated,
                sync_id
            ))

class ChatGPTSync:
    """ChatGPT data synchronization service."""
    
    def __init__(self, db_path: str = None):
        self.db_path = db_path or os.getenv("DB_PATH", "sidecar.db")
        self.status = SyncStatus(self.db_path)
    
    async def download_export(self, export_url: str, target_path: Path) -> bool:
        """Download ChatGPT export from URL."""
        try:
            async with httpx.AsyncClient(timeout=300.0) as client:
                logger.info(f"Downloading ChatGPT export from {export_url}")
                
                response = await client.get(export_url)
                response.raise_for_status()
                
                async with aiofiles.open(target_path, 'wb') as f:
                    await f.write(response.content)
                
                logger.info(f"Export downloaded successfully to {target_path}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to download export: {e}")
            raise ChatGPTSyncError(f"Download failed: {e}")
    
    def extract_and_validate_export(self, zip_path: Path, extract_dir: Path) -> List[Path]:
        """Extract and validate ChatGPT export ZIP file."""
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
            
            # Find conversation JSON files
            conversation_files = list(extract_dir.glob("conversations.json"))
            if not conversation_files:
                # Check for other patterns
                conversation_files = list(extract_dir.glob("**/conversations.json"))
            
            if not conversation_files:
                raise ChatGPTSyncError("No conversations.json file found in export")
            
            logger.info(f"Found {len(conversation_files)} conversation files")
            return conversation_files
            
        except zipfile.BadZipFile:
            raise ChatGPTSyncError("Invalid ZIP file")
        except Exception as e:
            raise ChatGPTSyncError(f"Extraction failed: {e}")
    
    def calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA-256 hash of file for deduplication."""
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    
    def is_file_already_processed(self, user_id: int, file_hash: str) -> bool:
        """Check if file has already been processed."""
        with DB(self.db_path) as db:
            cur = db.conn.execute("""
                SELECT 1 FROM sync_file_history 
                WHERE user_id = ? AND file_hash = ?
            """, (user_id, file_hash))
            return cur.fetchone() is not None
    
    def record_processed_file(self, user_id: int, file_path: str, file_hash: str, sync_id: str):
        """Record that a file has been processed."""
        with DB(self.db_path) as db:
            db.conn.execute("""
                INSERT OR REPLACE INTO sync_file_history 
                (user_id, file_path, file_hash, sync_id, processed_at)
                VALUES (?, ?, ?, ?, ?)
            """, (user_id, file_path, file_hash, sync_id, datetime.now().isoformat()))
    
    async def sync_user_data(self, user_id: int, export_url: str = None) -> Dict[str, Any]:
        """Sync ChatGPT data for a specific user."""
        sync_id = self.status.create_sync_record(user_id, "manual" if export_url else "scheduled")
        
        try:
            if not export_url:
                export_url = CHATGPT_EXPORT_URL
            
            if not export_url:
                raise ChatGPTSyncError("No export URL configured")
            
            # Create temporary directory for processing
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                zip_path = temp_path / "export.zip"
                extract_dir = temp_path / "extracted"
                extract_dir.mkdir()
                
                # Download export
                await self.download_export(export_url, zip_path)
                
                # Extract and validate
                conversation_files = self.extract_and_validate_export(zip_path, extract_dir)
                
                files_processed = 0
                conversations_added = 0
                conversations_updated = 0
                
                # Process each conversation file
                for conv_file in conversation_files:
                    file_hash = self.calculate_file_hash(conv_file)
                    
                    # Skip if already processed
                    if self.is_file_already_processed(user_id, file_hash):
                        logger.info(f"Skipping already processed file: {conv_file}")
                        continue
                    
                    # Import conversations
                    try:
                        logger.info(f"Processing conversation file: {conv_file}")
                        
                        # Use existing ChatGPT ingest function
                        with open(conv_file, 'r', encoding='utf-8') as f:
                            conversations_data = json.load(f)
                        
                        # Create project for this sync if it doesn't exist
                        project_name = f"ChatGPT Sync {datetime.now().strftime('%Y-%m')}"
                        
                        with DB(self.db_path) as db:
                            cur = db.conn.execute("""
                                SELECT id FROM project 
                                WHERE user_id = ? AND name = ?
                            """, (user_id, project_name))
                            
                            project_row = cur.fetchone()
                            if project_row:
                                project_id = project_row["id"]
                            else:
                                cur = db.conn.execute("""
                                    INSERT INTO project (user_id, name, description)
                                    VALUES (?, ?, ?)
                                """, (user_id, project_name, "Automatically synced ChatGPT conversations"))
                                project_id = cur.lastrowid
                        
                        # Import using existing ingest function
                        result = ingest_export(str(conv_file), project_id=project_id)
                        
                        conversations_added += len(conversations_data) if isinstance(conversations_data, list) else 1
                        files_processed += 1
                        
                        # Record that this file was processed
                        self.record_processed_file(user_id, str(conv_file), file_hash, sync_id)
                        
                        logger.info(f"Successfully processed {conv_file}")
                        
                    except Exception as e:
                        logger.error(f"Failed to process {conv_file}: {e}")
                        # Continue with other files
                        continue
                
                # Update sync record with success
                self.status.update_sync_record(
                    sync_id=sync_id,
                    status="completed",
                    files_processed=files_processed,
                    conversations_added=conversations_added,
                    conversations_updated=conversations_updated
                )
                
                return {
                    "sync_id": sync_id,
                    "status": "completed",
                    "files_processed": files_processed,
                    "conversations_added": conversations_added,
                    "conversations_updated": conversations_updated
                }
                
        except Exception as e:
            logger.error(f"Sync failed for user {user_id}: {e}")
            
            # Update sync record with failure
            self.status.update_sync_record(
                sync_id=sync_id,
                status="failed",
                error_message=str(e)
            )
            
            raise ChatGPTSyncError(f"Sync failed: {e}")

# Celery tasks
@celery_app.task(bind=True, max_retries=3, default_retry_delay=300)
def sync_user_chatgpt_data(self, user_id: int, export_url: str = None):
    """Celery task to sync ChatGPT data for a user."""
    try:
        sync_service = ChatGPTSync()
        
        # Run the async sync function
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(
                sync_service.sync_user_data(user_id, export_url)
            )
            return result
        finally:
            loop.close()
            
    except ChatGPTSyncError as e:
        logger.error(f"Sync task failed for user {user_id}: {e}")
        raise self.retry(exc=e)
    except Exception as e:
        logger.error(f"Unexpected error in sync task for user {user_id}: {e}")
        raise

@celery_app.task
def scheduled_sync_all_users():
    """Celery task to sync all users with automatic sync enabled."""
    if not CHATGPT_SYNC_ENABLED:
        logger.info("ChatGPT sync disabled, skipping scheduled sync")
        return
    
    sync_service = ChatGPTSync()
    
    # Get all users with sync enabled
    with DB() as db:
        cur = db.conn.execute("""
            SELECT DISTINCT user_id FROM user_sync_config 
            WHERE sync_enabled = 1 AND chatgpt_export_url IS NOT NULL
        """)
        
        users = cur.fetchall()
    
    if not users:
        logger.info("No users configured for automatic sync")
        return
    
    logger.info(f"Starting scheduled sync for {len(users)} users")
    
    for user in users:
        user_id = user["user_id"]
        
        # Get user's export URL
        with DB() as db:
            cur = db.conn.execute("""
                SELECT chatgpt_export_url FROM user_sync_config 
                WHERE user_id = ?
            """, (user_id,))
            
            config_row = cur.fetchone()
            if not config_row:
                continue
                
            export_url = config_row["chatgpt_export_url"]
        
        # Schedule sync task for this user
        sync_user_chatgpt_data.delay(user_id, export_url)
    
    logger.info("Scheduled sync tasks for all users")

# Celery beat schedule for automatic sync
celery_app.conf.beat_schedule = {
    'sync-chatgpt-data': {
        'task': 'app.sync_service.scheduled_sync_all_users',
        'schedule': timedelta(hours=CHATGPT_SYNC_INTERVAL_HOURS),
    },
}
celery_app.conf.timezone = 'UTC'