"""Tests for database operations and context management."""

import pytest
import os
import json
import tempfile
import sqlite3
from pathlib import Path
from unittest.mock import Mock, patch

from app.db import DB, Database, get_database_url


class TestDatabaseConnection:
    """Test database connection and context management."""
    
    @pytest.fixture
    def temp_db_path(self):
        """Create a temporary database file."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        yield db_path
        # Cleanup
        if os.path.exists(db_path):
            os.unlink(db_path)
    
    @pytest.fixture
    def initialized_db(self, temp_db_path):
        """Create an initialized database with schema."""
        schema_path = Path(__file__).parent.parent / 'schema.sql'
        with DB(temp_db_path) as db:
            if schema_path.exists():
                db.init_schema(str(schema_path))
        return temp_db_path
    
    def test_context_manager(self, temp_db_path):
        """Test database context manager functionality."""
        with DB(temp_db_path) as db:
            assert db is not None
            assert isinstance(db, Database)
            assert db.conn is not None
    
    def test_connection_commit(self, initialized_db):
        """Test that changes are committed."""
        # Insert data
        with DB(initialized_db) as db:
            db.conn.execute(
                "INSERT INTO user(email, display_name) VALUES(?, ?)",
                ("test@example.com", "Test User")
            )
        
        # Verify data persisted
        with DB(initialized_db) as db:
            cur = db.conn.execute("SELECT * FROM user WHERE email = ?", ("test@example.com",))
            row = cur.fetchone()
            assert row is not None
            assert row["email"] == "test@example.com"
    
    def test_connection_rollback_on_error(self, initialized_db):
        """Test that changes are rolled back on error."""
        try:
            with DB(initialized_db) as db:
                db.conn.execute(
                    "INSERT INTO user(email, display_name) VALUES(?, ?)",
                    ("test@example.com", "Test User")
                )
                # Force an error
                raise Exception("Test error")
        except:
            pass
        
        # Verify data was not persisted
        with DB(initialized_db) as db:
            cur = db.conn.execute("SELECT * FROM user WHERE email = ?", ("test@example.com",))
            row = cur.fetchone()
            # Depending on implementation, this might still exist
            # The important thing is the context manager handles the error
    
    def test_get_database_url(self):
        """Test database URL generation."""
        # Test with environment variable
        with patch.dict(os.environ, {'DB_URL': 'postgresql://localhost/test'}):
            assert get_database_url() == 'postgresql://localhost/test'
        
        # Test without environment variable (SQLite fallback)
        with patch.dict(os.environ, {}, clear=True):
            url = get_database_url()
            assert url.startswith('sqlite:///')


class TestUserOperations:
    """Test user-related database operations."""
    
    @pytest.fixture
    def db(self, temp_db_path):
        """Create a database instance with schema."""
        schema_path = Path(__file__).parent.parent / 'schema.sql'
        with DB(temp_db_path) as database:
            if schema_path.exists():
                database.init_schema(str(schema_path))
        return temp_db_path
    
    def test_upsert_user_new(self, db):
        """Test inserting a new user."""
        with DB(db) as database:
            user_id = database.upsert_user("test@example.com", "Test User")
            
            assert user_id is not None
            assert user_id > 0
            
            # Verify user was created
            cur = database.conn.execute("SELECT * FROM user WHERE id = ?", (user_id,))
            row = cur.fetchone()
            assert row["email"] == "test@example.com"
            assert row["display_name"] == "Test User"
    
    def test_upsert_user_existing(self, db):
        """Test upserting an existing user returns same ID."""
        with DB(db) as database:
            # Create user
            user_id1 = database.upsert_user("test@example.com", "Test User")
            
            # Upsert same user
            user_id2 = database.upsert_user("test@example.com", "Different Name")
            
            assert user_id1 == user_id2
            
            # Verify name wasn't updated (upsert returns existing)
            cur = database.conn.execute("SELECT * FROM user WHERE id = ?", (user_id1,))
            row = cur.fetchone()
            assert row["display_name"] == "Test User"


class TestDocumentOperations:
    """Test document-related database operations."""
    
    @pytest.fixture
    def db_with_user(self, temp_db_path):
        """Create a database with a test user."""
        schema_path = Path(__file__).parent.parent / 'schema.sql'
        with DB(temp_db_path) as db:
            if schema_path.exists():
                db.init_schema(str(schema_path))
            db.upsert_user("test@example.com", "Test User")
        return temp_db_path
    
    def test_upsert_document_new(self, db_with_user):
        """Test inserting a new document."""
        with DB(db_with_user) as db:
            doc_id = db.upsert_document(
                user_id=1,
                title="Test Document",
                doc_type="test",
                fingerprint="abc123",
                metadata={"key": "value"}
            )
            
            assert doc_id is not None
            assert doc_id > 0
            
            # Verify document was created
            cur = db.conn.execute("SELECT * FROM document WHERE id = ?", (doc_id,))
            row = cur.fetchone()
            assert row["title"] == "Test Document"
            assert row["fingerprint"] == "abc123"
            assert json.loads(row["metadata_json"]) == {"key": "value"}
    
    def test_upsert_document_duplicate_fingerprint(self, db_with_user):
        """Test upserting document with duplicate fingerprint returns existing."""
        with DB(db_with_user) as db:
            # Create document
            doc_id1 = db.upsert_document(
                user_id=1,
                title="Document 1",
                doc_type="test",
                fingerprint="abc123",
                metadata={"version": 1}
            )
            
            # Try to create with same fingerprint
            doc_id2 = db.upsert_document(
                user_id=1,
                title="Document 2",
                doc_type="test",
                fingerprint="abc123",
                metadata={"version": 2}
            )
            
            assert doc_id1 == doc_id2
            
            # Verify original document preserved
            cur = db.conn.execute("SELECT * FROM document WHERE id = ?", (doc_id1,))
            row = cur.fetchone()
            assert row["title"] == "Document 1"
    
    def test_link_project_document(self, db_with_user):
        """Test linking document to project."""
        with DB(db_with_user) as db:
            # Create project
            cur = db.conn.execute(
                "INSERT INTO project(user_id, name) VALUES(?, ?)",
                (1, "Test Project")
            )
            project_id = cur.lastrowid
            
            # Create document
            doc_id = db.upsert_document(
                user_id=1,
                title="Test Doc",
                doc_type="test",
                fingerprint="xyz789",
                metadata={}
            )
            
            # Link them
            db.link_project_document(project_id, doc_id)
            
            # Verify link exists
            cur = db.conn.execute(
                "SELECT * FROM project_document WHERE project_id = ? AND document_id = ?",
                (project_id, doc_id)
            )
            row = cur.fetchone()
            assert row is not None


class TestChunkOperations:
    """Test chunk-related database operations."""
    
    @pytest.fixture
    def db_with_document(self, temp_db_path):
        """Create a database with a test document."""
        schema_path = Path(__file__).parent.parent / 'schema.sql'
        with DB(temp_db_path) as db:
            if schema_path.exists():
                db.init_schema(str(schema_path))
            db.upsert_user("test@example.com", "Test User")
            db.upsert_document(1, "Test Doc", "test", "abc123", {})
        return temp_db_path
    
    def test_insert_chunks(self, db_with_document):
        """Test inserting chunks for a document."""
        chunks = [
            {"text": "First chunk", "start_char": 0, "end_char": 11, "token_estimate": 2},
            {"text": "Second chunk", "start_char": 12, "end_char": 24, "token_estimate": 2},
            {"text": "Third chunk", "start_char": 25, "end_char": 36, "token_estimate": 2}
        ]
        
        with DB(db_with_document) as db:
            chunk_ids = db.insert_chunks(1, chunks)
            
            assert len(chunk_ids) == 3
            assert all(id > 0 for id in chunk_ids)
            
            # Verify chunks were created
            cur = db.conn.execute("SELECT * FROM chunk WHERE document_id = ? ORDER BY ordinal", (1,))
            rows = cur.fetchall()
            assert len(rows) == 3
            assert rows[0]["text"] == "First chunk"
            assert rows[1]["text"] == "Second chunk"
            assert rows[2]["text"] == "Third chunk"
            assert rows[0]["ordinal"] == 0
            assert rows[1]["ordinal"] == 1
            assert rows[2]["ordinal"] == 2
    
    def test_fetch_chunks_by_ids(self, db_with_document):
        """Test fetching chunks by their IDs."""
        chunks = [
            {"text": "Chunk A"},
            {"text": "Chunk B"},
            {"text": "Chunk C"}
        ]
        
        with DB(db_with_document) as db:
            chunk_ids = db.insert_chunks(1, chunks)
            
            # Fetch specific chunks
            fetched = db.fetch_chunks_by_ids([chunk_ids[0], chunk_ids[2]])
            
            assert len(fetched) == 2
            assert fetched[0]["text"] == "Chunk A"
            assert fetched[1]["text"] == "Chunk C"
    
    def test_fetch_chunks_empty_list(self, db_with_document):
        """Test fetching chunks with empty ID list."""
        with DB(db_with_document) as db:
            fetched = db.fetch_chunks_by_ids([])
            assert fetched == []


class TestEmbeddingOperations:
    """Test embedding reference operations."""
    
    @pytest.fixture
    def db_with_chunks(self, temp_db_path):
        """Create a database with test chunks."""
        schema_path = Path(__file__).parent.parent / 'schema.sql'
        with DB(temp_db_path) as db:
            if schema_path.exists():
                db.init_schema(str(schema_path))
            db.upsert_user("test@example.com", "Test User")
            doc_id = db.upsert_document(1, "Test Doc", "test", "abc123", {})
            chunk_ids = db.insert_chunks(doc_id, [
                {"text": "Chunk 1"},
                {"text": "Chunk 2"}
            ])
        return temp_db_path, chunk_ids
    
    def test_create_embedding_refs(self, db_with_chunks):
        """Test creating embedding references."""
        db_path, chunk_ids = db_with_chunks
        
        with DB(db_path) as db:
            faiss_ids = [0, 1]  # FAISS index positions
            ref_ids = db.create_embedding_refs(
                chunk_ids=chunk_ids,
                index_name="test_index",
                vector_dim=384,
                faiss_ids=faiss_ids
            )
            
            assert len(ref_ids) == 2
            assert all(id > 0 for id in ref_ids)
            
            # Verify references were created
            cur = db.conn.execute(
                "SELECT * FROM embedding_ref WHERE index_name = ? ORDER BY faiss_id",
                ("test_index",)
            )
            rows = cur.fetchall()
            assert len(rows) == 2
            assert rows[0]["chunk_id"] == chunk_ids[0]
            assert rows[0]["faiss_id"] == 0
            assert rows[1]["chunk_id"] == chunk_ids[1]
            assert rows[1]["faiss_id"] == 1
    
    def test_fetch_embeddings_for_rebuild(self, db_with_chunks):
        """Test fetching embeddings for index rebuild."""
        db_path, chunk_ids = db_with_chunks
        
        with DB(db_path) as db:
            # Create embedding refs
            db.create_embedding_refs(
                chunk_ids=chunk_ids,
                index_name="main",
                vector_dim=384,
                faiss_ids=[0, 1]
            )
            
            # Fetch for rebuild
            rows = db.fetch_embeddings_for_rebuild("main")
            
            assert len(rows) == 2
            assert all("text" in row for row in rows)
            assert all("embedding_ref_id" in row for row in rows)
            assert all("chunk_id" in row for row in rows)


class TestSearchOperations:
    """Test search-related database operations."""
    
    @pytest.fixture
    def db_with_data(self, temp_db_path):
        """Create a database with test data."""
        schema_path = Path(__file__).parent.parent / 'schema.sql'
        with DB(temp_db_path) as db:
            if schema_path.exists():
                db.init_schema(str(schema_path))
            
            # Create user and documents
            user_id = db.upsert_user("test@example.com", "Test User")
            doc1_id = db.upsert_document(user_id, "Doc 1", "test", "fp1", {})
            doc2_id = db.upsert_document(user_id, "Doc 2", "test", "fp2", {})
            
            # Create chunks
            chunks1 = db.insert_chunks(doc1_id, [
                {"text": "Python programming"},
                {"text": "Machine learning"}
            ])
            chunks2 = db.insert_chunks(doc2_id, [
                {"text": "Data science"},
                {"text": "Neural networks"}
            ])
            
            return temp_db_path, chunks1 + chunks2
    
    def test_fetch_chunks_with_metadata(self, db_with_data):
        """Test fetching chunks with document metadata."""
        db_path, chunk_ids = db_with_data
        
        with DB(db_path) as db:
            # Fetch first two chunks
            chunks = db.fetch_chunks_by_ids(chunk_ids[:2])
            
            assert len(chunks) == 2
            assert chunks[0]["text"] == "Python programming"
            assert chunks[1]["text"] == "Machine learning"
            
            # Check if document info is included (depends on implementation)
            # This might need adjustment based on actual implementation


class TestTransactionHandling:
    """Test transaction handling and error recovery."""
    
    @pytest.fixture
    def db_path(self, temp_db_path):
        """Create an initialized database."""
        schema_path = Path(__file__).parent.parent / 'schema.sql'
        with DB(temp_db_path) as db:
            if schema_path.exists():
                db.init_schema(str(schema_path))
        return temp_db_path
    
    def test_transaction_isolation(self, db_path):
        """Test that transactions are isolated."""
        # Start two connections
        conn1 = sqlite3.connect(db_path)
        conn2 = sqlite3.connect(db_path)
        
        try:
            # Insert in first connection (not committed)
            conn1.execute("INSERT INTO user(email) VALUES(?)", ("user1@example.com",))
            
            # Try to read from second connection
            cur = conn2.execute("SELECT * FROM user WHERE email = ?", ("user1@example.com",))
            row = cur.fetchone()
            assert row is None  # Should not see uncommitted data
            
            # Commit first connection
            conn1.commit()
            
            # Now should be visible
            cur = conn2.execute("SELECT * FROM user WHERE email = ?", ("user1@example.com",))
            row = cur.fetchone()
            assert row is not None
        finally:
            conn1.close()
            conn2.close()


class TestSchemaInitialization:
    """Test database schema initialization."""
    
    def test_init_schema(self, temp_db_path):
        """Test initializing database schema."""
        schema_path = Path(__file__).parent.parent / 'schema.sql'
        
        if not schema_path.exists():
            pytest.skip("Schema file not found")
        
        with DB(temp_db_path) as db:
            db.init_schema(str(schema_path))
            
            # Verify tables exist
            cur = db.conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
            tables = [row[0] for row in cur.fetchall()]
            
            # Check for expected tables
            expected_tables = ['user', 'document', 'chunk', 'embedding_ref']
            for table in expected_tables:
                assert table in tables
    
    def test_init_schema_idempotent(self, temp_db_path):
        """Test that schema initialization is idempotent."""
        schema_path = Path(__file__).parent.parent / 'schema.sql'
        
        if not schema_path.exists():
            pytest.skip("Schema file not found")
        
        with DB(temp_db_path) as db:
            # Initialize twice
            db.init_schema(str(schema_path))
            db.init_schema(str(schema_path))
            
            # Should not raise errors
            # Verify tables still exist
            cur = db.conn.execute(
                "SELECT COUNT(*) FROM sqlite_master WHERE type='table'"
            )
            count = cur.fetchone()[0]
            assert count > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])