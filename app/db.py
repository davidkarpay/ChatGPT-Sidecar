
import sqlite3, json, os
from contextlib import contextmanager

DB_PATH = os.getenv("DB_PATH", "sidecar.db")

def get_database_url() -> str:
    """Get database URL for Alembic and PostgreSQL support."""
    db_url = os.getenv("DB_URL")
    if db_url:
        return db_url
    # Default to SQLite for local development
    return f"sqlite:///{DB_PATH}"

@contextmanager
def DB(path = None):
    p = path or DB_PATH
    conn = sqlite3.connect(p)
    conn.row_factory = sqlite3.Row
    try:
        yield Database(conn)
        conn.commit()
    finally:
        conn.close()

class Database:
    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    def init_schema(self, schema_path: str = "schema.sql"):
        with open(schema_path, "r", encoding="utf-8") as f:
            self.conn.executescript(f.read())

    def upsert_user(self, email: str, display_name: str = None) -> int:
        cur = self.conn.execute("SELECT id FROM user WHERE email=?", (email,))
        row = cur.fetchone()
        if row:
            return row["id"]
        cur = self.conn.execute("INSERT INTO user(email, display_name) VALUES(?,?)", (email, display_name))
        return cur.lastrowid

    def upsert_document(self, user_id: int, title: str, doc_type: str, fingerprint: str, metadata: dict) -> int:
        cur = self.conn.execute("SELECT id FROM document WHERE fingerprint=?", (fingerprint,))
        row = cur.fetchone()
        if row:
            return row["id"]
        cur = self.conn.execute(
            "INSERT INTO document(user_id,title,doc_type,fingerprint,metadata_json) VALUES(?,?,?,?,?)",
            (user_id, title, doc_type, fingerprint, json.dumps(metadata or {}))
        )
        return cur.lastrowid

    def link_project_document(self, project_id: int, document_id: int):
        self.conn.execute(
            "INSERT OR IGNORE INTO project_document(project_id, document_id) VALUES(?,?)",
            (project_id, document_id)
        )

    def insert_chunks(self, doc_id: int, chunks: list) -> list:
        ids = []
        for i, c in enumerate(chunks):
            cur = self.conn.execute(
                "INSERT INTO chunk(document_id, ordinal, text, start_char, end_char, token_count) VALUES(?,?,?,?,?,?)",
                (doc_id, i, c["text"], c.get("start_char"), c.get("end_char"), c.get("token_estimate", 0))
            )
            ids.append(cur.lastrowid)
        return ids

    def create_embedding_refs(self, chunk_ids: list, index_name: str, vector_dim: int, faiss_ids: list) -> list:
        out = []
        for chunk_id, faiss_id in zip(chunk_ids, faiss_ids):
            cur = self.conn.execute(
                "INSERT INTO embedding_ref(chunk_id, index_name, vector_dim, faiss_id) VALUES(?,?,?,?)",
                (chunk_id, index_name, vector_dim, faiss_id)
            )
            out.append(cur.lastrowid)
        return out

    def fetch_chunks_by_ids(self, chunk_ids: list[int]):
        if not chunk_ids:
            return []
        placeholders = ",".join("?" for _ in chunk_ids)  # parameterized, not interpolated
        q = f"""
        SELECT c.id as chunk_id, c.text, c.start_char, c.end_char,
               d.id as doc_id, d.title, d.doc_type
        FROM chunk c
        JOIN document d ON d.id = c.document_id
        WHERE c.id IN ({placeholders})
        """
        return self.conn.execute(q, chunk_ids).fetchall()
    
    def fetch_embeddings_for_rebuild(self, index_name: str = "main"):
        """Fetch all embeddings for rebuilding the FAISS index."""
        cur = self.conn.execute("""
            SELECT e.id as embedding_ref_id, e.chunk_id, e.faiss_id, c.text
            FROM embedding_ref e
            JOIN chunk c ON e.chunk_id = c.id
            WHERE e.index_name = ?
            ORDER BY e.faiss_id
        """, (index_name,))
        return [dict(r) for r in cur.fetchall()]

    def fetch_chunks_by_faiss_indices(self, faiss_indices: list, index_name: str):
        if not faiss_indices:
            return {}
        placeholders = ",".join("?" for _ in faiss_indices)
        q = f"""
        SELECT er.faiss_id, c.id as chunk_id, c.text, c.start_char, c.end_char,
               d.id as doc_id, d.title, d.doc_type
        FROM embedding_ref er
        JOIN chunk c ON c.id = er.chunk_id
        JOIN document d ON d.id = c.document_id
        WHERE er.index_name = ? AND er.faiss_id IN ({placeholders})
        """
        cur = self.conn.execute(q, (index_name, *faiss_indices))
        rows = cur.fetchall()
        return { r["faiss_id"]: r for r in rows }

    def get_document_by_id(self, doc_id: int):
        """Get document metadata by ID"""
        q = """
        SELECT id, title, doc_type, mime_type, source, metadata_json, created_at
        FROM document
        WHERE id = ?
        """
        return self.conn.execute(q, (doc_id,)).fetchone()

    def get_document_chunks(self, doc_id: int):
        """Get all chunks for a document ordered by position"""
        q = """
        SELECT id as chunk_id, ordinal, text, start_char, end_char, token_count
        FROM chunk
        WHERE document_id = ?
        ORDER BY ordinal
        """
        return self.conn.execute(q, (doc_id,)).fetchall()

    def get_chunk_with_context(self, doc_id: int, chunk_id: int, context_chunks: int = 2):
        """Get a specific chunk with surrounding context chunks"""
        q = """
        SELECT c.id as chunk_id, c.ordinal, c.text, c.start_char, c.end_char, c.token_count,
               d.title, d.doc_type
        FROM chunk c
        JOIN document d ON d.id = c.document_id
        WHERE c.document_id = ? AND c.id = ?
        """
        target_chunk = self.conn.execute(q, (doc_id, chunk_id)).fetchone()
        
        if not target_chunk:
            return None
            
        # Get surrounding chunks
        q = """
        SELECT id as chunk_id, ordinal, text, start_char, end_char, token_count
        FROM chunk
        WHERE document_id = ? AND ordinal BETWEEN ? AND ?
        ORDER BY ordinal
        """
        ordinal = target_chunk["ordinal"]
        start_ordinal = max(0, ordinal - context_chunks)
        end_ordinal = ordinal + context_chunks
        
        context_chunks_data = self.conn.execute(q, (doc_id, start_ordinal, end_ordinal)).fetchall()
        
        return {
            "target_chunk": target_chunk,
            "context_chunks": context_chunks_data,
            "document": {
                "title": target_chunk["title"],
                "doc_type": target_chunk["doc_type"]
            }
        }
    
    # Training data methods for local model fine-tuning
    
    def store_training_data(self, session_id: str, user_query: str, model_response: str, 
                          context_sources: list, model_name: str) -> int:
        """Store training data from a chat interaction"""
        cur = self.conn.execute(
            """INSERT INTO training_data 
               (session_id, user_query, model_response, context_json, model_name) 
               VALUES (?, ?, ?, ?, ?)""",
            (session_id, user_query, model_response, json.dumps(context_sources), model_name)
        )
        return cur.lastrowid
    
    def update_training_feedback(self, training_data_id: int, rating: int = None, 
                               correction: str = None):
        """Update training data with user feedback"""
        if rating is not None and correction is not None:
            self.conn.execute(
                """UPDATE training_data 
                   SET feedback_rating = ?, feedback_correction = ?, updated_at = CURRENT_TIMESTAMP 
                   WHERE id = ?""",
                (rating, correction, training_data_id)
            )
        elif rating is not None:
            self.conn.execute(
                """UPDATE training_data 
                   SET feedback_rating = ?, updated_at = CURRENT_TIMESTAMP 
                   WHERE id = ?""",
                (rating, training_data_id)
            )
        elif correction is not None:
            self.conn.execute(
                """UPDATE training_data 
                   SET feedback_correction = ?, updated_at = CURRENT_TIMESTAMP 
                   WHERE id = ?""",
                (correction, training_data_id)
            )
    
    def get_training_data(self, limit: int = 1000, min_rating: int = None, 
                         model_name: str = None) -> list:
        """Retrieve training data for fine-tuning"""
        conditions = []
        params = []
        
        if min_rating is not None:
            conditions.append("feedback_rating >= ?")
            params.append(min_rating)
        
        if model_name is not None:
            conditions.append("model_name = ?")
            params.append(model_name)
        
        where_clause = " WHERE " + " AND ".join(conditions) if conditions else ""
        params.append(limit)
        
        query = f"""SELECT session_id, user_query, model_response, context_json, 
                          feedback_rating, feedback_correction, model_name, created_at
                   FROM training_data {where_clause}
                   ORDER BY created_at DESC LIMIT ?"""
        
        return self.conn.execute(query, params).fetchall()
    
    def create_model_version(self, base_model: str, version_name: str, 
                           adapter_path: str = None, training_config: dict = None,
                           training_data_count: int = 0) -> int:
        """Create a new model version record"""
        cur = self.conn.execute(
            """INSERT INTO model_version 
               (base_model, version_name, adapter_path, training_config_json, training_data_count) 
               VALUES (?, ?, ?, ?, ?)""",
            (base_model, version_name, adapter_path, 
             json.dumps(training_config) if training_config else None, training_data_count)
        )
        return cur.lastrowid
    
    def set_active_model(self, model_version_id: int):
        """Set a model version as active and deactivate others"""
        # First deactivate all models for this base model
        base_model = self.conn.execute(
            "SELECT base_model FROM model_version WHERE id = ?", 
            (model_version_id,)
        ).fetchone()["base_model"]
        
        self.conn.execute(
            "UPDATE model_version SET is_active = 0 WHERE base_model = ?",
            (base_model,)
        )
        
        # Then activate the specified model
        self.conn.execute(
            "UPDATE model_version SET is_active = 1 WHERE id = ?",
            (model_version_id,)
        )
    
    def get_active_model(self, base_model: str = None):
        """Get the currently active model version"""
        if base_model:
            return self.conn.execute(
                "SELECT * FROM model_version WHERE base_model = ? AND is_active = 1",
                (base_model,)
            ).fetchone()
        else:
            return self.conn.execute(
                "SELECT * FROM model_version WHERE is_active = 1"
            ).fetchone()
    
    def create_training_session(self, model_version_id: int, training_data_filter: str,
                              config: dict) -> int:
        """Create a training session record"""
        cur = self.conn.execute(
            """INSERT INTO training_session 
               (model_version_id, training_data_filter, config_json, started_at) 
               VALUES (?, ?, ?, CURRENT_TIMESTAMP)""",
            (model_version_id, training_data_filter, json.dumps(config))
        )
        return cur.lastrowid
    
    def update_training_session(self, session_id: int, status: str, 
                              error_message: str = None, metrics: dict = None):
        """Update training session status and metrics"""
        if status == 'completed':
            self.conn.execute(
                """UPDATE training_session 
                   SET status = ?, completed_at = CURRENT_TIMESTAMP, metrics_json = ?
                   WHERE id = ?""",
                (status, json.dumps(metrics) if metrics else None, session_id)
            )
        elif status == 'failed':
            self.conn.execute(
                """UPDATE training_session 
                   SET status = ?, completed_at = CURRENT_TIMESTAMP, error_message = ?
                   WHERE id = ?""",
                (status, error_message, session_id)
            )
        else:
            self.conn.execute(
                "UPDATE training_session SET status = ? WHERE id = ?",
                (status, session_id)
            )
    
    # Evaluation and benchmarking methods
    
    def store_benchmark_prompt(self, prompt_id: str, category: str, prompt: str,
                              context: list, expected_qualities: list, scoring_criteria: dict):
        """Store a benchmark prompt in the database"""
        self.conn.execute(
            """INSERT OR REPLACE INTO benchmark_prompt 
               (id, category, prompt, context_json, expected_qualities_json, scoring_criteria_json, updated_at) 
               VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)""",
            (prompt_id, category, prompt, json.dumps(context), 
             json.dumps(expected_qualities), json.dumps(scoring_criteria))
        )
    
    def get_benchmark_prompts(self, category: str = None) -> list:
        """Get benchmark prompts, optionally filtered by category"""
        if category:
            query = "SELECT * FROM benchmark_prompt WHERE category = ? ORDER BY id"
            rows = self.conn.execute(query, (category,)).fetchall()
        else:
            query = "SELECT * FROM benchmark_prompt ORDER BY category, id"
            rows = self.conn.execute(query).fetchall()
        
        return [dict(row) for row in rows]
    
    def store_evaluation_result(self, prompt_id: str, model_version_id: int, response: str,
                               scores: dict, metadata: dict = None) -> int:
        """Store evaluation result in database"""
        cur = self.conn.execute(
            """INSERT INTO evaluation_result 
               (prompt_id, model_version_id, response, scores_json, metadata_json) 
               VALUES (?, ?, ?, ?, ?)""",
            (prompt_id, model_version_id, response, json.dumps(scores), 
             json.dumps(metadata) if metadata else None)
        )
        return cur.lastrowid
    
    def get_evaluation_results(self, model_version_id: int = None, prompt_id: str = None,
                              limit: int = 1000) -> list:
        """Get evaluation results with optional filtering"""
        conditions = []
        params = []
        
        if model_version_id is not None:
            conditions.append("model_version_id = ?")
            params.append(model_version_id)
        
        if prompt_id is not None:
            conditions.append("prompt_id = ?")
            params.append(prompt_id)
        
        where_clause = " WHERE " + " AND ".join(conditions) if conditions else ""
        params.append(limit)
        
        query = f"""SELECT er.*, bp.category, bp.prompt
                   FROM evaluation_result er
                   JOIN benchmark_prompt bp ON er.prompt_id = bp.id
                   {where_clause}
                   ORDER BY er.evaluated_at DESC LIMIT ?"""
        
        rows = self.conn.execute(query, params).fetchall()
        return [dict(row) for row in rows]
    
    def store_model_snapshot(self, model_version_id: int, parameter_stats: dict,
                            embedding_analysis: dict, performance_metrics: dict) -> int:
        """Store model snapshot in database"""
        cur = self.conn.execute(
            """INSERT INTO model_snapshot 
               (model_version_id, parameter_stats_json, embedding_analysis_json, performance_metrics_json) 
               VALUES (?, ?, ?, ?)""",
            (model_version_id, json.dumps(parameter_stats), 
             json.dumps(embedding_analysis), json.dumps(performance_metrics))
        )
        return cur.lastrowid
    
    def get_model_snapshots(self, model_version_id: int = None, limit: int = 100) -> list:
        """Get model snapshots with optional filtering"""
        if model_version_id is not None:
            query = """SELECT ms.*, mv.version_name, mv.base_model
                      FROM model_snapshot ms
                      JOIN model_version mv ON ms.model_version_id = mv.id
                      WHERE ms.model_version_id = ?
                      ORDER BY ms.created_at DESC LIMIT ?"""
            rows = self.conn.execute(query, (model_version_id, limit)).fetchall()
        else:
            query = """SELECT ms.*, mv.version_name, mv.base_model
                      FROM model_snapshot ms
                      JOIN model_version mv ON ms.model_version_id = mv.id
                      ORDER BY ms.created_at DESC LIMIT ?"""
            rows = self.conn.execute(query, (limit,)).fetchall()
        
        return [dict(row) for row in rows]
    
    def store_monthly_checkin(self, report_period: str, training_stats: dict,
                             performance_comparison: dict, recommendations: list,
                             report_data: dict) -> int:
        """Store monthly check-in report"""
        cur = self.conn.execute(
            """INSERT OR REPLACE INTO monthly_checkin 
               (report_period, training_stats_json, performance_comparison_json, 
                recommendations_json, report_data_json) 
               VALUES (?, ?, ?, ?, ?)""",
            (report_period, json.dumps(training_stats), json.dumps(performance_comparison),
             json.dumps(recommendations), json.dumps(report_data))
        )
        return cur.lastrowid
    
    def get_monthly_checkins(self, limit: int = 12) -> list:
        """Get recent monthly check-in reports"""
        rows = self.conn.execute(
            "SELECT * FROM monthly_checkin ORDER BY report_period DESC LIMIT ?",
            (limit,)
        ).fetchall()
        return [dict(row) for row in rows]
    
    def get_monthly_checkin(self, report_period: str):
        """Get specific monthly check-in report"""
        row = self.conn.execute(
            "SELECT * FROM monthly_checkin WHERE report_period = ?",
            (report_period,)
        ).fetchone()
        return dict(row) if row else None
    
    # User identity management methods
    
    def create_user_identity(self, user_id: int, full_name: str, role: str = None, 
                           email: str = None, preferences: dict = None) -> int:
        """Create a new user identity"""
        # Deactivate other identities for this user
        self.conn.execute(
            "UPDATE user_identity SET is_active = 0 WHERE user_id = ?",
            (user_id,)
        )
        
        cur = self.conn.execute(
            """INSERT INTO user_identity 
               (user_id, full_name, role, email, preferences_json, is_active) 
               VALUES (?, ?, ?, ?, ?, 1)""",
            (user_id, full_name, role, email, 
             json.dumps(preferences) if preferences else None)
        )
        return cur.lastrowid
    
    def get_active_user_identity(self, user_id: int = None):
        """Get the active user identity (for a specific user or any active user)"""
        if user_id:
            row = self.conn.execute(
                """SELECT ui.*, u.email as user_email 
                   FROM user_identity ui 
                   JOIN user u ON ui.user_id = u.id 
                   WHERE ui.user_id = ? AND ui.is_active = 1""",
                (user_id,)
            ).fetchone()
        else:
            # Get any active user identity (for single-user systems)
            row = self.conn.execute(
                """SELECT ui.*, u.email as user_email 
                   FROM user_identity ui 
                   JOIN user u ON ui.user_id = u.id 
                   WHERE ui.is_active = 1 
                   ORDER BY ui.updated_at DESC LIMIT 1"""
            ).fetchone()
        
        return dict(row) if row else None
    
    def update_user_identity(self, identity_id: int, full_name: str = None, 
                           role: str = None, preferences: dict = None):
        """Update user identity information"""
        updates = []
        params = []
        
        if full_name is not None:
            updates.append("full_name = ?")
            params.append(full_name)
        
        if role is not None:
            updates.append("role = ?")
            params.append(role)
        
        if preferences is not None:
            updates.append("preferences_json = ?")
            params.append(json.dumps(preferences))
        
        if updates:
            updates.append("updated_at = CURRENT_TIMESTAMP")
            params.append(identity_id)
            
            query = f"UPDATE user_identity SET {', '.join(updates)} WHERE id = ?"
            self.conn.execute(query, params)
    
    def create_user_session(self, session_id: str, user_identity_id: int, 
                          session_data: dict = None) -> int:
        """Create or update a user session"""
        # First try to update existing session
        existing = self.conn.execute(
            "SELECT id FROM user_session WHERE session_id = ?",
            (session_id,)
        ).fetchone()
        
        if existing:
            # Update existing session
            self.conn.execute(
                """UPDATE user_session 
                   SET user_identity_id = ?, last_active_at = CURRENT_TIMESTAMP,
                       session_data_json = ?
                   WHERE session_id = ?""",
                (user_identity_id, json.dumps(session_data) if session_data else None, session_id)
            )
            return existing["id"]
        else:
            # Create new session
            cur = self.conn.execute(
                """INSERT INTO user_session 
                   (session_id, user_identity_id, session_data_json) 
                   VALUES (?, ?, ?)""",
                (session_id, user_identity_id, json.dumps(session_data) if session_data else None)
            )
            return cur.lastrowid
    
    def get_user_session(self, session_id: str):
        """Get user session with identity information"""
        row = self.conn.execute(
            """SELECT us.*, ui.full_name, ui.role, ui.email, ui.preferences_json
               FROM user_session us
               JOIN user_identity ui ON us.user_identity_id = ui.id
               WHERE us.session_id = ?""",
            (session_id,)
        ).fetchone()
        return dict(row) if row else None
    
    def store_training_data_with_identity(self, session_id: str, user_query: str, 
                                        model_response: str, context_sources: list, 
                                        model_name: str, user_identity_id: int = None) -> int:
        """Store training data with user identity reference"""
        cur = self.conn.execute(
            """INSERT INTO training_data 
               (session_id, user_query, model_response, context_json, model_name, user_identity_id) 
               VALUES (?, ?, ?, ?, ?, ?)""",
            (session_id, user_query, model_response, json.dumps(context_sources), 
             model_name, user_identity_id)
        )
        return cur.lastrowid
