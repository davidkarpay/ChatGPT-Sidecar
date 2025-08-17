
import sqlite3, json, os
from contextlib import contextmanager

DB_PATH = os.getenv("DB_PATH", "sidecar.db")

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
