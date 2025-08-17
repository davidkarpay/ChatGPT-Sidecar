
-- Users
CREATE TABLE IF NOT EXISTS user (
  id INTEGER PRIMARY KEY,
  email TEXT UNIQUE,
  display_name TEXT,
  created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

-- Projects
CREATE TABLE IF NOT EXISTS project (
  id INTEGER PRIMARY KEY,
  user_id INTEGER NOT NULL,
  name TEXT NOT NULL,
  description TEXT,
  created_at TEXT DEFAULT CURRENT_TIMESTAMP,
  updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
  UNIQUE(user_id, name)
);

-- Documents
CREATE TABLE IF NOT EXISTS document (
  id INTEGER PRIMARY KEY,
  user_id INTEGER NOT NULL,
  title TEXT,
  doc_type TEXT NOT NULL,
  mime_type TEXT,
  source TEXT,
  fingerprint TEXT,
  metadata_json TEXT,
  created_at TEXT DEFAULT CURRENT_TIMESTAMP,
  updated_at TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_document_type ON document(doc_type);
CREATE INDEX IF NOT EXISTS idx_document_fp ON document(fingerprint);

-- Project <-> Document
CREATE TABLE IF NOT EXISTS project_document (
  project_id INTEGER NOT NULL,
  document_id INTEGER NOT NULL,
  PRIMARY KEY (project_id, document_id)
);

-- Chunks
CREATE TABLE IF NOT EXISTS chunk (
  id INTEGER PRIMARY KEY,
  document_id INTEGER NOT NULL,
  ordinal INTEGER NOT NULL,
  text TEXT NOT NULL,
  start_char INTEGER,
  end_char INTEGER,
  token_count INTEGER,
  created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_chunk_doc ON chunk(document_id);

-- Embedding references
CREATE TABLE IF NOT EXISTS embedding_ref (
  id INTEGER PRIMARY KEY,
  chunk_id INTEGER NOT NULL,
  index_name TEXT NOT NULL,
  vector_dim INTEGER NOT NULL,
  faiss_id INTEGER NOT NULL,
  created_at TEXT DEFAULT CURRENT_TIMESTAMP,
  UNIQUE(index_name, chunk_id)
);

-- Tags
CREATE TABLE IF NOT EXISTS tag (
  id INTEGER PRIMARY KEY,
  name TEXT NOT NULL UNIQUE
);
CREATE TABLE IF NOT EXISTS document_tag (
  document_id INTEGER NOT NULL,
  tag_id INTEGER NOT NULL,
  PRIMARY KEY (document_id, tag_id)
);
CREATE TABLE IF NOT EXISTS fact (
  id INTEGER PRIMARY KEY,
  user_id INTEGER NOT NULL,
  key TEXT NOT NULL,
  value TEXT NOT NULL,
  notes TEXT,
  created_at TEXT DEFAULT CURRENT_TIMESTAMP,
  updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
  UNIQUE(user_id, key)
);
CREATE TABLE IF NOT EXISTS fact_tag (
  fact_id INTEGER NOT NULL,
  tag_id INTEGER NOT NULL,
  PRIMARY KEY (fact_id, tag_id)
);

-- Citations: fact <-> chunk
CREATE TABLE IF NOT EXISTS citation (
  id INTEGER PRIMARY KEY,
  fact_id INTEGER NOT NULL,
  chunk_id INTEGER NOT NULL,
  excerpt TEXT,
  weight REAL DEFAULT 1.0
);

-- Conversation normalization
CREATE TABLE IF NOT EXISTS conversation (
  id INTEGER PRIMARY KEY,
  document_id INTEGER NOT NULL UNIQUE,
  title TEXT,
  created_at TEXT
);
CREATE TABLE IF NOT EXISTS message (
  id INTEGER PRIMARY KEY,
  conversation_id INTEGER NOT NULL,
  role TEXT NOT NULL,
  content TEXT NOT NULL,
  created_at TEXT,
  metadata_json TEXT
);

-- Source links
CREATE TABLE IF NOT EXISTS source_link (
  id INTEGER PRIMARY KEY,
  document_id INTEGER NOT NULL,
  link_type TEXT NOT NULL,
  href TEXT NOT NULL,
  notes TEXT
);
