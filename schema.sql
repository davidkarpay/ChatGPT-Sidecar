
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

-- Training data collection for local model fine-tuning
CREATE TABLE IF NOT EXISTS training_data (
  id INTEGER PRIMARY KEY,
  session_id TEXT NOT NULL,
  user_query TEXT NOT NULL,
  model_response TEXT NOT NULL,
  context_json TEXT, -- JSON array of context sources used
  feedback_rating INTEGER, -- 1 (bad) to 5 (good), NULL if no feedback
  feedback_correction TEXT, -- User's corrected response if provided
  model_name TEXT NOT NULL, -- Which model generated this response
  user_identity_id INTEGER, -- Links to user_identity for personalized training
  created_at TEXT DEFAULT CURRENT_TIMESTAMP,
  updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (user_identity_id) REFERENCES user_identity(id)
);
CREATE INDEX IF NOT EXISTS idx_training_data_session ON training_data(session_id);
CREATE INDEX IF NOT EXISTS idx_training_data_rating ON training_data(feedback_rating);
CREATE INDEX IF NOT EXISTS idx_training_data_model ON training_data(model_name);

-- Model versions and fine-tuning tracking
CREATE TABLE IF NOT EXISTS model_version (
  id INTEGER PRIMARY KEY,
  base_model TEXT NOT NULL, -- Base model name (e.g., "TinyLlama-1.1B")
  version_name TEXT NOT NULL, -- Version identifier (e.g., "v1.0", "user-tuned-1")
  adapter_path TEXT, -- Path to LoRA adapter files, NULL for base model
  training_config_json TEXT, -- JSON of training parameters used
  training_data_count INTEGER DEFAULT 0, -- Number of training examples used
  created_at TEXT DEFAULT CURRENT_TIMESTAMP,
  is_active INTEGER DEFAULT 0, -- 1 if this is the currently active model
  UNIQUE(base_model, version_name)
);

-- Session management for authentication
CREATE TABLE IF NOT EXISTS session (
    id INTEGER PRIMARY KEY,
    user_id INTEGER NOT NULL,
    token TEXT UNIQUE NOT NULL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    expires_at TEXT,
    is_active BOOLEAN DEFAULT 1,
    FOREIGN KEY (user_id) REFERENCES user(id)
);
CREATE INDEX IF NOT EXISTS idx_session_token ON session(token);
CREATE INDEX IF NOT EXISTS idx_session_user ON session(user_id);

-- Sync history for ChatGPT data synchronization
CREATE TABLE IF NOT EXISTS sync_history (
    sync_id TEXT PRIMARY KEY,
    user_id INTEGER NOT NULL,
    started_at TEXT DEFAULT CURRENT_TIMESTAMP,
    completed_at TEXT,
    status TEXT CHECK(status IN ('running', 'completed', 'failed')) DEFAULT 'running',
    source_url TEXT,
    files_processed INTEGER DEFAULT 0,
    conversations_added INTEGER DEFAULT 0,
    conversations_updated INTEGER DEFAULT 0,
    error_message TEXT,
    FOREIGN KEY (user_id) REFERENCES user(id)
);
CREATE INDEX IF NOT EXISTS idx_sync_user ON sync_history(user_id);
CREATE INDEX IF NOT EXISTS idx_sync_status ON sync_history(status);

-- Training sessions for tracking fine-tuning runs
CREATE TABLE IF NOT EXISTS training_session (
  id INTEGER PRIMARY KEY,
  model_version_id INTEGER NOT NULL,
  training_data_filter TEXT, -- SQL WHERE clause for selecting training data
  config_json TEXT NOT NULL, -- Training hyperparameters, LoRA settings, etc.
  status TEXT DEFAULT 'pending', -- pending, running, completed, failed
  started_at TEXT,
  completed_at TEXT,
  error_message TEXT,
  metrics_json TEXT, -- Training loss, validation metrics, etc.
  FOREIGN KEY (model_version_id) REFERENCES model_version(id)
);

-- Benchmark prompts for consistent model evaluation
CREATE TABLE IF NOT EXISTS benchmark_prompt (
  id TEXT PRIMARY KEY, -- prompt identifier (e.g., "legal_case_analysis")
  category TEXT NOT NULL, -- "legal", "factual", "coherence", "anti_hallucination"
  prompt TEXT NOT NULL,
  context_json TEXT, -- JSON array of context sources
  expected_qualities_json TEXT, -- JSON array of expected qualities
  scoring_criteria_json TEXT, -- JSON object with scoring guidelines
  created_at TEXT DEFAULT CURRENT_TIMESTAMP,
  updated_at TEXT DEFAULT CURRENT_TIMESTAMP
);

-- Evaluation results for benchmark prompt responses
CREATE TABLE IF NOT EXISTS evaluation_result (
  id INTEGER PRIMARY KEY,
  prompt_id TEXT NOT NULL,
  model_version_id INTEGER NOT NULL,
  response TEXT NOT NULL,
  scores_json TEXT NOT NULL, -- JSON object with scoring metrics
  metadata_json TEXT, -- Additional evaluation data
  evaluated_at TEXT DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (prompt_id) REFERENCES benchmark_prompt(id),
  FOREIGN KEY (model_version_id) REFERENCES model_version(id)
);
CREATE INDEX IF NOT EXISTS idx_evaluation_result_prompt ON evaluation_result(prompt_id);
CREATE INDEX IF NOT EXISTS idx_evaluation_result_model ON evaluation_result(model_version_id);
CREATE INDEX IF NOT EXISTS idx_evaluation_result_date ON evaluation_result(evaluated_at);

-- Model snapshots for tracking parameter and performance changes
CREATE TABLE IF NOT EXISTS model_snapshot (
  id INTEGER PRIMARY KEY,
  model_version_id INTEGER NOT NULL,
  parameter_stats_json TEXT NOT NULL, -- Statistical summaries of model parameters
  embedding_analysis_json TEXT NOT NULL, -- Embedding space analysis
  performance_metrics_json TEXT NOT NULL, -- Benchmark performance summary
  created_at TEXT DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (model_version_id) REFERENCES model_version(id)
);
CREATE INDEX IF NOT EXISTS idx_model_snapshot_version ON model_snapshot(model_version_id);
CREATE INDEX IF NOT EXISTS idx_model_snapshot_date ON model_snapshot(created_at);

-- Monthly check-in reports
CREATE TABLE IF NOT EXISTS monthly_checkin (
  id INTEGER PRIMARY KEY,
  report_period TEXT NOT NULL, -- YYYY-MM format
  training_stats_json TEXT NOT NULL, -- Training data statistics for the month
  performance_comparison_json TEXT, -- Model performance comparisons
  recommendations_json TEXT, -- JSON array of recommendations
  report_data_json TEXT, -- Complete report data
  generated_at TEXT DEFAULT CURRENT_TIMESTAMP,
  UNIQUE(report_period)
);
CREATE INDEX IF NOT EXISTS idx_monthly_checkin_period ON monthly_checkin(report_period);

-- User identity management for personalized model responses
CREATE TABLE IF NOT EXISTS user_identity (
  id INTEGER PRIMARY KEY,
  user_id INTEGER NOT NULL, -- Links to existing user table
  full_name TEXT NOT NULL, -- User's full name (e.g., "David Karpay")
  role TEXT, -- Professional role (e.g., "attorney", "researcher")
  email TEXT,
  preferences_json TEXT, -- User preferences and additional identity info
  is_active INTEGER DEFAULT 1, -- Whether this identity is currently active
  created_at TEXT DEFAULT CURRENT_TIMESTAMP,
  updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (user_id) REFERENCES user(id),
  UNIQUE(user_id, full_name)
);

-- User sessions linked to user identity for personalized responses
CREATE TABLE IF NOT EXISTS user_session (
  id INTEGER PRIMARY KEY,
  session_id TEXT NOT NULL UNIQUE, -- Chat session identifier
  user_identity_id INTEGER NOT NULL, -- Links to user_identity
  created_at TEXT DEFAULT CURRENT_TIMESTAMP,
  last_active_at TEXT DEFAULT CURRENT_TIMESTAMP,
  session_data_json TEXT, -- Additional session-specific data
  FOREIGN KEY (user_identity_id) REFERENCES user_identity(id)
);
CREATE INDEX IF NOT EXISTS idx_user_session_session_id ON user_session(session_id);
CREATE INDEX IF NOT EXISTS idx_user_session_identity ON user_session(user_identity_id);
CREATE INDEX IF NOT EXISTS idx_user_session_active ON user_session(last_active_at);
