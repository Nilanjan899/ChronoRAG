-- ==============================================================================
-- ChronoRAG Canonical SQLite Schema with Bitemporal Indices & FTS5
-- ==============================================================================

PRAGMA foreign_keys = ON;

-- 1. documents table
CREATE TABLE IF NOT EXISTS documents (
    document_id TEXT PRIMARY KEY,
    filename TEXT NOT NULL,
    file_type TEXT NOT NULL,
    content_hash TEXT NOT NULL UNIQUE,
    upload_timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    publication_timestamp TIMESTAMP,
    document_metadata TEXT DEFAULT '{}',
    processing_status TEXT DEFAULT 'pending'
);

-- 2. document_versions table
CREATE TABLE IF NOT EXISTS document_versions (
    version_id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL REFERENCES documents(document_id) ON DELETE CASCADE,
    version_number INTEGER NOT NULL DEFAULT 1,
    parent_version_id TEXT REFERENCES document_versions(version_id),
    original_file_reference TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(document_id, version_number)
);

-- 3. chunks table
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id TEXT PRIMARY KEY,
    document_version_id TEXT NOT NULL REFERENCES document_versions(version_id) ON DELETE CASCADE,
    chunk_text TEXT NOT NULL,
    page_number INTEGER,
    section_heading TEXT,
    chunk_index INTEGER NOT NULL DEFAULT 0,
    embedding TEXT, -- JSON array of floats
    chunk_metadata TEXT DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_chunks_version ON chunks(document_version_id);

-- FTS5 table for chunks
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    chunk_id UNINDEXED,
    chunk_text,
    tokenize='porter unicode61'
);

-- 4. entities table
CREATE TABLE IF NOT EXISTS entities (
    entity_id TEXT PRIMARY KEY,
    canonical_name TEXT NOT NULL UNIQUE,
    entity_type TEXT NOT NULL,
    aliases TEXT DEFAULT '[]',
    entity_metadata TEXT DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_entities_canonical ON entities(canonical_name);

-- 5. claims table
CREATE TABLE IF NOT EXISTS claims (
    claim_id TEXT PRIMARY KEY,
    subject_entity_id TEXT REFERENCES entities(entity_id) ON DELETE SET NULL,
    subject_text TEXT NOT NULL,
    predicate TEXT NOT NULL,
    object_value TEXT NOT NULL,
    object_entity_id TEXT REFERENCES entities(entity_id) ON DELETE SET NULL,
    conditions TEXT DEFAULT '{}',

    -- Valid-Time Dimension [valid_from, valid_until)
    valid_from TIMESTAMP,
    valid_until TIMESTAMP,
    valid_from_unknown BOOLEAN NOT NULL DEFAULT 0,
    valid_until_unknown BOOLEAN NOT NULL DEFAULT 0,

    -- Transaction-Time Dimension [recorded_from, recorded_until)
    recorded_from TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    recorded_until TIMESTAMP,

    confidence REAL NOT NULL DEFAULT 1.0,
    extraction_model TEXT DEFAULT 'Qwen2.5-1.5B-Instruct',
    extraction_version TEXT DEFAULT 'v1.0',
    verification_status TEXT DEFAULT 'verified',
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_claims_subject ON claims(subject_text);
CREATE INDEX IF NOT EXISTS idx_claims_predicate ON claims(predicate);
CREATE INDEX IF NOT EXISTS idx_claims_valid_time ON claims(valid_from, valid_until);
CREATE INDEX IF NOT EXISTS idx_claims_tx_time ON claims(recorded_from, recorded_until);

-- FTS5 table for claims
CREATE VIRTUAL TABLE IF NOT EXISTS claims_fts USING fts5(
    claim_id UNINDEXED,
    subject_text,
    predicate,
    object_value,
    tokenize='porter unicode61'
);

-- 6. claim_sources table
CREATE TABLE IF NOT EXISTS claim_sources (
    source_id TEXT PRIMARY KEY,
    claim_id TEXT NOT NULL REFERENCES claims(claim_id) ON DELETE CASCADE,
    chunk_id TEXT NOT NULL REFERENCES chunks(chunk_id) ON DELETE CASCADE,
    exact_quote TEXT NOT NULL,
    page_number INTEGER,
    extraction_confidence REAL DEFAULT 1.0,
    source_type TEXT DEFAULT 'primary_text'
);

CREATE INDEX IF NOT EXISTS idx_claim_sources_claim ON claim_sources(claim_id);
CREATE INDEX IF NOT EXISTS idx_claim_sources_chunk ON claim_sources(chunk_id);

-- 7. claim_relations table
CREATE TABLE IF NOT EXISTS claim_relations (
    relation_id TEXT PRIMARY KEY,
    source_claim_id TEXT NOT NULL REFERENCES claims(claim_id) ON DELETE CASCADE,
    target_claim_id TEXT NOT NULL REFERENCES claims(claim_id) ON DELETE CASCADE,
    relation_type TEXT NOT NULL, -- SUPERSEDES, AMENDS, CONTRADICTS, SUPPORTS, REFINES, DERIVED_FROM
    confidence REAL DEFAULT 1.0,
    evidence TEXT,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_relations_source ON claim_relations(source_claim_id);
CREATE INDEX IF NOT EXISTS idx_relations_target ON claim_relations(target_claim_id);

-- 8. extraction_runs table
CREATE TABLE IF NOT EXISTS extraction_runs (
    run_id TEXT PRIMARY KEY,
    model_name TEXT NOT NULL,
    model_version TEXT NOT NULL,
    prompt_version TEXT NOT NULL,
    processing_timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    documents_processed INTEGER DEFAULT 0,
    claims_extracted INTEGER DEFAULT 0,
    validation_errors INTEGER DEFAULT 0,
    processing_statistics TEXT DEFAULT '{}'
);
