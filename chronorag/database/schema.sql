-- ==============================================================================
-- ChronoRAG Canonical PostgreSQL Schema with pgvector and Bitemporal Range Types
-- ==============================================================================

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "vector";
CREATE EXTENSION IF NOT EXISTS "btree_gist";

-- 1. documents table
CREATE TABLE IF NOT EXISTS documents (
    document_id VARCHAR(36) PRIMARY KEY,
    filename VARCHAR(255) NOT NULL,
    file_type VARCHAR(50) NOT NULL,
    content_hash VARCHAR(64) NOT NULL UNIQUE,
    upload_timestamp TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    publication_timestamp TIMESTAMPTZ,
    document_metadata JSONB DEFAULT '{}'::jsonb,
    processing_status VARCHAR(50) DEFAULT 'pending'
);

-- 2. document_versions table
CREATE TABLE IF NOT EXISTS document_versions (
    version_id VARCHAR(36) PRIMARY KEY,
    document_id VARCHAR(36) NOT NULL REFERENCES documents(document_id) ON DELETE CASCADE,
    version_number INTEGER NOT NULL DEFAULT 1,
    parent_version_id VARCHAR(36) REFERENCES document_versions(version_id),
    original_file_reference VARCHAR(500) NOT NULL,
    content_hash VARCHAR(64) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT uq_doc_version UNIQUE (document_id, version_number)
);

-- 3. chunks table
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id VARCHAR(36) PRIMARY KEY,
    document_version_id VARCHAR(36) NOT NULL REFERENCES document_versions(version_id) ON DELETE CASCADE,
    chunk_text TEXT NOT NULL,
    page_number INTEGER,
    section_heading VARCHAR(255),
    chunk_index INTEGER NOT NULL DEFAULT 0,
    embedding vector(384),
    chunk_metadata JSONB DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS idx_chunks_version ON chunks(document_version_id);
CREATE INDEX IF NOT EXISTS idx_chunks_fts ON chunks USING gin(to_tsvector('english', chunk_text));
CREATE INDEX IF NOT EXISTS idx_chunks_embedding ON chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- 4. entities table
CREATE TABLE IF NOT EXISTS entities (
    entity_id VARCHAR(36) PRIMARY KEY,
    canonical_name VARCHAR(255) NOT NULL UNIQUE,
    entity_type VARCHAR(100) NOT NULL,
    aliases JSONB DEFAULT '[]'::jsonb,
    entity_metadata JSONB DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS idx_entities_canonical ON entities(canonical_name);

-- 5. claims table (Core Bitemporal Store)
CREATE TABLE IF NOT EXISTS claims (
    claim_id VARCHAR(36) PRIMARY KEY,
    subject_entity_id VARCHAR(36) REFERENCES entities(entity_id) ON DELETE SET NULL,
    subject_text VARCHAR(255) NOT NULL,
    predicate VARCHAR(255) NOT NULL,
    object_value TEXT NOT NULL,
    object_entity_id VARCHAR(36) REFERENCES entities(entity_id) ON DELETE SET NULL,
    conditions JSONB DEFAULT '{}'::jsonb,

    -- Valid-Time Dimension [valid_from, valid_until)
    valid_from TIMESTAMPTZ,
    valid_until TIMESTAMPTZ,
    valid_from_unknown BOOLEAN NOT NULL DEFAULT FALSE,
    valid_until_unknown BOOLEAN NOT NULL DEFAULT FALSE,

    -- Transaction-Time Dimension [recorded_from, recorded_until)
    recorded_from TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    recorded_until TIMESTAMPTZ,

    confidence REAL NOT NULL DEFAULT 1.0,
    extraction_model VARCHAR(100) DEFAULT 'Qwen2.5-1.5B-Instruct',
    extraction_version VARCHAR(50) DEFAULT 'v1.0',
    verification_status VARCHAR(50) DEFAULT 'verified',
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_claims_subject ON claims(subject_text);
CREATE INDEX IF NOT EXISTS idx_claims_predicate ON claims(predicate);
CREATE INDEX IF NOT EXISTS idx_claims_valid_time ON claims(valid_from, valid_until);
CREATE INDEX IF NOT EXISTS idx_claims_tx_time ON claims(recorded_from, recorded_until);
CREATE INDEX IF NOT EXISTS idx_claims_fts ON claims USING gin(
    to_tsvector('english', subject_text || ' ' || predicate || ' ' || object_value)
);

-- 6. claim_sources table
CREATE TABLE IF NOT EXISTS claim_sources (
    source_id VARCHAR(36) PRIMARY KEY,
    claim_id VARCHAR(36) NOT NULL REFERENCES claims(claim_id) ON DELETE CASCADE,
    chunk_id VARCHAR(36) NOT NULL REFERENCES chunks(chunk_id) ON DELETE CASCADE,
    exact_quote TEXT NOT NULL,
    page_number INTEGER,
    extraction_confidence REAL DEFAULT 1.0,
    source_type VARCHAR(50) DEFAULT 'primary_text'
);

CREATE INDEX IF NOT EXISTS idx_claim_sources_claim ON claim_sources(claim_id);
CREATE INDEX IF NOT EXISTS idx_claim_sources_chunk ON claim_sources(chunk_id);

-- 7. claim_relations table
CREATE TABLE IF NOT EXISTS claim_relations (
    relation_id VARCHAR(36) PRIMARY KEY,
    source_claim_id VARCHAR(36) NOT NULL REFERENCES claims(claim_id) ON DELETE CASCADE,
    target_claim_id VARCHAR(36) NOT NULL REFERENCES claims(claim_id) ON DELETE CASCADE,
    relation_type VARCHAR(50) NOT NULL, -- SUPERSEDES, AMENDS, CONTRADICTS, SUPPORTS, REFINES, DERIVED_FROM
    confidence REAL DEFAULT 1.0,
    evidence TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_relations_source ON claim_relations(source_claim_id);
CREATE INDEX IF NOT EXISTS idx_relations_target ON claim_relations(target_claim_id);

-- 8. extraction_runs table
CREATE TABLE IF NOT EXISTS extraction_runs (
    run_id VARCHAR(36) PRIMARY KEY,
    model_name VARCHAR(100) NOT NULL,
    model_version VARCHAR(50) NOT NULL,
    prompt_version VARCHAR(50) NOT NULL,
    processing_timestamp TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    documents_processed INTEGER DEFAULT 0,
    claims_extracted INTEGER DEFAULT 0,
    validation_errors INTEGER DEFAULT 0,
    processing_statistics JSONB DEFAULT '{}'::jsonb
);
