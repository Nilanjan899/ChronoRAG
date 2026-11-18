import uuid
from datetime import datetime, timezone
from typing import List, Optional
from sqlalchemy import (
    Column,
    String,
    Integer,
    Float,
    Boolean,
    DateTime,
    ForeignKey,
    Text,
    JSON,
    Index,
)
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


def generate_uuid() -> str:
    return str(uuid.uuid4())


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class Document(Base):
    __tablename__ = "documents"

    document_id = Column(String(36), primary_key=True, default=generate_uuid)
    filename = Column(String(255), nullable=False)
    file_type = Column(String(50), nullable=False)  # pdf, docx, txt, md, csv
    content_hash = Column(String(64), nullable=False, unique=True, index=True)
    upload_timestamp = Column(DateTime(timezone=True), default=utc_now, nullable=False)
    publication_timestamp = Column(DateTime(timezone=True), nullable=True)
    document_metadata = Column(JSON, default=dict)
    processing_status = Column(String(50), default="pending")  # pending, processed, failed

    versions = relationship("DocumentVersion", back_populates="document", cascade="all, delete-orphan")


class DocumentVersion(Base):
    __tablename__ = "document_versions"

    version_id = Column(String(36), primary_key=True, default=generate_uuid)
    document_id = Column(String(36), ForeignKey("documents.document_id", ondelete="CASCADE"), nullable=False)
    version_number = Column(Integer, default=1, nullable=False)
    parent_version_id = Column(String(36), ForeignKey("document_versions.version_id"), nullable=True)
    original_file_reference = Column(String(500), nullable=False)
    content_hash = Column(String(64), nullable=False)
    created_at = Column(DateTime(timezone=True), default=utc_now, nullable=False)

    document = relationship("Document", back_populates="versions")
    chunks = relationship("Chunk", back_populates="version", cascade="all, delete-orphan")
    parent_version = relationship("DocumentVersion", remote_side=[version_id])


class Chunk(Base):
    __tablename__ = "chunks"

    chunk_id = Column(String(36), primary_key=True, default=generate_uuid)
    document_version_id = Column(String(36), ForeignKey("document_versions.version_id", ondelete="CASCADE"), nullable=False)
    chunk_text = Column(Text, nullable=False)
    page_number = Column(Integer, nullable=True)
    section_heading = Column(String(255), nullable=True)
    chunk_index = Column(Integer, default=0, nullable=False)
    embedding = Column(JSON, nullable=True)  # List[float] stored as JSON or vector in PG
    chunk_metadata = Column(JSON, default=dict)

    version = relationship("DocumentVersion", back_populates="chunks")
    sources = relationship("ClaimSource", back_populates="chunk", cascade="all, delete-orphan")


class Entity(Base):
    __tablename__ = "entities"

    entity_id = Column(String(36), primary_key=True, default=generate_uuid)
    canonical_name = Column(String(255), nullable=False, unique=True, index=True)
    entity_type = Column(String(100), nullable=False)  # Organization, Policy, Department, Person, Product, Regulation
    aliases = Column(JSON, default=list)  # List[str]
    entity_metadata = Column(JSON, default=dict)

    claims_as_subject = relationship("Claim", foreign_keys="Claim.subject_entity_id", back_populates="subject_entity")
    claims_as_object = relationship("Claim", foreign_keys="Claim.object_entity_id", back_populates="object_entity")


class Claim(Base):
    """
    Bitemporal atomic claim:
    - Valid Time: [valid_from, valid_until)
    - Transaction Time: [recorded_from, recorded_until)
    """
    __tablename__ = "claims"

    claim_id = Column(String(36), primary_key=True, default=generate_uuid)
    subject_entity_id = Column(String(36), ForeignKey("entities.entity_id", ondelete="SET NULL"), nullable=True)
    subject_text = Column(String(255), nullable=False, index=True)
    predicate = Column(String(255), nullable=False, index=True)
    object_value = Column(Text, nullable=False)
    object_entity_id = Column(String(36), ForeignKey("entities.entity_id", ondelete="SET NULL"), nullable=True)
    conditions = Column(JSON, default=dict)  # applicability conditions (department, tenure, etc.)
    
    # Valid Time (Half-open [valid_from, valid_until))
    valid_from = Column(DateTime(timezone=True), nullable=True, index=True)
    valid_until = Column(DateTime(timezone=True), nullable=True, index=True)
    valid_from_unknown = Column(Boolean, default=False, nullable=False)
    valid_until_unknown = Column(Boolean, default=False, nullable=False)

    # Transaction Time (Half-open [recorded_from, recorded_until))
    recorded_from = Column(DateTime(timezone=True), default=utc_now, nullable=False, index=True)
    recorded_until = Column(DateTime(timezone=True), nullable=True, index=True)

    confidence = Column(Float, default=1.0, nullable=False)
    extraction_model = Column(String(100), default="Qwen2.5-1.5B-Instruct")
    extraction_version = Column(String(50), default="v1.0")
    verification_status = Column(String(50), default="verified")  # verified, unverified, disputed
    created_at = Column(DateTime(timezone=True), default=utc_now, nullable=False)

    subject_entity = relationship("Entity", foreign_keys=[subject_entity_id], back_populates="claims_as_subject")
    object_entity = relationship("Entity", foreign_keys=[object_entity_id], back_populates="claims_as_object")
    sources = relationship("ClaimSource", back_populates="claim", cascade="all, delete-orphan")
    relations_as_source = relationship("ClaimRelation", foreign_keys="ClaimRelation.source_claim_id", back_populates="source_claim")
    relations_as_target = relationship("ClaimRelation", foreign_keys="ClaimRelation.target_claim_id", back_populates="target_claim")


class ClaimSource(Base):
    __tablename__ = "claim_sources"

    source_id = Column(String(36), primary_key=True, default=generate_uuid)
    claim_id = Column(String(36), ForeignKey("claims.claim_id", ondelete="CASCADE"), nullable=False, index=True)
    chunk_id = Column(String(36), ForeignKey("chunks.chunk_id", ondelete="CASCADE"), nullable=False, index=True)
    exact_quote = Column(Text, nullable=False)
    page_number = Column(Integer, nullable=True)
    extraction_confidence = Column(Float, default=1.0)
    source_type = Column(String(50), default="primary_text")  # primary_text, amendment_clause, table

    claim = relationship("Claim", back_populates="sources")
    chunk = relationship("Chunk", back_populates="sources")


class ClaimRelation(Base):
    __tablename__ = "claim_relations"

    relation_id = Column(String(36), primary_key=True, default=generate_uuid)
    source_claim_id = Column(String(36), ForeignKey("claims.claim_id", ondelete="CASCADE"), nullable=False, index=True)
    target_claim_id = Column(String(36), ForeignKey("claims.claim_id", ondelete="CASCADE"), nullable=False, index=True)
    relation_type = Column(String(50), nullable=False)  # SUPERSEDES, AMENDS, CONTRADICTS, SUPPORTS, REFINES, DERIVED_FROM
    confidence = Column(Float, default=1.0)
    evidence = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), default=utc_now, nullable=False)

    source_claim = relationship("Claim", foreign_keys=[source_claim_id], back_populates="relations_as_source")
    target_claim = relationship("Claim", foreign_keys=[target_claim_id], back_populates="relations_as_target")


class ExtractionRun(Base):
    __tablename__ = "extraction_runs"

    run_id = Column(String(36), primary_key=True, default=generate_uuid)
    model_name = Column(String(100), nullable=False)
    model_version = Column(String(50), nullable=False)
    prompt_version = Column(String(50), nullable=False)
    processing_timestamp = Column(DateTime(timezone=True), default=utc_now, nullable=False)
    documents_processed = Column(Integer, default=0)
    claims_extracted = Column(Integer, default=0)
    validation_errors = Column(Integer, default=0)
    processing_statistics = Column(JSON, default=dict)
