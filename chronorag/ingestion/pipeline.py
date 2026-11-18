import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from sqlalchemy import text, select
from chronorag.config import settings
from chronorag.database.connection import get_db_session, is_postgres
from chronorag.database.models import Document, DocumentVersion, Chunk
from chronorag.ingestion.hasher import compute_sha256
from chronorag.ingestion.reader import DocumentReader, ParsedBlock
from chronorag.ingestion.chunker import StructureAwareChunker
from chronorag.retrieval.dense import get_embedder

logger = logging.getLogger(__name__)


class IngestionPipeline:
    """Production ingestion pipeline handling versioning, chunking, and embedding."""

    def __init__(self, chunk_size: Optional[int] = None, chunk_overlap: Optional[int] = None):
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap
        self.chunker = StructureAwareChunker(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        self.embedder = get_embedder()

    def ingest_file(
        self,
        file_path: Path,
        publication_timestamp: Optional[datetime] = None,
        custom_metadata: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Document, DocumentVersion, bool]:
        """
        Ingest a file.
        Returns (Document, DocumentVersion, is_new_or_amended).
        If duplicate content exists, returns the existing document/version.
        If filename matches existing document but content is different, creates a new version.
        """
        file_path = Path(file_path)
        content_hash = compute_sha256(file_path)
        filename = file_path.name
        file_type = file_path.suffix.lstrip(".").lower() or "txt"
        meta = custom_metadata or {}
        meta["file_size_bytes"] = file_path.stat().st_size

        with get_db_session() as session:
            # 1. Check for exact duplicate content
            existing_doc = session.query(Document).filter_by(content_hash=content_hash).first()
            if existing_doc:
                latest_ver = (
                    session.query(DocumentVersion)
                    .filter_by(document_id=existing_doc.document_id)
                    .order_by(DocumentVersion.version_number.desc())
                    .first()
                )
                logger.info(f"Duplicate document detected: {filename} (hash {content_hash[:8]}). Skipping ingestion.")
                return existing_doc, latest_ver, False

            # 2. Check if a document with same filename exists (New Version detection)
            existing_named_doc = session.query(Document).filter_by(filename=filename).first()

            if existing_named_doc:
                # New version of existing document!
                doc = existing_named_doc
                latest_ver = (
                    session.query(DocumentVersion)
                    .filter_by(document_id=doc.document_id)
                    .order_by(DocumentVersion.version_number.desc())
                    .first()
                )
                next_version_num = (latest_ver.version_number + 1) if latest_ver else 1
                parent_ver_id = latest_ver.version_id if latest_ver else None

                doc_version = DocumentVersion(
                    document_id=doc.document_id,
                    version_number=next_version_num,
                    parent_version_id=parent_ver_id,
                    original_file_reference=str(file_path),
                    content_hash=content_hash,
                )
                doc.processing_status = "processing"
                session.add(doc_version)
                session.flush()
                logger.info(
                    f"Creating new version {next_version_num} for document '{filename}' (Doc ID: {doc.document_id})"
                )
            else:
                # Brand new document
                doc = Document(
                    filename=filename,
                    file_type=file_type,
                    content_hash=content_hash,
                    publication_timestamp=publication_timestamp,
                    document_metadata=meta,
                    processing_status="processing",
                )
                session.add(doc)
                session.flush()

                doc_version = DocumentVersion(
                    document_id=doc.document_id,
                    version_number=1,
                    parent_version_id=None,
                    original_file_reference=str(file_path),
                    content_hash=content_hash,
                )
                session.add(doc_version)
                session.flush()
                logger.info(f"Ingesting new document '{filename}' (ID: {doc.document_id})")

            # 3. Read and parse document structure
            blocks = DocumentReader.read_file(file_path)

            # 4. Create structure-aware chunks
            doc_chunks = self.chunker.chunk_blocks(blocks)
            logger.info(f"Generated {len(doc_chunks)} chunks for {filename} (Version {doc_version.version_number})")

            # 5. Generate embeddings and persist chunks
            chunk_texts = [c.text for c in doc_chunks]
            embeddings = self.embedder.embed_batch(chunk_texts)

            chunk_records = []
            for idx, c in enumerate(doc_chunks):
                chunk_rec = Chunk(
                    document_version_id=doc_version.version_id,
                    chunk_text=c.text,
                    page_number=c.page_number,
                    section_heading=c.section_heading,
                    chunk_index=c.chunk_index,
                    embedding=embeddings[idx] if idx < len(embeddings) else None,
                    chunk_metadata=c.metadata,
                )
                chunk_records.append(chunk_rec)
                session.add(chunk_rec)

            session.flush()

            # 6. Update FTS index for SQLite if applicable
            if not is_postgres():
                for c_rec in chunk_records:
                    session.execute(
                        text("INSERT INTO chunks_fts (chunk_id, chunk_text) VALUES (:cid, :txt)"),
                        {"cid": c_rec.chunk_id, "txt": c_rec.chunk_text},
                    )

            doc.processing_status = "processed"
            session.commit()
            return doc, doc_version, True

    def ingest_text(
        self,
        filename: str,
        text_content: str,
        publication_timestamp: Optional[datetime] = None,
        custom_metadata: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Document, DocumentVersion, bool]:
        """Ingest text directly from memory, persisting to storage/raw_uploads."""
        upload_dir = settings.storage_dir if hasattr(settings, "storage_dir") else (Path(__file__).parent.parent.parent / "storage" / "raw_uploads")
        upload_dir.mkdir(parents=True, exist_ok=True)
        file_path = upload_dir / filename
        file_path.write_text(text_content, encoding="utf-8")
        return self.ingest_file(file_path, publication_timestamp, custom_metadata)
