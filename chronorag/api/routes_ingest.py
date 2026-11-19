import os
import shutil
import tempfile
from typing import List, Optional
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from chronorag.api.schemas import IngestTextRequest, IngestResponse, DocumentVersionSummary, ClaimResponse
from chronorag.ingestion.pipeline import IngestionPipeline
from chronorag.extraction.pipeline import ClaimExtractionPipeline
from chronorag.graph.relation_detector import RelationDetector
from chronorag.database.connection import get_db_session
from chronorag.database.models import Document, DocumentVersion, Claim
from chronorag.extraction.date_parser import DateParser

router = APIRouter(prefix="/api/ingest", tags=["Ingestion"])


@router.post("/text", response_model=IngestResponse)
def ingest_text(payload: IngestTextRequest):
    """Ingests raw text directly, chunks it, and extracts atomic temporal claims."""
    # Write to a temporary file
    temp_dir = tempfile.mkdtemp()
    temp_path = os.path.join(temp_dir, payload.filename)
    try:
        with open(temp_path, "w", encoding="utf-8") as f:
            f.write(payload.text_content)

        ingest_pipe = IngestionPipeline()
        doc, doc_version, is_new = ingest_pipe.ingest_file(
            file_path=temp_path,
            publication_date=payload.publication_date,
            custom_metadata=payload.metadata,
        )

        num_claims = 0
        if is_new:
            # Run extraction pipeline
            extract_pipe = ClaimExtractionPipeline()
            extracted_claims = extract_pipe.extract_from_document_version(
                document_version_id=doc_version.version_id,
            )
            num_claims = len(extracted_claims)

            # Detect relations
            detector = RelationDetector()
            detector.detect_and_store_relations()

        chunks_count = 0
        with get_db_session() as s:
            from chronorag.database.models import Chunk
            chunks_count = s.query(Chunk).filter_by(document_version_id=doc_version.version_id).count()

        return IngestResponse(
            document_id=doc.document_id,
            version_id=doc_version.version_id,
            version_number=doc_version.version_number,
            is_new_version=is_new,
            is_duplicate=not is_new,
            num_chunks=chunks_count,
            num_claims=num_claims,
            message="Document text ingested and processed successfully.",
        )
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


@router.post("/file", response_model=IngestResponse)
async def ingest_file(
    file: UploadFile = File(...),
    publication_date: Optional[str] = Form(None),
):
    """Ingests an uploaded document (PDF, DOCX, TXT, MD, CSV), creates version, and extracts claims."""
    temp_dir = tempfile.mkdtemp()
    temp_path = os.path.join(temp_dir, file.filename)
    try:
        with open(temp_path, "wb") as f:
            content = await file.read()
            f.write(content)

        ingest_pipe = IngestionPipeline()
        doc, doc_version, is_new = ingest_pipe.ingest_file(
            file_path=temp_path,
            publication_date=publication_date,
        )

        num_claims = 0
        if is_new:
            extract_pipe = ClaimExtractionPipeline()
            extracted_claims = extract_pipe.extract_from_document_version(
                document_version_id=doc_version.version_id,
            )
            num_claims = len(extracted_claims)

            detector = RelationDetector()
            detector.detect_and_store_relations()

        chunks_count = 0
        with get_db_session() as s:
            from chronorag.database.models import Chunk
            chunks_count = s.query(Chunk).filter_by(document_version_id=doc_version.version_id).count()

        return IngestResponse(
            document_id=doc.document_id,
            version_id=doc_version.version_id,
            version_number=doc_version.version_number,
            is_new_version=is_new,
            is_duplicate=not is_new,
            num_chunks=chunks_count,
            num_claims=num_claims,
            message=f"File '{file.filename}' processed successfully.",
        )
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


@router.get("/documents", response_model=List[DocumentVersionSummary])
def list_documents():
    """Lists all ingested documents and their version metadata."""
    summaries = []
    with get_db_session() as session:
        docs = session.query(Document).order_by(Document.upload_timestamp.desc()).all()
        for doc in docs:
            versions = session.query(DocumentVersion).filter_by(document_id=doc.document_id).all()
            latest = max(versions, key=lambda v: v.version_number) if versions else None
            summaries.append(
                DocumentVersionSummary(
                    document_id=doc.document_id,
                    filename=doc.filename,
                    current_version=latest.version_number if latest else 1,
                    content_hash=latest.content_hash if latest else doc.content_hash,
                    upload_timestamp=doc.upload_timestamp,
                    publication_timestamp=doc.publication_timestamp,
                    total_versions=len(versions),
                )
            )
    return summaries


@router.get("/claims", response_model=List[ClaimResponse])
def list_claims(limit: int = 100):
    """Retrieves extracted atomic claims."""
    claims_out = []
    with get_db_session() as session:
        claims = session.query(Claim).order_by(Claim.valid_from.desc()).limit(limit).all()
        for c in claims:
            claims_out.append(
                ClaimResponse(
                    claim_id=c.claim_id,
                    subject=c.subject_text or "Unknown",
                    predicate=c.predicate or "Unknown",
                    object_value=c.object_value or "",
                    conditions=c.conditions or {},
                    valid_from=c.valid_from.strftime("%Y-%m-%d") if c.valid_from else None,
                    valid_until=c.valid_until.strftime("%Y-%m-%d") if c.valid_until else None,
                    recorded_from=c.recorded_from.strftime("%Y-%m-%d") if c.recorded_from else None,
                    recorded_until=c.recorded_until.strftime("%Y-%m-%d") if c.recorded_until else None,
                    confidence=c.confidence or 1.0,
                    verification_status=c.verification_status or "verified",
                )
            )
    return claims_out
