import logging
from datetime import datetime, timezone
from typing import List, Optional, Tuple, Dict, Any
from sqlalchemy import text
from chronorag.config import settings
from chronorag.database.connection import get_db_session, is_postgres
from chronorag.database.models import Claim, ClaimSource, ExtractionRun, Chunk, DocumentVersion, Entity
from chronorag.extraction.models import RawExtractedClaim, ValidatedAtomicClaim
from chronorag.extraction.date_parser import parse_temporal_boundary
from chronorag.extraction.engine import get_extraction_engine

logger = logging.getLogger(__name__)


class ClaimExtractionPipeline:
    """End-to-end pipeline extracting atomic claims from chunks into bitemporal storage."""

    def __init__(self):
        self.engine = get_extraction_engine()

    def process_document_version(self, version_id: str) -> List[Claim]:
        """Extracts and stores claims for all chunks in a document version."""
        persisted_claims: List[Claim] = []
        validation_errors = 0
        total_chunks = 0

        with get_db_session() as session:
            chunks = session.query(Chunk).filter_by(document_version_id=version_id).all()
            total_chunks = len(chunks)

            # Get document publication timestamp for transaction/fallback reference
            doc_version = session.query(DocumentVersion).filter_by(version_id=version_id).first()
            doc = doc_version.document if doc_version else None
            # Transaction time is when this version was created/recorded in the system
            tx_time = doc_version.created_at if doc_version else datetime.now(timezone.utc)

            for chunk in chunks:
                try:
                    raw_claims = self.engine.extract_from_chunk(chunk.chunk_text, chunk.page_number)
                except Exception as e:
                    logger.error(f"Extraction failed for chunk {chunk.chunk_id}: {e}")
                    validation_errors += 1
                    continue

                for rc in raw_claims:
                    # Parse valid_from and valid_until
                    valid_from_dt, valid_from_unk = parse_temporal_boundary(rc.effective_start_date)
                    valid_until_dt, valid_until_unk = parse_temporal_boundary(rc.effective_end_date)

                    # Create or resolve entity canonical name
                    entity_name = rc.subject.strip()
                    entity = session.query(Entity).filter_by(canonical_name=entity_name).first()
                    if not entity:
                        entity = Entity(
                            canonical_name=entity_name,
                            entity_type="PolicySubject",
                            aliases=[entity_name.lower()],
                        )
                        session.add(entity)
                        session.flush()

                    claim = Claim(
                        subject_entity_id=entity.entity_id,
                        subject_text=entity_name,
                        predicate=rc.predicate.strip(),
                        object_value=rc.object_value.strip(),
                        conditions=rc.conditions,
                        valid_from=valid_from_dt,
                        valid_until=valid_until_dt,
                        valid_from_unknown=valid_from_unk,
                        valid_until_unknown=valid_until_unk,
                        recorded_from=tx_time,
                        recorded_until=None,
                        confidence=rc.confidence,
                        extraction_model=self.engine.model_name,
                        extraction_version=self.engine.prompt_version,
                        verification_status="verified",
                    )
                    session.add(claim)
                    session.flush()

                    # Save claim source
                    source = ClaimSource(
                        claim_id=claim.claim_id,
                        chunk_id=chunk.chunk_id,
                        exact_quote=rc.exact_quote,
                        page_number=rc.page_number,
                        extraction_confidence=rc.confidence,
                        source_type="primary_text",
                    )
                    session.add(source)

                    # Update FTS for SQLite
                    if not is_postgres():
                        session.execute(
                            text(
                                "INSERT INTO claims_fts (claim_id, subject_text, predicate, object_value) "
                                "VALUES (:cid, :subj, :pred, :obj)"
                            ),
                            {
                                "cid": claim.claim_id,
                                "subj": claim.subject_text,
                                "pred": claim.predicate,
                                "obj": claim.object_value,
                            },
                        )

                    persisted_claims.append(claim)

            # Record extraction run for auditability
            run = ExtractionRun(
                model_name=self.engine.model_name,
                model_version=self.engine.prompt_version,
                prompt_version=self.engine.prompt_version,
                documents_processed=1,
                claims_extracted=len(persisted_claims),
                validation_errors=validation_errors,
                processing_statistics={
                    "total_chunks": total_chunks,
                    "claims_count": len(persisted_claims),
                    "version_id": version_id,
                },
            )
            session.add(run)
            session.commit()

        logger.info(f"Extracted {len(persisted_claims)} claims for document version {version_id}")
        return persisted_claims
