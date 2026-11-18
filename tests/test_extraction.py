import pytest
from datetime import datetime, timezone
from pydantic import ValidationError
from chronorag.database.connection import init_db, get_db_session
from chronorag.database.models import Claim, ClaimSource, ExtractionRun
from chronorag.extraction.models import RawExtractedClaim, ClaimExtractionBatch
from chronorag.extraction.date_parser import parse_temporal_boundary
from chronorag.extraction.engine import LocalClaimExtractionEngine
from chronorag.extraction.pipeline import ClaimExtractionPipeline
from chronorag.ingestion.pipeline import IngestionPipeline


@pytest.fixture(scope="module", autouse=True)
def setup_db():
    init_db()


def test_date_parser_temporal_boundaries():
    # 1. Standard ISO date
    dt, is_unk = parse_temporal_boundary("2024-07-01")
    assert is_unk is False
    assert dt == datetime(2024, 7, 1, tzinfo=timezone.utc)

    # 2. Natural language date
    dt2, is_unk2 = parse_temporal_boundary("July 1, 2024")
    assert is_unk2 is False
    assert dt2.year == 2024 and dt2.month == 7 and dt2.day == 1

    # 3. Open-ended markers -> None, False
    dt3, is_unk3 = parse_temporal_boundary("current")
    assert dt3 is None
    assert is_unk3 is False

    # 4. Explicitly unknown markers -> None, True
    dt4, is_unk4 = parse_temporal_boundary("unknown")
    assert dt4 is None
    assert is_unk4 is True

    dt5, is_unk5 = parse_temporal_boundary("TBD")
    assert dt5 is None
    assert is_unk5 is True


def test_pydantic_claim_validation():
    # Valid claim
    claim = RawExtractedClaim(
        subject="Employees",
        predicate="annual leave entitlement",
        object_value="15 days",
        exact_quote="Employees receive 15 days of annual leave.",
        effective_start_date="2024-07-01",
        confidence=0.95,
    )
    assert claim.subject == "Employees"
    assert claim.confidence == 0.95

    # Invalid claim (confidence out of bounds)
    with pytest.raises(ValidationError):
        RawExtractedClaim(
            subject="Employees",
            predicate="leave",
            object_value="15 days",
            exact_quote="quote",
            confidence=1.5,  # must be <= 1.0
        )


def test_claim_extraction_from_text():
    engine = LocalClaimExtractionEngine(backend="deterministic")
    passage = "Effective July 1, 2024, employees receive 15 annual leaves instead of the previous entitlement of 12."
    claims = engine.extract_from_chunk(passage, page_number=3)
    assert len(claims) >= 1

    c = claims[0]
    assert "Employees" in c.subject
    assert "annual leave" in c.predicate
    assert "15" in c.object_value
    assert "2024" in c.effective_start_date
    assert c.previous_value == "12"
    assert c.page_number == 3


def test_full_extraction_pipeline(tmp_path):
    # Ingest document
    ingest = IngestionPipeline()
    p = tmp_path / "policy_doc.txt"
    p.write_text(
        "Effective January 1, 2025, employees receive 18 annual leaves instead of 15.",
        encoding="utf-8",
    )
    doc, ver, _ = ingest.ingest_file(p)

    # Run extraction pipeline
    extractor = ClaimExtractionPipeline()
    extracted_claims = extractor.process_document_version(ver.version_id)
    assert len(extracted_claims) >= 1

    # Verify database persistence
    with get_db_session() as session:
        stored_claim = session.query(Claim).filter_by(claim_id=extracted_claims[0].claim_id).first()
        assert stored_claim is not None
        assert stored_claim.valid_from.year == 2025
        assert stored_claim.valid_from.month == 1

        # Check provenance in claim_sources
        source = session.query(ClaimSource).filter_by(claim_id=stored_claim.claim_id).first()
        assert source is not None
        assert "18 annual leaves" in source.exact_quote

        # Check extraction audit log
        run = session.query(ExtractionRun).first()
        assert run is not None
        assert run.claims_extracted >= 1
