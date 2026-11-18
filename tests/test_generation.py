import pytest
from datetime import datetime, timezone
from chronorag.database.connection import init_db, get_db_session
from chronorag.database.models import Claim, ClaimSource, Chunk, Document, DocumentVersion
from chronorag.retrieval.engine import ChronoRetrievalEngine
from chronorag.generation.answer_engine import ChronoAnswerEngine, ChronoAnswer
from chronorag.generation.provenance import CitationValidator


@pytest.fixture(scope="module", autouse=True)
def setup_db():
    init_db()


def test_four_output_elements():
    with get_db_session() as session:
        # Create a document, version, chunk, claim, and source
        doc = Document(filename="HR_Leave_Policy_2024.txt", file_type="txt", content_hash="hash_m6_test_1")
        session.add(doc)
        session.flush()

        ver = DocumentVersion(document_id=doc.document_id, version_number=1, original_file_reference="HR_Leave_Policy_2024.txt", content_hash="hash_m6_test_1")
        session.add(ver)
        session.flush()

        chunk = Chunk(document_version_id=ver.version_id, chunk_text="Effective July 1, 2024, leave increases to 15 days.", page_number=2, chunk_index=0)
        session.add(chunk)
        session.flush()

        claim = Claim(
            subject_text="Employees",
            predicate="annual leave entitlement",
            object_value="15 days",
            valid_from=datetime(2024, 7, 1, tzinfo=timezone.utc),
            valid_until=None,
        )
        session.add(claim)
        session.flush()

        source = ClaimSource(claim_id=claim.claim_id, chunk_id=chunk.chunk_id, exact_quote="leave increases to 15 days", page_number=2)
        session.add(source)
        session.commit()

    retriever = ChronoRetrievalEngine()
    evidence = retriever.retrieve("What was the annual leave entitlement on September 1, 2024?")

    engine = ChronoAnswerEngine()
    result = engine.generate_answer(evidence)

    # 1. Natural language answer
    assert isinstance(result.answer, str)
    assert "15 days" in result.answer
    assert "[Doc:" in result.answer

    # 2. Supporting evidence
    assert len(result.supporting_evidence) >= 1
    assert result.supporting_evidence[0]["value"] == "15 days"
    assert result.supporting_evidence[0]["document"] == "HR_Leave_Policy_2024.txt"

    # 3. Historical timeline
    assert len(result.timeline) >= 1
    assert result.timeline[0]["value"] == "15 days"

    # 4. Retrieval reasoning trace
    assert "query_understanding" in result.reasoning_trace
    assert "retrieval_strategy" in result.reasoning_trace
    assert "deterministic_temporal_filters" in result.reasoning_trace


def test_abstention_on_unsupported_period():
    retriever = ChronoRetrievalEngine()
    # Query a date far in the past before any records existed
    evidence = retriever.retrieve("What was the leave entitlement on January 1, 1990?")

    engine = ChronoAnswerEngine()
    result = engine.generate_answer(evidence)

    assert result.is_abstention is True
    assert "Insufficient Evidence" in result.answer


def test_citation_validation():
    available_sources = [
        {"filename": "HR_Policy.pdf", "page_number": 3, "claim_id": "c-123", "source_id": "s-456"}
    ]

    # Valid citation
    text_valid = "Entitlement is 15 days [Doc: HR_Policy.pdf, p.3]."
    is_valid, valids, fabs = CitationValidator.validate_citations(text_valid, available_sources)
    assert is_valid is True
    assert len(fabs) == 0

    # Fabricated citation
    text_invalid = "Entitlement is 20 days [Doc: Unknown_Fake_Document.pdf, p.99]."
    is_valid_f, valids_f, fabs_f = CitationValidator.validate_citations(text_invalid, available_sources)
    assert is_valid_f is False
    assert len(fabs_f) == 1
