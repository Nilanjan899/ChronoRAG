import pytest
from datetime import datetime, timezone
from chronorag.database.connection import init_db, get_db_session
from chronorag.database.models import Claim, ClaimSource, Chunk, Document, DocumentVersion
from chronorag.retrieval.query_parser import TemporalQueryParser
from chronorag.retrieval.fusion import reciprocal_rank_fusion
from chronorag.retrieval.temporal_filter import DeterministicTemporalFilter
from chronorag.retrieval.conflict import ConflictDetector
from chronorag.retrieval.engine import ChronoRetrievalEngine


@pytest.fixture(scope="module", autouse=True)
def setup_db():
    init_db()


def test_query_parser_variants():
    parser = TemporalQueryParser()

    # 1. Historical Snapshot query
    q1 = parser.parse("What was the annual leave entitlement on March 1, 2024?")
    assert q1.query_type == "snapshot"
    assert q1.as_of_valid_time is not None
    assert q1.as_of_valid_time.year == 2024
    assert q1.as_of_valid_time.month == 3

    # 2. Comparison query
    q2 = parser.parse("Compare leave entitlement between March 1, 2024 and September 1, 2024.")
    assert q2.query_type == "comparison"
    assert q2.as_of_valid_time is not None
    assert q2.as_of_valid_time_b is not None

    # 3. Bitemporal / Transaction-time query
    q3 = parser.parse("What did the system know on January 5, 2025?")
    assert q3.query_type == "bitemporal"
    assert q3.as_of_transaction_time is not None
    assert q3.as_of_transaction_time.year == 2025
    assert q3.as_of_transaction_time.month == 1
    assert q3.as_of_transaction_time.day == 5

    # 4. Change detection query
    q4 = parser.parse("When did the annual leave entitlement change?")
    assert q4.query_type == "change_detection"


def test_reciprocal_rank_fusion():
    list1 = [("claim_A", 0.9), ("claim_B", 0.7), ("claim_C", 0.5)]
    list2 = [("claim_B", 0.8), ("claim_A", 0.6), ("claim_D", 0.4)]

    fused = reciprocal_rank_fusion([list1, list2], k=60)
    # Both claim_A and claim_B appeared in both lists, so they must be top
    top_ids = [item[0] for item in fused[:2]]
    assert "claim_A" in top_ids
    assert "claim_B" in top_ids


def test_conflict_detection():
    detector = ConflictDetector()

    # Two claims with same condition, same period, different values
    c1 = Claim(
        subject_text="Engineers",
        predicate="remote days",
        object_value="2 days",
        conditions={"role": "SWE"},
        valid_from=datetime(2024, 1, 1, tzinfo=timezone.utc),
        valid_until=datetime(2024, 12, 31, tzinfo=timezone.utc),
    )
    c2 = Claim(
        subject_text="Engineers",
        predicate="remote days",
        object_value="4 days",
        conditions={"role": "SWE"},
        valid_from=datetime(2024, 6, 1, tzinfo=timezone.utc),
        valid_until=datetime(2024, 12, 31, tzinfo=timezone.utc),
    )
    # Different condition (not a contradiction)
    c3 = Claim(
        subject_text="Engineers",
        predicate="remote days",
        object_value="5 days",
        conditions={"role": "Staff SWE"},
        valid_from=datetime(2024, 1, 1, tzinfo=timezone.utc),
        valid_until=datetime(2024, 12, 31, tzinfo=timezone.utc),
    )

    report = detector.analyze([c1, c2, c3])
    assert report.has_conflicts is True
    assert len(report.contradictions) >= 1
    assert len(report.condition_differences) >= 1


def test_chrono_retrieval_bitemporal_precision():
    """
    Verifies that temporal filtering rejects wrong-era evidence!
    March 2024 query must only retrieve 12-day entitlement, NOT 15 or 18!
    """
    with get_db_session() as session:
        # Create bitemporal records
        c_era1 = Claim(
            subject_text="Employees",
            predicate="annual leave entitlement",
            object_value="12 days",
            valid_from=datetime(2023, 1, 1, tzinfo=timezone.utc),
            valid_until=datetime(2024, 7, 1, tzinfo=timezone.utc),
            recorded_from=datetime(2023, 1, 1, tzinfo=timezone.utc),
        )
        c_era2 = Claim(
            subject_text="Employees",
            predicate="annual leave entitlement",
            object_value="15 days",
            valid_from=datetime(2024, 7, 1, tzinfo=timezone.utc),
            valid_until=datetime(2025, 1, 1, tzinfo=timezone.utc),
            recorded_from=datetime(2024, 6, 15, tzinfo=timezone.utc),
        )
        c_era3 = Claim(
            subject_text="Employees",
            predicate="annual leave entitlement",
            object_value="18 days",
            valid_from=datetime(2025, 1, 1, tzinfo=timezone.utc),
            valid_until=None,
            recorded_from=datetime(2025, 1, 15, tzinfo=timezone.utc),
        )
        session.add_all([c_era1, c_era2, c_era3])
        session.commit()

    engine = ChronoRetrievalEngine(enable_temporal_filtering=True)

    # Query March 1, 2024
    res_march = engine.retrieve("What was the annual leave entitlement on March 1, 2024?")
    retrieved_values = [c.object_value for c in res_march.claims]
    assert "12 days" in retrieved_values
    assert "15 days" not in retrieved_values
    assert "18 days" not in retrieved_values

    # Query September 1, 2024
    res_sept = engine.retrieve("What was the annual leave entitlement on September 1, 2024?")
    retrieved_values_sept = [c.object_value for c in res_sept.claims]
    assert "15 days" in retrieved_values_sept
    assert "12 days" not in retrieved_values_sept
    assert "18 days" not in retrieved_values_sept
