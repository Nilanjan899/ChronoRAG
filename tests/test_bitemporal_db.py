import pytest
from datetime import datetime, timezone, timedelta
from chronorag.database.connection import init_db, get_db_session, get_engine
from chronorag.database.models import Document, DocumentVersion, Chunk, Claim, Entity, ClaimRelation, ClaimSource
from chronorag.database.bitemporal import (
    BitemporalInterval,
    is_claim_valid_at,
    was_claim_recorded_at,
    is_bitemporally_active,
)


@pytest.fixture(scope="module", autouse=True)
def setup_database():
    init_db()


def test_bitemporal_interval_math():
    # Half-open [2024-01-01, 2024-07-01)
    i1 = BitemporalInterval(
        start=datetime(2024, 1, 1, tzinfo=timezone.utc),
        end=datetime(2024, 7, 1, tzinfo=timezone.utc),
    )
    # [2024-07-01, None) open-ended
    i2 = BitemporalInterval(
        start=datetime(2024, 7, 1, tzinfo=timezone.utc),
        end=None,
    )

    # Contains check: exactly start should be included, end should be excluded
    assert i1.contains(datetime(2024, 1, 1, tzinfo=timezone.utc)) is True
    assert i1.contains(datetime(2024, 6, 30, 23, 59, 59, tzinfo=timezone.utc)) is True
    assert i1.contains(datetime(2024, 7, 1, 0, 0, 0, tzinfo=timezone.utc)) is False

    # Open-ended check
    assert i2.contains(datetime(2024, 7, 1, tzinfo=timezone.utc)) is True
    assert i2.contains(datetime(2025, 12, 31, tzinfo=timezone.utc)) is True
    assert i2.contains(datetime(2024, 6, 30, tzinfo=timezone.utc)) is False

    # Adjacency check
    assert i1.is_adjacent(i2) is True

    # Non-overlapping
    assert i1.overlaps(i2) is False

    # Overlapping interval [2024-05-01, 2024-09-01)
    i_overlap = BitemporalInterval(
        start=datetime(2024, 5, 1, tzinfo=timezone.utc),
        end=datetime(2024, 9, 1, tzinfo=timezone.utc),
    )
    assert i1.overlaps(i_overlap) is True
    assert i2.overlaps(i_overlap) is True


def test_unknown_temporal_boundaries():
    i_unknown_start = BitemporalInterval(
        start=None,
        end=datetime(2024, 7, 1, tzinfo=timezone.utc),
        start_unknown=True,
    )
    # If start is explicitly unknown, we cannot guarantee point containment
    assert i_unknown_start.contains(datetime(2024, 3, 1, tzinfo=timezone.utc)) is False


def test_bitemporal_leave_policy_lifecycle():
    """
    Scenario from project specification:
    Claim 1: 12 annual leaves (valid from 2023-01-01, superseded on 2024-07-01; recorded on 2023-01-01)
    Claim 2: 15 annual leaves (valid from 2024-07-01, until 2025-01-01; recorded on 2024-06-15)
    Claim 3: 18 annual leaves (valid from 2025-01-01 open-ended; recorded on 2025-01-15 - delayed upload!)
    """
    with get_db_session() as session:
        # 1. Claim 1
        c1 = Claim(
            subject_text="Employees",
            predicate="annual leave entitlement",
            object_value="12 days",
            valid_from=datetime(2023, 1, 1, tzinfo=timezone.utc),
            valid_until=datetime(2024, 7, 1, tzinfo=timezone.utc),
            recorded_from=datetime(2023, 1, 1, tzinfo=timezone.utc),
            recorded_until=None,
        )
        # 2. Claim 2
        c2 = Claim(
            subject_text="Employees",
            predicate="annual leave entitlement",
            object_value="15 days",
            valid_from=datetime(2024, 7, 1, tzinfo=timezone.utc),
            valid_until=datetime(2025, 1, 1, tzinfo=timezone.utc),
            recorded_from=datetime(2024, 6, 15, tzinfo=timezone.utc),
            recorded_until=None,
        )
        # 3. Claim 3 - Amendment effective 2025-01-01, but uploaded/recorded 2025-01-15
        c3 = Claim(
            subject_text="Employees",
            predicate="annual leave entitlement",
            object_value="18 days",
            valid_from=datetime(2025, 1, 1, tzinfo=timezone.utc),
            valid_until=None,
            recorded_from=datetime(2025, 1, 15, tzinfo=timezone.utc),
            recorded_until=None,
        )
        session.add_all([c1, c2, c3])
        session.flush()

        # Query 1: What was the entitlement on March 1, 2024?
        # Real-world query: valid_time = 2024-03-01, current transaction time
        q1_time = datetime(2024, 3, 1, tzinfo=timezone.utc)
        assert is_claim_valid_at(c1.valid_from, c1.valid_until, q1_time) is True
        assert is_claim_valid_at(c2.valid_from, c2.valid_until, q1_time) is False
        assert is_claim_valid_at(c3.valid_from, c3.valid_until, q1_time) is False

        # Query 2: What was the entitlement on September 1, 2024?
        q2_time = datetime(2024, 9, 1, tzinfo=timezone.utc)
        assert is_claim_valid_at(c1.valid_from, c1.valid_until, q2_time) is False
        assert is_claim_valid_at(c2.valid_from, c2.valid_until, q2_time) is True
        assert is_claim_valid_at(c3.valid_from, c3.valid_until, q2_time) is False

        # Query 3: What did the knowledge system know on January 5, 2025?
        # On Jan 5, 2025: Claim 3 was NOT yet recorded (recorded 2025-01-15)!
        system_time_jan5 = datetime(2025, 1, 5, tzinfo=timezone.utc)
        assert was_claim_recorded_at(c1.recorded_from, c1.recorded_until, system_time_jan5) is True
        assert was_claim_recorded_at(c2.recorded_from, c2.recorded_until, system_time_jan5) is True
        assert was_claim_recorded_at(c3.recorded_from, c3.recorded_until, system_time_jan5) is False

        # Query 4: What would the system have answered if queried on January 5, 2025 for entitlement on Jan 5, 2025?
        # Valid time: 2025-01-05, Transaction time: 2025-01-05
        # Claim 3 was valid in the real world, BUT the system didn't know it yet!
        # Thus Claim 3 is NOT active bitemporally as of Jan 5, 2025
        assert is_bitemporally_active(
            c3.valid_from, c3.valid_until, c3.recorded_from, c3.recorded_until,
            as_of_valid_time=system_time_jan5, as_of_transaction_time=system_time_jan5
        ) is False

        # What was recorded and valid on Jan 20, 2025 for Jan 5, 2025?
        # Both are now active!
        system_time_jan20 = datetime(2025, 1, 20, tzinfo=timezone.utc)
        assert is_bitemporally_active(
            c3.valid_from, c3.valid_until, c3.recorded_from, c3.recorded_until,
            as_of_valid_time=system_time_jan5, as_of_transaction_time=system_time_jan20
        ) is True
