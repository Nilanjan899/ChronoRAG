from datetime import datetime, timezone
from typing import Optional, Tuple
from pydantic import BaseModel, Field


class BitemporalInterval(BaseModel):
    """
    Represents a half-open temporal interval [start, end)
    with explicit flags for unknown boundaries vs open-ended/unbounded.
    """
    start: Optional[datetime] = None
    end: Optional[datetime] = None
    start_unknown: bool = False
    end_unknown: bool = False

    def contains(self, point: datetime) -> bool:
        """
        Check if point falls within [start, end).
        If start is unknown, we cannot guarantee point is contained unless start_unknown is resolved.
        If start is None and not start_unknown, it means unbounded past (-inf).
        If end is None and not end_unknown, it means open-ended (+inf).
        """
        if self.start_unknown or self.end_unknown:
            return False

        if self.start is not None and point < self.start:
            return False
        if self.end is not None and point >= self.end:
            return False
        return True

    def overlaps(self, other: "BitemporalInterval") -> bool:
        """
        Check if two intervals overlap: max(start_A, start_B) < min(end_A, end_B)
        """
        if self.start_unknown or self.end_unknown or other.start_unknown or other.end_unknown:
            return False

        # If A starts after or at B's end
        if other.end is not None and self.start is not None and self.start >= other.end:
            return False
        # If B starts after or at A's end
        if self.end is not None and other.start is not None and other.start >= self.end:
            return False
        return True

    def is_adjacent(self, other: "BitemporalInterval") -> bool:
        """
        Two intervals are adjacent if self.end == other.start or other.end == self.start.
        """
        if self.end is not None and other.start is not None and self.end == other.start:
            return True
        if other.end is not None and self.start is not None and other.end == self.start:
            return True
        return False


def normalize_datetime(dt: Optional[datetime]) -> Optional[datetime]:
    """Ensure datetime is timezone-aware UTC."""
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def is_claim_valid_at(
    valid_from: Optional[datetime],
    valid_until: Optional[datetime],
    query_time: datetime,
    valid_from_unknown: bool = False,
    valid_until_unknown: bool = False,
) -> bool:
    """
    Checks if a claim was true in the real world at query_time.
    Interval is half-open: [valid_from, valid_until).
    """
    if valid_from_unknown:
        return False
    q = normalize_datetime(query_time)
    vf = normalize_datetime(valid_from)
    vu = normalize_datetime(valid_until)

    if vf is not None and q < vf:
        return False
    if vu is not None and q >= vu:
        return False
    return True


def was_claim_recorded_at(
    recorded_from: datetime,
    recorded_until: Optional[datetime],
    system_time: datetime,
) -> bool:
    """
    Checks if the system knew/recorded the claim at system_time.
    Interval is half-open: [recorded_from, recorded_until).
    If recorded_until is None, it is currently known and has not been invalidated.
    """
    st = normalize_datetime(system_time)
    rf = normalize_datetime(recorded_from)
    ru = normalize_datetime(recorded_until)

    if st < rf:
        return False
    if ru is not None and st >= ru:
        return False
    return True


def is_bitemporally_active(
    valid_from: Optional[datetime],
    valid_until: Optional[datetime],
    recorded_from: datetime,
    recorded_until: Optional[datetime],
    as_of_valid_time: datetime,
    as_of_transaction_time: datetime,
    valid_from_unknown: bool = False,
    valid_until_unknown: bool = False,
) -> bool:
    """
    True bitemporal check:
    1. Valid time condition: Was the fact true at as_of_valid_time?
    2. Transaction time condition: Did the system know this fact at as_of_transaction_time?
    """
    valid = is_claim_valid_at(
        valid_from, valid_until, as_of_valid_time, valid_from_unknown, valid_until_unknown
    )
    if not valid:
        return False

    known = was_claim_recorded_at(recorded_from, recorded_until, as_of_transaction_time)
    return known
