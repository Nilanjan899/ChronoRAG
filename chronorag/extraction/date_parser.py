import re
from datetime import datetime, timezone
from typing import Optional, Tuple
import dateparser

UNKNOWN_DATE_MARKERS = {
    "unknown",
    "unspecified",
    "tbd",
    "tba",
    "n/a",
    "none",
    "null",
    "open",
    "open-ended",
    "ongoing",
    "indefinite",
    "current",
    "present",
}


def parse_temporal_boundary(date_str: Optional[str]) -> Tuple[Optional[datetime], bool]:
    """
    Parses a date string into (normalized_datetime_utc, is_unknown).
    - If date_str is None or empty: returns (None, False) -> represents open-ended/unbounded.
    - If date_str is explicitly unknown ('unknown', 'TBD', etc.): returns (None, True).
    - If date_str is a valid date: returns (datetime, False).
    - If date_str is unparseable: returns (None, True) to avoid inventing dates!
    """
    if not date_str:
        return None, False

    clean_str = date_str.strip().lower()
    if clean_str in UNKNOWN_DATE_MARKERS:
        if clean_str in {"open", "open-ended", "ongoing", "indefinite", "current", "present"}:
            return None, False  # Open-ended
        return None, True  # Explicitly unknown

    # ISO Format regex check: YYYY-MM-DD
    iso_match = re.search(r"(\d{4})-(\d{2})-(\d{2})", date_str)
    if iso_match:
        try:
            year, month, day = map(int, iso_match.groups())
            return datetime(year, month, day, tzinfo=timezone.utc), False
        except ValueError:
            return None, True

    # Natural language parsing with dateparser
    parsed = dateparser.parse(
        date_str,
        settings={"PREFER_DAY_OF_MONTH": "first", "TIMEZONE": "UTC", "RETURN_AS_TIMEZONE_AWARE": True},
    )

    if parsed is not None:
        # Normalize to UTC
        return parsed.astimezone(timezone.utc), False

    # Could not parse: explicitly unknown rather than invented!
    return None, True


class ParsedDateResult:
    def __init__(self, dt: Optional[datetime], is_unknown: bool):
        self.dt = dt
        self.is_unknown = is_unknown


class DateParser:
    @staticmethod
    def parse(date_str: Optional[str]) -> Optional[ParsedDateResult]:
        dt, is_unknown = parse_temporal_boundary(date_str)
        return ParsedDateResult(dt, is_unknown)

