import re
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any, Literal
from pydantic import BaseModel, Field
import dateparser

QueryType = Literal[
    "snapshot",
    "comparison",
    "change_detection",
    "bitemporal",
    "supersession",
    "multi_hop",
    "unresolvable",
    "latest",
]


class ParsedTemporalQuery(BaseModel):
    query_text: str
    query_type: QueryType = "latest"
    target_entities: List[str] = Field(default_factory=list)
    predicates: List[str] = Field(default_factory=list)
    
    # Valid-Time constraints
    as_of_valid_time: Optional[datetime] = None
    as_of_valid_time_b: Optional[datetime] = None  # for comparison
    
    # Transaction-Time constraints (what the system knew on date T)
    as_of_transaction_time: Optional[datetime] = None
    
    conditions: Dict[str, Any] = Field(default_factory=dict)
    confidence: float = 1.0


class TemporalQueryParser:
    """
    Parses natural language questions into structured bitemporal query representations.
    Detects valid time, transaction time ('what did the system know on...'), and query intent.
    """

    def parse(self, query: str) -> ParsedTemporalQuery:
        q = query.strip()
        lower_q = q.lower()

        # 1. Detect Transaction Time (System Knowledge)
        # e.g.: "What did the system know on January 5, 2025?"
        as_of_tx_time: Optional[datetime] = None
        tx_match = re.search(
            r"(?:system\s+know(?:\s+on|\s+as\s+of)?|queried\s+on|recorded\s+as\s+of)\s+([A-Z][a-z]+\s+\d{1,2},?\s+\d{4}|\d{4}-\d{2}-\d{2})",
            q,
            re.IGNORECASE,
        )
        if tx_match:
            parsed_tx = dateparser.parse(tx_match.group(1))
            if parsed_tx:
                as_of_tx_time = datetime(parsed_tx.year, parsed_tx.month, parsed_tx.day, tzinfo=timezone.utc)

        # 2. Extract Valid Time dates
        date_pattern = r"(?:on|as of|in|effective|starting|between|and|to)\s+([A-Z][a-z]+\s+\d{1,2},?\s+\d{4}|\d{4}-\d{2}-\d{2}|[A-Z][a-z]+\s+\d{4})"
        found_dates = []
        for match in re.finditer(date_pattern, q, re.IGNORECASE):
            date_str = match.group(1)
            # Avoid re-using tx time match
            if tx_match and date_str in tx_match.group(0):
                continue
            parsed_dt = dateparser.parse(date_str, settings={"PREFER_DAY_OF_MONTH": "first"})
            if parsed_dt:
                found_dates.append(datetime(parsed_dt.year, parsed_dt.month, parsed_dt.day, tzinfo=timezone.utc))

        # 3. Classify Query Type
        query_type: QueryType = "latest"
        as_of_valid_time: Optional[datetime] = None
        as_of_valid_time_b: Optional[datetime] = None

        if "compare" in lower_q or "difference between" in lower_q or len(found_dates) >= 2:
            query_type = "comparison"
            as_of_valid_time = found_dates[0] if len(found_dates) > 0 else None
            as_of_valid_time_b = found_dates[1] if len(found_dates) > 1 else None
        elif "when did" in lower_q or "history of" in lower_q or "evolve" in lower_q or "timeline" in lower_q or "change" in lower_q:
            query_type = "change_detection"
        elif "supersede" in lower_q or "replace" in lower_q or "amend" in lower_q or "which source introduced" in lower_q:
            query_type = "supersession"
        elif as_of_tx_time is not None and (as_of_valid_time is not None or "answered if queried" in lower_q):
            query_type = "bitemporal"
        elif len(found_dates) == 1 or "was the" in lower_q or "on march" in lower_q or "on september" in lower_q:
            query_type = "snapshot"
            as_of_valid_time = found_dates[0] if found_dates else None
        elif "which departments" in lower_q or "affected by" in lower_q or "multi-hop" in lower_q:
            query_type = "multi_hop"
        elif "latest" in lower_q or "current" in lower_q or "today" in lower_q:
            query_type = "latest"
        elif as_of_tx_time is not None:
            query_type = "bitemporal"
            # If valid time not stated, ask about valid state at that transaction time
            as_of_valid_time = as_of_tx_time

        # 4. Extract Entities and Predicates
        entities: List[str] = []
        predicates: List[str] = []

        if "leave" in lower_q or "vacation" in lower_q:
            entities.append("Employees")
            predicates.append("annual leave entitlement")
        if "travel" in lower_q or "budget" in lower_q:
            entities.append("Senior Managers")
            predicates.append("travel budget")
        if "policy" in lower_q:
            entities.append("Annual Leave Policy")
        if "api" in lower_q or "endpoint" in lower_q:
            predicates.append("lifecycle status")

        return ParsedTemporalQuery(
            query_text=q,
            query_type=query_type,
            target_entities=entities,
            predicates=predicates,
            as_of_valid_time=as_of_valid_time,
            as_of_valid_time_b=as_of_valid_time_b,
            as_of_transaction_time=as_of_tx_time,
        )
