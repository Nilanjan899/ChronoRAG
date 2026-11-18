from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
from chronorag.database.models import Claim
from chronorag.database.bitemporal import normalize_datetime


class HistoricalTimelineBuilder:
    """Builds a chronological timeline of state changes and policy transitions."""

    @classmethod
    def build_timeline(cls, claims: List[Claim], graph_lineage: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
        events: List[Dict[str, Any]] = []

        # If lineage exists, use claims from lineage as well
        seen_ids = set()
        claim_pool: List[Claim] = list(claims)

        # Sort claims by valid_from (earliest first)
        def sort_key(c: Claim):
            vf = normalize_datetime(c.valid_from)
            return vf if vf is not None else datetime.min.replace(tzinfo=timezone.utc)

        claim_pool.sort(key=sort_key)

        for idx, c in enumerate(claim_pool):
            if c.claim_id in seen_ids:
                continue
            seen_ids.add(c.claim_id)

            start_str = c.valid_from.strftime("%Y-%m-%d") if c.valid_from else "Beginning of Record"
            end_str = c.valid_until.strftime("%Y-%m-%d") if c.valid_until else "Present / Ongoing"

            status = "Initial Policy" if idx == 0 else "Amended / Superseded"
            if c.valid_until is None:
                status = "Active / Current Policy"

            events.append(
                {
                    "step": idx + 1,
                    "claim_id": c.claim_id,
                    "subject": c.subject_text,
                    "predicate": c.predicate,
                    "value": c.object_value,
                    "valid_from": start_str,
                    "valid_until": end_str,
                    "status": status,
                    "conditions": c.conditions or {},
                    "confidence": c.confidence,
                }
            )

        return events
