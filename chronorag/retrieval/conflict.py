from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from chronorag.database.models import Claim
from chronorag.database.bitemporal import BitemporalInterval


@dataclass
class ConflictReport:
    has_conflicts: bool = False
    contradictions: List[Dict[str, Any]] = field(default_factory=list)
    supersessions: List[Dict[str, Any]] = field(default_factory=list)
    condition_differences: List[Dict[str, Any]] = field(default_factory=list)
    unresolved_dates: List[Dict[str, Any]] = field(default_factory=list)


class ConflictDetector:
    """Detects contradictions, supersessions, condition divergence, and date gaps."""

    def analyze(self, claims: List[Claim]) -> ConflictReport:
        report = ConflictReport()

        for i, c1 in enumerate(claims):
            # Check for unresolved/missing dates
            if c1.valid_from_unknown or c1.valid_until_unknown:
                report.unresolved_dates.append(
                    {
                        "claim_id": c1.claim_id,
                        "subject": c1.subject_text,
                        "predicate": c1.predicate,
                        "value": c1.object_value,
                        "reason": "Explicitly unknown validity boundary",
                    }
                )

            for c2 in claims[i + 1 :]:
                # Only compare claims with matching subject & predicate
                same_subject = c1.subject_text.lower() == c2.subject_text.lower()
                same_pred = c1.predicate.lower() == c2.predicate.lower()

                if not (same_subject and same_pred):
                    continue

                # Disregard generic non-atomic clauses from conflict detection
                GENERIC_NON_ATOMIC = {"policy statement", "statement", "clause", "document", "policy clause"}
                if (
                    c1.subject_text.lower() in GENERIC_NON_ATOMIC
                    or c1.predicate.lower() in GENERIC_NON_ATOMIC
                    or c2.subject_text.lower() in GENERIC_NON_ATOMIC
                    or c2.predicate.lower() in GENERIC_NON_ATOMIC
                ):
                    continue

                same_cond = c1.conditions == c2.conditions
                same_val = c1.object_value.strip().lower() == c2.object_value.strip().lower()

                int1 = BitemporalInterval(
                    start=c1.valid_from,
                    end=c1.valid_until,
                    start_unknown=bool(c1.valid_from_unknown or False),
                    end_unknown=bool(c1.valid_until_unknown or False),
                )
                int2 = BitemporalInterval(
                    start=c2.valid_from,
                    end=c2.valid_until,
                    start_unknown=bool(c2.valid_from_unknown or False),
                    end_unknown=bool(c2.valid_until_unknown or False),
                )

                # 1. Condition divergence (Not a contradiction!)
                if not same_cond and not same_val:
                    report.condition_differences.append(
                        {
                            "claim_1": {"id": c1.claim_id, "val": c1.object_value, "cond": c1.conditions},
                            "claim_2": {"id": c2.claim_id, "val": c2.object_value, "cond": c2.conditions},
                            "explanation": "Values differ but apply to different criteria/categories.",
                        }
                    )

                # 2. Genuine Contradiction: same conditions, different values, overlapping validity
                elif same_cond and not same_val and int1.overlaps(int2):
                    report.has_conflicts = True
                    report.contradictions.append(
                        {
                            "claim_1": {"id": c1.claim_id, "val": c1.object_value, "valid": str(c1.valid_from)},
                            "claim_2": {"id": c2.claim_id, "val": c2.object_value, "valid": str(c2.valid_from)},
                            "explanation": f"Mutual conflict: '{c1.object_value}' vs '{c2.object_value}' during overlapping validity period.",
                        }
                    )

                # 3. Supersession: one replaces the other sequentially
                elif same_cond and not same_val and not int1.overlaps(int2):
                    newer, older = (c1, c2) if (c1.valid_from or datetime.min) >= (c2.valid_from or datetime.min) else (c2, c1)
                    report.supersessions.append(
                        {
                            "newer_claim_id": newer.claim_id,
                            "newer_val": newer.object_value,
                            "older_claim_id": older.claim_id,
                            "older_val": older.object_value,
                            "transition_date": str(newer.valid_from),
                        }
                    )

        return report
