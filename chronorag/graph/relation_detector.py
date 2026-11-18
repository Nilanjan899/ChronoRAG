import logging
from typing import List, Optional, Tuple, Dict, Any
from datetime import datetime
from chronorag.database.connection import get_db_session
from chronorag.database.models import Claim, ClaimRelation
from chronorag.database.bitemporal import BitemporalInterval, is_claim_valid_at

logger = logging.getLogger(__name__)


class TemporalRelationDetector:
    """
    Detects relationships between atomic claims:
    - SUPERSEDES: Newer claim replaces older claim on same subject+predicate.
    - AMENDS: Modifies applicability, conditions, or scope.
    - CONTRADICTS: Overlapping validity intervals asserting conflicting values under same conditions.
    - SUPPORTS: Multiple claims affirming identical fact.
    - REFINES: Adds narrower condition or specialized subgroup detail.
    """

    def detect_relations_for_claim(self, claim: Claim) -> List[ClaimRelation]:
        detected: List[ClaimRelation] = []

        with get_db_session() as session:
            # Find candidate claims with matching or related subject & predicate
            candidates = (
                session.query(Claim)
                .filter(
                    Claim.claim_id != claim.claim_id,
                    Claim.predicate.ilike(f"%{claim.predicate}%"),
                )
                .all()
            )

            for cand in candidates:
                rel = self._evaluate_pair(claim, cand)
                if rel:
                    session.add(rel)
                    detected.append(rel)

            session.commit()
        return detected

    def detect_all_relations(self) -> int:
        """Scans all claims in the database and computes relationship graph."""
        count = 0
        with get_db_session() as session:
            claims = session.query(Claim).order_by(Claim.created_at.asc()).all()
            for i, c1 in enumerate(claims):
                for c2 in claims[i + 1 :]:
                    rel = self._evaluate_pair(c2, c1)
                    if rel:
                        # Check if relation already exists
                        exists = (
                            session.query(ClaimRelation)
                            .filter_by(
                                source_claim_id=rel.source_claim_id,
                                target_claim_id=rel.target_claim_id,
                                relation_type=rel.relation_type,
                            )
                            .first()
                        )
                        if not exists:
                            session.add(rel)
                            count += 1
            session.commit()
        return count

    def _evaluate_pair(self, newer: Claim, older: Claim) -> Optional[ClaimRelation]:
        """
        Evaluates relationship from newer (source) to older (target).
        """
        # Compare subject and predicate
        same_subject = (
            newer.subject_text.lower() == older.subject_text.lower()
            or (newer.subject_entity_id and newer.subject_entity_id == older.subject_entity_id)
        )
        same_predicate = newer.predicate.lower() == older.predicate.lower()

        if not (same_subject and same_predicate):
            return None

        # Build intervals
        int_newer = BitemporalInterval(
            start=newer.valid_from,
            end=newer.valid_until,
            start_unknown=newer.valid_from_unknown,
            end_unknown=newer.valid_until_unknown,
        )
        int_older = BitemporalInterval(
            start=older.valid_from,
            end=older.valid_until,
            start_unknown=older.valid_from_unknown,
            end_unknown=older.valid_until_unknown,
        )

        same_conditions = newer.conditions == older.conditions
        same_value = newer.object_value.strip().lower() == older.object_value.strip().lower()

        # 1. SUPPORTS: Identical value, overlapping or agreeing validity
        if same_value and same_conditions:
            return ClaimRelation(
                source_claim_id=newer.claim_id,
                target_claim_id=older.claim_id,
                relation_type="SUPPORTS",
                confidence=0.98,
                evidence=f"Claims agree on '{newer.object_value}' for {newer.subject_text}.",
            )

        # 2. SUPERSEDES: Different value, newer starts at or after older
        # e.g.: older was valid until 2024-07-01 or open-ended, newer starts 2024-07-01
        if not same_value and same_conditions:
            if newer.valid_from and older.valid_from and newer.valid_from >= older.valid_from:
                # If older interval was open-ended or ends at newer.valid_from
                return ClaimRelation(
                    source_claim_id=newer.claim_id,
                    target_claim_id=older.claim_id,
                    relation_type="SUPERSEDES",
                    confidence=0.95,
                    evidence=f"New entitlement '{newer.object_value}' replaces previous '{older.object_value}' as of {newer.valid_from.date() if newer.valid_from else 'effective date'}.",
                )

        # 3. REFINES: Same predicate, different conditions (e.g. tenure >= 2 years vs general)
        if not same_conditions and (newer.conditions or older.conditions):
            return ClaimRelation(
                source_claim_id=newer.claim_id,
                target_claim_id=older.claim_id,
                relation_type="REFINES",
                confidence=0.88,
                evidence=f"Refines conditions: {newer.conditions} vs {older.conditions}.",
            )

        # 4. CONTRADICTS: Overlapping validity intervals asserting different values with same conditions
        if not same_value and same_conditions and int_newer.overlaps(int_older):
            return ClaimRelation(
                source_claim_id=newer.claim_id,
                target_claim_id=older.claim_id,
                relation_type="CONTRADICTS",
                confidence=0.90,
                evidence=f"Conflicting values ('{newer.object_value}' vs '{older.object_value}') during overlapping period.",
            )

        # 5. AMENDS: General amendment
        if not same_value and newer.valid_from and older.valid_from and newer.valid_from > older.valid_from:
            return ClaimRelation(
                source_claim_id=newer.claim_id,
                target_claim_id=older.claim_id,
                relation_type="AMENDS",
                confidence=0.85,
                evidence=f"Amends earlier policy state.",
            )

        return None
