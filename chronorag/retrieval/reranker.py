import math
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
from chronorag.database.models import Claim
from chronorag.database.bitemporal import normalize_datetime


class TemporalReranker:
    """
    Multi-signal reranker combining semantic fusion score, entity overlap,
    temporal proximity to query anchor, and provenance completeness.
    """

    def __init__(
        self,
        weight_semantic: float = 0.40,
        weight_entity: float = 0.25,
        weight_temporal_decay: float = 0.20,
        weight_provenance: float = 0.15,
    ):
        self.w_sem = weight_semantic
        self.w_ent = weight_entity
        self.w_temp = weight_temporal_decay
        self.w_prov = weight_provenance

    def rerank(
        self,
        claims: List[Claim],
        query_text: str,
        target_entities: List[str],
        anchor_date: Optional[datetime] = None,
        base_scores: Optional[Dict[str, float]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Reranks claims and returns list of dictionaries with full scoring breakdown.
        """
        scored_claims = []
        base_map = base_scores or {}

        for c in claims:
            # 1. Semantic / RRF score
            sem_score = base_map.get(c.claim_id, 0.5)

            # 2. Entity match score
            ent_score = 0.0
            if target_entities:
                for target in target_entities:
                    if target.lower() in c.subject_text.lower():
                        ent_score = 1.0
                        break
            else:
                ent_score = 0.5

            # 3. Temporal proximity score (decay based on days difference from anchor date)
            temp_score = 1.0
            norm_anchor = normalize_datetime(anchor_date)
            norm_valid = normalize_datetime(c.valid_from)
            if norm_anchor and norm_valid:
                # Time delta in days
                delta_days = abs((norm_anchor - norm_valid).total_seconds()) / 86400.0
                # Exponential decay half-life = 365 days
                temp_score = math.exp(-delta_days / 365.0)

            # 4. Provenance & Confidence score
            has_sources = len(c.sources) > 0 if hasattr(c, "sources") else True
            prov_score = c.confidence * (1.0 if has_sources else 0.7)

            # Total weighted score
            total_score = (
                self.w_sem * sem_score
                + self.w_ent * ent_score
                + self.w_temp * temp_score
                + self.w_prov * prov_score
            )

            scored_claims.append(
                {
                    "claim": c,
                    "claim_id": c.claim_id,
                    "total_score": round(total_score, 4),
                    "semantic_score": round(sem_score, 4),
                    "entity_score": round(ent_score, 4),
                    "temporal_score": round(temp_score, 4),
                    "provenance_score": round(prov_score, 4),
                }
            )

        # Sort descending by total_score
        scored_claims.sort(key=lambda x: x["total_score"], reverse=True)
        return scored_claims
