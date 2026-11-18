from typing import Dict, Any, List
from chronorag.retrieval.engine import RetrievedEvidence


class RetrievalReasoningTrace:
    """Formats an auditable, operational explanation of the retrieval and filtering process."""

    @classmethod
    def build_trace(cls, evidence: RetrievedEvidence) -> Dict[str, Any]:
        q = evidence.query
        return {
            "query_understanding": {
                "raw_query": q.query_text,
                "classified_intent": q.query_type,
                "valid_time_constraint": q.as_of_valid_time.strftime("%Y-%m-%d") if q.as_of_valid_time else "Unbounded / Latest",
                "valid_time_constraint_b": q.as_of_valid_time_b.strftime("%Y-%m-%d") if q.as_of_valid_time_b else None,
                "transaction_time_constraint": q.as_of_transaction_time.strftime("%Y-%m-%d %H:%M:%S") if q.as_of_transaction_time else "Current Knowledge State",
                "extracted_entities": q.target_entities,
                "extracted_predicates": q.predicates,
            },
            "retrieval_strategy": {
                "dense_model": "sentence-transformers/all-MiniLM-L6-v2 (Cosine Similarity)",
                "keyword_model": "Full-Text Search (BM25 / ts_rank)",
                "fusion_method": "Reciprocal Rank Fusion (RRF, k=60)",
                "total_candidates_examined": evidence.total_candidates_examined,
            },
            "deterministic_temporal_filters": {
                "filter_enforced": True if (q.as_of_valid_time or q.as_of_transaction_time) else False,
                "rejected_claims_count": evidence.rejected_claims_count,
                "exclusion_reason": "Fact invalid at requested valid time, or claim unrecorded at requested transaction time.",
            },
            "conflict_analysis": {
                "has_conflicts": evidence.conflict_report.has_conflicts,
                "contradictions": evidence.conflict_report.contradictions,
                "supersessions": evidence.conflict_report.supersessions,
                "condition_divergence": evidence.conflict_report.condition_differences,
            },
            "evidence_selection": {
                "selected_claims_count": len(evidence.claims),
                "selected_claim_ids": [c.claim_id for c in evidence.claims],
                "sources_supplied_to_generator": len(evidence.sources),
                "scoring_breakdown": evidence.scoring_trace,
            },
        }
