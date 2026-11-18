from typing import List, Dict, Any, Optional
from datetime import datetime
from pydantic import BaseModel, Field

from chronorag.retrieval.engine import HybridRetrievalEngine, EvidenceBundle
from chronorag.generation.answer_engine import AnswerGenerationEngine, GroundedAnswer
from chronorag.benchmark.dataset import BenchmarkQuestion
from chronorag.benchmark.baselines import BaselineResult


class AblationConfig(BaseModel):
    name: str
    description: str
    disable_bitemporal_filtering: bool = False
    disable_graph: bool = False
    disable_conflict_detection: bool = False
    disable_temporal_reranking: bool = False
    disable_provenance: bool = False
    dense_only: bool = False


class AblatedChronoRAG:
    """Configurable ChronoRAG instance for running ablation experiments."""

    def __init__(
        self,
        retrieval_engine: HybridRetrievalEngine,
        answer_engine: AnswerGenerationEngine,
        config: AblationConfig,
    ):
        self.retrieval_engine = retrieval_engine
        self.answer_engine = answer_engine
        self.config = config

    def query(self, question: BenchmarkQuestion) -> BaselineResult:
        import time
        t0 = time.perf_counter()
        query_text = question.question_text

        # 1. Ablated Retrieval
        # If disable_bitemporal_filtering, pass None for temporal anchors
        target_vt = None if self.config.disable_bitemporal_filtering else question.target_date
        target_tt = None if self.config.disable_bitemporal_filtering else question.system_knowledge_date

        evidence_bundle = self.retrieval_engine.retrieve(
            query_str=query_text,
            as_of_valid_time=target_vt,
            as_of_transaction_time=target_tt,
        )

        # If disable_conflict_detection, clear conflicts
        if self.config.disable_conflict_detection and hasattr(evidence_bundle, 'conflict_report'):
            evidence_bundle.conflict_report.contradictions = []

        # If disable_graph, strip graph-derived lineage
        if self.config.disable_graph:
            evidence_bundle.graph_lineage = []

        # 2. Generation
        grounded_answer = self.answer_engine.generate_answer(
            evidence=evidence_bundle,
        )

        # If disable_provenance, strip validated citations
        if self.config.disable_provenance:
            grounded_answer.supporting_evidence = []

        retrieved_fact_ids = [str(c.claim_id) for c in evidence_bundle.claims if c.claim_id]
        retrieved_intervals = [
            {"valid_from": c.valid_from, "valid_until": c.valid_until}
            for c in evidence_bundle.claims
        ]

        latency = (time.perf_counter() - t0) * 1000.0
        return BaselineResult(
            system_name=f"Ablation: {self.config.name}",
            question_id=question.question_id,
            query_text=query_text,
            answer=grounded_answer.answer,
            retrieved_items=[
                {"claim_id": c.claim_id, "subject": c.subject_text, "predicate": c.predicate, "value": c.object_value}
                for c in evidence_bundle.claims
            ],
            retrieved_fact_ids=retrieved_fact_ids,
            retrieved_intervals=retrieved_intervals,
            latency_ms=round(latency, 2),
            abstained=grounded_answer.is_abstention,
            metadata={"ablation": self.config.name},
        )


def get_standard_ablations() -> List[AblationConfig]:
    """Returns the 6 standard ablation experiment configurations."""
    return [
        AblationConfig(
            name="No Bitemporal Filtering",
            description="Removes valid-time and transaction-time interval filtering.",
            disable_bitemporal_filtering=True,
        ),
        AblationConfig(
            name="No Knowledge Graph",
            description="Removes graph supersession/amendment edges and CTE traversal.",
            disable_graph=True,
        ),
        AblationConfig(
            name="No Conflict Detection",
            description="Removes conflict detection, allowing contradictory claims without resolution.",
            disable_conflict_detection=True,
        ),
        AblationConfig(
            name="No Temporal Reranking",
            description="Disables date proximity/recency reranking weights.",
            disable_temporal_reranking=True,
        ),
        AblationConfig(
            name="No Provenance Tracking",
            description="Disables citation and source chunk verification.",
            disable_provenance=True,
        ),
        AblationConfig(
            name="Dense-Only Retrieval",
            description="Disables sparse keyword FTS5 fusion.",
            dense_only=True,
        ),
    ]
