import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from chronorag.config import settings
from chronorag.database.connection import get_db_session
from chronorag.database.models import Claim, Chunk, Document, DocumentVersion, ClaimSource, ClaimRelation
from chronorag.retrieval.dense import get_embedder
from chronorag.retrieval.keyword import KeywordRetriever
from chronorag.retrieval.fusion import reciprocal_rank_fusion
from chronorag.retrieval.query_parser import TemporalQueryParser, ParsedTemporalQuery
from chronorag.retrieval.temporal_filter import DeterministicTemporalFilter
from chronorag.retrieval.conflict import ConflictDetector, ConflictReport
from chronorag.retrieval.reranker import TemporalReranker
from chronorag.graph.graph_store import TemporalGraphStore

logger = logging.getLogger(__name__)


@dataclass
class RetrievedEvidence:
    query: ParsedTemporalQuery
    claims: List[Claim]
    sources: List[Dict[str, Any]]
    conflict_report: ConflictReport
    scoring_trace: List[Dict[str, Any]]
    graph_lineage: List[Dict[str, Any]]
    total_candidates_examined: int
    rejected_claims_count: int


class ChronoRetrievalEngine:
    """Integrated ChronoRAG Hybrid Bitemporal Retrieval Engine."""

    def __init__(
        self,
        enable_temporal_filtering: bool = True,
        enable_graph_traversal: bool = True,
        enable_conflict_detection: bool = True,
        enable_temporal_reranking: bool = True,
        evidence_budget: Optional[int] = None,
    ):
        self.enable_temporal_filtering = enable_temporal_filtering
        self.enable_graph_traversal = enable_graph_traversal
        self.enable_conflict_detection = enable_conflict_detection
        self.enable_temporal_reranking = enable_temporal_reranking
        self.evidence_budget = evidence_budget or settings.final_evidence_budget

        self.query_parser = TemporalQueryParser()
        self.embedder = get_embedder()
        self.keyword_retriever = KeywordRetriever()
        self.conflict_detector = ConflictDetector()
        self.reranker = TemporalReranker()
        self.graph_store = TemporalGraphStore()

    def retrieve(
        self,
        query_str: str,
        as_of_valid_time: Optional[Any] = None,
        as_of_transaction_time: Optional[Any] = None,
    ) -> RetrievedEvidence:
        """Executes full bitemporal retrieval pipeline."""
        # 1. Parse natural language query
        parsed_q = self.query_parser.parse(query_str)
        if as_of_valid_time is not None:
            if isinstance(as_of_valid_time, str):
                from chronorag.extraction.date_parser import DateParser
                p = DateParser.parse(as_of_valid_time)
                if p:
                    parsed_q.as_of_valid_time = p.dt
            else:
                parsed_q.as_of_valid_time = as_of_valid_time

        if as_of_transaction_time is not None:
            if isinstance(as_of_transaction_time, str):
                from chronorag.extraction.date_parser import DateParser
                p = DateParser.parse(as_of_transaction_time)
                if p:
                    parsed_q.as_of_transaction_time = p.dt
            else:
                parsed_q.as_of_transaction_time = as_of_transaction_time

        logger.info(f"Parsed Query Type: {parsed_q.query_type}, ValidTime: {parsed_q.as_of_valid_time}, TxTime: {parsed_q.as_of_transaction_time}")

        # 2. Sparse / Keyword retrieval
        sparse_hits = self.keyword_retriever.search_claims(query_str, top_k=settings.sparse_top_k)

        # 3. Dense retrieval
        query_vec = self.embedder.embed_text(query_str)
        dense_hits = self._dense_search_claims(query_vec, top_k=settings.dense_top_k)

        # 4. Reciprocal Rank Fusion
        fused_hits = reciprocal_rank_fusion([dense_hits, sparse_hits], k=settings.rrf_k)
        fused_map = {item_id: score for item_id, score in fused_hits}

        with get_db_session() as session:
            # Load candidate claims
            candidate_ids = [item_id for item_id, _ in fused_hits]
            candidates = []
            if candidate_ids:
                candidates = session.query(Claim).filter(Claim.claim_id.in_(candidate_ids)).all()

            # Query intent matching: include claims matching parsed predicates or target entities
            if parsed_q.predicates:
                for pred in parsed_q.predicates:
                    pred_claims = session.query(Claim).filter(Claim.predicate.ilike(f"%{pred}%")).limit(30).all()
                    for pc in pred_claims:
                        if pc.claim_id not in [c.claim_id for c in candidates]:
                            candidates.append(pc)

            if not candidates:
                # Fallback to all claims
                candidates = session.query(Claim).limit(50).all()

            total_examined = len(candidates)

            # 5. Deterministic Temporal Filtering
            rejected_count = 0
            if self.enable_temporal_filtering and (
                parsed_q.as_of_valid_time is not None or parsed_q.as_of_transaction_time is not None
            ):
                eligible_claims = DeterministicTemporalFilter.filter_claim_list(
                    candidates,
                    as_of_valid_time=parsed_q.as_of_valid_time,
                    as_of_transaction_time=parsed_q.as_of_transaction_time,
                )
                rejected_count = total_examined - len(eligible_claims)
            else:
                eligible_claims = candidates

            # If snapshot query returned empty, try fallback to latest known if not strictly historical
            if not eligible_claims and candidates and parsed_q.query_type == "latest":
                eligible_claims = candidates

            # 6. Conflict Detection
            if self.enable_conflict_detection and eligible_claims:
                conflict_report = self.conflict_detector.analyze(eligible_claims)
            else:
                conflict_report = ConflictReport()

            # 7. Temporal Reranking
            if self.enable_temporal_reranking and eligible_claims:
                anchor = parsed_q.as_of_valid_time
                scored_results = self.reranker.rerank(
                    claims=eligible_claims,
                    query_text=query_str,
                    target_entities=parsed_q.target_entities,
                    anchor_date=anchor,
                    base_scores=fused_map,
                )
                final_claims = [item["claim"] for item in scored_results[: self.evidence_budget]]
                scoring_trace = scored_results[: self.evidence_budget]
            else:
                final_claims = eligible_claims[: self.evidence_budget]
                scoring_trace = [{"claim_id": c.claim_id, "total_score": 1.0} for c in final_claims]

            # 8. Graph Lineage Expansion
            graph_lineage = []
            if self.enable_graph_traversal and final_claims:
                top_claim = final_claims[0]
                if parsed_q.query_type in ["change_detection", "supersession", "comparison"]:
                    graph_lineage = self.graph_store.get_supersession_chain(top_claim.claim_id, direction="backward")
                elif parsed_q.query_type == "multi_hop" and parsed_q.target_entities:
                    graph_lineage = self.graph_store.traverse_multi_hop_entities(parsed_q.target_entities[0])

            # 9. Collect Provenance Sources
            sources = []
            final_claim_ids = [c.claim_id for c in final_claims]
            if final_claim_ids:
                claim_sources = (
                    session.query(ClaimSource)
                    .filter(ClaimSource.claim_id.in_(final_claim_ids))
                    .all()
                )
                for s in claim_sources:
                    chunk = session.query(Chunk).filter_by(chunk_id=s.chunk_id).first()
                    doc_ver = chunk.version if chunk else None
                    doc = doc_ver.document if doc_ver else None
                    sources.append(
                        {
                            "claim_id": s.claim_id,
                            "source_id": s.source_id,
                            "exact_quote": s.exact_quote,
                            "page_number": s.page_number,
                            "filename": doc.filename if doc else "unknown",
                            "version_number": doc_ver.version_number if doc_ver else 1,
                        }
                    )

            return RetrievedEvidence(
                query=parsed_q,
                claims=final_claims,
                sources=sources,
                conflict_report=conflict_report,
                scoring_trace=scoring_trace,
                graph_lineage=graph_lineage,
                total_candidates_examined=total_examined,
                rejected_claims_count=rejected_count,
            )

    def _dense_search_claims(self, query_vec: List[float], top_k: int = 10) -> List[Tuple[str, float]]:
        """Computes cosine similarity between query and candidate chunks/claims."""
        hits: List[Tuple[str, float]] = []
        with get_db_session() as session:
            chunks = session.query(Chunk).filter(Chunk.embedding != None).all()
            for ch in chunks:
                if ch.embedding:
                    sim = self.embedder.cosine_similarity(query_vec, ch.embedding)
                    # Map chunk to associated claims
                    for src in ch.sources:
                        hits.append((src.claim_id, sim))

        # Sort and deduplicate
        hits.sort(key=lambda x: x[1], reverse=True)
        seen = set()
        deduped = []
        for cid, score in hits:
            if cid not in seen:
                seen.add(cid)
                deduped.append((cid, score))
            if len(deduped) >= top_k:
                break
        return deduped


HybridRetrievalEngine = ChronoRetrievalEngine
EvidenceBundle = RetrievedEvidence

