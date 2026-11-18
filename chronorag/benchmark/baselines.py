import time
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
from pydantic import BaseModel, Field

from chronorag.retrieval.dense import DenseRetriever
from chronorag.retrieval.keyword import KeywordRetriever
from chronorag.retrieval.fusion import ReciprocalRankFusion
from chronorag.retrieval.engine import HybridRetrievalEngine
from chronorag.generation.answer_engine import AnswerGenerationEngine, GroundedAnswer
from chronorag.benchmark.dataset import BenchmarkQuestion, GroundTruthFact
from chronorag.extraction.date_parser import DateParser


class BaselineResult(BaseModel):
    system_name: str
    question_id: str
    query_text: str
    answer: str
    retrieved_items: List[Dict[str, Any]] = Field(default_factory=list)
    retrieved_fact_ids: List[str] = Field(default_factory=list)
    retrieved_intervals: List[Dict[str, Optional[datetime]]] = Field(default_factory=list)
    latency_ms: float
    abstained: bool = False
    metadata: Dict[str, Any] = Field(default_factory=dict)


class BaseRAGSystem:
    def __init__(self, name: str):
        self.name = name

    def query(self, question: BenchmarkQuestion) -> BaselineResult:
        raise NotImplementedError


class VanillaDenseRAG(BaseRAGSystem):
    """System A: Traditional Dense-only Vector RAG without temporal awareness."""

    def __init__(
        self,
        dense_retriever: DenseRetriever,
        chunk_corpus: List[Dict[str, Any]],
        top_k: int = 5
    ):
        super().__init__("Vanilla Dense RAG (System A)")
        self.dense_retriever = dense_retriever
        self.chunk_corpus = chunk_corpus
        self.top_k = top_k
        self._index_corpus()

    def _index_corpus(self):
        if not self.chunk_corpus:
            return
        texts = [c["chunk_text"] for c in self.chunk_corpus]
        self.embeddings = self.dense_retriever.embed_texts(texts)

    def query(self, question: BenchmarkQuestion) -> BaselineResult:
        t0 = time.perf_counter()
        query_text = question.question_text
        
        q_emb = self.dense_retriever.embed_query(query_text)
        
        # Dense cosine similarity
        scored_chunks = []
        for i, chunk in enumerate(self.chunk_corpus):
            chunk_emb = self.embeddings[i]
            # Dot product for normalized vectors
            sim = sum(a * b for a, b in zip(q_emb, chunk_emb))
            scored_chunks.append((sim, chunk))
            
        scored_chunks.sort(key=lambda x: x[0], reverse=True)
        top_chunks = [c for _, c in scored_chunks[:self.top_k]]
        
        # Build prompt & answer
        context_parts = []
        retrieved_fact_ids = []
        retrieved_intervals = []
        
        for c in top_chunks:
            context_parts.append(f"- {c['chunk_text']}")
            if "fact_id" in c:
                retrieved_fact_ids.append(c["fact_id"])
            retrieved_intervals.append({
                "valid_from": c.get("valid_from"),
                "valid_until": c.get("valid_until"),
            })
            
        context_str = "\n".join(context_parts)
        
        # Synthetic answer based on top match
        best_chunk = top_chunks[0]["chunk_text"] if top_chunks else "No relevant documents found."
        answer = f"Based on retrieved documents: {best_chunk}"
        
        latency = (time.perf_counter() - t0) * 1000.0
        return BaselineResult(
            system_name=self.name,
            question_id=question.question_id,
            query_text=query_text,
            answer=answer,
            retrieved_items=top_chunks,
            retrieved_fact_ids=retrieved_fact_ids,
            retrieved_intervals=retrieved_intervals,
            latency_ms=round(latency, 2),
            abstained=False,
        )


class HybridRAG(BaseRAGSystem):
    """System B: Hybrid Dense + Keyword RAG (RRF) without temporal reasoning."""

    def __init__(
        self,
        dense_retriever: DenseRetriever,
        keyword_retriever: KeywordRetriever,
        chunk_corpus: List[Dict[str, Any]],
        top_k: int = 5
    ):
        super().__init__("Hybrid RAG (System B)")
        self.dense_retriever = dense_retriever
        self.keyword_retriever = keyword_retriever
        self.chunk_corpus = chunk_corpus
        self.top_k = top_k
        self._index_corpus()

    def _index_corpus(self):
        if not self.chunk_corpus:
            return
        texts = [c["chunk_text"] for c in self.chunk_corpus]
        self.embeddings = self.dense_retriever.embed_texts(texts)

    def query(self, question: BenchmarkQuestion) -> BaselineResult:
        t0 = time.perf_counter()
        query_text = question.question_text
        
        # Dense ranking
        q_emb = self.dense_retriever.embed_query(query_text)
        dense_ranks = {}
        scored_dense = []
        for i, chunk in enumerate(self.chunk_corpus):
            chunk_emb = self.embeddings[i]
            sim = sum(a * b for a, b in zip(q_emb, chunk_emb))
            scored_dense.append((sim, i, chunk))
        scored_dense.sort(key=lambda x: x[0], reverse=True)
        for rank, (_, idx, _) in enumerate(scored_dense):
            dense_ranks[idx] = rank + 1
            
        # Keyword ranking
        keyword_results = self.keyword_retriever.search_chunks(query_text, limit=len(self.chunk_corpus))
        keyword_ranks = {}
        for rank, item in enumerate(keyword_results):
            # match chunk_id
            for idx, c in enumerate(self.chunk_corpus):
                if c.get("chunk_id") == item["chunk_id"]:
                    keyword_ranks[idx] = rank + 1
                    break
                    
        # RRF
        k = 60
        rrf_scores = []
        for idx in range(len(self.chunk_corpus)):
            score = 0.0
            if idx in dense_ranks:
                score += 1.0 / (k + dense_ranks[idx])
            if idx in keyword_ranks:
                score += 1.0 / (k + keyword_ranks[idx])
            rrf_scores.append((score, self.chunk_corpus[idx]))
            
        rrf_scores.sort(key=lambda x: x[0], reverse=True)
        top_chunks = [c for _, c in rrf_scores[:self.top_k]]
        
        retrieved_fact_ids = []
        retrieved_intervals = []
        for c in top_chunks:
            if "fact_id" in c:
                retrieved_fact_ids.append(c["fact_id"])
            retrieved_intervals.append({
                "valid_from": c.get("valid_from"),
                "valid_until": c.get("valid_until"),
            })
            
        best_chunk = top_chunks[0]["chunk_text"] if top_chunks else "No relevant documents found."
        answer = f"Based on hybrid retrieval: {best_chunk}"
        
        latency = (time.perf_counter() - t0) * 1000.0
        return BaselineResult(
            system_name=self.name,
            question_id=question.question_id,
            query_text=query_text,
            answer=answer,
            retrieved_items=top_chunks,
            retrieved_fact_ids=retrieved_fact_ids,
            retrieved_intervals=retrieved_intervals,
            latency_ms=round(latency, 2),
            abstained=False,
        )


class TemporalFilteredRAG(BaseRAGSystem):
    """System C: Hybrid RAG with naive document/chunk metadata timestamp filtering."""

    def __init__(
        self,
        dense_retriever: DenseRetriever,
        chunk_corpus: List[Dict[str, Any]],
        top_k: int = 5
    ):
        super().__init__("Temporal-Filtered RAG (System C)")
        self.dense_retriever = dense_retriever
        self.chunk_corpus = chunk_corpus
        self.top_k = top_k
        self._index_corpus()

    def _index_corpus(self):
        if not self.chunk_corpus:
            return
        texts = [c["chunk_text"] for c in self.chunk_corpus]
        self.embeddings = self.dense_retriever.embed_texts(texts)

    def query(self, question: BenchmarkQuestion) -> BaselineResult:
        t0 = time.perf_counter()
        query_text = question.question_text
        
        # Parse simple target date
        target_dt = None
        if question.target_date:
            parsed = DateParser.parse(question.target_date)
            if parsed:
                target_dt = parsed.dt
                
        # Filter chunks by naive document timestamp <= target_dt (if available)
        eligible_chunks = []
        for i, c in enumerate(self.chunk_corpus):
            # Naive metadata filtering on publication or recorded timestamp
            pub_date = c.get("publication_date") or c.get("recorded_from") or c.get("valid_from")
            if target_dt and pub_date:
                if isinstance(pub_date, str):
                    p = DateParser.parse(pub_date)
                    pub_dt = p.dt if p else None
                else:
                    pub_dt = pub_date
                    
                if pub_dt and pub_dt > target_dt:
                    # Filtered out because document is published after target date
                    continue
            eligible_chunks.append((i, c))
            
        if not eligible_chunks:
            # Fall back to all chunks if filtering eliminated everything
            eligible_chunks = list(enumerate(self.chunk_corpus))

        q_emb = self.dense_retriever.embed_query(query_text)
        scored = []
        for idx, chunk in eligible_chunks:
            chunk_emb = self.embeddings[idx]
            sim = sum(a * b for a, b in zip(q_emb, chunk_emb))
            scored.append((sim, chunk))
            
        scored.sort(key=lambda x: x[0], reverse=True)
        top_chunks = [c for _, c in scored[:self.top_k]]
        
        retrieved_fact_ids = []
        retrieved_intervals = []
        for c in top_chunks:
            if "fact_id" in c:
                retrieved_fact_ids.append(c["fact_id"])
            retrieved_intervals.append({
                "valid_from": c.get("valid_from"),
                "valid_until": c.get("valid_until"),
            })
            
        best_chunk = top_chunks[0]["chunk_text"] if top_chunks else "No relevant documents found."
        answer = f"Based on temporally filtered documents: {best_chunk}"
        
        latency = (time.perf_counter() - t0) * 1000.0
        return BaselineResult(
            system_name=self.name,
            question_id=question.question_id,
            query_text=query_text,
            answer=answer,
            retrieved_items=top_chunks,
            retrieved_fact_ids=retrieved_fact_ids,
            retrieved_intervals=retrieved_intervals,
            latency_ms=round(latency, 2),
            abstained=False,
        )


class FullChronoRAG(BaseRAGSystem):
    """System D: Full ChronoRAG Engine with bitemporal filtering, knowledge graph, conflict resolution & timeline synthesis."""

    def __init__(
        self,
        retrieval_engine: HybridRetrievalEngine,
        answer_engine: AnswerGenerationEngine,
    ):
        super().__init__("ChronoRAG (System D)")
        self.retrieval_engine = retrieval_engine
        self.answer_engine = answer_engine

    def query(self, question: BenchmarkQuestion) -> BaselineResult:
        t0 = time.perf_counter()
        query_text = question.question_text
        
        # Execute temporal retrieval
        evidence_bundle = self.retrieval_engine.retrieve(
            query_str=query_text,
            as_of_valid_time=question.target_date,
            as_of_transaction_time=question.system_knowledge_date,
        )
        
        # Synthesize grounded answer
        grounded_answer: GroundedAnswer = self.answer_engine.generate_answer(
            evidence=evidence_bundle,
        )
        
        retrieved_fact_ids = []
        retrieved_intervals = []
        for claim in evidence_bundle.claims:
            if claim.claim_id:
                retrieved_fact_ids.append(str(claim.claim_id))
            retrieved_intervals.append({
                "valid_from": claim.valid_from,
                "valid_until": claim.valid_until,
            })
            
        latency = (time.perf_counter() - t0) * 1000.0
        conflicts_count = len(evidence_bundle.conflict_report.contradictions) if hasattr(evidence_bundle, 'conflict_report') and evidence_bundle.conflict_report else 0
        return BaselineResult(
            system_name=self.name,
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
            metadata={
                "timeline_events": len(grounded_answer.timeline),
                "conflicts_detected": conflicts_count,
                "citations": len(grounded_answer.supporting_evidence),
            }
        )
