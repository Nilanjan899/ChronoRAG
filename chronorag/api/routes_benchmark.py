import logging
from datetime import datetime, timezone
from typing import Optional
from fastapi import APIRouter, HTTPException
from chronorag.api.schemas import BenchmarkRunRequest, BenchmarkRunResponse
from chronorag.benchmark.runner import ChronoBenchRunner, BenchmarkReport
from chronorag.benchmark.dataset import ChronoBenchDatasetGenerator
from chronorag.benchmark.baselines import (
    VanillaDenseRAG,
    HybridRAG,
    TemporalFilteredRAG,
    FullChronoRAG,
)
from chronorag.benchmark.ablations import AblatedChronoRAG, get_standard_ablations
from chronorag.retrieval.dense import DenseRetriever
from chronorag.retrieval.keyword import KeywordRetriever
from chronorag.retrieval.engine import ChronoRetrievalEngine
from chronorag.generation.answer_engine import ChronoAnswerEngine

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/benchmark", tags=["ChronoBench"])

_latest_report: Optional[BenchmarkReport] = None


@router.post("/run", response_model=BenchmarkRunResponse)
def run_benchmark(payload: BenchmarkRunRequest):
    """
    Executes the ChronoBench evaluation suite across:
    1. System A: Vanilla Dense RAG
    2. System B: Hybrid RAG
    3. System C: Temporal-Filtered RAG
    4. System D: Full ChronoRAG
    Optionally evaluates the 6 ablation variants.
    """
    global _latest_report
    try:
        gen = ChronoBenchDatasetGenerator(seed=42)
        facts, questions = gen.generate_benchmark_suite(num_questions=payload.num_questions)

        # Build chunk corpus from facts
        chunk_corpus = []
        for f in facts:
            chunk_corpus.append({
                "chunk_id": f.fact_id,
                "chunk_text": f.source_passage,
                "fact_id": f.fact_id,
                "valid_from": f.valid_from,
                "valid_until": f.valid_until,
                "recorded_from": f.recorded_from,
                "recorded_until": f.recorded_until,
                "publication_date": f.recorded_from,
            })

        dense_retriever = DenseRetriever()
        keyword_retriever = KeywordRetriever()
        retrieval_engine = ChronoRetrievalEngine()
        answer_engine = ChronoAnswerEngine()

        sys_a = VanillaDenseRAG(dense_retriever, chunk_corpus, top_k=3)
        sys_b = HybridRAG(dense_retriever, keyword_retriever, chunk_corpus, top_k=3)
        sys_c = TemporalFilteredRAG(dense_retriever, chunk_corpus, top_k=3)
        sys_d = FullChronoRAG(retrieval_engine, answer_engine)

        runner = ChronoBenchRunner(questions=questions)

        logger.info(f"Evaluating System A on {len(questions)} questions...")
        sum_a = runner.evaluate_system(sys_a)

        logger.info(f"Evaluating System B...")
        sum_b = runner.evaluate_system(sys_b)

        logger.info(f"Evaluating System C...")
        sum_c = runner.evaluate_system(sys_c)

        logger.info(f"Evaluating System D (Full ChronoRAG)...")
        sum_d = runner.evaluate_system(sys_d)

        systems = [sum_a, sum_b, sum_c, sum_d]
        ablations = []

        if payload.include_ablations:
            configs = get_standard_ablations()
            for cfg in configs:
                abl_sys = AblatedChronoRAG(retrieval_engine, answer_engine, cfg)
                abl_sum = runner.evaluate_system(abl_sys)
                ablations.append(abl_sum)

        timestamp_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        report = BenchmarkReport(
            run_timestamp=timestamp_str,
            num_questions=len(questions),
            systems=systems,
            ablations=ablations,
        )
        _latest_report = report

        markdown = runner.generate_markdown_report(report)

        return BenchmarkRunResponse(
            run_timestamp=timestamp_str,
            total_questions=len(questions),
            report_markdown=markdown,
            systems_summary=[s.model_dump() for s in systems],
            ablations_summary=[a.model_dump() for a in ablations],
        )
    except Exception as e:
        logger.error(f"Error running benchmark: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/latest", response_model=Optional[BenchmarkRunResponse])
def get_latest_benchmark():
    """Returns the most recent cached benchmark report, if any."""
    if _latest_report is None:
        return None
    runner = ChronoBenchRunner(questions=[])
    markdown = runner.generate_markdown_report(_latest_report)
    return BenchmarkRunResponse(
        run_timestamp=_latest_report.run_timestamp,
        total_questions=_latest_report.num_questions,
        report_markdown=markdown,
        systems_summary=[s.model_dump() for s in _latest_report.systems],
        ablations_summary=[a.model_dump() for a in _latest_report.ablations],
    )
