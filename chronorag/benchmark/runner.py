import json
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from pydantic import BaseModel, Field

from chronorag.benchmark.dataset import BenchmarkQuestion, ChronoBenchDatasetGenerator
from chronorag.benchmark.baselines import (
    BaseRAGSystem,
    VanillaDenseRAG,
    HybridRAG,
    TemporalFilteredRAG,
    FullChronoRAG,
    BaselineResult,
)
from chronorag.benchmark.metrics import BenchmarkMetrics, EvaluationResult
from chronorag.benchmark.ablations import AblatedChronoRAG, get_standard_ablations
from chronorag.extraction.date_parser import DateParser

logger = logging.getLogger(__name__)


class BenchmarkSummary(BaseModel):
    system_name: str
    total_questions: int
    exact_match: float
    token_f1: float
    historical_accuracy: float
    temporal_precision_at_k: float
    temporal_coverage_at_k: float
    wrong_era_retrieval_rate: float
    abstention_accuracy: float
    mean_latency_ms: float
    p95_latency_ms: float
    by_domain: Dict[str, Dict[str, float]] = Field(default_factory=dict)
    by_category: Dict[str, Dict[str, float]] = Field(default_factory=dict)


class BenchmarkReport(BaseModel):
    run_timestamp: str
    num_questions: int
    systems: List[BenchmarkSummary]
    ablations: List[BenchmarkSummary] = Field(default_factory=list)
    detailed_evaluations: List[EvaluationResult] = Field(default_factory=list)


class ChronoBenchRunner:
    """Executes benchmark evaluation across all systems and ablations."""

    def __init__(self, questions: Optional[List[BenchmarkQuestion]] = None):
        if questions is None:
            gen = ChronoBenchDatasetGenerator(seed=42)
            _, self.questions = gen.generate_benchmark_suite(num_questions=50)
        else:
            self.questions = questions

    def evaluate_system(
        self,
        system: BaseRAGSystem,
        questions: Optional[List[BenchmarkQuestion]] = None
    ) -> BenchmarkSummary:
        eval_qs = questions or self.questions
        eval_results: List[EvaluationResult] = []

        for q in eval_qs:
            # Parse target date for metric checks
            target_dt = None
            if q.target_date:
                parsed = DateParser.parse(q.target_date)
                if parsed:
                    target_dt = parsed.dt

            try:
                res: BaselineResult = system.query(q)
            except Exception as e:
                logger.error(f"Error querying {system.name} on {q.question_id}: {e}")
                res = BaselineResult(
                    system_name=system.name,
                    question_id=q.question_id,
                    query_text=q.question_text,
                    answer=f"Error: {str(e)}",
                    latency_ms=0.0,
                    abstained=False,
                )

            # Compute metrics
            em = BenchmarkMetrics.compute_exact_match(res.answer, q.expected_answer)
            f1 = BenchmarkMetrics.compute_token_f1(res.answer, q.expected_answer)
            hist_acc = BenchmarkMetrics.compute_historical_accuracy(
                prediction=res.answer,
                ground_truth=q.expected_answer,
                should_abstain=q.should_abstain,
            )
            prec_k = BenchmarkMetrics.compute_temporal_precision_at_k(
                retrieved_intervals=res.retrieved_intervals,
                target_date=target_dt,
            )
            cov_k = BenchmarkMetrics.compute_temporal_coverage_at_k(
                retrieved_fact_ids=res.retrieved_fact_ids,
                expected_fact_ids=q.expected_fact_ids,
            )
            wrong_era = BenchmarkMetrics.compute_wrong_era_retrieval_rate(
                retrieved_intervals=res.retrieved_intervals,
                target_date=target_dt,
            )
            abstain_acc = BenchmarkMetrics.compute_abstention_accuracy(
                prediction=res.answer,
                should_abstain=q.should_abstain,
            )

            eval_results.append(
                EvaluationResult(
                    question_id=q.question_id,
                    system_name=system.name,
                    predicted_answer=res.answer,
                    ground_truth_answer=q.expected_answer,
                    exact_match=em,
                    token_f1=f1,
                    historical_accuracy=hist_acc,
                    temporal_precision_at_k=prec_k,
                    temporal_coverage_at_k=cov_k,
                    wrong_era_retrieval_rate=wrong_era,
                    abstention_accuracy=abstain_acc,
                    latency_ms=res.latency_ms,
                    retrieved_fact_ids=res.retrieved_fact_ids,
                    metadata={"domain": q.domain, "category": q.category},
                )
            )

        return self._aggregate_summary(system.name, eval_qs, eval_results)

    def _aggregate_summary(
        self,
        system_name: str,
        questions: List[BenchmarkQuestion],
        evals: List[EvaluationResult],
    ) -> BenchmarkSummary:
        n = len(evals)
        if n == 0:
            return BenchmarkSummary(
                system_name=system_name,
                total_questions=0,
                exact_match=0.0,
                token_f1=0.0,
                historical_accuracy=0.0,
                temporal_precision_at_k=0.0,
                temporal_coverage_at_k=0.0,
                wrong_era_retrieval_rate=0.0,
                abstention_accuracy=0.0,
                mean_latency_ms=0.0,
                p95_latency_ms=0.0,
            )

        latencies = sorted([e.latency_ms for e in evals])
        p95_idx = int(n * 0.95)
        p95 = latencies[min(p95_idx, n - 1)]

        # Breakdown by domain
        domains = set(q.domain for q in questions)
        by_domain = {}
        for d in domains:
            d_evals = [e for e in evals if e.metadata.get("domain") == d]
            if d_evals:
                by_domain[d] = {
                    "historical_accuracy": round(sum(e.historical_accuracy for e in d_evals) / len(d_evals), 4),
                    "token_f1": round(sum(e.token_f1 for e in d_evals) / len(d_evals), 4),
                    "temporal_precision": round(sum(e.temporal_precision_at_k for e in d_evals) / len(d_evals), 4),
                }

        # Breakdown by category
        categories = set(q.category for q in questions)
        by_category = {}
        for c in categories:
            c_evals = [e for e in evals if e.metadata.get("category") == c]
            if c_evals:
                by_category[c] = {
                    "historical_accuracy": round(sum(e.historical_accuracy for e in c_evals) / len(c_evals), 4),
                    "token_f1": round(sum(e.token_f1 for e in c_evals) / len(c_evals), 4),
                    "temporal_precision": round(sum(e.temporal_precision_at_k for e in c_evals) / len(c_evals), 4),
                }

        return BenchmarkSummary(
            system_name=system_name,
            total_questions=n,
            exact_match=round(sum(e.exact_match for e in evals) / n, 4),
            token_f1=round(sum(e.token_f1 for e in evals) / n, 4),
            historical_accuracy=round(sum(e.historical_accuracy for e in evals) / n, 4),
            temporal_precision_at_k=round(sum(e.temporal_precision_at_k for e in evals) / n, 4),
            temporal_coverage_at_k=round(sum(e.temporal_coverage_at_k for e in evals) / n, 4),
            wrong_era_retrieval_rate=round(sum(e.wrong_era_retrieval_rate for e in evals) / n, 4),
            abstention_accuracy=round(sum(e.abstention_accuracy for e in evals) / n, 4),
            mean_latency_ms=round(sum(latencies) / n, 2),
            p95_latency_ms=round(p95, 2),
            by_domain=by_domain,
            by_category=by_category,
        )

    @staticmethod
    def generate_markdown_report(report: BenchmarkReport) -> str:
        lines = [
            f"# ChronoBench Evaluation Report ({report.run_timestamp})",
            f"**Total Benchmark Questions Evaluated**: {report.num_questions}\n",
            "## 1. System Performance Comparison",
            "| System | Historical Acc | Token F1 | Temp Precision@K | Temp Coverage@K | Wrong-Era Rate | Abstention Acc | Mean Latency (ms) |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for s in report.systems:
            lines.append(
                f"| {s.system_name} | {s.historical_accuracy:.1%} | {s.token_f1:.3f} | "
                f"{s.temporal_precision_at_k:.1%} | {s.temporal_coverage_at_k:.1%} | "
                f"{s.wrong_era_retrieval_rate:.1%} | {s.abstention_accuracy:.1%} | {s.mean_latency_ms:.1f}ms |"
            )

        if report.ablations:
            lines.append("\n## 2. ChronoRAG Ablation Studies")
            lines.append("| Ablation Component Removed | Historical Acc | Token F1 | Temp Precision@K | Temp Coverage@K | Wrong-Era Rate |")
            lines.append("|---|---|---|---|---|---|")
            for a in report.ablations:
                lines.append(
                    f"| {a.system_name} | {a.historical_accuracy:.1%} | {a.token_f1:.3f} | "
                    f"{a.temporal_precision_at_k:.1%} | {a.temporal_coverage_at_k:.1%} | "
                    f"{a.wrong_era_retrieval_rate:.1%} |"
                )

        return "\n".join(lines)
