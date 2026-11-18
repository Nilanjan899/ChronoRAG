import pytest
from datetime import datetime, timezone

from chronorag.benchmark.dataset import ChronoBenchDatasetGenerator, BenchmarkQuestion
from chronorag.benchmark.metrics import BenchmarkMetrics
from chronorag.benchmark.baselines import (
    VanillaDenseRAG,
    HybridRAG,
    TemporalFilteredRAG,
    FullChronoRAG,
)
from chronorag.benchmark.runner import ChronoBenchRunner
from chronorag.retrieval.dense import DenseRetriever
from chronorag.retrieval.keyword import KeywordRetriever
from chronorag.retrieval.engine import HybridRetrievalEngine
from chronorag.generation.answer_engine import AnswerGenerationEngine
from chronorag.database.connection import get_engine, init_db


def test_dataset_generation():
    gen = ChronoBenchDatasetGenerator(seed=42)
    facts, questions = gen.generate_benchmark_suite(num_questions=30)
    assert len(facts) >= 5
    assert len(questions) > 0
    domains = set(q.domain for q in questions)
    assert len(domains) >= 2


def test_metrics_calculation():
    em = BenchmarkMetrics.compute_exact_match("The leave is 15 days.", "15 days")
    assert em == 1.0

    f1 = BenchmarkMetrics.compute_token_f1("The leave is 15 days.", "15 days")
    assert f1 > 0.5

    hist_acc = BenchmarkMetrics.compute_historical_accuracy(
        "Effective July 1, 2024, leave is 15 days.", "15 days"
    )
    assert hist_acc == 1.0

    abstain_acc = BenchmarkMetrics.compute_abstention_accuracy(
        "There is insufficient information for this period.", should_abstain=True
    )
    assert abstain_acc == 1.0

    wrong_abstain = BenchmarkMetrics.compute_abstention_accuracy(
        "There is insufficient information for this period.", should_abstain=False
    )
    assert wrong_abstain == 0.0

    target_dt = datetime(2024, 8, 1, tzinfo=timezone.utc)
    intervals = [
        {
            "valid_from": datetime(2024, 7, 1, tzinfo=timezone.utc),
            "valid_until": datetime(2025, 1, 1, tzinfo=timezone.utc),
        },
        {
            "valid_from": datetime(2023, 1, 1, tzinfo=timezone.utc),
            "valid_until": datetime(2024, 7, 1, tzinfo=timezone.utc),
        },
    ]
    prec = BenchmarkMetrics.compute_temporal_precision_at_k(intervals, target_dt)
    assert prec == 0.5
    wrong_era = BenchmarkMetrics.compute_wrong_era_retrieval_rate(intervals, target_dt)
    assert wrong_era == 0.5


def test_baselines_execution():
    init_db()
    dense_retriever = DenseRetriever()
    keyword_retriever = KeywordRetriever()

    chunks = [
        {
            "chunk_id": "c1",
            "chunk_text": "Employees receive 12 annual leaves.",
            "fact_id": "hr_leave_v1",
            "valid_from": datetime(2023, 1, 1, tzinfo=timezone.utc),
            "valid_until": datetime(2024, 7, 1, tzinfo=timezone.utc),
        },
        {
            "chunk_id": "c2",
            "chunk_text": "Effective July 1, 2024, the annual leave entitlement increases to 15 days.",
            "fact_id": "hr_leave_v2",
            "valid_from": datetime(2024, 7, 1, tzinfo=timezone.utc),
            "valid_until": datetime(2025, 1, 1, tzinfo=timezone.utc),
        },
    ]

    sys_a = VanillaDenseRAG(dense_retriever, chunks, top_k=2)
    sys_c = TemporalFilteredRAG(dense_retriever, chunks, top_k=2)

    q = BenchmarkQuestion(
        question_id="test_q1",
        domain="Corporate and HR policies",
        category="snapshot",
        question_text="What was the annual leave entitlement on March 1, 2024?",
        expected_answer="12 days",
        target_date="2024-03-01",
        expected_fact_ids=["hr_leave_v1"],
    )

    res_a = sys_a.query(q)
    assert res_a.answer is not None
    assert len(res_a.retrieved_items) > 0

    res_c = sys_c.query(q)
    assert res_c.answer is not None

    runner = ChronoBenchRunner(questions=[q])
    summary_a = runner.evaluate_system(sys_a)
    assert summary_a.total_questions == 1
