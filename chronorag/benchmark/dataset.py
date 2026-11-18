import random
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any, Optional, Tuple
from pydantic import BaseModel, Field


class GroundTruthFact(BaseModel):
    fact_id: str
    domain: str  # HR, Software_API, Regulation
    entity: str
    predicate: str
    object_value: str
    conditions: Dict[str, Any] = Field(default_factory=dict)
    valid_from: datetime
    valid_until: Optional[datetime] = None  # None = open-ended
    recorded_from: datetime
    recorded_until: Optional[datetime] = None
    source_doc_name: str
    source_passage: str


class BenchmarkQuestion(BaseModel):
    question_id: str
    domain: str
    category: str  # snapshot, comparison, supersession, bitemporal, conflict, multi_hop, unresolvable
    question_text: str
    expected_answer: str
    target_date: Optional[str] = None
    target_date_b: Optional[str] = None
    system_knowledge_date: Optional[str] = None
    expected_fact_ids: List[str] = Field(default_factory=list)
    should_abstain: bool = False
    is_conflicting: bool = False
    difficulty: str = "medium"


class ChronoBenchDatasetGenerator:
    """Generates ground-truth facts and corresponding benchmark questions across 3 domains."""

    def __init__(self, seed: int = 42):
        self.seed = seed
        random.seed(seed)

    def generate_benchmark_suite(self, num_questions: int = 200) -> Tuple[List[GroundTruthFact], List[BenchmarkQuestion]]:
        facts: List[GroundTruthFact] = []
        questions: List[BenchmarkQuestion] = []

        # Domain 1: Corporate & HR Policies (Leave, Travel, Health Insurance, Parental Leave, Remote Work)
        hr_facts, hr_qs = self._generate_hr_suite()
        facts.extend(hr_facts)
        questions.extend(hr_qs)

        # Domain 2: Software & API Documentation (Rate Limits, Deprecations, SDK Versions, Token Lifetimes)
        api_facts, api_qs = self._generate_api_suite()
        facts.extend(api_facts)
        questions.extend(api_qs)

        # Domain 3: Regulations and Amendments (Tax Compliance, Data Privacy, Minimum Wage, Environmental Standards)
        reg_facts, reg_qs = self._generate_regulation_suite()
        facts.extend(reg_facts)
        questions.extend(reg_qs)

        # Subsample or scale to num_questions
        if len(questions) > num_questions:
            questions = random.sample(questions, num_questions)

        return facts, questions

    def _generate_hr_suite(self):
        facts: List[GroundTruthFact] = []
        questions: List[BenchmarkQuestion] = []

        # Timeline for Annual Leave
        f1 = GroundTruthFact(
            fact_id="hr_leave_v1",
            domain="Corporate and HR policies",
            entity="Employees",
            predicate="annual leave entitlement",
            object_value="12 days",
            valid_from=datetime(2023, 1, 1, tzinfo=timezone.utc),
            valid_until=datetime(2024, 7, 1, tzinfo=timezone.utc),
            recorded_from=datetime(2023, 1, 1, tzinfo=timezone.utc),
            source_doc_name="Employee_Handbook_2023.txt",
            source_passage="Employees receive 12 annual leaves.",
        )
        f2 = GroundTruthFact(
            fact_id="hr_leave_v2",
            domain="Corporate and HR policies",
            entity="Employees",
            predicate="annual leave entitlement",
            object_value="15 days",
            valid_from=datetime(2024, 7, 1, tzinfo=timezone.utc),
            valid_until=datetime(2025, 1, 1, tzinfo=timezone.utc),
            recorded_from=datetime(2024, 6, 15, tzinfo=timezone.utc),
            source_doc_name="Policy_Amendment_July_2024.txt",
            source_passage="Effective July 1, 2024, the annual leave entitlement increases to 15 days.",
        )
        f3 = GroundTruthFact(
            fact_id="hr_leave_v3",
            domain="Corporate and HR policies",
            entity="Employees",
            predicate="annual leave entitlement",
            object_value="18 days",
            valid_from=datetime(2025, 1, 1, tzinfo=timezone.utc),
            valid_until=None,
            recorded_from=datetime(2025, 1, 15, tzinfo=timezone.utc),  # Uploaded on Jan 15!
            source_doc_name="Amendment_2025_Delayed.txt",
            source_passage="Document uploaded January 15, 2025, describing the amendment effective January 1, 2025: entitlement increases to 18 days.",
        )
        facts.extend([f1, f2, f3])

        # Snapshot questions
        questions.append(
            BenchmarkQuestion(
                question_id="hr_q1_snap_march",
                domain="Corporate and HR policies",
                category="snapshot",
                question_text="What was the annual leave entitlement on March 1, 2024?",
                expected_answer="12 days",
                target_date="2024-03-01",
                expected_fact_ids=["hr_leave_v1"],
            )
        )
        questions.append(
            BenchmarkQuestion(
                question_id="hr_q2_snap_sept",
                domain="Corporate and HR policies",
                category="snapshot",
                question_text="What was the annual leave entitlement on September 1, 2024?",
                expected_answer="15 days",
                target_date="2024-09-01",
                expected_fact_ids=["hr_leave_v2"],
            )
        )
        # Bitemporal question (System Knowledge on Jan 5, 2025)
        questions.append(
            BenchmarkQuestion(
                question_id="hr_q3_bitemporal_jan5",
                domain="Corporate and HR policies",
                category="bitemporal",
                question_text="What did the knowledge system know on January 5, 2025?",
                expected_answer="15 days",
                system_knowledge_date="2025-01-05",
                expected_fact_ids=["hr_leave_v2"],
            )
        )
        # Supersession question
        questions.append(
            BenchmarkQuestion(
                question_id="hr_q4_supersession",
                domain="Corporate and HR policies",
                category="supersession",
                question_text="When did the annual leave entitlement change from 12 days to 15 days?",
                expected_answer="July 1, 2024",
                expected_fact_ids=["hr_leave_v2"],
            )
        )
        # Comparison question
        questions.append(
            BenchmarkQuestion(
                question_id="hr_q5_comparison",
                domain="Corporate and HR policies",
                category="comparison",
                question_text="Compare leave entitlement between March 1, 2024 and September 1, 2024.",
                expected_answer="12 days vs 15 days",
                target_date="2024-03-01",
                target_date_b="2024-09-01",
                expected_fact_ids=["hr_leave_v1", "hr_leave_v2"],
            )
        )
        # Unresolvable / Abstention question
        questions.append(
            BenchmarkQuestion(
                question_id="hr_q6_abstention",
                domain="Corporate and HR policies",
                category="unresolvable",
                question_text="What was the leave entitlement on June 1, 2010?",
                expected_answer="Insufficient Evidence",
                should_abstain=True,
                expected_fact_ids=[],
            )
        )

        return facts, questions

    def _generate_api_suite(self):
        facts: List[GroundTruthFact] = []
        questions: List[BenchmarkQuestion] = []

        f1 = GroundTruthFact(
            fact_id="api_rate_v1",
            domain="Software and API documentation",
            entity="Payment API",
            predicate="rate limit",
            object_value="100 requests per minute",
            valid_from=datetime(2023, 6, 1, tzinfo=timezone.utc),
            valid_until=datetime(2024, 10, 1, tzinfo=timezone.utc),
            recorded_from=datetime(2023, 6, 1, tzinfo=timezone.utc),
            source_doc_name="Payment_API_v1.txt",
            source_passage="Payment API rate limit is 100 requests per minute for standard tiers.",
        )
        f2 = GroundTruthFact(
            fact_id="api_rate_v2",
            domain="Software and API documentation",
            entity="Payment API",
            predicate="rate limit",
            object_value="500 requests per minute",
            valid_from=datetime(2024, 10, 1, tzinfo=timezone.utc),
            valid_until=None,
            recorded_from=datetime(2024, 9, 20, tzinfo=timezone.utc),
            source_doc_name="Payment_API_v2_Migration.txt",
            source_passage="Effective October 1, 2024, Payment API rate limit increases to 500 requests per minute.",
        )
        facts.extend([f1, f2])

        questions.append(
            BenchmarkQuestion(
                question_id="api_q1_snap_aug",
                domain="Software and API documentation",
                category="snapshot",
                question_text="What was the Payment API rate limit on August 15, 2024?",
                expected_answer="100 requests per minute",
                target_date="2024-08-15",
                expected_fact_ids=["api_rate_v1"],
            )
        )
        questions.append(
            BenchmarkQuestion(
                question_id="api_q2_snap_nov",
                domain="Software and API documentation",
                category="snapshot",
                question_text="What was the Payment API rate limit on November 15, 2024?",
                expected_answer="500 requests per minute",
                target_date="2024-11-15",
                expected_fact_ids=["api_rate_v2"],
            )
        )
        return facts, questions

    def _generate_regulation_suite(self):
        facts: List[GroundTruthFact] = []
        questions: List[BenchmarkQuestion] = []

        f1 = GroundTruthFact(
            fact_id="reg_tax_v1",
            domain="Regulations and amendments",
            entity="Digital Services",
            predicate="corporate tax compliance rate",
            object_value="15%",
            valid_from=datetime(2022, 1, 1, tzinfo=timezone.utc),
            valid_until=datetime(2024, 4, 1, tzinfo=timezone.utc),
            recorded_from=datetime(2022, 1, 1, tzinfo=timezone.utc),
            source_doc_name="Digital_Tax_Statute_2022.txt",
            source_passage="Corporate tax compliance rate for Digital Services is 15%.",
        )
        f2 = GroundTruthFact(
            fact_id="reg_tax_v2",
            domain="Regulations and amendments",
            entity="Digital Services",
            predicate="corporate tax compliance rate",
            object_value="21%",
            valid_from=datetime(2024, 4, 1, tzinfo=timezone.utc),
            valid_until=None,
            recorded_from=datetime(2024, 3, 1, tzinfo=timezone.utc),
            source_doc_name="Statute_Amendment_Act_2024.txt",
            source_passage="Effective April 1, 2024, corporate tax compliance rate increases to 21%.",
        )
        facts.extend([f1, f2])

        questions.append(
            BenchmarkQuestion(
                question_id="reg_q1_snap_feb",
                domain="Regulations and amendments",
                category="snapshot",
                question_text="What was the corporate tax compliance rate for Digital Services on February 1, 2024?",
                expected_answer="15%",
                target_date="2024-02-01",
                expected_fact_ids=["reg_tax_v1"],
            )
        )
        questions.append(
            BenchmarkQuestion(
                question_id="reg_q2_snap_may",
                domain="Regulations and amendments",
                category="snapshot",
                question_text="What was the corporate tax compliance rate for Digital Services on May 1, 2024?",
                expected_answer="21%",
                target_date="2024-05-01",
                expected_fact_ids=["reg_tax_v2"],
            )
        )
        return facts, questions
