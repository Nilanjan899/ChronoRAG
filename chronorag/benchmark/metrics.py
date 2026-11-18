import re
import string
from typing import List, Dict, Any, Optional
from datetime import datetime
from pydantic import BaseModel, Field


class EvaluationResult(BaseModel):
    question_id: str
    system_name: str
    predicted_answer: str
    ground_truth_answer: str
    exact_match: float
    token_f1: float
    historical_accuracy: float
    temporal_precision_at_k: float
    temporal_coverage_at_k: float
    wrong_era_retrieval_rate: float
    abstention_accuracy: float
    latency_ms: float
    retrieved_fact_ids: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class BenchmarkMetrics:
    """Calculates evaluation metrics for temporal RAG systems."""

    @staticmethod
    def normalize_text(text: str) -> str:
        """Lowercases text, removes punctuation, articles and extra whitespace."""
        text = text.lower()
        # Remove punctuation
        text = text.translate(str.maketrans("", "", string.punctuation))
        # Remove common articles
        text = re.sub(r"\b(a|an|the)\b", " ", text)
        # Collapse whitespace
        text = " ".join(text.split())
        return text

    @classmethod
    def compute_exact_match(cls, prediction: str, ground_truth: str) -> float:
        norm_pred = cls.normalize_text(prediction)
        norm_gt = cls.normalize_text(ground_truth)
        if not norm_gt:
            return 1.0 if not norm_pred else 0.0
        return 1.0 if norm_gt in norm_pred or norm_pred == norm_gt else 0.0

    @classmethod
    def compute_token_f1(cls, prediction: str, ground_truth: str) -> float:
        norm_pred = cls.normalize_text(prediction)
        norm_gt = cls.normalize_text(ground_truth)
        
        pred_tokens = norm_pred.split()
        gt_tokens = norm_gt.split()
        
        if not pred_tokens or not gt_tokens:
            return 1.0 if pred_tokens == gt_tokens else 0.0
        
        common = set(pred_tokens) & set(gt_tokens)
        num_same = sum(min(pred_tokens.count(tok), gt_tokens.count(tok)) for tok in common)
        
        if num_same == 0:
            return 0.0
        
        precision = 1.0 * num_same / len(pred_tokens)
        recall = 1.0 * num_same / len(gt_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        return round(f1, 4)

    @classmethod
    def compute_historical_accuracy(
        cls, 
        prediction: str, 
        ground_truth: str, 
        should_abstain: bool = False
    ) -> float:
        norm_pred = cls.normalize_text(prediction)
        
        # Check abstention
        abstain_keywords = [
            "insufficient", "unknown", "no information", "cannot be determined",
            "abstain", "not available", "unrecorded", "not recorded"
        ]
        pred_abstained = any(kw in norm_pred for kw in abstain_keywords)
        
        if should_abstain:
            return 1.0 if pred_abstained else 0.0
        
        if pred_abstained:
            return 0.0
        
        # Ground truth matching
        norm_gt = cls.normalize_text(ground_truth)
        if norm_gt in norm_pred:
            return 1.0
        
        # Check key numbers or terms
        gt_numbers = re.findall(r"\d+", norm_gt)
        if gt_numbers:
            pred_numbers = re.findall(r"\d+", norm_pred)
            if all(n in pred_numbers for n in gt_numbers):
                return 1.0

        return cls.compute_token_f1(prediction, ground_truth) >= 0.5 and 1.0 or 0.0

    @classmethod
    def compute_temporal_precision_at_k(
        cls,
        retrieved_intervals: List[Dict[str, Optional[datetime]]],
        target_date: Optional[datetime],
    ) -> float:
        """Fraction of retrieved items whose validity interval covers target_date."""
        if not retrieved_intervals or not target_date:
            return 1.0
        
        valid_count = 0
        for item in retrieved_intervals:
            v_from = item.get("valid_from")
            v_until = item.get("valid_until")
            
            # Check half-open [v_from, v_until)
            is_valid = True
            if v_from and target_date < v_from:
                is_valid = False
            if v_until and target_date >= v_until:
                is_valid = False
            
            if is_valid:
                valid_count += 1
                
        return round(valid_count / len(retrieved_intervals), 4)

    @classmethod
    def compute_temporal_coverage_at_k(
        cls,
        retrieved_fact_ids: List[str],
        expected_fact_ids: List[str],
    ) -> float:
        """Whether expected fact IDs are found in retrieved items."""
        if not expected_fact_ids:
            return 1.0
        if not retrieved_fact_ids:
            return 0.0
        
        retrieved_set = set(retrieved_fact_ids)
        expected_set = set(expected_fact_ids)
        
        covered = expected_set & retrieved_set
        return round(len(covered) / len(expected_set), 4)

    @classmethod
    def compute_wrong_era_retrieval_rate(
        cls,
        retrieved_intervals: List[Dict[str, Optional[datetime]]],
        target_date: Optional[datetime],
    ) -> float:
        """Fraction of retrieved items that clearly belong to another era (expired or future)."""
        if not retrieved_intervals or not target_date:
            return 0.0
        
        prec = cls.compute_temporal_precision_at_k(retrieved_intervals, target_date)
        return round(1.0 - prec, 4)

    @classmethod
    def compute_abstention_accuracy(
        cls,
        prediction: str,
        should_abstain: bool
    ) -> float:
        norm_pred = cls.normalize_text(prediction)
        abstain_keywords = [
            "insufficient", "unknown", "no information", "cannot be determined",
            "abstain", "not available", "unrecorded", "not recorded"
        ]
        abstained = any(kw in norm_pred for kw in abstain_keywords)
        
        if should_abstain:
            return 1.0 if abstained else 0.0
        else:
            return 0.0 if abstained else 1.0
