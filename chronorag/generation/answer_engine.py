import logging
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from chronorag.retrieval.engine import RetrievedEvidence
from chronorag.generation.timeline import HistoricalTimelineBuilder
from chronorag.generation.trace import RetrievalReasoningTrace
from chronorag.generation.provenance import CitationValidator

logger = logging.getLogger(__name__)


class ChronoAnswer(BaseModel):
    """The complete four-element grounded answer package."""
    answer: str = Field(..., description="Natural-language grounded answer with citations")
    supporting_evidence: List[Dict[str, Any]] = Field(default_factory=list, description="Supporting claims and source passages")
    timeline: List[Dict[str, Any]] = Field(default_factory=list, description="Chronological timeline of policy transitions")
    reasoning_trace: Dict[str, Any] = Field(default_factory=dict, description="Auditable retrieval reasoning trace")
    is_abstention: bool = Field(default=False, description="True if the system correctly refused to answer due to insufficient evidence")
    has_conflict: bool = Field(default=False, description="True if unresolved contradictory evidence was detected")
    citation_validation: Dict[str, Any] = Field(default_factory=dict, description="Audit of citation faithfulness")


class ChronoAnswerEngine:
    """Evidence-grounded temporal question-answering and explainability engine."""

    def __init__(self, use_llm: bool = False):
        self.use_llm = use_llm

    def generate_answer(self, evidence: RetrievedEvidence) -> ChronoAnswer:
        q = evidence.query
        timeline = HistoricalTimelineBuilder.build_timeline(evidence.claims, evidence.graph_lineage)
        reasoning_trace = RetrievalReasoningTrace.build_trace(evidence)

        # 1. Check for Abstention (No verified claims survived temporal filters)
        if not evidence.claims:
            target_time_str = (
                q.as_of_valid_time.strftime("%B %d, %Y")
                if q.as_of_valid_time
                else "the requested period"
            )
            abstention_msg = (
                f"Insufficient Evidence: ChronoRAG cannot verify any active or authoritative policy "
                f"for '{q.query_text}' on {target_time_str}. The knowledge base contains no verified claims "
                f"valid for this historical timeframe."
            )
            return ChronoAnswer(
                answer=abstention_msg,
                supporting_evidence=[],
                timeline=[],
                reasoning_trace=reasoning_trace,
                is_abstention=True,
                has_conflict=False,
                citation_validation={"is_valid": True, "fabricated": []},
            )

        # 2. Check for Contradictions
        if evidence.conflict_report.has_conflicts:
            conflict_descs = [
                f"- Disagreement: {c['explanation']}"
                for c in evidence.conflict_report.contradictions
            ]
            conflict_msg = (
                f"Conflict Detected in Evidence:\n"
                f"The knowledge repository contains mutually conflicting assertions valid during this period:\n"
                + "\n".join(conflict_descs)
                + "\n\nCannot provide a singular definitive answer without policy reconciliation."
            )
            return ChronoAnswer(
                answer=conflict_msg,
                supporting_evidence=self._format_evidence(evidence),
                timeline=timeline,
                reasoning_trace=reasoning_trace,
                is_abstention=False,
                has_conflict=True,
                citation_validation={"is_valid": True, "fabricated": []},
            )

        # 3. Formulate Natural-Language Answer based on query intent
        answer_text = self._synthesize_answer(evidence, timeline)

        # 4. Validate Citations
        is_cit_valid, valid_cits, fab_cits = CitationValidator.validate_citations(answer_text, evidence.sources)

        return ChronoAnswer(
            answer=answer_text,
            supporting_evidence=self._format_evidence(evidence),
            timeline=timeline,
            reasoning_trace=reasoning_trace,
            is_abstention=False,
            has_conflict=False,
            citation_validation={
                "is_valid": is_cit_valid,
                "valid_citations": valid_cits,
                "fabricated_citations": fab_cits,
            },
        )

    def _synthesize_answer(self, evidence: RetrievedEvidence, timeline: List[Dict[str, Any]]) -> str:
        q = evidence.query
        top_claim = evidence.claims[0]
        top_source = evidence.sources[0] if evidence.sources else {}
        doc_ref = f"[Doc: {top_source.get('filename', 'Policy Record')}, p.{top_source.get('page_number', 1)}]"

        # Format temporal interpretation
        if q.query_type == "snapshot" and q.as_of_valid_time:
            date_str = q.as_of_valid_time.strftime("%B %d, %Y")
            return (
                f"As of {date_str}, {top_claim.subject_text} had a {top_claim.predicate} of "
                f"**{top_claim.object_value}** {doc_ref}. "
                f"This policy was valid from {top_claim.valid_from.strftime('%Y-%m-%d') if top_claim.valid_from else 'initial record'} "
                f"to {top_claim.valid_until.strftime('%Y-%m-%d') if top_claim.valid_until else 'ongoing'}."
            )

        elif q.query_type == "comparison":
            d1_str = q.as_of_valid_time.strftime("%B %d, %Y") if q.as_of_valid_time else "earlier period"
            d2_str = q.as_of_valid_time_b.strftime("%B %d, %Y") if q.as_of_valid_time_b else "later period"
            # Compare timeline states
            if len(timeline) >= 2:
                c_prev = timeline[0]
                c_curr = timeline[-1]
                return (
                    f"Historical Comparison:\n"
                    f"- On {c_prev['valid_from']}: {c_prev['subject']} {c_prev['predicate']} was **{c_prev['value']}**.\n"
                    f"- On {c_curr['valid_from']}: The entitlement transitioned to **{c_curr['value']}** {doc_ref}."
                )
            else:
                return (
                    f"Between {d1_str} and {d2_str}, the recorded state for {top_claim.subject_text} was "
                    f"**{top_claim.object_value}** {doc_ref}."
                )

        elif q.query_type == "bitemporal":
            tx_str = q.as_of_transaction_time.strftime("%B %d, %Y") if q.as_of_transaction_time else "the query date"
            return (
                f"As of transaction time {tx_str}, the system knew that {top_claim.subject_text} had "
                f"a {top_claim.predicate} of **{top_claim.object_value}** {doc_ref}. "
                f"Any later amendments recorded after {tx_str} were not yet present in the system's knowledge base."
            )

        elif q.query_type == "change_detection":
            transitions = []
            for t in timeline:
                transitions.append(f"- Effective {t['valid_from']}: changed to **{t['value']}** ({t['status']})")
            return (
                f"Evolution of {top_claim.subject_text} {top_claim.predicate}:\n"
                + "\n".join(transitions)
                + f"\n\nSource: {doc_ref}."
            )

        elif q.query_type == "supersession":
            older = timeline[0] if timeline else None
            return (
                f"The current policy of **{top_claim.object_value}** became effective on "
                f"{top_claim.valid_from.strftime('%Y-%m-%d') if top_claim.valid_from else 'effective date'} {doc_ref}, "
                f"superseding the previous entitlement of **{older['value'] if older else 'earlier policy'}**."
            )

        # Default latest / fact query
        return (
            f"According to the latest policy in the knowledge base, {top_claim.subject_text} "
            f"receive **{top_claim.object_value}** for {top_claim.predicate} {doc_ref}."
        )

    def _format_evidence(self, evidence: RetrievedEvidence) -> List[Dict[str, Any]]:
        formatted = []
        for c in evidence.claims:
            matching_sources = [s for s in evidence.sources if s.get("claim_id") == c.claim_id]
            quote = matching_sources[0]["exact_quote"] if matching_sources else "Document passage"
            doc_name = matching_sources[0]["filename"] if matching_sources else "Document"
            page = matching_sources[0]["page_number"] if matching_sources else 1
            formatted.append(
                {
                    "claim_id": c.claim_id,
                    "subject": c.subject_text,
                    "predicate": c.predicate,
                    "value": c.object_value,
                    "valid_from": c.valid_from.strftime("%Y-%m-%d") if c.valid_from else "Beginning",
                    "valid_until": c.valid_until.strftime("%Y-%m-%d") if c.valid_until else "Ongoing",
                    "exact_quote": quote,
                    "document": doc_name,
                    "page": page,
                    "confidence": c.confidence,
                }
            )
        return formatted


AnswerGenerationEngine = ChronoAnswerEngine
GroundedAnswer = ChronoAnswer

