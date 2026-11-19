import logging
from fastapi import APIRouter, HTTPException
from chronorag.api.schemas import QueryRequest, QueryResponse
from chronorag.retrieval.engine import ChronoRetrievalEngine
from chronorag.generation.answer_engine import ChronoAnswerEngine

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/query", tags=["Query"])

_retrieval_engine = None
_answer_engine = None


def get_engines():
    global _retrieval_engine, _answer_engine
    if _retrieval_engine is None:
        _retrieval_engine = ChronoRetrievalEngine()
    if _answer_engine is None:
        _answer_engine = ChronoAnswerEngine()
    return _retrieval_engine, _answer_engine


@router.post("", response_model=QueryResponse)
def query_chronorag(payload: QueryRequest):
    """
    Core bitemporal question answering endpoint.
    Returns:
    1. Grounded natural language answer with inline citations.
    2. Supporting evidence claims with verified quotes.
    3. Chronological timeline of policy state transitions.
    4. Auditable operational retrieval reasoning trace.
    """
    retrieval_engine, answer_engine = get_engines()
    try:
        evidence = retrieval_engine.retrieve(
            query_str=payload.query,
            as_of_valid_time=payload.as_of_valid_time,
            as_of_transaction_time=payload.as_of_transaction_time,
        )

        chrono_answer = answer_engine.generate_answer(evidence)

        return QueryResponse(
            query=payload.query,
            answer=chrono_answer.answer,
            supporting_evidence=chrono_answer.supporting_evidence,
            timeline=chrono_answer.timeline,
            reasoning_trace=chrono_answer.reasoning_trace,
            is_abstention=chrono_answer.is_abstention,
            has_conflict=chrono_answer.has_conflict,
            citation_validation=chrono_answer.citation_validation,
        )
    except Exception as e:
        logger.error(f"Error answering query '{payload.query}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
