import pytest
from chronorag.api.server import health_check
from chronorag.api.schemas import IngestTextRequest, QueryRequest
from chronorag.api.routes_ingest import ingest_text, list_documents, list_claims
from chronorag.api.routes_query import query_chronorag
from chronorag.api.routes_graph import export_graph, get_claim_lineage
from chronorag.database.connection import init_db


@pytest.fixture(autouse=True)
def setup_db():
    init_db()


def test_health_endpoint():
    res = health_check()
    assert res["status"] == "healthy"
    assert res["service"] == "ChronoRAG"
    assert "database_type" in res


def test_api_ingest_and_query_flow():
    # 1. Ingest document via route function
    req = IngestTextRequest(
        filename="Test_HR_Policy_2024.txt",
        text_content="Effective July 1, 2024, the annual leave entitlement increases to 15 days.",
        publication_date="2024-06-15",
        metadata={"department": "HR"},
    )
    ingest_res = ingest_text(req)
    assert ingest_res.document_id is not None
    assert ingest_res.version_number >= 1

    # 2. Query bitemporal route
    q_req = QueryRequest(
        query="What is the annual leave entitlement?",
        as_of_valid_time="2024-08-01",
    )
    q_res = query_chronorag(q_req)
    assert q_res.answer is not None
    assert "15 days" in q_res.answer
    assert isinstance(q_res.timeline, list)
    assert isinstance(q_res.reasoning_trace, dict)


def test_api_graph_endpoints():
    graph_data = export_graph()
    assert len(graph_data.nodes) >= 1
    assert isinstance(graph_data.edges, list)

    claims = list_claims(limit=5)
    assert len(claims) >= 1
    target_cid = claims[0].claim_id

    lineage = get_claim_lineage(target_cid)
    assert lineage.claim_id == target_cid
    assert isinstance(lineage.lineage_path, list)
