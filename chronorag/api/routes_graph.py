from fastapi import APIRouter, HTTPException
from chronorag.api.schemas import GraphExportResponse, LineageResponse
from chronorag.graph.visualizer import GraphVisualizer
from chronorag.graph.graph_store import TemporalGraphStore

router = APIRouter(prefix="/api/graph", tags=["Knowledge Graph"])


@router.get("/export", response_model=GraphExportResponse)
def export_graph():
    """Exports graph nodes and edges for visual rendering."""
    data = GraphVisualizer.get_graph_dict()
    return GraphExportResponse(nodes=data["nodes"], edges=data["edges"])


@router.get("/lineage/{claim_id}", response_model=LineageResponse)
def get_claim_lineage(claim_id: str):
    """Executes recursive CTE to trace historical supersession lineage for a claim."""
    store = TemporalGraphStore()
    chain = store.get_supersession_chain(start_claim_id=str(claim_id), direction="backward")
    return LineageResponse(claim_id=str(claim_id), lineage_path=chain)
