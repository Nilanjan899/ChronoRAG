import json
from typing import Dict, Any, Optional
from pathlib import Path
from chronorag.database.connection import get_db_session
from chronorag.database.models import Claim, ClaimRelation, Entity


class GraphVisualizer:
    """Generates visual graph representations for Streamlit and PyVis."""

    @staticmethod
    def generate_pyvis_html(output_path: Optional[Path] = None) -> str:
        from pyvis.network import Network

        net = Network(height="600px", width="100%", directed=True, bgcolor="#222222", font_color="white")
        net.force_atlas_2based()

        with get_db_session() as session:
            claims = session.query(Claim).limit(100).all()
            relations = session.query(ClaimRelation).limit(200).all()

            for c in claims:
                valid_str = f"Valid: {c.valid_from.date() if c.valid_from else 'past'} to {c.valid_until.date() if c.valid_until else 'open'}"
                label = f"{c.subject_text}\n{c.predicate}: {c.object_value}"
                title = f"Claim ID: {c.claim_id}\n{valid_str}\nStatus: {c.verification_status}"
                color = "#4CAF50" if not c.valid_until else "#2196F3"
                net.add_node(c.claim_id, label=label, title=title, color=color, shape="box")

            for r in relations:
                color_map = {
                    "SUPERSEDES": "#F44336",
                    "AMENDS": "#FF9800",
                    "CONTRADICTS": "#E91E63",
                    "SUPPORTS": "#8BC34A",
                    "REFINES": "#9C27B0",
                }
                net.add_edge(
                    r.source_claim_id,
                    r.target_claim_id,
                    title=r.relation_type,
                    label=r.relation_type,
                    color=color_map.get(r.relation_type, "#9E9E9E"),
                )

        if output_path:
            net.save_graph(str(output_path))
            return output_path.read_text(encoding="utf-8")
        return net.generate_html()

    @staticmethod
    def get_graph_dict() -> Dict[str, Any]:
        """Returns JSON-serializable graph data for Plotly / cytoscape."""
        nodes = []
        edges = []
        with get_db_session() as session:
            claims = session.query(Claim).limit(100).all()
            relations = session.query(ClaimRelation).limit(200).all()

            for c in claims:
                nodes.append(
                    {
                        "id": c.claim_id,
                        "label": f"{c.subject_text}: {c.predicate}",
                        "value": c.object_value,
                        "valid_from": str(c.valid_from) if c.valid_from else None,
                        "valid_until": str(c.valid_until) if c.valid_until else None,
                    }
                )
            for r in relations:
                edges.append(
                    {
                        "source": r.source_claim_id,
                        "target": r.target_claim_id,
                        "relation": r.relation_type,
                    }
                )
        return {"nodes": nodes, "edges": edges}
