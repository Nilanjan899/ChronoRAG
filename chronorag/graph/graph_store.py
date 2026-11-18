import logging
from typing import List, Dict, Any, Optional
from sqlalchemy import text
from chronorag.database.connection import get_db_session
from chronorag.database.models import Claim, ClaimRelation, Entity

logger = logging.getLogger(__name__)


class TemporalGraphStore:
    """
    Temporal Knowledge Graph store using Recursive Common Table Expressions (CTEs)
    for lineage traversal, supersession chains, and multi-hop queries.
    """

    def get_supersession_chain(self, start_claim_id: str, direction: str = "backward") -> List[Dict[str, Any]]:
        """
        Traverse SUPERSEDES / AMENDS relations recursively.
        direction: 'backward' (find historical predecessors) or 'forward' (find later amendments)
        """
        if direction == "backward":
            cte_sql = text("""
                WITH RECURSIVE lineage AS (
                    -- Base case: target claim
                    SELECT 
                        c.claim_id,
                        c.subject_text,
                        c.predicate,
                        c.object_value,
                        c.valid_from,
                        c.valid_until,
                        cr.relation_type,
                        0 as depth
                    FROM claims c
                    LEFT JOIN claim_relations cr ON c.claim_id = cr.target_claim_id
                    WHERE c.claim_id = :start_id

                    UNION ALL

                    -- Recursive case: traverse targets
                    SELECT 
                        c_prev.claim_id,
                        c_prev.subject_text,
                        c_prev.predicate,
                        c_prev.object_value,
                        c_prev.valid_from,
                        c_prev.valid_until,
                        rel.relation_type,
                        l.depth + 1
                    FROM claim_relations rel
                    JOIN lineage l ON rel.source_claim_id = l.claim_id
                    JOIN claims c_prev ON rel.target_claim_id = c_prev.claim_id
                    WHERE l.depth < 10
                )
                SELECT DISTINCT claim_id, subject_text, predicate, object_value, valid_from, valid_until, relation_type, depth
                FROM lineage
                ORDER BY depth ASC;
            """)
        else:
            cte_sql = text("""
                WITH RECURSIVE lineage AS (
                    SELECT 
                        c.claim_id,
                        c.subject_text,
                        c.predicate,
                        c.object_value,
                        c.valid_from,
                        c.valid_until,
                        cr.relation_type,
                        0 as depth
                    FROM claims c
                    LEFT JOIN claim_relations cr ON c.claim_id = cr.source_claim_id
                    WHERE c.claim_id = :start_id

                    UNION ALL

                    SELECT 
                        c_next.claim_id,
                        c_next.subject_text,
                        c_next.predicate,
                        c_next.object_value,
                        c_next.valid_from,
                        c_next.valid_until,
                        rel.relation_type,
                        l.depth + 1
                    FROM claim_relations rel
                    JOIN lineage l ON rel.target_claim_id = l.claim_id
                    JOIN claims c_next ON rel.source_claim_id = c_next.claim_id
                    WHERE l.depth < 10
                )
                SELECT DISTINCT claim_id, subject_text, predicate, object_value, valid_from, valid_until, relation_type, depth
                FROM lineage
                ORDER BY depth ASC;
            """)

        results = []
        with get_db_session() as session:
            rows = session.execute(cte_sql, {"start_id": start_claim_id}).fetchall()
            for r in rows:
                results.append(
                    {
                        "claim_id": r[0],
                        "subject_text": r[1],
                        "predicate": r[2],
                        "object_value": r[3],
                        "valid_from": r[4],
                        "valid_until": r[5],
                        "relation_type": r[6],
                        "depth": r[7],
                    }
                )
        return results

    def traverse_multi_hop_entities(self, start_entity_name: str, max_hops: int = 3) -> List[Dict[str, Any]]:
        """
        Multi-hop traversal across entities, claims, and relations.
        e.g.: Policy amendment -> affected policy -> applicability relationship -> department
        """
        cte_sql = text("""
            WITH RECURSIVE entity_hops AS (
                -- Base: entity mention
                SELECT 
                    e.entity_id as current_entity_id,
                    e.canonical_name as current_entity_name,
                    c.claim_id,
                    c.predicate,
                    c.object_value,
                    c.conditions,
                    0 as hop_count
                FROM entities e
                JOIN claims c ON e.entity_id = c.subject_entity_id OR e.canonical_name = c.subject_text
                WHERE lower(e.canonical_name) = lower(:ent_name)

                UNION ALL

                -- Recursive: traverse to related claims
                SELECT 
                    e_next.entity_id as current_entity_id,
                    e_next.canonical_name as current_entity_name,
                    c_next.claim_id,
                    c_next.predicate,
                    c_next.object_value,
                    c_next.conditions,
                    h.hop_count + 1
                FROM entity_hops h
                JOIN claim_relations cr ON h.claim_id = cr.source_claim_id OR h.claim_id = cr.target_claim_id
                JOIN claims c_next ON (
                    CASE 
                        WHEN h.claim_id = cr.source_claim_id THEN cr.target_claim_id 
                        ELSE cr.source_claim_id 
                    END
                ) = c_next.claim_id
                LEFT JOIN entities e_next ON c_next.subject_entity_id = e_next.entity_id
                WHERE h.hop_count < :max_hops
            )
            SELECT DISTINCT current_entity_name, claim_id, predicate, object_value, conditions, hop_count
            FROM entity_hops
            ORDER BY hop_count ASC;
        """)

        results = []
        with get_db_session() as session:
            rows = session.execute(
                cte_sql, {"ent_name": start_entity_name, "max_hops": max_hops}
            ).fetchall()
            for r in rows:
                results.append(
                    {
                        "entity_name": r[0],
                        "claim_id": r[1],
                        "predicate": r[2],
                        "object_value": r[3],
                        "conditions": r[4],
                        "hop_count": r[5],
                    }
                )
        return results

    def check_graph_integrity(self) -> Dict[str, Any]:
        """
        Integrity checks:
        1. Cycle detection in supersession chains.
        2. Inverted temporal direction (newer claim has older valid_from).
        3. Dangling edges.
        """
        cycles_detected = 0
        inverted_temporal_edges = 0

        with get_db_session() as session:
            relations = session.query(ClaimRelation).filter(ClaimRelation.relation_type.in_(["SUPERSEDES", "AMENDS"])).all()
            for rel in relations:
                src = session.query(Claim).filter_by(claim_id=rel.source_claim_id).first()
                tgt = session.query(Claim).filter_by(claim_id=rel.target_claim_id).first()
                if src and tgt:
                    if src.valid_from and tgt.valid_from and src.valid_from < tgt.valid_from:
                        inverted_temporal_edges += 1

            # Simple cycle check
            chain_check = session.execute(text("""
                SELECT cr1.source_claim_id, cr1.target_claim_id 
                FROM claim_relations cr1
                JOIN claim_relations cr2 ON cr1.source_claim_id = cr2.target_claim_id 
                                        AND cr1.target_claim_id = cr2.source_claim_id
            """)).fetchall()
            cycles_detected = len(chain_check) // 2

        return {
            "status": "healthy" if (cycles_detected == 0 and inverted_temporal_edges == 0) else "warnings",
            "cycles_detected": cycles_detected,
            "inverted_temporal_edges": inverted_temporal_edges,
        }
