import pytest
from datetime import datetime, timezone
from chronorag.database.connection import init_db, get_db_session
from chronorag.database.models import Claim, ClaimRelation, Entity
from chronorag.graph.entity_resolver import EntityResolver
from chronorag.graph.relation_detector import TemporalRelationDetector
from chronorag.graph.graph_store import TemporalGraphStore


@pytest.fixture(scope="module", autouse=True)
def setup_db():
    init_db()


def test_entity_resolution():
    resolver = EntityResolver(similarity_threshold=0.80)
    import uuid
    suffix = uuid.uuid4().hex[:12]
    name1 = f"{suffix}_Alpha_Safety_Policy"
    name2 = f"{suffix}_Alpha_Safety_Policy_Document"
    name3 = f"{suffix}_Zeta_Network_Encryption"

    # 1. Create first entity
    ent1, is_new1, score1 = resolver.resolve_entity(name1, entity_type="Policy")
    assert is_new1 is True

    # 2. Similar alias should resolve to ent1
    ent2, is_new2, score2 = resolver.resolve_entity(name2, entity_type="Policy")
    assert is_new2 is False
    assert ent2.entity_id == ent1.entity_id

    # 3. Completely different entity should NOT merge
    ent3, is_new3, score3 = resolver.resolve_entity(name3, entity_type="Policy")
    assert is_new3 is True
    assert ent3.entity_id != ent1.entity_id


def test_relation_detector_supersedes_and_supports():
    detector = TemporalRelationDetector()

    with get_db_session() as session:
        # Older claim: 12 leaves valid from 2023-01-01 to 2024-07-01
        c_old = Claim(
            subject_text="Employees",
            predicate="annual leave entitlement",
            object_value="12 days",
            valid_from=datetime(2023, 1, 1, tzinfo=timezone.utc),
            valid_until=datetime(2024, 7, 1, tzinfo=timezone.utc),
        )
        # Newer claim: 15 leaves valid from 2024-07-01 onwards
        c_new = Claim(
            subject_text="Employees",
            predicate="annual leave entitlement",
            object_value="15 days",
            valid_from=datetime(2024, 7, 1, tzinfo=timezone.utc),
            valid_until=None,
        )
        # Supporting claim: confirms 15 leaves
        c_sup = Claim(
            subject_text="Employees",
            predicate="annual leave entitlement",
            object_value="15 days",
            valid_from=datetime(2024, 7, 1, tzinfo=timezone.utc),
            valid_until=None,
        )
        session.add_all([c_old, c_new, c_sup])
        session.flush()

        rel_sup = detector._evaluate_pair(c_new, c_old)
        assert rel_sup is not None
        assert rel_sup.relation_type == "SUPERSEDES"

        rel_agree = detector._evaluate_pair(c_sup, c_new)
        assert rel_agree is not None
        assert rel_agree.relation_type == "SUPPORTS"


def test_recursive_cte_graph_traversal():
    store = TemporalGraphStore()
    detector = TemporalRelationDetector()

    with get_db_session() as session:
        # Build 3-step supersession chain: v1 (12d) -> v2 (15d) -> v3 (18d)
        c1 = Claim(
            subject_text="Staff",
            predicate="vacation days",
            object_value="12 days",
            valid_from=datetime(2022, 1, 1, tzinfo=timezone.utc),
            valid_until=datetime(2023, 1, 1, tzinfo=timezone.utc),
        )
        c2 = Claim(
            subject_text="Staff",
            predicate="vacation days",
            object_value="15 days",
            valid_from=datetime(2023, 1, 1, tzinfo=timezone.utc),
            valid_until=datetime(2024, 1, 1, tzinfo=timezone.utc),
        )
        c3 = Claim(
            subject_text="Staff",
            predicate="vacation days",
            object_value="18 days",
            valid_from=datetime(2024, 1, 1, tzinfo=timezone.utc),
            valid_until=None,
        )
        session.add_all([c1, c2, c3])
        session.flush()

        # Link relations
        r1 = ClaimRelation(source_claim_id=c2.claim_id, target_claim_id=c1.claim_id, relation_type="SUPERSEDES")
        r2 = ClaimRelation(source_claim_id=c3.claim_id, target_claim_id=c2.claim_id, relation_type="SUPERSEDES")
        session.add_all([r1, r2])
        session.commit()

        # Backward lineage from c3
        chain = store.get_supersession_chain(c3.claim_id, direction="backward")
        assert len(chain) >= 3
        claim_ids_in_chain = [item["claim_id"] for item in chain]
        assert c3.claim_id in claim_ids_in_chain
        assert c2.claim_id in claim_ids_in_chain
        assert c1.claim_id in claim_ids_in_chain


def test_graph_integrity_check():
    store = TemporalGraphStore()
    integrity = store.check_graph_integrity()
    assert "status" in integrity
    assert "cycles_detected" in integrity
    assert integrity["cycles_detected"] == 0
