import pytest
from pathlib import Path
from chronorag.database.connection import init_db, get_db_session
from chronorag.database.models import Document, DocumentVersion, Chunk
from chronorag.ingestion.hasher import compute_sha256
from chronorag.ingestion.reader import DocumentReader
from chronorag.ingestion.chunker import StructureAwareChunker
from chronorag.ingestion.pipeline import IngestionPipeline


@pytest.fixture(scope="module", autouse=True)
def setup_db():
    init_db()


def test_sha256_hashing(tmp_path):
    f1 = tmp_path / "doc1.txt"
    f1.write_text("Hello ChronoRAG world", encoding="utf-8")
    hash1 = compute_sha256(f1)
    assert len(hash1) == 64

    # Identical content yields identical hash
    hash2 = compute_sha256("Hello ChronoRAG world")
    assert hash1 == hash2

    # Different content yields different hash
    hash3 = compute_sha256("Hello ChronoRAG world modified")
    assert hash1 != hash3


def test_markdown_and_txt_reading(tmp_path):
    md_file = tmp_path / "policy.md"
    md_file.write_text(
        "# Annual Leave Policy\n\n"
        "Employees receive 12 annual leaves.\n\n"
        "## Amendment 2024\n\n"
        "Effective July 1, 2024, leaves increase to 15 days.",
        encoding="utf-8",
    )
    blocks = DocumentReader.read_file(md_file)
    assert len(blocks) >= 4
    headings = [b.text for b in blocks if b.block_type == "heading"]
    assert "Annual Leave Policy" in headings
    assert "Amendment 2024" in headings


def test_structure_aware_chunking(tmp_path):
    chunker = StructureAwareChunker(chunk_size=100, chunk_overlap=30)
    md_file = tmp_path / "structure.md"
    md_file.write_text(
        "# Section 1: Overview\n"
        "This is paragraph one explaining the core policy terms in full detail.\n\n"
        "# Section 2: Amendments\n"
        "This is paragraph two explaining the new amendments introduced in 2024.",
        encoding="utf-8",
    )
    blocks = DocumentReader.read_file(md_file)
    chunks = chunker.chunk_blocks(blocks)
    assert len(chunks) >= 2
    for c in chunks:
        assert c.chunk_index >= 0
        assert c.page_number is not None


def test_ingestion_duplicate_and_versioning(tmp_path):
    import uuid
    u_id = uuid.uuid4().hex[:8]
    pipeline = IngestionPipeline()
    file_path = tmp_path / f"travel_policy_{u_id}.txt"
    file_path.write_text(f"Travel budget is $500 per day for senior managers {u_id}.", encoding="utf-8")

    # 1. Initial Ingestion
    doc, ver, is_new = pipeline.ingest_file(file_path)
    assert is_new is True
    assert ver.version_number == 1
    assert ver.parent_version_id is None

    # 2. Duplicate Ingestion Attempt (same content)
    doc_dup, ver_dup, is_new_dup = pipeline.ingest_file(file_path)
    assert is_new_dup is False
    assert doc_dup.document_id == doc.document_id
    assert ver_dup.version_id == ver.version_id

    # 3. New Version (same filename, modified content)
    file_path.write_text(f"Effective June 2025, travel budget is $650 per day for senior managers {u_id}.", encoding="utf-8")
    doc_v2, ver_v2, is_new_v2 = pipeline.ingest_file(file_path)
    assert is_new_v2 is True
    assert doc_v2.document_id == doc.document_id
    assert ver_v2.version_number == 2
    assert ver_v2.parent_version_id == ver.version_id
    assert ver_v2.version_id != ver.version_id

    # Verify both versions exist in database
    with get_db_session() as session:
        all_versions = session.query(DocumentVersion).filter_by(document_id=doc.document_id).all()
        assert len(all_versions) == 2
