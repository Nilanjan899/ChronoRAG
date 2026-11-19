import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

# Set up path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chronorag.database.connection import init_db, get_db_session
from chronorag.database.models import Document, DocumentVersion, Chunk, Claim, ClaimRelation
from chronorag.ingestion.pipeline import IngestionPipeline
from chronorag.extraction.pipeline import ClaimExtractionPipeline
from chronorag.graph.relation_detector import RelationDetector


def seed_knowledge_base():
    print("Initializing database...")
    init_db()

    docs_data = [
        {
            "filename": "Employee_Handbook_2023.txt",
            "publication_date": "2023-01-01",
            "content": (
                "Section 4.1: Annual Leave Policy.\n"
                "Employees receive 12 annual leaves.\n"
                "Leave must be requested at least two weeks in advance."
            ),
        },
        {
            "filename": "Leave_Policy_Amendment_2024.txt",
            "publication_date": "2024-06-15",
            "content": (
                "Executive Amendment 2024-03.\n"
                "Effective July 1, 2024, the annual leave entitlement increases to 15 days for all full-time employees.\n"
                "This policy supersedes the 2023 handbook entitlement."
            ),
        },
        {
            "filename": "Leave_Policy_Amendment_2025.txt",
            "publication_date": "2025-01-15",
            "content": (
                "Executive Amendment 2025-01.\n"
                "Document uploaded to the knowledge system on January 15, 2025.\n"
                "Effective January 1, 2025, the entitlement increases to 18 days for all employees."
            ),
        },
        {
            "filename": "Payment_API_Spec_v1.txt",
            "publication_date": "2023-06-01",
            "content": (
                "API Documentation v1.2.\n"
                "Payment API rate limit is 100 requests per minute for standard tiers.\n"
                "Authentication token lifetime is 24 hours."
            ),
        },
        {
            "filename": "Payment_API_Spec_v2.txt",
            "publication_date": "2024-09-20",
            "content": (
                "API Documentation v2.0 Migration Guide.\n"
                "Effective October 1, 2024, Payment API rate limit increases to 500 requests per minute.\n"
                "Authentication token lifetime is reduced to 1 hour for enhanced security."
            ),
        },
        {
            "filename": "Corporate_Tax_Statute_2022.txt",
            "publication_date": "2022-01-01",
            "content": (
                "Digital Services Tax Act 2022.\n"
                "Corporate tax compliance rate for Digital Services is 15%."
            ),
        },
        {
            "filename": "Corporate_Tax_Amendment_2024.txt",
            "publication_date": "2024-03-01",
            "content": (
                "Statute Amendment Act 2024.\n"
                "Effective April 1, 2024, corporate tax compliance rate increases to 21% for Digital Services."
            ),
        },
    ]

    ingest_pipe = IngestionPipeline()
    extract_pipe = ClaimExtractionPipeline()

    for item in docs_data:
        print(f"Processing '{item['filename']}'...")
        with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{item['filename']}", mode="w", encoding="utf-8") as tmp:
            tmp.write(item["content"])
            tmp_path = tmp.name

        try:
            doc, doc_version, is_new = ingest_pipe.ingest_file(
                file_path=tmp_path,
                publication_date=item["publication_date"],
            )
            print(f"  -> Ingested version {doc_version.version_number} (ID: {doc_version.version_id[:8]}...)")

            if is_new:
                claims = extract_pipe.extract_from_document_version(
                    document_version_id=doc_version.version_id,
                )
                print(f"  -> Extracted {len(claims)} atomic claims.")
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    # Detect relations across all claims
    print("Computing Knowledge Graph relations (SUPERSEDES, AMENDS, CONTRADICTS, SUPPORTS)...")
    detector = RelationDetector()
    rel_count = detector.detect_all_relations()
    print(f"Detected and linked {rel_count} graph relations.")

    # Print summary
    with get_db_session() as s:
        print("\n--- Seeding Complete ---")
        print(f"Total Documents: {s.query(Document).count()}")
        print(f"Total Versions: {s.query(DocumentVersion).count()}")
        print(f"Total Chunks: {s.query(Chunk).count()}")
        print(f"Total Claims: {s.query(Claim).count()}")
        print(f"Total Relations: {s.query(ClaimRelation).count()}")


if __name__ == "__main__":
    seed_knowledge_base()
