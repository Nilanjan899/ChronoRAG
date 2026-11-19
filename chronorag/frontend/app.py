import os
import sys
import json
from datetime import datetime, timezone
from pathlib import Path
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit.components.v1 as components

# Ensure chronorag is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from chronorag.config import settings
from chronorag.database.connection import init_db, get_db_session
from chronorag.database.models import Document, DocumentVersion, Claim, ClaimRelation
from chronorag.retrieval.engine import ChronoRetrievalEngine
from chronorag.generation.answer_engine import ChronoAnswerEngine
from chronorag.ingestion.pipeline import IngestionPipeline
from chronorag.extraction.pipeline import ClaimExtractionPipeline
from chronorag.graph.relation_detector import RelationDetector
from chronorag.graph.visualizer import GraphVisualizer
from chronorag.benchmark.dataset import ChronoBenchDatasetGenerator
from chronorag.benchmark.baselines import (
    VanillaDenseRAG,
    HybridRAG,
    TemporalFilteredRAG,
    FullChronoRAG,
)
from chronorag.benchmark.ablations import AblatedChronoRAG, get_standard_ablations
from chronorag.benchmark.runner import ChronoBenchRunner, BenchmarkReport
from chronorag.retrieval.dense import DenseRetriever
from chronorag.retrieval.keyword import KeywordRetriever


# Page configuration
st.set_page_config(
    page_title="ChronoRAG — Bitemporal Knowledge Retrieval & Historical Reasoning",
    page_icon="⏱️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize database schema on load
init_db()


@st.cache_resource
def get_engines():
    retrieval = ChronoRetrievalEngine()
    answer = ChronoAnswerEngine()
    return retrieval, answer


retrieval_engine, answer_engine = get_engines()

# Custom CSS for clean, professional look
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E88E5;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #616161;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 1rem;
        border-left: 4px solid #1E88E5;
        margin-bottom: 0.8rem;
    }
    .timeline-badge {
        display: inline-block;
        padding: 0.25em 0.6em;
        font-size: 0.85em;
        font-weight: 700;
        line-height: 1;
        text-align: center;
        white-space: nowrap;
        vertical-align: baseline;
        border-radius: 0.25rem;
        color: #fff;
        background-color: #0288D1;
    }
    .trace-box {
        font-family: 'Courier New', Courier, monospace;
        font-size: 0.85rem;
        background-color: #1e1e1e;
        color: #d4d4d4;
        padding: 1rem;
        border-radius: 6px;
        overflow-x: auto;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/clock--v1.png", width=64)
    st.markdown("### **ChronoRAG Engine**")
    st.markdown(
        "**Bitemporal Knowledge Retrieval & Evidence Reconstruction**\n\n"
        "• Valid Time: Real-world truth\n"
        "• Transaction Time: System knowledge\n"
        "• Zero-cost & 100% local operation"
    )
    st.divider()

    st.markdown("#### **System Diagnostics**")
    st.caption(f"**Database:** {settings.database_type.upper()}")
    st.caption(f"**Embedding Model:** {Path(settings.embedding_model_name).name}")
    st.caption(f"**Inference:** 100% Local (CPU/GPU)")

    with get_db_session() as s:
        doc_count = s.query(Document).count()
        ver_count = s.query(DocumentVersion).count()
        claim_count = s.query(Claim).count()
        rel_count = s.query(ClaimRelation).count()

    st.markdown("#### **Knowledge Base Stats**")
    c1, c2 = st.columns(2)
    c1.metric("Documents", doc_count)
    c2.metric("Versions", ver_count)
    c1.metric("Claims", claim_count)
    c2.metric("Relations", rel_count)


# Header
st.markdown('<div class="main-title">⏱️ ChronoRAG</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Bitemporal Knowledge Retrieval, Historical Reasoning & Evidence Reconstruction Engine</div>',
    unsafe_allow_html=True,
)

# Main Navigation Tabs
tab_chat, tab_explorer, tab_bench = st.tabs(
    [
        "💬 Section A: Temporal Knowledge Chat",
        "🧭 Section B: Knowledge Explorer & Graph",
        "📊 Section C: ChronoBench Evaluation Dashboard",
    ]
)

# ==============================================================================
# TAB 1: TEMPORAL KNOWLEDGE CHAT
# ==============================================================================
with tab_chat:
    st.markdown("### Temporal Querying with Bitemporal Filters")
    st.caption("Ask questions across valid time (what was true) and transaction time (what the system knew).")

    col_q, col_vt, col_tt = st.columns([3, 1.2, 1.2])

    with col_q:
        user_query = st.text_input(
            "Natural Language Query",
            value="What was the annual leave entitlement for employees on March 1, 2024?",
            placeholder="e.g., What was the entitlement on September 1, 2024?",
        )

    with col_vt:
        vt_input = st.text_input(
            "As-of Valid Time (Optional)",
            value="",
            placeholder="YYYY-MM-DD (e.g. 2024-03-01)",
            help="Specify when the fact had to be valid in the real world. Leave blank to let query parser detect.",
        )

    with col_tt:
        tt_input = st.text_input(
            "As-of Transaction Time (Optional)",
            value="",
            placeholder="YYYY-MM-DD (e.g. 2025-01-05)",
            help="Specify what the system recorded at that point in time (retroactive knowledge).",
        )

    btn_search = st.button("🔍 Execute Bitemporal Retrieval & Reasoning", type="primary", use_container_width=True)

    # Preset query chips
    st.markdown("**Sample Temporal Queries:**")
    preset_cols = st.columns(4)
    if preset_cols[0].button("📅 March 1, 2024 (Before Amendment)"):
        st.session_state["query_text"] = "What was the employee annual leave entitlement on March 1, 2024?"
        st.rerun()
    if preset_cols[1].button("📅 Sept 1, 2024 (After Amendment)"):
        st.session_state["query_text"] = "What was the employee annual leave entitlement on September 1, 2024?"
        st.rerun()
    if preset_cols[2].button("🔄 Policy Evolution Timeline"):
        st.session_state["query_text"] = "How did the employee annual leave policy change over time?"
        st.rerun()
    if preset_cols[3].button("🕰️ System Knowledge on Jan 5, 2025"):
        st.session_state["query_text"] = "What would the system have answered on January 5, 2025 regarding leave entitlement?"
        st.rerun()

    if "query_text" in st.session_state:
        user_query = st.session_state["query_text"]

    if btn_search or ("query_text" in st.session_state and not btn_search):
        with st.spinner("Executing bitemporal retrieval, recursive CTE graph traversal, and synthesis..."):
            as_of_vt = vt_input.strip() if vt_input.strip() else None
            as_of_tt = tt_input.strip() if tt_input.strip() else None

            evidence = retrieval_engine.retrieve(
                query_str=user_query,
                as_of_valid_time=as_of_vt,
                as_of_transaction_time=as_of_tt,
            )
            ans = answer_engine.generate_answer(evidence)

        # 1. Natural Language Answer
        st.markdown("#### 1. Grounded Answer")
        if ans.is_abstention:
            st.warning(f"⚠️ **Abstention**: {ans.answer}")
        elif ans.has_conflict:
            st.error(f"⚔️ **Temporal Contradiction Detected**:\n\n{ans.answer}")
        else:
            st.success(ans.answer)

        # 2. Historical Timeline
        st.markdown("#### 2. Historical State Timeline")
        if ans.timeline:
            timeline_items = []
            for ev in ans.timeline:
                vt_range = f"{ev.get('valid_from', 'Start')} → {ev.get('valid_until', 'Present/Ongoing')}"
                timeline_items.append(
                    {
                        "Period": vt_range,
                        "Entitlement / Value": ev.get("object_value", "N/A"),
                        "Status": ev.get("status", "Active"),
                        "Source Document": ev.get("source_document", "Knowledge Base"),
                    }
                )
            st.table(pd.DataFrame(timeline_items))
        else:
            st.info("No chronological progression records detected for this query.")

        # 3. Supporting Evidence
        st.markdown("#### 3. Supporting Claim Evidence & Citations")
        if ans.supporting_evidence:
            for idx, ev in enumerate(ans.supporting_evidence, start=1):
                with st.expander(f"Citation [{idx}] — {ev.get('subject')}: {ev.get('value')} (Source: {ev.get('document')}, p. {ev.get('page')})"):
                    st.markdown(f"**Exact Quote:** *\"{ev.get('exact_quote')}\"*")
                    col_e1, col_e2, col_e3 = st.columns(3)
                    col_e1.markdown(f"**Valid From:** `{ev.get('valid_from')}`")
                    col_e2.markdown(f"**Valid Until:** `{ev.get('valid_until')}`")
                    col_e3.markdown(f"**Extraction Confidence:** `{ev.get('confidence')}`")
        else:
            st.caption("No direct supporting claims available.")

        # 4. Auditable Retrieval Reasoning Trace
        st.markdown("#### 4. Operational Retrieval Reasoning Trace")
        with st.expander("🛠️ View Full Retrieval Trace & Deterministic Decision Log", expanded=False):
            st.json(ans.reasoning_trace)


# ==============================================================================
# TAB 2: KNOWLEDGE EXPLORER & GRAPH
# ==============================================================================
with tab_explorer:
    st.markdown("### Knowledge Base Explorer & Lineage Visualizer")

    subtab_ingest, subtab_docs, subtab_claims, subtab_graph = st.tabs(
        [
            "📥 Ingest New Document / Version",
            "📄 Document Version Tree",
            "🧩 Atomic Claims Table",
            "🕸️ Knowledge Graph",
        ]
    )

    # Ingestion Subtab
    with subtab_ingest:
        st.markdown("#### Ingest or Update Knowledge Documents")
        col_in1, col_in2 = st.columns([1, 1])

        with col_in1:
            st.markdown("**Option 1: Upload File (PDF, DOCX, TXT, MD, CSV)**")
            uploaded_file = st.file_uploader("Choose a file", type=["pdf", "docx", "txt", "md", "csv"])
            file_pub_date = st.text_input("Document Publication Date", placeholder="YYYY-MM-DD (e.g. 2024-07-01)")
            if st.button("Process Uploaded File", type="primary") and uploaded_file:
                import tempfile
                with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{uploaded_file.name}") as tmp:
                    tmp.write(uploaded_file.getvalue())
                    tmp_path = tmp.name

                with st.spinner("Ingesting file and extracting atomic claims..."):
                    pipe = IngestionPipeline()
                    doc, doc_ver, is_new = pipe.ingest_file(tmp_path, publication_date=file_pub_date if file_pub_date else None)

                    if is_new:
                        ext_pipe = ClaimExtractionPipeline()
                        claims = ext_pipe.extract_from_document_version(doc_ver.version_id)
                        det = RelationDetector()
                        det.detect_all_relations()
                        st.success(f"Ingested {uploaded_file.name} as Version {doc_ver.version_number}. Extracted {len(claims)} atomic claims.")
                    else:
                        st.warning(f"File content matches existing document {doc.document_id} (Version {doc_ver.version_number}). No duplicate created.")
                os.remove(tmp_path)
                st.rerun()

        with col_in2:
            st.markdown("**Option 2: Direct Text Ingestion (Amendments / Policies)**")
            doc_name = st.text_input("Document Filename", value="Policy_Update_2025.txt")
            doc_text = st.text_area(
                "Document Content",
                value="Effective January 1, 2025, the annual leave entitlement increases to 18 days.",
                height=150,
            )
            pub_date = st.text_input("Effective / Publication Date", value="2025-01-15")

            if st.button("Ingest Text Document"):
                import tempfile
                with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{doc_name}", mode="w", encoding="utf-8") as tmp:
                    tmp.write(doc_text)
                    tmp_path = tmp.name

                with st.spinner("Processing document text..."):
                    pipe = IngestionPipeline()
                    doc, doc_ver, is_new = pipe.ingest_file(tmp_path, publication_date=pub_date)
                    if is_new:
                        ext_pipe = ClaimExtractionPipeline()
                        claims = ext_pipe.extract_from_document_version(doc_ver.version_id)
                        det = RelationDetector()
                        det.detect_all_relations()
                        st.success(f"Successfully processed {doc_name} (v{doc_ver.version_number}) with {len(claims)} claims extracted.")
                    else:
                        st.warning("Content identical to existing version.")
                os.remove(tmp_path)
                st.rerun()

    # Document Versions Subtab
    with subtab_docs:
        st.markdown("#### Ingested Documents & Immutable Versions")
        with get_db_session() as s:
            docs = s.query(Document).order_by(Document.created_at.desc()).all()
            doc_rows = []
            for d in docs:
                versions = s.query(DocumentVersion).filter_by(document_id=d.document_id).all()
                for v in versions:
                    doc_rows.append(
                        {
                            "Document ID": d.document_id[:8] + "...",
                            "Filename": d.filename,
                            "Version": f"v{v.version_number}",
                            "Content Hash (SHA-256)": v.content_hash[:16] + "...",
                            "Publication Date": d.publication_timestamp.strftime("%Y-%m-%d") if d.publication_timestamp else "N/A",
                            "Uploaded At": v.created_at.strftime("%Y-%m-%d %H:%M"),
                        }
                    )
        if doc_rows:
            st.dataframe(pd.DataFrame(doc_rows), use_container_width=True)
        else:
            st.info("No documents currently in knowledge base. Ingest a document above or run seed data script.")

    # Claims Subtab
    with subtab_claims:
        st.markdown("#### Atomic Bitemporal Claims")
        with get_db_session() as s:
            claims = s.query(Claim).order_by(Claim.valid_from.desc()).all()
            claim_rows = []
            for c in claims:
                claim_rows.append(
                    {
                        "ID": c.claim_id,
                        "Subject": c.subject_text,
                        "Predicate": c.predicate,
                        "Object Value": c.object_value,
                        "Valid From": c.valid_from.strftime("%Y-%m-%d") if c.valid_from else "Beginning",
                        "Valid Until": c.valid_until.strftime("%Y-%m-%d") if c.valid_until else "Ongoing",
                        "Recorded From": c.recorded_from.strftime("%Y-%m-%d") if c.recorded_from else "Unknown",
                        "Confidence": c.confidence,
                        "Status": c.verification_status,
                    }
                )
        if claim_rows:
            st.dataframe(pd.DataFrame(claim_rows), use_container_width=True)
        else:
            st.info("No claims found.")

    # Graph Visualizer Subtab
    with subtab_graph:
        st.markdown("#### Interactive Temporal Knowledge Graph")
        st.caption("Visual representation of claims and temporal relationship edges (SUPERSEDES, AMENDS, CONTRADICTS, SUPPORTS).")
        try:
            html_content = GraphVisualizer.generate_pyvis_html()
            components.html(html_content, height=600, scrolling=True)
        except Exception as e:
            st.warning(f"Could not render PyVis interactive canvas: {e}")
            graph_data = GraphVisualizer.get_graph_dict()
            st.write(f"Total Nodes: {len(graph_data['nodes'])}, Total Edges: {len(graph_data['edges'])}")


# ==============================================================================
# TAB 3: CHRONOBENCH EVALUATION DASHBOARD
# ==============================================================================
with tab_bench:
    st.markdown("### ChronoBench Evaluation Dashboard")
    st.caption("Rigorous benchmark evaluation across 4 independent RAG systems and 6 ablation configurations.")

    col_b1, col_b2 = st.columns([1, 2])
    with col_b1:
        st.markdown("#### Benchmark Configuration")
        num_qs = st.slider("Number of Benchmark Questions", min_value=10, max_value=60, value=20, step=5)
        run_ablations = st.checkbox("Include 6 Ablation Studies", value=True)
        btn_run_bench = st.button("🚀 Run ChronoBench Suite", type="primary", use_container_width=True)

    with col_b2:
        st.markdown("#### Systems Under Evaluation")
        st.markdown(
            """
            - **System A (Vanilla Dense RAG)**: Dense embeddings only (`MiniLM-L6-v2`), top-k chunk retrieval.
            - **System B (Hybrid RAG)**: Dense + SQLite FTS5 BM25 with Reciprocal Rank Fusion (RRF, k=60).
            - **System C (Temporal-Filtered RAG)**: Hybrid retrieval with naive document metadata date filter.
            - **System D (ChronoRAG)**: Full bitemporal engine, half-open interval math, CTE graph traversal, conflict detection & timeline generation.
            """
        )

    if btn_run_bench:
        with st.spinner(f"Running ChronoBench with {num_qs} queries across 4 systems..."):
            gen = ChronoBenchDatasetGenerator(seed=42)
            facts, questions = gen.generate_benchmark_suite(num_questions=num_qs)

            chunk_corpus = []
            for f in facts:
                chunk_corpus.append({
                    "chunk_id": f.fact_id,
                    "chunk_text": f.source_passage,
                    "fact_id": f.fact_id,
                    "valid_from": f.valid_from,
                    "valid_until": f.valid_until,
                    "recorded_from": f.recorded_from,
                    "recorded_until": f.recorded_until,
                    "publication_date": f.recorded_from,
                })

            dense_retriever = DenseRetriever()
            keyword_retriever = KeywordRetriever()

            sys_a = VanillaDenseRAG(dense_retriever, chunk_corpus, top_k=3)
            sys_b = HybridRAG(dense_retriever, keyword_retriever, chunk_corpus, top_k=3)
            sys_c = TemporalFilteredRAG(dense_retriever, chunk_corpus, top_k=3)
            sys_d = FullChronoRAG(retrieval_engine, answer_engine)

            runner = ChronoBenchRunner(questions=questions)

            sum_a = runner.evaluate_system(sys_a)
            sum_b = runner.evaluate_system(sys_b)
            sum_c = runner.evaluate_system(sys_c)
            sum_d = runner.evaluate_system(sys_d)

            systems_res = [sum_a, sum_b, sum_c, sum_d]
            ablations_res = []

            if run_ablations:
                for cfg in get_standard_ablations():
                    abl = AblatedChronoRAG(retrieval_engine, answer_engine, cfg)
                    ablations_res.append(runner.evaluate_system(abl))

            st.session_state["benchmark_results"] = {
                "systems": systems_res,
                "ablations": ablations_res,
                "num_questions": len(questions),
            }

    if "benchmark_results" in st.session_state:
        bdata = st.session_state["benchmark_results"]
        sys_list = bdata["systems"]

        st.divider()
        st.markdown("#### 1. Overall System Performance Comparison")

        metrics_table = []
        for s in sys_list:
            metrics_table.append(
                {
                    "System": s.system_name,
                    "Historical Accuracy": f"{s.historical_accuracy:.1%}",
                    "Token F1": f"{s.token_f1:.3f}",
                    "Temporal Precision@K": f"{s.temporal_precision_at_k:.1%}",
                    "Temporal Coverage@K": f"{s.temporal_coverage_at_k:.1%}",
                    "Wrong-Era Rate": f"{s.wrong_era_retrieval_rate:.1%}",
                    "Abstention Acc": f"{s.abstention_accuracy:.1%}",
                    "Latency (ms)": f"{s.mean_latency_ms:.1f}ms",
                }
            )
        st.table(pd.DataFrame(metrics_table))

        # Comparative Visualization Charts
        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            st.markdown("##### Historical Accuracy & Temporal Precision")
            chart_df = pd.DataFrame([
                {"System": s.system_name, "Historical Accuracy": s.historical_accuracy, "Temporal Precision@K": s.temporal_precision_at_k}
                for s in sys_list
            ])
            fig1 = px.bar(
                chart_df,
                x="System",
                y=["Historical Accuracy", "Temporal Precision@K"],
                barmode="group",
                color_discrete_sequence=["#1976D2", "#43A047"],
                height=350,
            )
            st.plotly_chart(fig1, use_container_width=True)

        with chart_col2:
            st.markdown("##### Wrong-Era Retrieval Rate (Lower is Better)")
            chart_df2 = pd.DataFrame([
                {"System": s.system_name, "Wrong-Era Rate": s.wrong_era_retrieval_rate}
                for s in sys_list
            ])
            fig2 = px.bar(
                chart_df2,
                x="System",
                y="Wrong-Era Rate",
                color="System",
                color_discrete_sequence=["#E53935", "#FB8C00", "#FDD835", "#43A047"],
                height=350,
            )
            st.plotly_chart(fig2, use_container_width=True)

        # Ablation Studies
        if bdata.get("ablations"):
            st.markdown("#### 2. ChronoRAG Component Ablation Studies")
            st.caption("Drop in historical accuracy when key architectural components are removed.")
            abl_table = []
            for a in bdata["ablations"]:
                abl_table.append({
                    "Ablation Component": a.system_name.replace("Ablation: ", ""),
                    "Historical Accuracy": f"{a.historical_accuracy:.1%}",
                    "Token F1": f"{a.token_f1:.3f}",
                    "Temporal Precision@K": f"{a.temporal_precision_at_k:.1%}",
                    "Wrong-Era Rate": f"{a.wrong_era_retrieval_rate:.1%}",
                })
            st.table(pd.DataFrame(abl_table))

            abl_chart_df = pd.DataFrame([
                {"Component": a.system_name.replace("Ablation: ", ""), "Accuracy": a.historical_accuracy}
                for a in bdata["ablations"]
            ])
            fig_abl = px.bar(
                abl_chart_df,
                x="Component",
                y="Accuracy",
                color="Accuracy",
                color_continuous_scale="Blues",
                height=300,
            )
            st.plotly_chart(fig_abl, use_container_width=True)
