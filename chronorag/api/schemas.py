from typing import List, Dict, Any, Optional
from datetime import datetime
from pydantic import BaseModel, Field


# ------------------- Ingestion Schemas -------------------
class IngestTextRequest(BaseModel):
    filename: str
    text_content: str
    publication_date: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class IngestResponse(BaseModel):
    document_id: str
    version_id: str
    version_number: int
    is_new_version: bool
    is_duplicate: bool
    num_chunks: int
    num_claims: int
    message: str


class DocumentVersionSummary(BaseModel):
    document_id: str
    filename: str
    current_version: int
    content_hash: str
    upload_timestamp: Optional[datetime] = None
    publication_timestamp: Optional[datetime] = None
    total_versions: int


class ClaimResponse(BaseModel):
    claim_id: str
    subject: str
    predicate: str
    object_value: str
    conditions: Dict[str, Any]
    valid_from: Optional[str]
    valid_until: Optional[str]
    recorded_from: Optional[str]
    recorded_until: Optional[str]
    confidence: float
    verification_status: str


# ------------------- Query Schemas -------------------
class QueryRequest(BaseModel):
    query: str
    as_of_valid_time: Optional[str] = None
    as_of_transaction_time: Optional[str] = None


class QueryResponse(BaseModel):
    query: str
    answer: str
    supporting_evidence: List[Dict[str, Any]] = Field(default_factory=list)
    timeline: List[Dict[str, Any]] = Field(default_factory=list)
    reasoning_trace: Dict[str, Any] = Field(default_factory=dict)
    is_abstention: bool = False
    has_conflict: bool = False
    citation_validation: Dict[str, Any] = Field(default_factory=dict)


# ------------------- Graph Schemas -------------------
class GraphExportResponse(BaseModel):
    nodes: List[Dict[str, Any]] = Field(default_factory=list)
    edges: List[Dict[str, Any]] = Field(default_factory=list)


class LineageResponse(BaseModel):
    claim_id: str
    lineage_path: List[Dict[str, Any]] = Field(default_factory=list)


# ------------------- Benchmark Schemas -------------------
class BenchmarkRunRequest(BaseModel):
    num_questions: int = 30
    include_ablations: bool = True


class BenchmarkRunResponse(BaseModel):
    run_timestamp: str
    total_questions: int
    report_markdown: str
    systems_summary: List[Dict[str, Any]]
    ablations_summary: List[Dict[str, Any]] = Field(default_factory=list)
