from datetime import datetime
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field, field_validator


class RawExtractedClaim(BaseModel):
    """Raw structured claim extracted by LLM/engine before normalization."""
    subject: str = Field(..., description="Entity or topic the claim is about")
    predicate: str = Field(..., description="Attribute, relation or property being asserted")
    object_value: str = Field(..., description="The value or state of the attribute")
    conditions: Dict[str, Any] = Field(default_factory=dict, description="Applicability conditions e.g. tenure, department")
    effective_start_date: Optional[str] = Field(None, description="When this fact became valid (ISO or text like 'July 1, 2024')")
    effective_end_date: Optional[str] = Field(None, description="When this fact ended (or null if currently valid/open-ended)")
    previous_value: Optional[str] = Field(None, description="Previous value if this amends an older policy")
    exact_quote: str = Field(..., description="Exact verbatim passage supporting this claim")
    page_number: Optional[int] = Field(None, description="Page number where quote appears")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Model extraction confidence score")


class ValidatedAtomicClaim(BaseModel):
    """Normalized, validated bitemporal atomic claim."""
    subject_text: str
    predicate: str
    object_value: str
    conditions: Dict[str, Any] = Field(default_factory=dict)
    
    # Valid Time
    valid_from: Optional[datetime] = None
    valid_until: Optional[datetime] = None
    valid_from_unknown: bool = False
    valid_until_unknown: bool = False

    # Previous historical value if superseded
    previous_value: Optional[str] = None

    # Provenance
    exact_quote: str
    page_number: Optional[int] = None
    confidence: float = 1.0
    extraction_model: str = "Qwen2.5-1.5B-Instruct"
    extraction_version: str = "v1.0"


class ClaimExtractionBatch(BaseModel):
    claims: List[RawExtractedClaim] = Field(default_factory=list)
