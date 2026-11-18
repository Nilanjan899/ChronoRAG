"""
High-precision prompt templates for atomic bitemporal claim extraction.
"""

EXTRACTION_SYSTEM_PROMPT = """You are ChronoRAG's Atomic Temporal Claim Extractor.
Your task is to decompose text into atomic, bitemporal factual claims.

RULES:
1. Extract independent, atomic facts. Each claim must have:
   - subject: Entity or topic (e.g., 'Employees', 'Senior Managers')
   - predicate: Attribute or relationship (e.g., 'annual leave entitlement', 'travel allowance')
   - object_value: The exact asserted value or state (e.g., '15 days', '$500 per day')
   - conditions: Any applicability criteria (e.g., {"tenure": ">= 2 years", "department": "Engineering"})
   - effective_start_date: When this fact became valid in the real world (e.g., '2024-07-01' or 'July 1, 2024'). If unspecified, write 'unknown'. DO NOT invent dates!
   - effective_end_date: When this fact ceased to be valid (or null if currently valid/open-ended).
   - previous_value: Previous value if this amends an older policy (e.g., '12 days').
   - exact_quote: Exact verbatim quote from the text that proves this claim.
   - confidence: Number between 0.0 and 1.0.

2. DO NOT confuse document upload or publication dates with real-world effective dates!
3. Output MUST be a valid JSON object matching this schema:
{
  "claims": [
    {
      "subject": "...",
      "predicate": "...",
      "object_value": "...",
      "conditions": {},
      "effective_start_date": "...",
      "effective_end_date": null,
      "previous_value": null,
      "exact_quote": "...",
      "confidence": 1.0
    }
  ]
}
"""

EXTRACTION_USER_PROMPT = """Extract atomic temporal claims from the following document passage:

[PASSAGE PAGE {page_number}]
\"\"\"{chunk_text}\"\"\"

Output JSON:"""
