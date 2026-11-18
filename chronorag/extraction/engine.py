import json
import logging
import re
from typing import List, Optional, Dict, Any
from chronorag.config import settings
from chronorag.extraction.models import RawExtractedClaim, ClaimExtractionBatch
from chronorag.extraction.prompt import EXTRACTION_SYSTEM_PROMPT, EXTRACTION_USER_PROMPT

logger = logging.getLogger(__name__)

_engine_instance = None


class LocalClaimExtractionEngine:
    """Configurable extraction engine supporting llama.cpp, local transformers, and deterministic rule engine."""

    def __init__(self, backend: Optional[str] = None):
        self.backend = backend or settings.llm_backend
        self.model_name = settings.llm_model_name
        self.prompt_version = "v1.0"
        self._cache: Dict[str, List[RawExtractedClaim]] = {}
        self._init_backend()

    def _init_backend(self):
        if self.backend == "llama_cpp" and settings.llm_model_path:
            try:
                import llama_cpp
                logger.info(f"Loading llama.cpp model from {settings.llm_model_path}")
                self.llm = llama_cpp.Llama(
                    model_path=settings.llm_model_path,
                    n_ctx=2048,
                    verbose=False,
                )
            except Exception as e:
                logger.warning(f"Could not initialize llama.cpp ({e}). Falling back to deterministic engine.")
                self.backend = "deterministic"
                self.llm = None
        else:
            self.llm = None

    def extract_from_chunk(self, chunk_text: str, page_number: Optional[int] = 1) -> List[RawExtractedClaim]:
        """Extract atomic claims from chunk with caching and Pydantic validation."""
        import hashlib
        cache_key = hashlib.sha256((chunk_text + str(page_number)).encode("utf-8")).hexdigest()
        if cache_key in self._cache:
            return self._cache[cache_key]

        claims: List[RawExtractedClaim] = []

        if self.backend == "llama_cpp" and self.llm is not None:
            claims = self._extract_with_llama(chunk_text, page_number)

        # If LLM produces empty or fails, use deterministic regex/pattern extraction
        if not claims:
            claims = self._extract_deterministic(chunk_text, page_number)

        self._cache[cache_key] = claims
        return claims

    def _extract_with_llama(self, chunk_text: str, page_number: Optional[int]) -> List[RawExtractedClaim]:
        prompt = EXTRACTION_USER_PROMPT.format(page_number=page_number or 1, chunk_text=chunk_text)
        try:
            response = self.llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=800,
            )
            raw_text = response["choices"][0]["message"]["content"]
            # Extract JSON block
            json_match = re.search(r"\{[\s\S]*\}", raw_text)
            if json_match:
                data = json.loads(json_match.group(0))
                batch = ClaimExtractionBatch.model_validate(data)
                return batch.claims
        except Exception as e:
            logger.warning(f"llama.cpp extraction error ({e}). Falling back.")
        return []

    def _extract_deterministic(self, chunk_text: str, page_number: Optional[int]) -> List[RawExtractedClaim]:
        """
        High-precision deterministic pattern extractor for temporal policies and claims.
        Recognizes:
        - "Effective [Date], [Subject] [Predicate] [Object] instead of previous [Prev]"
        - "[Subject] receive [Object] [Predicate]"
        - "Effective [Date], the [Predicate] increases/changes to [Object]"
        - "[Entity] amends/supersedes [Target]"
        """
        claims: List[RawExtractedClaim] = []
        sentences = re.split(r"(?<=[.!?])\s+", chunk_text)

        date_pattern = r"(?:Effective\s+|Starting\s+|As of\s+|From\s+)?([A-Z][a-z]+\s+\d{1,2},\s+\d{4}|\d{4}-\d{2}-\d{2}|[A-Z][a-z]+\s+\d{4})"
        instead_pattern = r"instead\s+of\s+(?:the\s+)?(?:previous\s+)?(?:entitlement\s+of\s+)?([0-9]+(?:\s+[a-zA-Z]+)?|\$[0-9]+)"

        for s in sentences:
            s_clean = s.strip()
            if not s_clean:
                continue

            # Check for amendment with previous value
            # e.g.: "Effective July 1, 2024, employees receive 15 annual leaves instead of 12."
            eff_match = re.search(date_pattern, s_clean, re.IGNORECASE)
            effective_date = eff_match.group(1) if eff_match else None

            prev_match = re.search(instead_pattern, s_clean, re.IGNORECASE)
            prev_value = prev_match.group(1) if prev_match else None

            # Pattern 1: Leave entitlement
            leave_match = re.search(
                r"(employees(?:.*?)?)\s+(?:receive|entitled to|granted)\s+(\d+\s+days|\d+\s+annual\s+leaves|\d+\s+leaves)",
                s_clean,
                re.IGNORECASE,
            )
            if leave_match:
                subj = leave_match.group(1).strip()
                obj = leave_match.group(2).strip()
                claims.append(
                    RawExtractedClaim(
                        subject=subj.capitalize(),
                        predicate="annual leave entitlement",
                        object_value=obj,
                        conditions={"tenure": "standard"} if "two years" not in s_clean else {"tenure": ">= 2 years"},
                        effective_start_date=effective_date or "unknown",
                        effective_end_date=None,
                        previous_value=prev_value,
                        exact_quote=s_clean,
                        page_number=page_number,
                        confidence=0.95,
                    )
                )
                continue

            # Pattern 2: Budget / Allowance (e.g. Travel budget is $500 per day)
            budget_match = re.search(
                r"([A-Za-z\s]+budget|[A-Za-z\s]+allowance|[A-Za-z\s]+rate)\s+(?:is|set to|increases to)\s+(\$[\d,]+(?:\s+per\s+[a-z]+)?)",
                s_clean,
                re.IGNORECASE,
            )
            if budget_match:
                pred = budget_match.group(1).strip().lower()
                val = budget_match.group(2).strip()
                claims.append(
                    RawExtractedClaim(
                        subject="Corporate Policy",
                        predicate=pred,
                        object_value=val,
                        conditions={},
                        effective_start_date=effective_date or "unknown",
                        effective_end_date=None,
                        previous_value=prev_value,
                        exact_quote=s_clean,
                        page_number=page_number,
                        confidence=0.95,
                    )
                )
                continue

            # Pattern 3: General API / Software / Regulation statement
            # e.g.: "Effective version 2.0, API endpoint /v1/users is deprecated."
            dep_match = re.search(r"(?:endpoint|parameter|method|api)\s+([/\w]+)\s+(?:is|became)\s+(deprecated|removed|active)", s_clean, re.IGNORECASE)
            if dep_match:
                endpoint = dep_match.group(1).strip()
                status = dep_match.group(2).strip()
                claims.append(
                    RawExtractedClaim(
                        subject=endpoint,
                        predicate="lifecycle status",
                        object_value=status,
                        conditions={},
                        effective_start_date=effective_date or "unknown",
                        effective_end_date=None,
                        previous_value=None,
                        exact_quote=s_clean,
                        page_number=page_number,
                        confidence=0.90,
                    )
                )
                continue

            # Fallback: if effective date present and asserted fact
            if effective_date and len(s_clean) > 20:
                claims.append(
                    RawExtractedClaim(
                        subject="Policy Statement",
                        predicate="policy clause",
                        object_value=s_clean,
                        conditions={},
                        effective_start_date=effective_date,
                        effective_end_date=None,
                        previous_value=prev_value,
                        exact_quote=s_clean,
                        page_number=page_number,
                        confidence=0.85,
                    )
                )

        return claims


def get_extraction_engine() -> LocalClaimExtractionEngine:
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = LocalClaimExtractionEngine()
    return _engine_instance
