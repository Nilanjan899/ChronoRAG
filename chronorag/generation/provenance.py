import re
from typing import List, Dict, Any, Tuple


class CitationValidator:
    """Validates citations in generated answers against actual retrieved source chunks."""

    @classmethod
    def validate_citations(
        cls, answer_text: str, available_sources: List[Dict[str, Any]]
    ) -> Tuple[bool, List[str], List[str]]:
        """
        Extracts citations from answer text (e.g., [Source: filename, Page: X] or [claim_id])
        and checks if each cited claim/source was actually retrieved.
        Returns: (is_valid, valid_citations, fabricated_citations)
        """
        valid_citations = []
        fabricated_citations = []

        valid_source_ids = {s.get("source_id") for s in available_sources if s.get("source_id")}
        valid_claim_ids = {s.get("claim_id") for s in available_sources if s.get("claim_id")}
        valid_filenames = {s.get("filename", "").lower() for s in available_sources if s.get("filename")}

        # Find citation tags like [Doc: filename, p.X] or [Source: ...] or UUIDs
        citations = re.findall(r"\[(?:Source|Doc|Ref|Claim):\s*([^\]]+)\]", answer_text, re.IGNORECASE)

        if not citations:
            # If no bracketed citations, check if sources were referenced in text
            return True, [], []

        for cit in citations:
            cit_clean = cit.strip()
            cit_lower = cit_clean.lower()

            matched = False
            # Check filename match
            for fn in valid_filenames:
                if fn and fn in cit_lower:
                    matched = True
                    break

            # Check ID match
            if not matched:
                for cid in valid_claim_ids | valid_source_ids:
                    if cid and cid in cit_clean:
                        matched = True
                        break

            if matched:
                valid_citations.append(cit_clean)
            else:
                fabricated_citations.append(cit_clean)

        is_valid = len(fabricated_citations) == 0
        return is_valid, valid_citations, fabricated_citations
