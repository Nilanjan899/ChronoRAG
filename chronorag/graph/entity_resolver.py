import re
import difflib
from typing import Optional, List, Tuple, Dict, Any
from chronorag.database.connection import get_db_session
from chronorag.database.models import Entity


class EntityResolver:
    """
    Context-aware entity resolution.
    Avoids aggressive false merges by requiring name similarity AND category/type compatibility.
    """

    def __init__(self, similarity_threshold: float = 0.82):
        self.similarity_threshold = similarity_threshold

    def resolve_entity(
        self,
        name: str,
        entity_type: str = "PolicySubject",
        context_predicates: Optional[List[str]] = None,
    ) -> Tuple[Entity, bool, float]:
        """
        Resolves a mention to an existing canonical entity or creates a new one.
        Returns: (Entity, is_new_entity, confidence_score)
        """
        clean_name = self._normalize_name(name)
        with get_db_session() as session:
            # 1. Exact match on canonical_name
            exact = session.query(Entity).filter_by(canonical_name=name).first()
            if exact:
                return exact, False, 1.0

            # 2. Match in aliases JSON
            all_entities = session.query(Entity).all()
            for ent in all_entities:
                aliases = ent.aliases or []
                if clean_name in [self._normalize_name(a) for a in aliases]:
                    return ent, False, 0.95

            # 3. Fuzzy similarity search with contextual safeguards
            best_match: Optional[Entity] = None
            best_score: float = 0.0

            for ent in all_entities:
                # Type must match or be generic
                if ent.entity_type != "Generic" and entity_type != "Generic" and ent.entity_type != entity_type:
                    continue

                # String & token similarity
                sim = self._compute_similarity(name, ent.canonical_name)
                if sim > best_score:
                    best_score = sim
                    best_match = ent

            if best_match and best_score >= self.similarity_threshold:
                # Add to aliases if not already present
                existing_aliases = list(best_match.aliases or [])
                if clean_name not in existing_aliases:
                    existing_aliases.append(name)
                    best_match.aliases = existing_aliases
                    session.add(best_match)
                    session.commit()
                return best_match, False, best_score

            # 4. No high-confidence match found -> create new canonical entity
            new_entity = Entity(
                canonical_name=name,
                entity_type=entity_type,
                aliases=[name, clean_name],
                entity_metadata={"context_predicates": context_predicates or []},
            )
            session.add(new_entity)
            session.commit()
            return new_entity, True, 1.0

    @classmethod
    def _normalize_name(cls, name: str) -> str:
        s = name.lower()
        # Remove common corporate document and policy qualifiers
        s = re.sub(r"\b(the|a|an|policy|amendment|document|procedure|guideline|plan)\b", "", s)
        s = re.sub(r"[^\w\s]", "", s)
        return " ".join(s.split())

    def _compute_similarity(self, name_a: str, name_b: str) -> float:
        norm_a = self._normalize_name(name_a)
        norm_b = self._normalize_name(name_b)
        if norm_a == norm_b:
            return 1.0

        seq_sim = difflib.SequenceMatcher(None, norm_a, norm_b).ratio()

        # Token set overlap (Jaccard on words)
        words_a = set(norm_a.split())
        words_b = set(norm_b.split())
        if words_a and words_b:
            jaccard = len(words_a & words_b) / len(words_a | words_b)
            # If one is a complete subset of the other (e.g. 'annual leave' inside 'annual leave v2')
            if words_a.issubset(words_b) or words_b.issubset(words_a):
                subset_sim = len(words_a & words_b) / min(len(words_a), len(words_b))
                return max(seq_sim, jaccard, subset_sim * 0.9)
            return max(seq_sim, jaccard)
        return seq_sim
