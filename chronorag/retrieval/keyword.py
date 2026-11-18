import re
import logging
from typing import List, Tuple, Dict, Any
from sqlalchemy import text
from chronorag.database.connection import get_db_session, is_postgres
from chronorag.database.models import Claim, Chunk

logger = logging.getLogger(__name__)


class KeywordRetriever:
    """Keyword retriever using PostgreSQL full-text search or SQLite FTS5."""

    def search_claims(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Searches claims via full-text search. Returns list of (claim_id, bm25_score)."""
        clean_query = self._sanitize_fts_query(query)
        if not clean_query:
            return []

        results: List[Tuple[str, float]] = []
        with get_db_session() as session:
            if not is_postgres():
                sql = text("""
                    SELECT claim_id, bm25(claims_fts) as score
                    FROM claims_fts
                    WHERE claims_fts MATCH :q
                    ORDER BY score ASC
                    LIMIT :top_k;
                """)
                try:
                    rows = session.execute(sql, {"q": clean_query, "top_k": top_k}).fetchall()
                    # In SQLite FTS5 bm25(), lower/more negative is better rank, so invert
                    for r in rows:
                        score = 1.0 / (1.0 + abs(float(r[1])))
                        results.append((r[0], score))
                except Exception as e:
                    logger.warning(f"FTS5 match failed for '{clean_query}': {e}. Using LIKE fallback.")
                    return self._fallback_like_search_claims(session, query, top_k)
            else:
                sql = text("""
                    SELECT claim_id, ts_rank(to_tsvector('english', subject_text || ' ' || predicate || ' ' || object_value), plainto_tsquery('english', :q)) as score
                    FROM claims
                    WHERE to_tsvector('english', subject_text || ' ' || predicate || ' ' || object_value) @@ plainto_tsquery('english', :q)
                    ORDER BY score DESC
                    LIMIT :top_k;
                """)
                rows = session.execute(sql, {"q": query, "top_k": top_k}).fetchall()
                for r in rows:
                    results.append((r[0], float(r[1])))

        return results

    def search_chunks(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Searches document chunks via full-text search. Returns list of (chunk_id, score)."""
        clean_query = self._sanitize_fts_query(query)
        if not clean_query:
            return []

        results: List[Tuple[str, float]] = []
        with get_db_session() as session:
            if not is_postgres():
                sql = text("""
                    SELECT chunk_id, bm25(chunks_fts) as score
                    FROM chunks_fts
                    WHERE chunks_fts MATCH :q
                    ORDER BY score ASC
                    LIMIT :top_k;
                """)
                try:
                    rows = session.execute(sql, {"q": clean_query, "top_k": top_k}).fetchall()
                    for r in rows:
                        score = 1.0 / (1.0 + abs(float(r[1])))
                        results.append((r[0], score))
                except Exception:
                    return self._fallback_like_search_chunks(session, query, top_k)
            else:
                sql = text("""
                    SELECT chunk_id, ts_rank(to_tsvector('english', chunk_text), plainto_tsquery('english', :q)) as score
                    FROM chunks
                    WHERE to_tsvector('english', chunk_text) @@ plainto_tsquery('english', :q)
                    ORDER BY score DESC
                    LIMIT :top_k;
                """)
                rows = session.execute(sql, {"q": query, "top_k": top_k}).fetchall()
                for r in rows:
                    results.append((r[0], float(r[1])))

        return results

    @staticmethod
    def _sanitize_fts_query(query: str) -> str:
        # Keep alphanumeric words for FTS MATCH
        words = re.findall(r"[A-Za-z0-9]+", query)
        if not words:
            return ""
        # Combine with OR or NEAR
        return " OR ".join(f'"{w}"' for w in words[:8])

    @staticmethod
    def _fallback_like_search_claims(session, query: str, top_k: int) -> List[Tuple[str, float]]:
        words = query.lower().split()
        if not words:
            return []
        term = f"%{words[0]}%"
        claims = session.query(Claim).filter(
            (Claim.subject_text.ilike(term)) | (Claim.predicate.ilike(term)) | (Claim.object_value.ilike(term))
        ).limit(top_k).all()
        return [(c.claim_id, 0.5) for c in claims]

    @staticmethod
    def _fallback_like_search_chunks(session, query: str, top_k: int) -> List[Tuple[str, float]]:
        words = query.lower().split()
        if not words:
            return []
        term = f"%{words[0]}%"
        chunks = session.query(Chunk).filter(Chunk.chunk_text.ilike(term)).limit(top_k).all()
        return [(c.chunk_id, 0.5) for c in chunks]
