import logging
import math
from typing import List, Optional, Union
import numpy as np
from chronorag.config import settings

logger = logging.getLogger(__name__)

_embedder_instance = None


class LocalDenseEmbedder:
    """Local embedding generator using SentenceTransformers with robust offline fallback."""

    def __init__(self, model_name: Optional[str] = None, dim: Optional[int] = None):
        self.model_name = model_name or settings.embedding_model_name
        self.dim = dim or settings.embedding_dim
        self.model = None
        self._load_model()

    def _load_model(self):
        try:
            import os
            from pathlib import Path
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
            from sentence_transformers import SentenceTransformer
            model_path = str(Path(self.model_name).resolve()) if Path(self.model_name).exists() else self.model_name
            logger.info(f"Loading local embedding model: {model_path}")
            self.model = SentenceTransformer(model_path, device=settings.device)
            logger.info("SentenceTransformer model loaded successfully.")
        except Exception as e:
            logger.warning(
                f"Could not load SentenceTransformer ({e}). Using deterministic offline embedding fallback."
            )
            self.model = None

    def embed_text(self, text: str) -> List[float]:
        if not text:
            return [0.0] * self.dim

        if self.model is not None:
            emb = self.model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
            return emb.tolist()
        return self._deterministic_fallback_embed(text)

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        if self.model is not None:
            embs = self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
            return [e.tolist() for e in embs]
        return [self._deterministic_fallback_embed(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self.embed_text(text)

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        return self.embed_batch(texts)

    def _deterministic_fallback_embed(self, text: str) -> List[float]:
        """
        Deterministic, seedable projection into R^dim for offline operation
        preserving n-gram frequency signals.
        """
        import hashlib
        vec = np.zeros(self.dim, dtype=np.float32)
        words = text.lower().split()
        for idx, w in enumerate(words):
            h = int(hashlib.md5(w.encode("utf-8")).hexdigest(), 16)
            pos = h % self.dim
            val = ((h >> 8) % 1000) / 1000.0 - 0.5
            vec[pos] += val * (1.0 / (math.log(idx + 2)))

        norm = np.linalg.norm(vec)
        if norm > 1e-6:
            vec = vec / norm
        return vec.tolist()

    @staticmethod
    def cosine_similarity(v1: List[float], v2: List[float]) -> float:
        a = np.array(v1, dtype=np.float32)
        b = np.array(v2, dtype=np.float32)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a < 1e-7 or norm_b < 1e-7:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))


def get_embedder() -> LocalDenseEmbedder:
    global _embedder_instance
    if _embedder_instance is None:
        _embedder_instance = LocalDenseEmbedder()
    return _embedder_instance


DenseRetriever = LocalDenseEmbedder

