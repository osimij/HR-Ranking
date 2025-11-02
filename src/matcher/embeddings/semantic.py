"""Sentence embedding helpers using sentence-transformers."""

from __future__ import annotations

from functools import lru_cache
from typing import Iterable, List

import numpy as np

try:  # pragma: no cover - heavy dependency
    from sentence_transformers import SentenceTransformer  # type: ignore
except ImportError as exc:  # pragma: no cover - optional dependency
    SentenceTransformer = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


class SemanticEmbedder:
    """Caches sentence embeddings for skill/experience text."""

    def __init__(self, model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2") -> None:
        if SentenceTransformer is None:
            raise RuntimeError(
                "sentence-transformers package required for semantic embeddings"
            ) from _IMPORT_ERROR
        self.model = SentenceTransformer(model_name)

    @lru_cache(maxsize=8192)
    def encode(self, text: str) -> np.ndarray:
        if not text.strip():
            return np.zeros(self.model.get_sentence_embedding_dimension())
        return np.asarray(self.model.encode(text, show_progress_bar=False))

    def batch_encode(self, texts: Iterable[str]) -> List[np.ndarray]:
        return [self.encode(text) for text in texts]

    @staticmethod
    def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        if not vec_a.any() or not vec_b.any():
            return 0.0
        denom = np.linalg.norm(vec_a) * np.linalg.norm(vec_b)
        if denom == 0:
            return 0.0
        return float(np.dot(vec_a, vec_b) / denom)

