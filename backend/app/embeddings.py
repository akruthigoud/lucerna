from functools import lru_cache

import numpy as np

from app.config import get_settings


@lru_cache
def _model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(get_settings().embedding_model)


def embed(texts: list[str]) -> np.ndarray:
    """Return L2-normalised float32 embeddings, shape (len(texts), dim)."""
    if not texts:
        return np.zeros((0, dimension()), dtype="float32")
    vectors = _model().encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return np.asarray(vectors, dtype="float32")


@lru_cache
def dimension() -> int:
    return _model().get_sentence_embedding_dimension()
