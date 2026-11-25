from functools import lru_cache

from app.config import get_settings


@lru_cache
def _model():
    from sentence_transformers import CrossEncoder

    return CrossEncoder(get_settings().reranker_model)


def rerank(question: str, passages: list[str]) -> list[float]:
    """Score each passage against the question with a cross-encoder."""
    if not passages:
        return []
    return [float(s) for s in _model().predict([(question, p) for p in passages])]
