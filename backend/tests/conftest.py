import re
import zlib

import numpy as np
import pytest

from app.llm import LLM
from app.store import Store

DIM = 64


def fake_embed(texts):
    """Bag-of-words hashed into DIM buckets, L2-normalised."""
    out = np.zeros((len(texts), DIM), dtype="float32")
    for i, t in enumerate(texts):
        for w in re.findall(r"[a-z0-9]+", t.lower()):
            out[i, zlib.crc32(w.encode()) % DIM] += 1
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    return out / np.where(norms == 0, 1, norms)


def fake_rerank(question, passages):
    q = set(re.findall(r"[a-z0-9]+", question.lower()))
    return [float(len(q & set(re.findall(r"[a-z0-9]+", p.lower())))) for p in passages]


class FakeLLM(LLM):
    name = "fake"

    def __init__(self):
        self.last_context = None

    def answer(self, question, context):
        self.last_context = context
        return "answer from context"


@pytest.fixture
def store(tmp_path):
    return Store(tmp_path / "index", DIM)
