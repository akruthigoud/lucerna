import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from app.chunking import chunk_text
from app.llm import LLM
from app.parsing import extract_text
from app.store import ChunkRecord, Document, Hit, Store, now_iso

EmbedFn = Callable[[list[str]], np.ndarray]
RerankFn = Callable[[str, list[str]], list[float]]


def ingest(path: Path, filename: str, store: Store, embed: EmbedFn,
           chunk_size: int = 800, overlap: int = 120) -> Document:
    text = extract_text(path)
    chunks = chunk_text(text, chunk_size, overlap)
    if not chunks:
        raise ValueError(f"No text could be extracted from {filename}")

    doc = Document(id=uuid.uuid4().hex[:12], filename=filename, chunks=len(chunks),
                   characters=len(text), uploaded_at=now_iso())
    records = [ChunkRecord(doc.id, c.index, c.text) for c in chunks]
    store.add(doc, records, embed([c.text for c in chunks]))
    return doc


@dataclass
class Answer:
    text: str
    hits: list[Hit]
    provider: str
    latency_ms: float


def ask(question: str, store: Store, llm: LLM, embed: EmbedFn, rerank: RerankFn,
        top_k: int = 8, top_n: int = 3, document_ids: set[str] | None = None) -> Answer:
    start = time.perf_counter()
    candidates = store.search(embed([question])[0], top_k, document_ids)

    if not candidates:
        return Answer("There are no matching documents to search yet.", [], llm.name,
                      (time.perf_counter() - start) * 1000)

    scores = rerank(question, [h.record.text for h in candidates])
    ranked = sorted((Hit(h.record, s) for h, s in zip(candidates, scores)),
                    key=lambda h: h.score, reverse=True)[:top_n]

    context = "\n\n".join(
        f"[{store.filename(h.record.document_id)}, chunk {h.record.index}]\n{h.record.text}" for h in ranked
    )
    text = llm.answer(question, context)
    return Answer(text, ranked, llm.name, (time.perf_counter() - start) * 1000)
