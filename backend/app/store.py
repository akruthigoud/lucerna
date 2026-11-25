import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import faiss
import numpy as np


@dataclass
class Document:
    id: str
    filename: str
    chunks: int
    characters: int
    uploaded_at: str  # ISO 8601


@dataclass
class ChunkRecord:
    document_id: str
    index: int
    text: str


@dataclass
class Hit:
    record: ChunkRecord
    score: float


class Store:
    """FAISS index plus document and chunk metadata, saved under `path`."""

    def __init__(self, path: str | Path, dim: int):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.dim = dim
        self.documents: dict[str, Document] = {}
        self.records: list[ChunkRecord] = []
        self.vectors = np.zeros((0, dim), dtype="float32")
        self._load()
        self._rebuild_index()

    # -- writes --
    def add(self, doc: Document, records: list[ChunkRecord], vectors: np.ndarray) -> None:
        if len(records) != len(vectors):
            raise ValueError("records and vectors must have the same length")
        self.documents[doc.id] = doc
        self.records.extend(records)
        self.vectors = np.vstack([self.vectors, vectors.astype("float32")])
        self.index.add(vectors.astype("float32"))
        self._save()

    def delete(self, document_id: str) -> bool:
        if document_id not in self.documents:
            return False
        keep = [i for i, r in enumerate(self.records) if r.document_id != document_id]
        self.records = [self.records[i] for i in keep]
        self.vectors = self.vectors[keep] if keep else np.zeros((0, self.dim), dtype="float32")
        del self.documents[document_id]
        self._rebuild_index()
        self._save()
        return True

    # -- reads --
    def search(self, query: np.ndarray, k: int, document_ids: set[str] | None = None) -> list[Hit]:
        if self.index.ntotal == 0:
            return []
        fetch = self.index.ntotal if document_ids else min(k, self.index.ntotal)
        scores, ids = self.index.search(query.reshape(1, -1).astype("float32"), fetch)
        hits = []
        for score, i in zip(scores[0], ids[0]):
            if i < 0:
                continue
            rec = self.records[i]
            if document_ids and rec.document_id not in document_ids:
                continue
            hits.append(Hit(rec, float(score)))
            if len(hits) == k:
                break
        return hits

    def filename(self, document_id: str) -> str:
        return self.documents[document_id].filename

    # -- persistence --
    def _rebuild_index(self) -> None:
        self.index = faiss.IndexFlatIP(self.dim)
        if len(self.vectors):
            self.index.add(self.vectors)

    def _save(self) -> None:
        np.save(self.path / "vectors.npy", self.vectors)
        meta = {
            "documents": [asdict(d) for d in self.documents.values()],
            "records": [asdict(r) for r in self.records],
        }
        (self.path / "meta.json").write_text(json.dumps(meta))

    def _load(self) -> None:
        meta_file, vec_file = self.path / "meta.json", self.path / "vectors.npy"
        if not (meta_file.exists() and vec_file.exists()):
            return
        meta = json.loads(meta_file.read_text())
        self.documents = {d["id"]: Document(**d) for d in meta["documents"]}
        self.records = [ChunkRecord(**r) for r in meta["records"]]
        self.vectors = np.load(vec_file)


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")
