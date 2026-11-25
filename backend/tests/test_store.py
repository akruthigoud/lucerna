import numpy as np
import pytest

from app.store import ChunkRecord, Document, Store
from tests.conftest import DIM, fake_embed


def _doc(doc_id, n=1):
    return Document(doc_id, f"{doc_id}.txt", n, 100, "2026-11-26T10:00:00+05:30")


def _add(store, doc_id, texts):
    recs = [ChunkRecord(doc_id, i, t) for i, t in enumerate(texts)]
    store.add(_doc(doc_id, len(texts)), recs, fake_embed(texts))


def test_empty_search(store):
    assert store.search(fake_embed(["anything"])[0], 3) == []


def test_best_match_first(store):
    _add(store, "a", ["refund policy and refunds", "shipping times", "office hours"])
    hits = store.search(fake_embed(["refund policy"])[0], 2)
    assert hits[0].record.text == "refund policy and refunds"
    assert hits[0].score >= hits[1].score


def test_filter_by_document(store):
    _add(store, "a", ["refund policy"])
    _add(store, "b", ["refund policy"])
    hits = store.search(fake_embed(["refund policy"])[0], 5, {"b"})
    assert [h.record.document_id for h in hits] == ["b"]


def test_delete_removes_chunks(store):
    _add(store, "a", ["refund policy"])
    _add(store, "b", ["shipping times", "returns"])
    assert store.delete("a")
    assert "a" not in store.documents
    assert len(store.records) == 2 and store.index.ntotal == 2
    assert all(h.record.document_id == "b" for h in store.search(fake_embed(["refund"])[0], 5))


def test_delete_unknown(store):
    assert store.delete("nope") is False


def test_length_mismatch(store):
    with pytest.raises(ValueError):
        store.add(_doc("a"), [ChunkRecord("a", 0, "x")], np.zeros((2, DIM), dtype="float32"))


def test_persists_documents_and_vectors(tmp_path):
    s1 = Store(tmp_path / "idx", DIM)
    _add(s1, "a", ["kept text"])
    s2 = Store(tmp_path / "idx", DIM)
    assert s2.documents["a"].uploaded_at == "2026-11-26T10:00:00+05:30"
    assert s2.search(fake_embed(["kept text"])[0], 1)[0].record.text == "kept text"
