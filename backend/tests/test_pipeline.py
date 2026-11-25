import pytest

from app.pipeline import ask, ingest
from tests.conftest import FakeLLM, fake_embed, fake_rerank


@pytest.fixture
def policy(tmp_path):
    f = tmp_path / "policy.txt"
    f.write_text(
        "Refunds are processed within five business days. "
        "Shipping usually takes three to seven days. "
        "Our office is open on weekdays from nine to five. " * 3
    )
    return f


def test_ingest_records_document(store, policy):
    doc = ingest(policy, "policy.txt", store, fake_embed, chunk_size=120, overlap=0)
    assert doc.filename == "policy.txt" and doc.chunks > 1
    assert store.documents[doc.id].uploaded_at
    assert len(store.records) == doc.chunks


def test_ingest_empty_file(store, tmp_path):
    f = tmp_path / "empty.txt"
    f.write_text("   ")
    with pytest.raises(ValueError):
        ingest(f, "empty.txt", store, fake_embed)


def test_ask_uses_top_reranked_chunks(store, policy):
    ingest(policy, "policy.txt", store, fake_embed, chunk_size=120, overlap=0)
    llm = FakeLLM()
    result = ask("How long do refunds take?", store, llm, fake_embed, fake_rerank, top_k=6, top_n=2)
    assert result.text == "answer from context"
    assert len(result.hits) == 2
    assert result.hits[0].score >= result.hits[1].score
    assert "Refunds" in llm.last_context and "policy.txt" in llm.last_context


def test_ask_with_no_documents(store):
    result = ask("Anything?", store, FakeLLM(), fake_embed, fake_rerank)
    assert result.hits == [] and "no matching documents" in result.text


def test_ask_filtered_to_missing_document(store, policy):
    ingest(policy, "policy.txt", store, fake_embed)
    result = ask("refunds", store, FakeLLM(), fake_embed, fake_rerank, document_ids={"other"})
    assert result.hits == []
