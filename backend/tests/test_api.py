import pytest
from fastapi.testclient import TestClient

from app import api
from app.llm import get_llm
from app.main import app
from tests.conftest import FakeLLM, fake_embed, fake_rerank


@pytest.fixture
def client(store):
    app.dependency_overrides = {
        api.get_store: lambda: store,
        api.get_embedder: lambda: fake_embed,
        api.get_reranker: lambda: fake_rerank,
        get_llm: lambda: FakeLLM(),
    }
    yield TestClient(app)
    app.dependency_overrides = {}


def _upload(client, name="notes.txt", body=b"Refunds take five days. Shipping takes a week."):
    return client.post("/api/documents", files={"file": (name, body, "text/plain")})


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200 and r.json()["documents"] == 0


def test_upload_and_list(client):
    r = _upload(client)
    assert r.status_code == 201
    doc = r.json()
    assert doc["filename"] == "notes.txt" and doc["chunks"] == 1
    listed = client.get("/api/documents").json()
    assert [d["id"] for d in listed] == [doc["id"]]
    assert listed[0]["uploaded_at"] == doc["uploaded_at"]


def test_upload_unsupported_type(client):
    assert _upload(client, name="virus.exe").status_code == 415


def test_upload_empty_file(client):
    assert _upload(client, body=b"   ").status_code == 422


def test_query_returns_citations(client):
    doc_id = _upload(client).json()["id"]
    r = client.post("/api/query", json={"question": "How long do refunds take?"})
    body = r.json()
    assert r.status_code == 200 and body["answer"] == "answer from context"
    assert body["citations"][0]["document_id"] == doc_id
    assert body["citations"][0]["filename"] == "notes.txt"


def test_delete_document(client):
    doc_id = _upload(client).json()["id"]
    assert client.delete(f"/api/documents/{doc_id}").status_code == 204
    assert client.get("/api/documents").json() == []
    assert client.delete(f"/api/documents/{doc_id}").status_code == 404
