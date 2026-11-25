import shutil
import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, UploadFile

from app import embeddings, reranker
from app.config import Settings, get_settings
from app.llm import LLM, get_llm
from app.parsing import SUPPORTED, UnsupportedFile
from app.pipeline import ask, ingest
from app.schemas import Citation, DocumentOut, Health, QueryIn, QueryOut
from app.store import Store

router = APIRouter()
_store: Store | None = None


def get_store() -> Store:
    global _store
    if _store is None:
        _store = Store(Path(get_settings().data_dir) / "index", embeddings.dimension())
    return _store


def get_embedder():
    return embeddings.embed


def get_reranker():
    return reranker.rerank


def _doc_out(d) -> DocumentOut:
    return DocumentOut(id=d.id, filename=d.filename, chunks=d.chunks,
                       characters=d.characters, uploaded_at=d.uploaded_at)


@router.get("/health", response_model=Health)
def health(store: Store = Depends(get_store), s: Settings = Depends(get_settings)):
    return Health(status="ok", llm_provider=s.llm_provider,
                  documents=len(store.documents), chunks=len(store.records))


@router.get("/documents", response_model=list[DocumentOut])
def list_documents(store: Store = Depends(get_store)):
    docs = sorted(store.documents.values(), key=lambda d: d.uploaded_at, reverse=True)
    return [_doc_out(d) for d in docs]


@router.post("/documents", response_model=DocumentOut, status_code=201)
def upload(file: UploadFile, store: Store = Depends(get_store), embed=Depends(get_embedder),
           s: Settings = Depends(get_settings)):
    ext = Path(file.filename or "").suffix.lower()
    if ext not in SUPPORTED:
        raise HTTPException(415, f"Unsupported file type '{ext}'. Use PDF, DOCX, TXT or MD.")

    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = Path(tmp.name)
    try:
        if tmp_path.stat().st_size > s.max_upload_mb * 1024 * 1024:
            raise HTTPException(413, f"File is larger than {s.max_upload_mb} MB")
        doc = ingest(tmp_path, file.filename, store, embed, s.chunk_size, s.chunk_overlap)
    except UnsupportedFile as e:
        raise HTTPException(415, str(e)) from e
    except ValueError as e:
        raise HTTPException(422, str(e)) from e
    finally:
        tmp_path.unlink(missing_ok=True)
    return _doc_out(doc)


@router.delete("/documents/{document_id}", status_code=204)
def delete_document(document_id: str, store: Store = Depends(get_store)):
    if not store.delete(document_id):
        raise HTTPException(404, "Document not found")


@router.post("/query", response_model=QueryOut)
def query(body: QueryIn, store: Store = Depends(get_store), llm: LLM = Depends(get_llm),
          embed=Depends(get_embedder), rerank=Depends(get_reranker), s: Settings = Depends(get_settings)):
    result = ask(body.question, store, llm, embed, rerank, s.top_k, s.rerank_top_n,
                 set(body.document_ids) if body.document_ids else None)
    citations = [
        Citation(document_id=h.record.document_id, filename=store.filename(h.record.document_id),
                 chunk=h.record.index, text=h.record.text, score=round(h.score, 4))
        for h in result.hits
    ]
    return QueryOut(answer=result.text, citations=citations, provider=result.provider,
                    latency_ms=round(result.latency_ms, 1))
