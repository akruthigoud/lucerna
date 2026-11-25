from datetime import datetime

from pydantic import BaseModel, Field


class DocumentOut(BaseModel):
    id: str
    filename: str
    chunks: int
    characters: int
    uploaded_at: datetime


class Citation(BaseModel):
    document_id: str
    filename: str
    chunk: int
    text: str
    score: float


class QueryIn(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    document_ids: list[str] | None = None


class QueryOut(BaseModel):
    answer: str
    citations: list[Citation]
    provider: str
    latency_ms: float


class Health(BaseModel):
    status: str
    llm_provider: str
    documents: int
    chunks: int
