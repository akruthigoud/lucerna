from abc import ABC, abstractmethod
from functools import lru_cache

from app.config import Settings, get_settings

INSTRUCTIONS = (
    "Answer the question using only the context below. "
    "If the context does not contain the answer, say you could not find it in the documents."
)


class LLM(ABC):
    name: str

    @abstractmethod
    def answer(self, question: str, context: str) -> str: ...


class LocalLLM(LLM):
    def __init__(self, model: str):
        from transformers import pipeline

        self.name = f"local:{model.split('/')[-1]}"
        self._pipe = pipeline("text2text-generation", model=model, max_new_tokens=256)

    def answer(self, question: str, context: str) -> str:
        prompt = f"{INSTRUCTIONS}\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"
        return self._pipe(prompt, do_sample=False)[0]["generated_text"].strip()


class ChatLLM(LLM):
    """OpenAI-compatible chat API (used for OpenAI and Groq)."""

    def __init__(self, client, model: str, label: str):
        self._client, self._model = client, model
        self.name = f"{label}:{model}"

    def answer(self, question: str, context: str) -> str:
        resp = self._client.chat.completions.create(
            model=self._model,
            temperature=0.2,
            messages=[
                {"role": "system", "content": INSTRUCTIONS},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
            ],
        )
        return resp.choices[0].message.content.strip()


def build_llm(s: Settings) -> LLM:
    if s.llm_provider == "openai":
        if not s.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is not set")
        from openai import OpenAI

        return ChatLLM(OpenAI(api_key=s.openai_api_key), s.openai_model, "openai")
    if s.llm_provider == "groq":
        if not s.groq_api_key:
            raise RuntimeError("GROQ_API_KEY is not set")
        from groq import Groq

        return ChatLLM(Groq(api_key=s.groq_api_key), s.groq_model, "groq")
    return LocalLLM(s.local_model)


@lru_cache
def get_llm() -> LLM:
    return build_llm(get_settings())
