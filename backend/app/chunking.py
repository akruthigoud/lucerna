import re
from dataclasses import dataclass

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")


@dataclass
class Chunk:
    index: int
    text: str


def split_sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    return _SENTENCE_END.split(text) if text else []


def chunk_text(text: str, size: int = 800, overlap: int = 120) -> list[Chunk]:
    """Pack whole sentences into chunks of about `size` characters.

    The last sentences of each chunk (up to `overlap` characters) are repeated
    at the start of the next one.
    """
    chunks: list[Chunk] = []
    current: list[str] = []
    length = 0

    for sentence in split_sentences(text):
        if current and length + len(sentence) + 1 > size:
            chunks.append(Chunk(len(chunks), " ".join(current)))
            tail: list[str] = []
            tail_len = 0
            for s in reversed(current):
                if tail_len + len(s) + 1 > overlap:
                    break
                tail.insert(0, s)
                tail_len += len(s) + 1
            current, length = tail, tail_len
        current.append(sentence)
        length += len(sentence) + 1

    if current:
        chunks.append(Chunk(len(chunks), " ".join(current)))
    return chunks
