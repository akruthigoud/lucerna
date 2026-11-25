from pathlib import Path

SUPPORTED = {".pdf", ".docx", ".txt", ".md"}


class UnsupportedFile(ValueError):
    pass


def extract_text(path: Path) -> str:
    ext = path.suffix.lower()
    if ext == ".pdf":
        from pypdf import PdfReader

        return "\n\n".join(page.extract_text() or "" for page in PdfReader(str(path)).pages)
    if ext == ".docx":
        from docx import Document

        return "\n\n".join(p.text for p in Document(str(path)).paragraphs if p.text.strip())
    if ext in (".txt", ".md"):
        return path.read_text(encoding="utf-8", errors="ignore")
    raise UnsupportedFile(f"Unsupported file type: {ext}")
