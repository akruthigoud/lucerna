from pathlib import Path

import pytest

from app.parsing import UnsupportedFile, extract_text


def test_txt(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("plain text")
    assert extract_text(f) == "plain text"


def test_markdown(tmp_path):
    f = tmp_path / "a.md"
    f.write_text("# Title\n\nBody")
    assert "Body" in extract_text(f)


def test_docx(tmp_path):
    docx = pytest.importorskip("docx")
    f = tmp_path / "a.docx"
    d = docx.Document()
    d.add_paragraph("First paragraph.")
    d.add_paragraph("Second paragraph.")
    d.save(f)
    text = extract_text(f)
    assert "First paragraph." in text and "Second paragraph." in text


def test_unsupported():
    with pytest.raises(UnsupportedFile):
        extract_text(Path("file.exe"))
