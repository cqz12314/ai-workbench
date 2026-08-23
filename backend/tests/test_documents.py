from types import SimpleNamespace

import pytest

from app.services.documents import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DocumentProcessingError,
    extract_text,
    process_document,
    split_text,
)


@pytest.mark.parametrize(("file_type", "suffix"), [("txt", ".txt"), ("markdown", ".md")])
def test_processes_utf8_text_documents(tmp_path, file_type: str, suffix: str) -> None:
    path = tmp_path / f"document{suffix}"
    path.write_text("标题\r\n\r\n正文内容", encoding="utf-8")

    assert process_document(path, file_type) == ["标题\n\n正文内容"]


def test_extracts_text_from_each_pdf_page(tmp_path, monkeypatch) -> None:
    pages = [
        SimpleNamespace(extract_text=lambda: "第一页"),
        SimpleNamespace(extract_text=lambda: None),
        SimpleNamespace(extract_text=lambda: "第三页"),
    ]
    monkeypatch.setattr(
        "app.services.documents.PdfReader",
        lambda _path: SimpleNamespace(pages=pages),
    )

    assert extract_text(tmp_path / "document.pdf", "pdf") == "第一页\n\n\n\n第三页"


def test_split_text_uses_bounded_overlapping_chunks() -> None:
    text = "段落内容。" * 500

    chunks = split_text(text)

    assert len(chunks) > 1
    assert all(len(chunk) <= CHUNK_SIZE for chunk in chunks)
    assert chunks[0][-CHUNK_OVERLAP:].strip() in chunks[1]


def test_rejects_document_without_extractable_text() -> None:
    with pytest.raises(DocumentProcessingError, match="no extractable text"):
        split_text(" \n\n ")
