from pathlib import Path

from pypdf import PdfReader

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
MIN_BOUNDARY_POSITION = 500
CHUNK_BOUNDARIES = ("\n\n", "\n", "。", "！", "？", ". ", "! ", "? ")


class DocumentProcessingError(RuntimeError):
    """Raised when an uploaded document cannot produce usable text."""


def extract_text(path: Path, file_type: str) -> str:
    try:
        if file_type in {"txt", "markdown"}:
            return path.read_text(encoding="utf-8-sig")
        if file_type == "pdf":
            reader = PdfReader(path)
            return "\n\n".join(page.extract_text() or "" for page in reader.pages)
    except (OSError, UnicodeError, ValueError) as exc:
        raise DocumentProcessingError("Document text extraction failed") from exc
    except Exception as exc:
        if file_type == "pdf":
            raise DocumentProcessingError("PDF text extraction failed") from exc
        raise
    raise DocumentProcessingError("Unsupported document type")


def normalize_text(text: str) -> str:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip() for line in normalized.split("\n")]
    return "\n".join(lines).strip()


def find_chunk_end(text: str, start: int, maximum_end: int) -> int:
    minimum_end = min(start + MIN_BOUNDARY_POSITION, maximum_end)
    best_end = -1
    for boundary in CHUNK_BOUNDARIES:
        position = text.rfind(boundary, minimum_end, maximum_end)
        if position >= 0:
            best_end = max(best_end, position + len(boundary))
    return best_end if best_end > start else maximum_end


def split_text(text: str) -> list[str]:
    text = normalize_text(text)
    if not text:
        raise DocumentProcessingError("Document contains no extractable text")

    chunks: list[str] = []
    start = 0
    while start < len(text):
        maximum_end = min(start + CHUNK_SIZE, len(text))
        end = (
            find_chunk_end(text, start, maximum_end)
            if maximum_end < len(text)
            else maximum_end
        )
        content = text[start:end].strip()
        if content:
            chunks.append(content)
        if end >= len(text):
            break
        start = max(end - CHUNK_OVERLAP, start + 1)
    return chunks


def process_document(path: Path, file_type: str) -> list[str]:
    return split_text(extract_text(path, file_type))
