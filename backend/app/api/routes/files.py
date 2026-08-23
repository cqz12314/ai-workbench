import logging
from pathlib import Path
from typing import Annotated
from uuid import uuid4

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.models import Document, DocumentChunk
from app.schemas.files import DocumentResponse
from app.services.documents import DocumentProcessingError, process_document
from app.services.vector_store import (
    VectorChunk,
    VectorStoreError,
    get_vector_store,
)

router = APIRouter(prefix="/files")
logger = logging.getLogger(__name__)
DbSession = Annotated[Session, Depends(get_db)]

UPLOAD_DIRECTORY = Path(__file__).resolve().parents[3] / "data" / "uploads"
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
UPLOAD_CHUNK_BYTES = 1024 * 1024
SUPPORTED_FILES = {
    ".pdf": ("pdf", {"application/pdf"}),
    ".txt": ("txt", {"text/plain"}),
    ".md": ("markdown", {"text/markdown", "text/plain", "text/x-markdown"}),
    ".markdown": ("markdown", {"text/markdown", "text/plain", "text/x-markdown"}),
}


def validate_upload(file: UploadFile) -> tuple[str, str, str]:
    supplied_name = file.filename or ""
    filename = supplied_name.replace("\\", "/").rsplit("/", 1)[-1].strip()
    if not filename or len(filename) > 255:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="A valid filename is required",
        )

    extension = Path(filename).suffix.lower()
    supported = SUPPORTED_FILES.get(extension)
    if supported is None:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Only PDF, TXT, and Markdown files are supported",
        )

    file_type, allowed_content_types = supported
    content_type = (file.content_type or "").lower()
    if content_type not in allowed_content_types:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="The file content type does not match its extension",
        )
    return filename, extension, file_type


@router.post(
    "/upload",
    response_model=DocumentResponse,
    status_code=status.HTTP_201_CREATED,
)
async def upload_file(
    file: Annotated[UploadFile, File(description="PDF, TXT, or Markdown file")],
    db: DbSession,
) -> Document:
    filename, extension, file_type = validate_upload(file)
    UPLOAD_DIRECTORY.mkdir(parents=True, exist_ok=True)
    stored_filename = f"{uuid4().hex}{extension}"
    destination = UPLOAD_DIRECTORY / stored_filename
    temporary = destination.with_suffix(f"{destination.suffix}.part")
    size = 0

    try:
        with temporary.open("xb") as output:
            while chunk := await file.read(UPLOAD_CHUNK_BYTES):
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    raise HTTPException(
                        status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                        detail="File exceeds the 20 MiB upload limit",
                    )
                output.write(chunk)
        temporary.replace(destination)
    except HTTPException:
        temporary.unlink(missing_ok=True)
        raise
    except OSError as exc:
        temporary.unlink(missing_ok=True)
        logger.exception("Failed to store uploaded file")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="File storage is unavailable",
        ) from exc
    finally:
        await file.close()

    try:
        chunks = process_document(destination, file_type)
    except DocumentProcessingError as exc:
        destination.unlink(missing_ok=True)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(exc),
        ) from exc

    relative_path = Path("data") / "uploads" / stored_filename
    document = Document(
        filename=filename,
        file_type=file_type,
        file_path=relative_path.as_posix(),
    )
    document.chunks.extend(
        DocumentChunk(chunk_index=index, content=content)
        for index, content in enumerate(chunks)
    )
    db.add(document)
    vector_ids: list[str] = []
    try:
        db.flush()
        vector_ids = get_vector_store().upsert_chunks(
            [
                VectorChunk(
                    id=chunk.id,
                    document_id=document.id,
                    chunk_index=chunk.chunk_index,
                    filename=document.filename,
                    content=chunk.content,
                )
                for chunk in document.chunks
            ]
        )
        db.commit()
        db.refresh(document)
    except VectorStoreError as exc:
        db.rollback()
        destination.unlink(missing_ok=True)
        logger.exception("Failed to index uploaded document")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Document vector storage is unavailable",
        ) from exc
    except SQLAlchemyError as exc:
        db.rollback()
        destination.unlink(missing_ok=True)
        if vector_ids:
            try:
                get_vector_store().delete_vectors(vector_ids)
            except VectorStoreError:
                logger.exception("Failed to compensate document vectors")
        logger.exception("Failed to persist uploaded document")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Document storage is unavailable",
        ) from exc
    return document


@router.get("", response_model=list[DocumentResponse])
async def list_files(db: DbSession) -> list[Document]:
    return list(
        db.scalars(select(Document).order_by(Document.created_at.desc(), Document.id.desc()))
    )
