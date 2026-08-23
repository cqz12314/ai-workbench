import asyncio

import pytest
from httpx import ASGITransport, AsyncClient, Response
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.db.base import Base
from app.db.session import get_db
from app.main import app
from app.models import Document, DocumentChunk
from app.services.vector_store import VectorStoreError


class FakeVectorStore:
    def __init__(self) -> None:
        self.indexed_chunks = []
        self.deleted_ids: list[str] = []

    def upsert_chunks(self, chunks):
        self.indexed_chunks.extend(chunks)
        return [f"chunk:{chunk.id}" for chunk in chunks]

    def delete_vectors(self, vector_ids: list[str]) -> None:
        self.deleted_ids.extend(vector_ids)


@pytest.fixture(autouse=True)
def fake_vector_store(monkeypatch):
    store = FakeVectorStore()
    monkeypatch.setattr("app.api.routes.files.get_vector_store", lambda: store)
    return store


@pytest.fixture(autouse=True)
def isolated_file_storage(tmp_path, monkeypatch):
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    testing_session = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    Base.metadata.create_all(engine)
    upload_directory = tmp_path / "uploads"
    monkeypatch.setattr("app.api.routes.files.UPLOAD_DIRECTORY", upload_directory)

    async def override_get_db():
        with testing_session() as session:
            yield session

    app.dependency_overrides[get_db] = override_get_db
    yield testing_session, upload_directory
    app.dependency_overrides.clear()
    Base.metadata.drop_all(engine)
    engine.dispose()


async def upload(filename: str, content: bytes, content_type: str) -> Response:
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        return await client.post(
            "/api/v1/files/upload",
            files={"file": (filename, content, content_type)},
        )


async def list_files() -> Response:
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        return await client.get("/api/v1/files")


@pytest.mark.parametrize(
    ("filename", "content_type", "expected_type"),
    [
        ("guide.pdf", "application/pdf", "pdf"),
        ("notes.txt", "text/plain", "txt"),
        ("README.md", "text/markdown", "markdown"),
    ],
)
def test_upload_supported_files(
    filename: str,
    content_type: str,
    expected_type: str,
    isolated_file_storage,
    monkeypatch,
) -> None:
    testing_session, upload_directory = isolated_file_storage
    monkeypatch.setattr(
        "app.api.routes.files.process_document", lambda _path, _file_type: ["document content"]
    )

    response = asyncio.run(upload(filename, b"document content", content_type))

    assert response.status_code == 201
    payload = response.json()
    assert payload["filename"] == filename
    assert payload["file_type"] == expected_type
    stored_files = list(upload_directory.iterdir())
    assert len(stored_files) == 1
    assert stored_files[0].read_bytes() == b"document content"
    assert stored_files[0].suffix == f".{filename.rsplit('.', 1)[-1].lower()}"
    with testing_session() as session:
        assert session.query(Document).count() == 1
        assert session.query(DocumentChunk).count() == 1


def test_file_list_returns_newest_first(isolated_file_storage) -> None:
    first = asyncio.run(upload("first.txt", b"first", "text/plain"))
    second = asyncio.run(upload("second.md", b"second", "text/markdown"))

    response = asyncio.run(list_files())

    assert response.status_code == 200
    assert [document["id"] for document in response.json()] == [
        second.json()["id"],
        first.json()["id"],
    ]


@pytest.mark.parametrize(
    ("filename", "content_type"),
    [
        ("malware.exe", "application/octet-stream"),
        ("fake.pdf", "text/plain"),
    ],
)
def test_upload_rejects_unsupported_files(
    filename: str,
    content_type: str,
    isolated_file_storage,
) -> None:
    testing_session, upload_directory = isolated_file_storage

    response = asyncio.run(upload(filename, b"not allowed", content_type))

    assert response.status_code == 415
    assert not upload_directory.exists()
    with testing_session() as session:
        assert session.query(Document).count() == 0


def test_upload_sanitizes_filename(isolated_file_storage) -> None:
    response = asyncio.run(upload("../../notes.txt", b"safe", "text/plain"))

    assert response.status_code == 201
    assert response.json()["filename"] == "notes.txt"


def test_upload_limit_removes_partial_file(isolated_file_storage, monkeypatch) -> None:
    testing_session, upload_directory = isolated_file_storage
    monkeypatch.setattr("app.api.routes.files.MAX_UPLOAD_BYTES", 4)

    response = asyncio.run(upload("large.txt", b"12345", "text/plain"))

    assert response.status_code == 413
    assert list(upload_directory.iterdir()) == []
    with testing_session() as session:
        assert session.query(Document).count() == 0


def test_text_upload_creates_ordered_chunks(
    isolated_file_storage, fake_vector_store
) -> None:
    testing_session, _ = isolated_file_storage
    content = ("第一段文档内容。" * 180).encode()

    response = asyncio.run(upload("long.txt", content, "text/plain"))

    assert response.status_code == 201
    document_id = response.json()["id"]
    with testing_session() as session:
        chunks = (
            session.query(DocumentChunk)
            .filter_by(document_id=document_id)
            .order_by(DocumentChunk.chunk_index)
            .all()
        )
        assert len(chunks) > 1
        assert [chunk.chunk_index for chunk in chunks] == list(range(len(chunks)))
        assert all(chunk.content for chunk in chunks)
    assert [chunk.chunk_index for chunk in fake_vector_store.indexed_chunks] == list(
        range(len(fake_vector_store.indexed_chunks))
    )


def test_invalid_text_rolls_back_file_and_database(isolated_file_storage) -> None:
    testing_session, upload_directory = isolated_file_storage

    response = asyncio.run(upload("invalid.txt", b"\xff\xfe\xfa", "text/plain"))

    assert response.status_code == 422
    assert list(upload_directory.iterdir()) == []
    with testing_session() as session:
        assert session.query(Document).count() == 0
        assert session.query(DocumentChunk).count() == 0


def test_vector_failure_rolls_back_file_and_database(
    isolated_file_storage, fake_vector_store, monkeypatch
) -> None:
    testing_session, upload_directory = isolated_file_storage

    def fail_index(_chunks):
        raise VectorStoreError("chroma unavailable")

    monkeypatch.setattr(fake_vector_store, "upsert_chunks", fail_index)

    response = asyncio.run(upload("notes.txt", b"searchable content", "text/plain"))

    assert response.status_code == 503
    assert list(upload_directory.iterdir()) == []
    with testing_session() as session:
        assert session.query(Document).count() == 0
        assert session.query(DocumentChunk).count() == 0
