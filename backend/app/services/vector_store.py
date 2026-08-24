import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import chromadb
from chromadb.api import ClientAPI
from chromadb.config import Settings

from app.services.embeddings import EmbeddingService

CHROMA_DIRECTORY = Path(__file__).resolve().parents[2] / "data" / "chroma"
COLLECTION_NAME = "document_chunks"
logger = logging.getLogger(__name__)


class VectorStoreError(RuntimeError):
    """Raised when vector persistence or retrieval fails."""


@lru_cache
def get_chroma_client() -> ClientAPI:
    CHROMA_DIRECTORY.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(
        path=str(CHROMA_DIRECTORY),
        settings=Settings(anonymized_telemetry=False),
    )


@dataclass(frozen=True)
class VectorChunk:
    id: int
    document_id: int
    chunk_index: int
    filename: str
    content: str


@dataclass(frozen=True)
class SearchResult:
    chunk_id: int
    document_id: int
    chunk_index: int
    filename: str
    content: str
    distance: float


class VectorStore:
    def __init__(
        self,
        client: ClientAPI,
        embedding_service: EmbeddingService | None = None,
        collection_name: str = COLLECTION_NAME,
    ) -> None:
        self.embedding_service = embedding_service or EmbeddingService()
        self.collection = client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    @staticmethod
    def vector_id(chunk_id: int) -> str:
        return f"chunk:{chunk_id}"

    def upsert_chunks(self, chunks: list[VectorChunk]) -> list[str]:
        if not chunks:
            return []
        ids = [self.vector_id(chunk.id) for chunk in chunks]
        try:
            self.collection.upsert(
                ids=ids,
                embeddings=self.embedding_service.embed_many(
                    [chunk.content for chunk in chunks]
                ),
                documents=[chunk.content for chunk in chunks],
                metadatas=[
                    {
                        "chunk_id": chunk.id,
                        "document_id": chunk.document_id,
                        "chunk_index": chunk.chunk_index,
                        "filename": chunk.filename,
                    }
                    for chunk in chunks
                ],
            )
        except Exception as exc:
            try:
                self.collection.delete(ids=ids)
            except Exception:
                logger.exception("Failed to clean up partially indexed vectors")
            raise VectorStoreError("Failed to index document chunks") from exc
        return ids

    def delete_vectors(self, vector_ids: list[str]) -> None:
        if not vector_ids:
            return
        try:
            self.collection.delete(ids=vector_ids)
        except Exception as exc:
            raise VectorStoreError("Failed to remove document vectors") from exc

    def search(self, query: str, limit: int) -> list[SearchResult]:
        if self.collection.count() == 0:
            return []
        try:
            response: dict[str, Any] = self.collection.query(
                query_embeddings=[self.embedding_service.embed(query)],
                n_results=min(limit, self.collection.count()),
                include=["documents", "metadatas", "distances"],
            )
            documents = response.get("documents") or [[]]
            metadatas = response.get("metadatas") or [[]]
            distances = response.get("distances") or [[]]
            return [
                SearchResult(
                    chunk_id=int(metadata["chunk_id"]),
                    document_id=int(metadata["document_id"]),
                    chunk_index=int(metadata["chunk_index"]),
                    filename=str(metadata["filename"]),
                    content=str(content),
                    distance=float(distance),
                )
                for content, metadata, distance in zip(
                    documents[0], metadatas[0], distances[0], strict=True
                )
            ]
        except Exception as exc:
            raise VectorStoreError("Vector search failed") from exc


@lru_cache
def get_vector_store() -> VectorStore:
    return VectorStore(get_chroma_client())
