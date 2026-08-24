import hashlib
import logging
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from chromadb.api import ClientAPI

from app.services.embeddings import EmbeddingService
from app.services.vector_store import VectorStoreError, get_chroma_client

CODE_COLLECTION_NAME = "code_chunks"
MAX_PUBLIC_CODE_RESULTS = 10
MAX_HYBRID_VECTOR_CANDIDATES = 100
MAX_EXACT_SYMBOL_CANDIDATES = 20
MAX_TARGETED_PATH_CANDIDATES = 40
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CodeVectorChunk:
    workspace_id: str
    project: str
    relative_path: str
    language: str
    symbol_name: str
    symbol_type: str
    start_line: int
    end_line: int
    content: str
    file_hash: str
    chunk_index: int


@dataclass(frozen=True)
class CodeSearchResult:
    relative_path: str
    language: str
    symbol_name: str
    symbol_type: str
    start_line: int
    end_line: int
    content: str
    distance: float


class CodeVectorStore:
    def __init__(
        self,
        client: ClientAPI,
        embedding_service: EmbeddingService | None = None,
        collection_name: str = CODE_COLLECTION_NAME,
    ) -> None:
        self.embedding_service = embedding_service or EmbeddingService()
        self.collection = client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    @staticmethod
    def vector_id(chunk: CodeVectorChunk) -> str:
        identity = ":".join(
            (
                chunk.workspace_id,
                chunk.relative_path,
                chunk.file_hash,
                str(chunk.chunk_index),
            )
        )
        return f"code:{hashlib.sha256(identity.encode()).hexdigest()}"

    def embed_chunks(self, chunks: list[CodeVectorChunk]) -> list[list[float]]:
        if not chunks:
            return []
        try:
            return self.embedding_service.embed_many([chunk.content for chunk in chunks])
        except Exception as exc:
            raise VectorStoreError("Failed to embed code chunks") from exc

    def upsert_chunks(
        self, chunks: list[CodeVectorChunk], embeddings: list[list[float]]
    ) -> list[str]:
        if not chunks:
            return []
        if len(chunks) != len(embeddings):
            raise VectorStoreError("Code chunk embeddings are incomplete")
        ids = [self.vector_id(chunk) for chunk in chunks]
        try:
            self.collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=[chunk.content for chunk in chunks],
                metadatas=[
                    {
                        "workspace_id": chunk.workspace_id,
                        "project": chunk.project,
                        "relative_path": chunk.relative_path,
                        "language": chunk.language,
                        "symbol_name": chunk.symbol_name,
                        "symbol_type": chunk.symbol_type,
                        "start_line": chunk.start_line,
                        "end_line": chunk.end_line,
                        "file_hash": chunk.file_hash,
                        "chunk_index": chunk.chunk_index,
                    }
                    for chunk in chunks
                ],
            )
        except Exception as exc:
            raise VectorStoreError("Failed to index code chunks") from exc
        return ids

    def indexed_files(self, workspace_id: str) -> dict[str, str]:
        try:
            response = self.collection.get(
                where={"workspace_id": workspace_id}, include=["metadatas"]
            )
        except Exception as exc:
            raise VectorStoreError("Failed to inspect code index") from exc
        files: dict[str, str] = {}
        for metadata in response.get("metadatas") or []:
            if metadata:
                files[str(metadata["relative_path"])] = str(metadata["file_hash"])
        return files

    @staticmethod
    def _file_filter(workspace_id: str, relative_path: str) -> dict[str, Any]:
        return {
            "$and": [
                {"workspace_id": {"$eq": workspace_id}},
                {"relative_path": {"$eq": relative_path}},
            ]
        }

    def delete_file(self, workspace_id: str, relative_path: str) -> None:
        try:
            self.collection.delete(where=self._file_filter(workspace_id, relative_path))
        except Exception as exc:
            raise VectorStoreError("Failed to remove code vectors") from exc

    def replace_file(
        self,
        workspace_id: str,
        relative_path: str,
        chunks: list[CodeVectorChunk],
        embeddings: list[list[float]],
    ) -> list[str]:
        self.delete_file(workspace_id, relative_path)
        return self.upsert_chunks(chunks, embeddings)

    def search(self, workspace_id: str, query: str, limit: int) -> list[CodeSearchResult]:
        if (
            isinstance(limit, bool)
            or not isinstance(limit, int)
            or not 1 <= limit <= MAX_PUBLIC_CODE_RESULTS
        ):
            raise VectorStoreError("Code search limit must be between 1 and 10")
        try:
            embedding = self.embedding_service.embed(query)
            return self._query_candidates(workspace_id, embedding, limit)
        except Exception as exc:
            if isinstance(exc, VectorStoreError):
                raise
            raise VectorStoreError("Code search failed") from exc

    def _hybrid_candidates(
        self,
        workspace_id: str,
        query: str,
        *,
        semantic_limit: int,
        exact_symbol: str | None = None,
        relative_paths: list[str] | None = None,
    ) -> list[CodeSearchResult]:
        """Retrieve bounded hybrid candidates without changing the public search contract."""
        if not 1 <= semantic_limit <= MAX_HYBRID_VECTOR_CANDIDATES:
            raise VectorStoreError("Hybrid candidate limit is invalid")
        paths = list(dict.fromkeys(relative_paths or []))[:MAX_TARGETED_PATH_CANDIDATES]
        try:
            embedding = self.embedding_service.embed(query)
            candidates = self._query_candidates(workspace_id, embedding, semantic_limit)
            if exact_symbol:
                candidates.extend(
                    self._query_candidates(
                        workspace_id,
                        embedding,
                        MAX_EXACT_SYMBOL_CANDIDATES,
                        metadata_filter={"symbol_name": {"$eq": exact_symbol}},
                    )
                )
            remaining_path_candidates = MAX_TARGETED_PATH_CANDIDATES
            for index, relative_path in enumerate(paths):
                paths_left = len(paths) - index
                path_limit = max(1, remaining_path_candidates // paths_left)
                path_results = self._query_candidates(
                    workspace_id,
                    embedding,
                    path_limit,
                    metadata_filter={"relative_path": {"$eq": relative_path}},
                )
                candidates.extend(path_results)
                remaining_path_candidates -= len(path_results)
                if remaining_path_candidates <= 0:
                    break
            return candidates
        except Exception as exc:
            if isinstance(exc, VectorStoreError):
                raise
            raise VectorStoreError("Hybrid code candidate retrieval failed") from exc

    def _query_candidates(
        self,
        workspace_id: str,
        embedding: list[float],
        limit: int,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[CodeSearchResult]:
        count = self.collection.count()
        if count == 0 or limit <= 0:
            return []
        filters: list[dict[str, Any]] = [{"workspace_id": {"$eq": workspace_id}}]
        if metadata_filter:
            filters.append(metadata_filter)
        where: dict[str, Any] = filters[0] if len(filters) == 1 else {"$and": filters}
        response: dict[str, Any] = self.collection.query(
            query_embeddings=[embedding],
            n_results=min(limit, count),
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        return self._search_results(response)

    @staticmethod
    def _search_results(response: dict[str, Any]) -> list[CodeSearchResult]:
        documents = response.get("documents") or [[]]
        metadatas = response.get("metadatas") or [[]]
        distances = response.get("distances") or [[]]
        return [
            CodeSearchResult(
                relative_path=str(metadata["relative_path"]),
                language=str(metadata["language"]),
                symbol_name=str(metadata["symbol_name"]),
                symbol_type=str(metadata["symbol_type"]),
                start_line=int(metadata["start_line"]),
                end_line=int(metadata["end_line"]),
                content=str(content),
                distance=float(distance),
            )
            for content, metadata, distance in zip(
                documents[0], metadatas[0], distances[0], strict=True
            )
        ]


@lru_cache
def get_code_vector_store() -> CodeVectorStore:
    return CodeVectorStore(get_chroma_client())
