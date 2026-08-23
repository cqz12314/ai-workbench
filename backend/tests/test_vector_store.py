import chromadb

from app.services.embeddings import EmbeddingService
from app.services.vector_store import VectorChunk, VectorStore


def test_chroma_indexes_and_retrieves_relevant_chunks() -> None:
    store = VectorStore(
        chromadb.EphemeralClient(),
        EmbeddingService(dimensions=64),
        collection_name="retrieval_test_chunks",
    )
    chunks = [
        VectorChunk(
            id=1,
            document_id=10,
            chunk_index=0,
            filename="phones.md",
            content="苹果手机电池和充电使用指南",
        ),
        VectorChunk(
            id=2,
            document_id=11,
            chunk_index=0,
            filename="cooking.txt",
            content="番茄意大利面烹饪方法和配料",
        ),
    ]

    vector_ids = store.upsert_chunks(chunks)
    results = store.search("苹果手机如何充电", limit=2)

    assert vector_ids == ["chunk:1", "chunk:2"]
    assert results[0].chunk_id == 1
    assert results[0].filename == "phones.md"
    assert results[0].content == chunks[0].content


def test_chroma_deletes_vectors() -> None:
    store = VectorStore(
        chromadb.EphemeralClient(),
        EmbeddingService(dimensions=32),
        collection_name="deletion_test_chunks",
    )
    vector_ids = store.upsert_chunks(
        [VectorChunk(1, 1, 0, "notes.txt", "需要删除的文档内容")]
    )

    store.delete_vectors(vector_ids)

    assert store.search("文档", limit=5) == []
