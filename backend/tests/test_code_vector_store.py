import chromadb
import pytest

from app.services.code_vector_store import CodeVectorChunk, CodeVectorStore
from app.services.embeddings import EmbeddingService
from app.services.vector_store import VectorChunk, VectorStore, VectorStoreError


def code_chunk(
    workspace_id: str,
    path: str,
    content: str,
    symbol_name: str = "login",
) -> CodeVectorChunk:
    return CodeVectorChunk(
        workspace_id=workspace_id,
        project="demo",
        relative_path=path,
        language="python",
        symbol_name=symbol_name,
        symbol_type="function",
        start_line=1,
        end_line=2,
        content=content,
        file_hash="abc",
        chunk_index=0,
    )


def test_code_collection_isolated_by_workspace_and_from_documents() -> None:
    client = chromadb.EphemeralClient()
    embeddings = EmbeddingService(dimensions=64)
    code_store = CodeVectorStore(client, embeddings)
    document_store = VectorStore(client, embeddings)
    first = code_chunk("workspace-a", "auth.py", "login authentication session")
    second = code_chunk("workspace-b", "auth.py", "cooking recipe tomato")
    code_store.upsert_chunks([first], code_store.embed_chunks([first]))
    code_store.upsert_chunks([second], code_store.embed_chunks([second]))
    document_store.upsert_chunks([VectorChunk(1, 1, 0, "guide.md", "document only")])

    results = code_store.search("workspace-a", "login", 10)

    assert [result.content for result in results] == [first.content]
    assert code_store.indexed_files("workspace-a") == {"auth.py": "abc"}
    code_store.delete_file("workspace-a", "auth.py")
    assert code_store.search("workspace-a", "login", 10) == []
    assert code_store.search("workspace-b", "recipe", 10)[0].content == second.content
    assert document_store.search("document", 5)[0].filename == "guide.md"


def test_public_code_search_limit_remains_ten() -> None:
    store = CodeVectorStore(chromadb.EphemeralClient(), EmbeddingService(dimensions=32))

    with pytest.raises(VectorStoreError, match="between 1 and 10"):
        store.search("workspace", "query", 11)


def test_hybrid_exact_symbol_recovers_outside_semantic_top_one_with_real_distance() -> None:
    client = chromadb.EphemeralClient()
    store = CodeVectorStore(client, EmbeddingService(dimensions=64), "hybrid_isolation")
    target = code_chunk(
        "workspace-a",
        "backend/app/services/code_index.py",
        "quiet implementation details with deliberately weak semantic content",
        "index_codebase",
    )
    semantic_top = code_chunk(
        "workspace-a",
        "backend/app/other.py",
        "index_codebase",
        "other",
    )
    foreign_target = code_chunk(
        "workspace-b",
        "backend/app/services/code_index.py",
        "index_codebase foreign workspace exact match",
        "index_codebase",
    )
    chunks = [target, semantic_top, foreign_target]
    store.upsert_chunks(chunks, store.embed_chunks(chunks))
    query_embedding = store.embedding_service.embed("index_codebase")

    semantic_candidates = store._query_candidates("workspace-a", query_embedding, limit=1)

    assert [result.content for result in semantic_candidates] == [semantic_top.content]
    assert target.content not in {result.content for result in semantic_candidates}

    candidates = store._hybrid_candidates(
        "workspace-a",
        "index_codebase",
        semantic_limit=1,
        exact_symbol="index_codebase",
    )

    targeted = [result for result in candidates if result.content == target.content]
    assert len(targeted) == 1
    assert foreign_target.content not in {result.content for result in candidates}
    assert isinstance(targeted[0].distance, float)
    assert 0.0 <= targeted[0].distance <= 2.0

    filtered = store.collection.query(
        query_embeddings=[query_embedding],
        n_results=1,
        where={
            "$and": [
                {"workspace_id": {"$eq": "workspace-a"}},
                {"symbol_name": {"$eq": "index_codebase"}},
            ]
        },
        include=["distances"],
    )
    assert targeted[0].distance == pytest.approx(filtered["distances"][0][0])


def test_hybrid_targeted_path_recovers_outside_semantic_top_one_and_is_isolated() -> None:
    store = CodeVectorStore(
        chromadb.EphemeralClient(),
        EmbeddingService(dimensions=64),
        "hybrid_path_isolation",
    )
    target = code_chunk(
        "workspace-a",
        "backend/app/services/agent.py",
        "quiet orchestration implementation with weak semantic content",
        "prepare_agent_response",
    )
    semantic_top = code_chunk(
        "workspace-a",
        "backend/app/other.py",
        "agent.py",
        "other",
    )
    foreign_target = code_chunk(
        "workspace-b",
        "backend/app/services/agent.py",
        "agent.py foreign workspace exact match",
        "foreign_agent",
    )
    chunks = [target, semantic_top, foreign_target]
    store.upsert_chunks(chunks, store.embed_chunks(chunks))
    query_embedding = store.embedding_service.embed("agent.py")

    semantic_candidates = store._query_candidates("workspace-a", query_embedding, limit=1)

    assert [result.content for result in semantic_candidates] == [semantic_top.content]
    assert target.content not in {result.content for result in semantic_candidates}

    candidates = store._hybrid_candidates(
        "workspace-a",
        "agent.py",
        semantic_limit=1,
        relative_paths=["backend/app/services/agent.py"],
    )

    assert target.content in {result.content for result in candidates}
    assert foreign_target.content not in {result.content for result in candidates}


def test_hybrid_candidate_retrieval_embeds_query_once() -> None:
    class CountingEmbeddings(EmbeddingService):
        def __init__(self):
            super().__init__(dimensions=32)
            self.query_calls = 0

        def embed(self, text: str) -> list[float]:
            self.query_calls += 1
            return super().embed(text)

    embeddings = CountingEmbeddings()
    store = CodeVectorStore(chromadb.EphemeralClient(), embeddings, "hybrid_embedding_once")
    chunk = code_chunk("workspace", "agent.py", "def get_auto_llm_tools(): pass")
    store.upsert_chunks([chunk], store.embed_chunks([chunk]))
    embeddings.query_calls = 0

    store._hybrid_candidates(
        "workspace",
        "agent.py",
        semantic_limit=100,
        relative_paths=["agent.py"],
    )

    assert embeddings.query_calls == 1
