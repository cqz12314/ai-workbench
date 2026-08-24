import chromadb

from app.services.code_vector_store import CodeVectorChunk, CodeVectorStore
from app.services.embeddings import EmbeddingService
from app.services.vector_store import VectorChunk, VectorStore


def code_chunk(workspace_id: str, path: str, content: str) -> CodeVectorChunk:
    return CodeVectorChunk(
        workspace_id=workspace_id,
        project="demo",
        relative_path=path,
        language="python",
        symbol_name="login",
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
