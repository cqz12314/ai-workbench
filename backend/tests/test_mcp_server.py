import asyncio
import json
from unittest.mock import AsyncMock

import pytest
from mcp import Client
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app import mcp_server
from app.db.base import Base
from app.models import Conversation, Document, Message
from app.services.llm import CompletionResult
from app.services.rag import RAGRetrievalError
from app.services.vector_store import SearchResult


@pytest.fixture(autouse=True)
def isolated_mcp_database(monkeypatch):
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    testing_session = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    Base.metadata.create_all(engine)
    monkeypatch.setattr(mcp_server, "SessionLocal", testing_session)
    yield testing_session
    Base.metadata.drop_all(engine)
    engine.dispose()


async def call_tool(name: str, arguments: dict | None = None):
    async with Client(mcp_server.mcp) as client:
        return await client.call_tool(name, arguments)


def result_json(result) -> dict:
    assert not result.is_error
    return json.loads(result.content[0].text)


def result_text(result) -> str:
    return "\n".join(content.text for content in result.content if hasattr(content, "text"))


def test_mcp_exposes_only_read_only_ai_tools() -> None:
    async def list_registered_tools():
        async with Client(mcp_server.mcp) as client:
            result = await client.list_tools()
            return result.tools

    tools = asyncio.run(list_registered_tools())
    assert {tool.name for tool in tools} == {
        "search_knowledge",
        "list_documents",
        "ask_workbench",
    }
    assert all(tool.annotations.read_only_hint for tool in tools)
    assert all(tool.annotations.destructive_hint is False for tool in tools)


def test_search_knowledge_returns_source_information(monkeypatch) -> None:
    expected = SearchResult(
        chunk_id=12,
        document_id=4,
        chunk_index=2,
        filename="guide.md",
        content="知识库检索内容",
        distance=0.18,
    )
    def search(_query: str, _limit: int) -> list[SearchResult]:
        return [expected]

    monkeypatch.setattr(mcp_server, "retrieve_knowledge", search)

    result = asyncio.run(
        call_tool("search_knowledge", {"query": "知识库", "limit": 3})
    )

    assert result_json(result) == {
        "results": [
            {
                "chunk_id": 12,
                "document_id": 4,
                "chunk_index": 2,
                "filename": "guide.md",
                "content": "知识库检索内容",
                "distance": 0.18,
            }
        ]
    }


def test_list_documents_omits_filesystem_paths(isolated_mcp_database) -> None:
    with isolated_mcp_database() as session:
        session.add(
            Document(
                filename="private-notes.txt",
                file_type="txt",
                file_path="/secret/server/path/private-notes.txt",
            )
        )
        session.commit()

    result = asyncio.run(call_tool("list_documents"))
    payload = result_json(result)

    assert payload["documents"][0]["filename"] == "private-notes.txt"
    assert set(payload["documents"][0]) == {"id", "filename", "file_type", "created_at"}
    assert "/secret/server/path" not in json.dumps(payload)


def test_ask_workbench_reuses_rag_and_llm_without_chat_history(
    isolated_mcp_database, monkeypatch
) -> None:
    prepared_messages = [
        {"role": "system", "content": "retrieved context"},
        {"role": "user", "content": "问题"},
    ]
    def prepare(_messages, request_rag_enabled):
        assert request_rag_enabled is True
        return prepared_messages

    completion = AsyncMock(
        return_value=CompletionResult(content="工作台回答", model="test-model")
    )
    monkeypatch.setattr(mcp_server, "prepare_chat_messages", prepare)
    monkeypatch.setattr(mcp_server, "create_completion", completion)

    result = asyncio.run(call_tool("ask_workbench", {"question": "问题"}))

    assert result_json(result) == {
        "answer": "工作台回答",
        "model": "test-model",
        "used_rag": True,
    }
    completion.assert_awaited_once_with(prepared_messages)
    with isolated_mcp_database() as session:
        assert session.query(Conversation).count() == 0
        assert session.query(Message).count() == 0


def test_ask_workbench_supports_non_rag_mode(monkeypatch) -> None:
    completion = AsyncMock(return_value=CompletionResult(content="普通回答", model="test-model"))
    monkeypatch.setattr(mcp_server, "create_completion", completion)

    result = asyncio.run(
        call_tool("ask_workbench", {"question": "普通问题", "use_rag": False})
    )

    assert result_json(result)["used_rag"] is False
    completion.assert_awaited_once_with([{"role": "user", "content": "普通问题"}])


@pytest.mark.parametrize(
    ("tool_name", "arguments"),
    [
        ("search_knowledge", {"query": "   "}),
        ("search_knowledge", {"query": "valid", "limit": 21}),
        ("ask_workbench", {"question": "   "}),
    ],
)
def test_mcp_tool_parameter_validation(tool_name: str, arguments: dict) -> None:
    result = asyncio.run(call_tool(tool_name, arguments))

    assert result.is_error


def test_mcp_errors_do_not_expose_internal_details(monkeypatch) -> None:
    def fail_search(_query: str, _limit: int):
        raise RAGRetrievalError("/secret/chroma/path")

    monkeypatch.setattr(mcp_server, "retrieve_knowledge", fail_search)

    result = asyncio.run(call_tool("search_knowledge", {"query": "test"}))

    assert result.is_error
    assert "Knowledge-base search is unavailable" in result_text(result)
    assert "/secret/chroma/path" not in result_text(result)


def test_ask_workbench_hides_provider_errors(monkeypatch) -> None:
    completion = AsyncMock(side_effect=RuntimeError("DEEPSEEK_API_KEY=secret-value"))
    monkeypatch.setattr(mcp_server, "create_completion", completion)

    result = asyncio.run(
        call_tool("ask_workbench", {"question": "test", "use_rag": False})
    )

    assert result.is_error
    assert "AI provider request failed" in result_text(result)
    assert "secret-value" not in result_text(result)
