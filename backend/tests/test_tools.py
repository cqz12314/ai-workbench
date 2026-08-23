import asyncio
from datetime import UTC, datetime

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.db.base import Base
from app.models import Document
from app.services import tools
from app.services.tools import ToolArgumentsError, ToolNotFoundError
from app.services.vector_store import SearchResult


def test_registry_exposes_only_expected_read_only_tools() -> None:
    assert set(tools.TOOL_REGISTRY) == {
        "search_knowledge",
        "list_documents",
        "github_list_repositories",
        "github_get_file",
        "github_search_code",
        "github_list_issues",
        "analyze_repository",
        "review_repository",
        "read_file",
        "propose_change",
        "apply_change",
        "git_status",
        "git_diff",
        "git_create_branch",
        "git_commit",
        "github_create_branch",
        "github_push_branch",
        "github_create_pull_request",
        "task_plan",
        "run_task",
    }
    schemas = tools.get_llm_tools()
    assert {schema["function"]["name"] for schema in schemas} == set(tools.TOOL_REGISTRY)
    assert not any(
        schema["function"]["name"] in {"push", "commit", "delete", "create_repository", "shell"}
        for schema in schemas
    )


def test_search_tool_validates_and_executes(monkeypatch) -> None:
    expected = SearchResult(
        chunk_id=1,
        document_id=2,
        chunk_index=3,
        filename="guide.md",
        content="Agent knowledge",
        distance=0.2,
    )
    calls = []

    def search(query: str, limit: int):
        calls.append((query, limit))
        return [expected]

    monkeypatch.setattr(tools, "search_knowledge", search)
    result = asyncio.run(
        tools.execute_tool("search_knowledge", '{"query": "  Agent  ", "limit": 3}')
    )

    assert calls == [("Agent", 3)]
    assert result["results"][0]["filename"] == "guide.md"
    assert result["results"][0]["content"] == "Agent knowledge"


def test_list_documents_never_returns_file_path(monkeypatch) -> None:
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    testing_session = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    Base.metadata.create_all(engine)
    with testing_session() as session:
        session.add(
            Document(
                filename="safe.pdf",
                file_type="pdf",
                file_path="/private/uploads/safe.pdf",
                created_at=datetime.now(UTC),
            )
        )
        session.commit()
    monkeypatch.setattr(tools, "SessionLocal", testing_session)

    result = asyncio.run(tools.execute_tool("list_documents", {}))

    assert result["documents"][0]["filename"] == "safe.pdf"
    assert "file_path" not in result["documents"][0]
    engine.dispose()


@pytest.mark.parametrize(
    ("name", "arguments", "error"),
    [
        ("shell", {}, ToolNotFoundError),
        ("search_knowledge", "not-json", ToolArgumentsError),
        ("search_knowledge", {"query": "   "}, ToolArgumentsError),
        ("list_documents", {"path": "/tmp"}, ToolArgumentsError),
    ],
)
def test_tool_rejects_unknown_tools_and_invalid_arguments(
    name: str, arguments: str | dict, error: type[Exception]
) -> None:
    with pytest.raises(error):
        asyncio.run(tools.execute_tool(name, arguments))
