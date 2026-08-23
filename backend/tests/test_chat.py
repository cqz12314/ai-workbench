import asyncio
import json
from unittest.mock import AsyncMock

import pytest
from httpx import ASGITransport, AsyncClient, Response
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.core.config import settings
from app.db.base import Base
from app.db.session import get_db
from app.main import app
from app.models import Conversation, Message
from app.services.llm import CompletionResult, LLMNotConfiguredError, StreamingCompletion
from app.services.vector_store import SearchResult


@pytest.fixture(autouse=True)
def isolated_database():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    testing_session = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    Base.metadata.create_all(engine)

    async def override_get_db():
        with testing_session() as session:
            yield session

    app.dependency_overrides[get_db] = override_get_db
    yield testing_session
    app.dependency_overrides.clear()
    Base.metadata.drop_all(engine)
    engine.dispose()


async def post_chat(payload: dict) -> Response:
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        return await client.post("/api/v1/chat", json=payload)


async def request(method: str, path: str, payload: dict | None = None) -> Response:
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        return await client.request(method, path, json=payload)


def test_chat_returns_assistant_message(monkeypatch) -> None:
    completion = AsyncMock(
        return_value=CompletionResult(content="你好！有什么可以帮你？", model="test-model")
    )
    monkeypatch.setattr("app.api.routes.chat.create_completion", completion)

    response = asyncio.run(post_chat({"messages": [{"role": "user", "content": "你好"}]}))

    assert response.status_code == 200
    assert response.json() == {
        "message": {"role": "assistant", "content": "你好！有什么可以帮你？"},
        "model": "test-model",
        "conversation_id": 1,
    }
    completion.assert_awaited_once_with([{"role": "user", "content": "你好"}])


def test_chat_uses_agent_when_enabled(monkeypatch) -> None:
    agent_completion = AsyncMock(
        return_value=CompletionResult(content="Agent 回复", model="agent-model")
    )
    direct_completion = AsyncMock()
    monkeypatch.setattr(settings, "agent_enabled", True)
    monkeypatch.setattr("app.api.routes.chat.create_agent_completion", agent_completion)
    monkeypatch.setattr("app.api.routes.chat.create_completion", direct_completion)

    response = asyncio.run(post_chat({"messages": [{"role": "user", "content": "查文档"}]}))

    assert response.status_code == 200
    assert response.json()["message"]["content"] == "Agent 回复"
    agent_completion.assert_awaited_once_with([{"role": "user", "content": "查文档"}])
    direct_completion.assert_not_awaited()


def test_chat_rejects_empty_messages() -> None:
    response = asyncio.run(post_chat({"messages": []}))

    assert response.status_code == 422


def test_chat_reports_missing_model(monkeypatch) -> None:
    completion = AsyncMock(side_effect=LLMNotConfiguredError())
    monkeypatch.setattr("app.api.routes.chat.create_completion", completion)

    response = asyncio.run(post_chat({"messages": [{"role": "user", "content": "你好"}]}))

    assert response.status_code == 503
    assert response.json() == {"detail": "AI model is not configured"}
    history = asyncio.run(get_latest_conversation())
    assert history.json()["messages"] == [{"role": "user", "content": "你好"}]


def test_chat_hides_provider_error_details(monkeypatch) -> None:
    completion = AsyncMock(side_effect=RuntimeError("provider rejected secret-key-value"))
    monkeypatch.setattr("app.api.routes.chat.create_completion", completion)

    response = asyncio.run(post_chat({"messages": [{"role": "user", "content": "你好"}]}))

    assert response.status_code == 502
    assert response.json() == {"detail": "AI provider request failed"}
    assert "secret-key-value" not in response.text


def test_chat_persists_and_loads_conversation(monkeypatch, isolated_database) -> None:
    completion = AsyncMock(
        return_value=CompletionResult(content="第一条回复", model="test-model")
    )
    monkeypatch.setattr("app.api.routes.chat.create_completion", completion)

    first = asyncio.run(post_chat({"messages": [{"role": "user", "content": "第一条"}]}))
    conversation_id = first.json()["conversation_id"]
    completion.return_value = CompletionResult(content="第二条回复", model="test-model")
    second = asyncio.run(
        post_chat(
            {
                "conversation_id": conversation_id,
                "messages": [
                    {"role": "user", "content": "第一条"},
                    {"role": "assistant", "content": "第一条回复"},
                    {"role": "user", "content": "第二条"},
                ],
            }
        )
    )

    assert second.status_code == 200
    history = asyncio.run(get_latest_conversation())
    assert history.status_code == 200
    assert history.json()["id"] == conversation_id
    assert history.json()["messages"] == [
        {"role": "user", "content": "第一条"},
        {"role": "assistant", "content": "第一条回复"},
        {"role": "user", "content": "第二条"},
        {"role": "assistant", "content": "第二条回复"},
    ]

    with isolated_database() as session:
        assert session.query(Conversation).count() == 1


def test_conversation_crud_preserves_other_conversations(isolated_database) -> None:
    first = asyncio.run(request("POST", "/api/v1/conversations", {"title": "第一段对话"}))
    second = asyncio.run(request("POST", "/api/v1/conversations", {"title": "第二段对话"}))

    assert first.status_code == 201
    assert first.json()["messages"] == []
    first_id = first.json()["id"]
    second_id = second.json()["id"]

    renamed = asyncio.run(
        request("PATCH", f"/api/v1/conversations/{first_id}", {"title": "  新标题  "})
    )
    assert renamed.status_code == 200
    assert renamed.json()["title"] == "新标题"

    listing = asyncio.run(request("GET", "/api/v1/conversations"))
    assert listing.status_code == 200
    assert {item["id"] for item in listing.json()} == {first_id, second_id}

    detail = asyncio.run(request("GET", f"/api/v1/conversations/{first_id}"))
    assert detail.status_code == 200
    assert detail.json()["title"] == "新标题"

    deleted = asyncio.run(request("DELETE", f"/api/v1/conversations/{first_id}"))
    assert deleted.status_code == 204
    assert asyncio.run(request("GET", f"/api/v1/conversations/{first_id}")).status_code == 404
    assert asyncio.run(request("GET", f"/api/v1/conversations/{second_id}")).status_code == 200

    with isolated_database() as session:
        assert session.query(Conversation).count() == 1


def test_delete_conversation_deletes_messages(isolated_database) -> None:
    with isolated_database() as session:
        conversation = Conversation(title="待删除")
        conversation.messages.append(Message(role="user", content="测试消息"))
        session.add(conversation)
        session.commit()
        conversation_id = conversation.id

    response = asyncio.run(request("DELETE", f"/api/v1/conversations/{conversation_id}"))

    assert response.status_code == 204
    with isolated_database() as session:
        assert session.query(Message).count() == 0


@pytest.mark.parametrize("title", ["", "   ", "x" * 201])
def test_conversation_title_validation(title: str) -> None:
    response = asyncio.run(request("POST", "/api/v1/conversations", {"title": title}))

    assert response.status_code == 422


def test_first_user_message_generates_conversation_title(monkeypatch) -> None:
    completion = AsyncMock(return_value=CompletionResult(content="回复", model="test-model"))
    monkeypatch.setattr("app.api.routes.chat.create_completion", completion)

    response = asyncio.run(post_chat({"messages": [{"role": "user", "content": "标题来源"}]}))
    conversation_id = response.json()["conversation_id"]
    detail = asyncio.run(request("GET", f"/api/v1/conversations/{conversation_id}"))

    assert detail.json()["title"] == "标题来源"


async def get_latest_conversation() -> Response:
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        return await client.get("/api/v1/conversations/latest")


def stream_chunks(*chunks: str, error: Exception | None = None):
    async def generate():
        for chunk in chunks:
            yield chunk
        if error is not None:
            raise error

    return generate()


def parse_stream_events(response: Response) -> list[dict]:
    return [json.loads(line) for line in response.text.splitlines()]


def test_stream_chat_emits_deltas_and_persists_complete_reply(
    monkeypatch, isolated_database
) -> None:
    completion = AsyncMock(
        return_value=StreamingCompletion(
            chunks=stream_chunks("第一段", "第二段"), model="test-stream-model"
        )
    )
    monkeypatch.setattr("app.api.routes.chat.create_streaming_completion", completion)

    response = asyncio.run(
        request(
            "POST",
            "/api/v1/chat/stream",
            {"messages": [{"role": "user", "content": "流式测试"}]},
        )
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/x-ndjson")
    events = parse_stream_events(response)
    assert [event["type"] for event in events] == ["start", "delta", "delta", "done"]
    assert [event["content"] for event in events if event["type"] == "delta"] == [
        "第一段",
        "第二段",
    ]
    conversation_id = events[0]["conversation_id"]
    with isolated_database() as session:
        messages = session.query(Message).filter_by(conversation_id=conversation_id).all()
        assert [(message.role, message.content) for message in messages] == [
            ("user", "流式测试"),
            ("assistant", "第一段第二段"),
        ]


def test_stream_chat_reports_interruption_without_saving_partial_reply(
    monkeypatch, isolated_database
) -> None:
    completion = AsyncMock(
        return_value=StreamingCompletion(
            chunks=stream_chunks("未完成", error=RuntimeError("provider disconnected")),
            model="test-stream-model",
        )
    )
    monkeypatch.setattr("app.api.routes.chat.create_streaming_completion", completion)

    response = asyncio.run(
        request(
            "POST",
            "/api/v1/chat/stream",
            {"messages": [{"role": "user", "content": "中断测试"}]},
        )
    )

    assert response.status_code == 200
    events = parse_stream_events(response)
    assert [event["type"] for event in events] == ["start", "delta", "error"]
    assert events[-1]["detail"] == "AI response stream was interrupted"
    assert "provider disconnected" not in response.text
    with isolated_database() as session:
        messages = session.query(Message).all()
        assert [(message.role, message.content) for message in messages] == [
            ("user", "中断测试")
        ]


def test_stream_chat_rejects_provider_start_error(monkeypatch) -> None:
    completion = AsyncMock(side_effect=RuntimeError("secret provider failure"))
    monkeypatch.setattr("app.api.routes.chat.create_streaming_completion", completion)

    response = asyncio.run(
        request(
            "POST",
            "/api/v1/chat/stream",
            {"messages": [{"role": "user", "content": "启动失败"}]},
        )
    )

    assert response.status_code == 502
    assert response.json() == {"detail": "AI provider request failed"}
    assert "secret provider failure" not in response.text


class RAGSearchStore:
    def __init__(self, results: list[SearchResult]) -> None:
        self.results = results
        self.queries: list[tuple[str, int]] = []

    def search(self, query: str, limit: int) -> list[SearchResult]:
        self.queries.append((query, limit))
        return self.results


def relevant_chunk() -> SearchResult:
    return SearchResult(
        chunk_id=9,
        document_id=3,
        chunk_index=2,
        filename="workbench-guide.md",
        content="AI Workbench 的知识库模式使用文档分块回答问题。",
        distance=0.2,
    )


def test_rag_search_results_are_added_to_completion_prompt(monkeypatch) -> None:
    store = RAGSearchStore([relevant_chunk()])
    completion = AsyncMock(
        return_value=CompletionResult(content="知识库回答", model="test-model")
    )
    monkeypatch.setattr(settings, "rag_enabled", True)
    monkeypatch.setattr("app.services.rag.get_vector_store", lambda: store)
    monkeypatch.setattr("app.api.routes.chat.create_completion", completion)

    response = asyncio.run(
        post_chat(
            {
                "messages": [{"role": "user", "content": "知识库模式是什么？"}],
                "rag_enabled": True,
            }
        )
    )

    assert response.status_code == 200
    assert store.queries == [("知识库模式是什么？", 5)]
    prompt = completion.await_args.args[0]
    assert prompt[-1] == {"role": "user", "content": "知识库模式是什么？"}
    assert prompt[0]["role"] == "system"
    assert "workbench-guide.md" in prompt[0]["content"]
    assert relevant_chunk().content in prompt[0]["content"]


def test_rag_without_relevant_documents_uses_original_messages(monkeypatch) -> None:
    weak_match = SearchResult(
        chunk_id=10,
        document_id=4,
        chunk_index=0,
        filename="unrelated.txt",
        content="不相关内容",
        distance=0.95,
    )
    store = RAGSearchStore([weak_match])
    completion = AsyncMock(return_value=CompletionResult(content="正常回答", model="test-model"))
    monkeypatch.setattr(settings, "rag_enabled", True)
    monkeypatch.setattr("app.services.rag.get_vector_store", lambda: store)
    monkeypatch.setattr("app.api.routes.chat.create_completion", completion)
    messages = [{"role": "user", "content": "没有文档也能回答吗？"}]

    response = asyncio.run(post_chat({"messages": messages, "rag_enabled": True}))

    assert response.status_code == 200
    completion.assert_awaited_once_with(messages)


def test_rag_deepseek_failure_remains_hidden(monkeypatch) -> None:
    store = RAGSearchStore([relevant_chunk()])
    completion = AsyncMock(side_effect=RuntimeError("deepseek secret failure"))
    monkeypatch.setattr(settings, "rag_enabled", True)
    monkeypatch.setattr("app.services.rag.get_vector_store", lambda: store)
    monkeypatch.setattr("app.api.routes.chat.create_completion", completion)

    response = asyncio.run(
        post_chat(
            {
                "messages": [{"role": "user", "content": "从知识库回答"}],
                "rag_enabled": True,
            }
        )
    )

    assert response.status_code == 502
    assert response.json() == {"detail": "AI provider request failed"}
    assert "deepseek secret failure" not in response.text


def test_streaming_chat_uses_rag_augmented_prompt(monkeypatch) -> None:
    store = RAGSearchStore([relevant_chunk()])
    completion = AsyncMock(
        return_value=StreamingCompletion(
            chunks=stream_chunks("知识库", "流式回答"), model="test-model"
        )
    )
    monkeypatch.setattr(settings, "rag_enabled", True)
    monkeypatch.setattr("app.services.rag.get_vector_store", lambda: store)
    monkeypatch.setattr("app.api.routes.chat.create_streaming_completion", completion)

    response = asyncio.run(
        request(
            "POST",
            "/api/v1/chat/stream",
            {
                "messages": [{"role": "user", "content": "流式知识库回答"}],
                "rag_enabled": True,
            },
        )
    )

    assert response.status_code == 200
    assert [event["type"] for event in parse_stream_events(response)] == [
        "start",
        "delta",
        "delta",
        "done",
    ]
    prompt = completion.await_args.args[0]
    assert prompt[0]["role"] == "system"
    assert "workbench-guide.md" in prompt[0]["content"]
