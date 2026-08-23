import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from pydantic import SecretStr

from app.core.config import settings
from app.services.llm import (
    LLMInvalidResponseError,
    LLMNotConfiguredError,
    create_completion,
    create_streaming_completion,
    select_tools,
)


def completion_response(content: str = "你好") -> SimpleNamespace:
    message = SimpleNamespace(content=content)
    choice = SimpleNamespace(message=message)
    return SimpleNamespace(choices=[choice], model="deepseek-v4-flash")


def test_deepseek_completion_uses_configured_credentials(monkeypatch) -> None:
    completion = AsyncMock(return_value=completion_response())
    monkeypatch.setattr("app.services.llm.acompletion", completion)
    monkeypatch.setattr(settings, "litellm_model", "deepseek/deepseek-v4-flash")
    monkeypatch.setattr(settings, "deepseek_api_key", SecretStr("test-secret"))
    monkeypatch.setattr(settings, "deepseek_api_base", "https://api.deepseek.com")
    messages = [{"role": "user", "content": "你好"}]

    result = asyncio.run(create_completion(messages))

    assert result.content == "你好"
    completion.assert_awaited_once_with(
        model="deepseek/deepseek-v4-flash",
        messages=messages,
        api_key="test-secret",
        api_base="https://api.deepseek.com",
    )


@pytest.mark.parametrize("api_key", [None, SecretStr("")])
def test_deepseek_completion_requires_api_key(
    monkeypatch, api_key: SecretStr | None
) -> None:
    completion = AsyncMock()
    monkeypatch.setattr("app.services.llm.acompletion", completion)
    monkeypatch.setattr(settings, "litellm_model", "deepseek/deepseek-v4-pro")
    monkeypatch.setattr(settings, "deepseek_api_key", api_key)

    with pytest.raises(LLMNotConfiguredError, match="DEEPSEEK_API_KEY"):
        asyncio.run(create_completion([{"role": "user", "content": "你好"}]))

    completion.assert_not_awaited()


def test_other_litellm_provider_remains_generic(monkeypatch) -> None:
    completion = AsyncMock(return_value=completion_response("hello"))
    monkeypatch.setattr("app.services.llm.acompletion", completion)
    monkeypatch.setattr(settings, "litellm_model", "openai/gpt-4o-mini")
    messages = [{"role": "user", "content": "hello"}]

    result = asyncio.run(create_completion(messages))

    assert result.content == "hello"
    completion.assert_awaited_once_with(model="openai/gpt-4o-mini", messages=messages)


def test_streaming_completion_normalizes_litellm_chunks(monkeypatch) -> None:
    class Stream:
        model = "deepseek-stream-model"

        def __aiter__(self):
            async def chunks():
                yield SimpleNamespace(
                    choices=[SimpleNamespace(delta=SimpleNamespace(content="你"))]
                )
                yield SimpleNamespace(
                    choices=[SimpleNamespace(delta=SimpleNamespace(content=None))]
                )
                yield SimpleNamespace(
                    choices=[SimpleNamespace(delta=SimpleNamespace(content="好"))]
                )

            return chunks()

    completion = AsyncMock(return_value=Stream())
    monkeypatch.setattr("app.services.llm.acompletion", completion)
    monkeypatch.setattr(settings, "litellm_model", "openai/test-model")
    messages = [{"role": "user", "content": "你好"}]

    async def collect_stream():
        result = await create_streaming_completion(messages)
        return result.model, [chunk async for chunk in result.chunks]

    model, chunks = asyncio.run(collect_stream())

    assert model == "deepseek-stream-model"
    assert chunks == ["你", "好"]
    completion.assert_awaited_once_with(
        model="openai/test-model", messages=messages, stream=True
    )


def test_tool_selection_normalizes_litellm_tool_calls(monkeypatch) -> None:
    tool_call = SimpleNamespace(
        id="call-1",
        function=SimpleNamespace(name="search_knowledge", arguments='{"query":"CAXA"}'),
    )
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=None, tool_calls=[tool_call])
            )
        ],
        model="tool-model",
    )
    completion = AsyncMock(return_value=response)
    monkeypatch.setattr("app.services.llm.acompletion", completion)
    monkeypatch.setattr(settings, "litellm_model", "openai/test-model")
    messages = [{"role": "user", "content": "查找 CAXA"}]
    tool_schemas = [{"type": "function", "function": {"name": "search_knowledge"}}]

    result = asyncio.run(select_tools(messages, tool_schemas))

    assert result.model == "tool-model"
    assert result.tool_calls[0].id == "call-1"
    assert result.tool_calls[0].name == "search_knowledge"
    completion.assert_awaited_once_with(
        model="openai/test-model",
        messages=messages,
        tools=tool_schemas,
        tool_choice="auto",
    )


def test_tool_selection_rejects_empty_provider_response(monkeypatch) -> None:
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=None, tool_calls=[]))],
        model="tool-model",
    )
    monkeypatch.setattr("app.services.llm.acompletion", AsyncMock(return_value=response))
    monkeypatch.setattr(settings, "litellm_model", "openai/test-model")

    with pytest.raises(LLMInvalidResponseError):
        asyncio.run(select_tools([{"role": "user", "content": "test"}], []))
