from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

from litellm import acompletion

from app.core.config import settings


class LLMNotConfiguredError(RuntimeError):
    """Raised when no LiteLLM model has been configured."""


class LLMInvalidResponseError(RuntimeError):
    """Raised when the provider returns no usable assistant message."""


@dataclass(frozen=True)
class CompletionResult:
    content: str
    model: str


@dataclass(frozen=True)
class StreamingCompletion:
    chunks: AsyncIterator[str]
    model: str


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: str


@dataclass(frozen=True)
class ToolSelectionResult:
    content: str | None
    tool_calls: list[ToolCall]
    model: str


def get_completion_options(messages: list[dict[str, Any]]) -> dict[str, Any]:
    if not settings.litellm_model:
        raise LLMNotConfiguredError("LITELLM_MODEL is not configured")

    completion_options: dict[str, Any] = {
        "model": settings.litellm_model,
        "messages": messages,
    }
    if settings.litellm_model.lower().startswith("deepseek/"):
        if settings.deepseek_api_key is None or not settings.deepseek_api_key.get_secret_value():
            raise LLMNotConfiguredError("DEEPSEEK_API_KEY is not configured")
        completion_options.update(
            api_key=settings.deepseek_api_key.get_secret_value(),
            api_base=settings.deepseek_api_base,
        )
    return completion_options


async def create_completion(messages: list[dict[str, Any]]) -> CompletionResult:
    """Call the configured model through LiteLLM and normalize its response."""
    completion_options = get_completion_options(messages)

    response: Any = await acompletion(**completion_options)

    try:
        content = response.choices[0].message.content
    except (AttributeError, IndexError, TypeError) as exc:
        raise LLMInvalidResponseError("The model returned an invalid response") from exc

    if not isinstance(content, str) or not content.strip():
        raise LLMInvalidResponseError("The model returned an empty response")

    response_model = getattr(response, "model", None)
    model = response_model if isinstance(response_model, str) else settings.litellm_model
    return CompletionResult(content=content, model=model)


async def create_streaming_completion(
    messages: list[dict[str, Any]],
) -> StreamingCompletion:
    """Start a LiteLLM stream and expose normalized text chunks."""
    completion_options = get_completion_options(messages)
    response: Any = await acompletion(**completion_options, stream=True)
    if not hasattr(response, "__aiter__"):
        raise LLMInvalidResponseError("The model returned an invalid stream")

    response_model = getattr(response, "model", None)
    model = response_model if isinstance(response_model, str) else settings.litellm_model

    async def content_chunks() -> AsyncIterator[str]:
        async for chunk in response:
            try:
                content = chunk.choices[0].delta.content
            except (AttributeError, IndexError, TypeError):
                continue
            if isinstance(content, str) and content:
                yield content

    return StreamingCompletion(chunks=content_chunks(), model=model)


async def select_tools(
    messages: list[dict[str, Any]], tools: list[dict[str, Any]]
) -> ToolSelectionResult:
    """Ask the configured model to answer directly or select allowlisted tools."""
    completion_options = get_completion_options(messages)
    response: Any = await acompletion(
        **completion_options,
        tools=tools,
        tool_choice="auto",
    )

    try:
        message = response.choices[0].message
    except (AttributeError, IndexError, TypeError) as exc:
        raise LLMInvalidResponseError("The model returned an invalid response") from exc

    content = getattr(message, "content", None)
    if content is not None and not isinstance(content, str):
        raise LLMInvalidResponseError("The model returned an invalid response")

    normalized_calls: list[ToolCall] = []
    raw_calls = getattr(message, "tool_calls", None) or []
    for raw_call in raw_calls:
        try:
            call_id = raw_call.id
            name = raw_call.function.name
            arguments = raw_call.function.arguments
        except AttributeError as exc:
            raise LLMInvalidResponseError("The model returned an invalid tool call") from exc
        if not all(isinstance(value, str) and value for value in (call_id, name, arguments)):
            raise LLMInvalidResponseError("The model returned an invalid tool call")
        normalized_calls.append(ToolCall(id=call_id, name=name, arguments=arguments))

    if not normalized_calls and (not isinstance(content, str) or not content.strip()):
        raise LLMInvalidResponseError("The model returned an empty response")

    response_model = getattr(response, "model", None)
    model = response_model if isinstance(response_model, str) else settings.litellm_model
    return ToolSelectionResult(content=content, tool_calls=normalized_calls, model=model)
