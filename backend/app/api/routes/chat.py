import asyncio
import json
import logging
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, selectinload

from app.core.config import settings
from app.db.migrations import DEFAULT_CONVERSATION_TITLE, MAX_GENERATED_TITLE_LENGTH
from app.db.session import get_db
from app.models import Conversation, Message
from app.schemas.chat import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    ConversationCreate,
    ConversationResponse,
    ConversationSummary,
    ConversationUpdate,
)
from app.services.agent import (
    create_agent_completion,
    create_agent_streaming_completion,
)
from app.services.llm import (
    LLMInvalidResponseError,
    LLMNotConfiguredError,
    create_completion,
    create_streaming_completion,
)
from app.services.rag import RAGRetrievalError, prepare_chat_messages
from app.services.tools import ToolError, ToolExecutionError

router = APIRouter()
logger = logging.getLogger(__name__)
DbSession = Annotated[Session, Depends(get_db)]


def get_conversation(db: Session, conversation_id: int | None) -> Conversation:
    if conversation_id is None:
        conversation = Conversation()
        db.add(conversation)
        return conversation

    conversation = db.get(Conversation, conversation_id)
    if conversation is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found")
    return conversation


def save_message(db: Session, conversation: Conversation, message: ChatMessage) -> None:
    if message.role == "user" and (
        not conversation.title or conversation.title == DEFAULT_CONVERSATION_TITLE
    ):
        conversation.title = message.content.strip()[:MAX_GENERATED_TITLE_LENGTH]
    conversation.messages.append(Message(role=message.role, content=message.content))
    conversation.updated_at = datetime.now(UTC)
    try:
        db.commit()
        db.refresh(conversation)
    except SQLAlchemyError as exc:
        db.rollback()
        logger.exception("Failed to persist chat message")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Conversation storage is unavailable",
        ) from exc


def serialize_conversation(conversation: Conversation) -> ConversationResponse:
    return ConversationResponse(
        id=conversation.id,
        title=conversation.title,
        messages=[
            ChatMessage(role=message.role, content=message.content)
            for message in conversation.messages
        ],
        created_at=conversation.created_at,
        updated_at=conversation.updated_at,
    )


@router.get("/conversations", response_model=list[ConversationSummary])
async def list_conversations(db: DbSession) -> list[Conversation]:
    return list(
        db.scalars(
            select(Conversation).order_by(
                Conversation.updated_at.desc(), Conversation.id.desc()
            )
        )
    )


@router.post(
    "/conversations",
    response_model=ConversationResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_conversation(request: ConversationCreate, db: DbSession) -> ConversationResponse:
    conversation = Conversation(title=request.title)
    db.add(conversation)
    try:
        db.commit()
        db.refresh(conversation)
    except SQLAlchemyError as exc:
        db.rollback()
        logger.exception("Failed to create conversation")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Conversation storage is unavailable",
        ) from exc
    return serialize_conversation(conversation)


@router.get("/conversations/latest", response_model=ConversationResponse | None)
async def latest_conversation(db: DbSession) -> ConversationResponse | None:
    conversation = db.scalar(
        select(Conversation)
        .options(selectinload(Conversation.messages))
        .order_by(Conversation.updated_at.desc(), Conversation.id.desc())
        .limit(1)
    )
    if conversation is None:
        return None

    return serialize_conversation(conversation)


@router.get("/conversations/{conversation_id}", response_model=ConversationResponse)
async def conversation_detail(conversation_id: int, db: DbSession) -> ConversationResponse:
    conversation = db.scalar(
        select(Conversation)
        .options(selectinload(Conversation.messages))
        .where(Conversation.id == conversation_id)
    )
    if conversation is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found")
    return serialize_conversation(conversation)


@router.patch("/conversations/{conversation_id}", response_model=ConversationSummary)
async def rename_conversation(
    conversation_id: int, request: ConversationUpdate, db: DbSession
) -> Conversation:
    conversation = get_conversation(db, conversation_id)
    conversation.title = request.title
    conversation.updated_at = datetime.now(UTC)
    try:
        db.commit()
        db.refresh(conversation)
    except SQLAlchemyError as exc:
        db.rollback()
        logger.exception("Failed to rename conversation")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Conversation storage is unavailable",
        ) from exc
    return conversation


@router.delete("/conversations/{conversation_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_conversation(conversation_id: int, db: DbSession) -> None:
    conversation = get_conversation(db, conversation_id)
    try:
        db.delete(conversation)
        db.commit()
    except SQLAlchemyError as exc:
        db.rollback()
        logger.exception("Failed to delete conversation")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Conversation storage is unavailable",
        ) from exc


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, db: DbSession) -> ChatResponse:
    user_message = request.messages[-1]
    if user_message.role != "user":
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="The last message must have the user role",
        )

    conversation = get_conversation(db, request.conversation_id)
    save_message(db, conversation, user_message)

    try:
        messages = [message.model_dump() for message in request.messages]
        if settings.agent_enabled:
            result = await create_agent_completion(messages)
        else:
            result = await create_completion(
                prepare_chat_messages(messages, request.rag_enabled)
            )
    except RAGRetrievalError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Knowledge base search is unavailable",
        ) from exc
    except LLMNotConfiguredError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="AI model is not configured",
        ) from exc
    except LLMInvalidResponseError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="AI provider returned an invalid response",
        ) from exc
    except ToolExecutionError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    except ToolError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="AI agent requested an invalid tool call",
        ) from exc
    except Exception as exc:
        logger.exception("LiteLLM chat completion failed")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="AI provider request failed",
        ) from exc

    assistant_message = ChatMessage(role="assistant", content=result.content)
    save_message(db, conversation, assistant_message)
    return ChatResponse(
        message=assistant_message,
        model=result.model,
        conversation_id=conversation.id,
    )


def stream_event(event_type: str, **payload: object) -> str:
    return json.dumps({"type": event_type, **payload}, ensure_ascii=False) + "\n"


@router.post("/chat/stream", response_class=StreamingResponse)
async def stream_chat(request: ChatRequest, db: DbSession) -> StreamingResponse:
    user_message = request.messages[-1]
    if user_message.role != "user":
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="The last message must have the user role",
        )

    conversation = get_conversation(db, request.conversation_id)
    save_message(db, conversation, user_message)

    try:
        messages = [message.model_dump() for message in request.messages]
        if settings.agent_enabled:
            result = await create_agent_streaming_completion(messages)
        else:
            result = await create_streaming_completion(
                prepare_chat_messages(messages, request.rag_enabled)
            )
    except RAGRetrievalError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Knowledge base search is unavailable",
        ) from exc
    except LLMNotConfiguredError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="AI model is not configured",
        ) from exc
    except LLMInvalidResponseError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="AI provider returned an invalid response",
        ) from exc
    except ToolExecutionError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    except ToolError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="AI agent requested an invalid tool call",
        ) from exc
    except Exception as exc:
        logger.exception("LiteLLM streaming chat completion failed to start")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="AI provider request failed",
        ) from exc

    async def generate() -> AsyncIterator[str]:
        parts: list[str] = []
        yield stream_event(
            "start", conversation_id=conversation.id, model=result.model
        )
        try:
            async for content in result.chunks:
                parts.append(content)
                yield stream_event("delta", content=content)

            complete_content = "".join(parts)
            if not complete_content.strip():
                yield stream_event("error", detail="AI provider returned an empty response")
                return

            save_message(
                db,
                conversation,
                ChatMessage(role="assistant", content=complete_content),
            )
            yield stream_event("done", conversation_id=conversation.id, model=result.model)
        except asyncio.CancelledError:
            logger.info(
                "Streaming chat request was cancelled",
                extra={"conversation_id": conversation.id},
            )
            raise
        except HTTPException:
            yield stream_event("error", detail="Failed to save the complete AI response")
        except Exception:
            logger.exception("LiteLLM chat stream was interrupted")
            yield stream_event("error", detail="AI response stream was interrupted")

    return StreamingResponse(
        generate(),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
