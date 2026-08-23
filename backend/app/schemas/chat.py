from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator

DEFAULT_CONVERSATION_TITLE = "新对话"


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1, max_length=32_000)


class ChatRequest(BaseModel):
    messages: list[ChatMessage] = Field(min_length=1, max_length=100)
    conversation_id: int | None = Field(default=None, gt=0)
    rag_enabled: bool = False


class ChatResponse(BaseModel):
    message: ChatMessage
    model: str
    conversation_id: int


class ConversationTitle(BaseModel):
    title: str = Field(min_length=1, max_length=200)

    @field_validator("title")
    @classmethod
    def normalize_title(cls, value: str) -> str:
        title = value.strip()
        if not title:
            raise ValueError("Conversation title must not be blank")
        return title


class ConversationCreate(ConversationTitle):
    title: str = Field(default=DEFAULT_CONVERSATION_TITLE, min_length=1, max_length=200)


class ConversationUpdate(ConversationTitle):
    pass


class ConversationSummary(BaseModel):
    id: int
    title: str
    created_at: datetime
    updated_at: datetime


class ConversationResponse(BaseModel):
    id: int
    title: str
    messages: list[ChatMessage]
    created_at: datetime
    updated_at: datetime
