import logging
from typing import Annotated

from mcp.server import MCPServer
from mcp.types import ToolAnnotations
from pydantic import Field
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError

from app.db.session import SessionLocal, initialize_database
from app.models import Document
from app.services.llm import (
    LLMInvalidResponseError,
    LLMNotConfiguredError,
    create_completion,
)
from app.services.rag import (
    RAGRetrievalError,
    prepare_chat_messages,
)
from app.services.rag import search_knowledge as retrieve_knowledge

logger = logging.getLogger(__name__)
mcp = MCPServer(
    "AI Workbench",
    instructions=(
        "Read-only access to AI Workbench documents, knowledge search, and AI answers. "
        "This server cannot modify files, run shell commands, or perform Git operations."
    ),
)

Query = Annotated[str, Field(min_length=1, max_length=1000)]
Question = Annotated[str, Field(min_length=1, max_length=32_000)]
SearchLimit = Annotated[int, Field(ge=1, le=20)]
READ_ONLY_TOOL = ToolAnnotations(
    readOnlyHint=True,
    destructiveHint=False,
    idempotentHint=True,
    openWorldHint=False,
)
AI_READ_ONLY_TOOL = ToolAnnotations(
    readOnlyHint=True,
    destructiveHint=False,
    idempotentHint=False,
    openWorldHint=True,
)


def normalize_text(value: str, field_name: str, maximum_length: int) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must not be blank")
    if len(normalized) > maximum_length:
        raise ValueError(f"{field_name} is too long")
    return normalized


def validate_limit(limit: int) -> int:
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 20:
        raise ValueError("limit must be an integer between 1 and 20")
    return limit


@mcp.tool(annotations=READ_ONLY_TOOL)
async def search_knowledge(query: Query, limit: SearchLimit = 5) -> dict:
    """Search uploaded AI Workbench documents for relevant text chunks."""
    query = normalize_text(query, "query", 1000)
    limit = validate_limit(limit)
    try:
        results = retrieve_knowledge(query, limit)
    except RAGRetrievalError as exc:
        logger.exception("MCP knowledge search failed")
        raise RuntimeError("Knowledge-base search is unavailable") from exc
    return {
        "results": [
            {
                "chunk_id": result.chunk_id,
                "document_id": result.document_id,
                "chunk_index": result.chunk_index,
                "filename": result.filename,
                "content": result.content,
                "distance": result.distance,
            }
            for result in results
        ]
    }


@mcp.tool(annotations=READ_ONLY_TOOL)
async def list_documents() -> dict:
    """List uploaded documents without exposing their server filesystem paths."""
    try:
        with SessionLocal() as session:
            documents = list(
                session.scalars(
                    select(Document).order_by(Document.created_at.desc(), Document.id.desc())
                )
            )
    except SQLAlchemyError as exc:
        logger.exception("MCP document listing failed")
        raise RuntimeError("Document listing is unavailable") from exc
    return {
        "documents": [
            {
                "id": document.id,
                "filename": document.filename,
                "file_type": document.file_type,
                "created_at": document.created_at.isoformat(),
            }
            for document in documents
        ]
    }


@mcp.tool(annotations=AI_READ_ONLY_TOOL)
async def ask_workbench(question: Question, use_rag: bool = True) -> dict:
    """Ask the configured AI model, optionally using read-only knowledge retrieval."""
    question = normalize_text(question, "question", 32_000)
    try:
        messages = prepare_chat_messages(
            [{"role": "user", "content": question}],
            request_rag_enabled=use_rag,
        )
        result = await create_completion(messages)
    except RAGRetrievalError as exc:
        raise RuntimeError("Knowledge-base search is unavailable") from exc
    except LLMNotConfiguredError as exc:
        raise RuntimeError("AI model is not configured") from exc
    except LLMInvalidResponseError as exc:
        raise RuntimeError("AI provider returned an invalid response") from exc
    except Exception as exc:
        logger.exception("MCP AI request failed")
        raise RuntimeError("AI provider request failed") from exc
    return {"answer": result.content, "model": result.model, "used_rag": use_rag}


def main() -> None:
    initialize_database()
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
