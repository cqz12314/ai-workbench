import logging
from typing import Annotated

from fastapi import APIRouter, HTTPException, Query, status

from app.schemas.search import SearchResultResponse
from app.services.vector_store import VectorStoreError, get_vector_store

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/search", response_model=list[SearchResultResponse])
async def search_documents(
    query: Annotated[str, Query(min_length=1, max_length=1000)],
    limit: Annotated[int, Query(ge=1, le=20)] = 5,
) -> list[SearchResultResponse]:
    normalized_query = query.strip()
    if not normalized_query:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="Search query must not be blank",
        )
    try:
        results = get_vector_store().search(normalized_query, limit)
    except VectorStoreError as exc:
        logger.exception("Document vector search failed")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Document search is unavailable",
        ) from exc
    return [SearchResultResponse.model_validate(result) for result in results]
