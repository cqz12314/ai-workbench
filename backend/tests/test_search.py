import asyncio

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app
from app.services.vector_store import SearchResult, VectorStoreError


class FakeSearchStore:
    def search(self, query: str, limit: int) -> list[SearchResult]:
        assert query == "苹果手机"
        assert limit == 3
        return [
            SearchResult(
                chunk_id=7,
                document_id=2,
                chunk_index=1,
                filename="guide.md",
                content="苹果手机充电指南",
                distance=0.12,
            )
        ]


async def search(path: str):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        return await client.get(path)


def test_search_returns_relevant_chunks(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.api.routes.search.get_vector_store", lambda: FakeSearchStore()
    )

    response = asyncio.run(search("/api/v1/search?query=苹果手机&limit=3"))

    assert response.status_code == 200
    assert response.json() == [
        {
            "chunk_id": 7,
            "document_id": 2,
            "chunk_index": 1,
            "filename": "guide.md",
            "content": "苹果手机充电指南",
            "distance": 0.12,
        }
    ]


@pytest.mark.parametrize("query", ["", "   "])
def test_search_rejects_empty_query(query: str) -> None:
    response = asyncio.run(search(f"/api/v1/search?query={query}"))

    assert response.status_code == 422


def test_search_hides_vector_store_errors(monkeypatch) -> None:
    class FailingStore:
        def search(self, _query: str, _limit: int):
            raise VectorStoreError("internal chroma path")

    monkeypatch.setattr("app.api.routes.search.get_vector_store", lambda: FailingStore())

    response = asyncio.run(search("/api/v1/search?query=test"))

    assert response.status_code == 503
    assert response.json() == {"detail": "Document search is unavailable"}
    assert "internal chroma path" not in response.text
