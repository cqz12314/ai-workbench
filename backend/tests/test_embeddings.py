import math

import pytest

from app.services.embeddings import EmbeddingError, EmbeddingService


def test_embedding_is_deterministic_and_normalized() -> None:
    service = EmbeddingService(dimensions=64)

    first = service.embed("苹果手机使用指南")
    second = service.embed("苹果手机使用指南")

    assert first == second
    assert len(first) == 64
    assert math.sqrt(sum(value * value for value in first)) == pytest.approx(1.0)


def test_embedding_rejects_blank_text() -> None:
    with pytest.raises(EmbeddingError, match="empty text"):
        EmbeddingService().embed("  \n ")
