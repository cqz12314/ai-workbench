import hashlib
import math
import re

EMBEDDING_DIMENSIONS = 384
WORD_PATTERN = re.compile(r"[a-z0-9_]+", re.IGNORECASE)


class EmbeddingError(RuntimeError):
    """Raised when text cannot be converted into an embedding."""


class EmbeddingService:
    """Create deterministic local embeddings without external model credentials."""

    def __init__(self, dimensions: int = EMBEDDING_DIMENSIONS) -> None:
        if dimensions <= 0:
            raise ValueError("Embedding dimensions must be positive")
        self.dimensions = dimensions

    def tokenize(self, text: str) -> list[str]:
        normalized = text.casefold().strip()
        characters = [character for character in normalized if not character.isspace()]
        tokens = [f"char:{character}" for character in characters]
        tokens.extend(
            f"bigram:{characters[index]}{characters[index + 1]}"
            for index in range(len(characters) - 1)
        )
        tokens.extend(f"word:{word}" for word in WORD_PATTERN.findall(normalized))
        return tokens

    def embed(self, text: str) -> list[float]:
        tokens = self.tokenize(text)
        if not tokens:
            raise EmbeddingError("Cannot embed empty text")

        vector = [0.0] * self.dimensions
        for token in tokens:
            digest = hashlib.blake2b(token.encode(), digest_size=8).digest()
            bucket = int.from_bytes(digest[:4], "big") % self.dimensions
            sign = 1.0 if digest[4] & 1 else -1.0
            vector[bucket] += sign

        magnitude = math.sqrt(sum(value * value for value in vector))
        if magnitude == 0:
            raise EmbeddingError("Embedding has zero magnitude")
        return [value / magnitude for value in vector]

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(text) for text in texts]
