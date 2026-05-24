"""Voyage AI embedding wrapper — voyage-3-large only."""

import voyageai
import voyageai.error

EMBEDDING_MODEL = "voyage-3-large"
EMBEDDING_DIM = 1024

_BATCH_SIZE = 128


class EmbeddingError(Exception):
    """Raised when the Voyage API call fails after the SDK's internal retries."""


_client: voyageai.AsyncClient | None = None


def _get_client() -> voyageai.AsyncClient:
    global _client
    if _client is None:
        _client = voyageai.AsyncClient()
    return _client


async def embed_texts(texts: list[str], input_type: str = "document") -> list[list[float]]:
    if input_type not in ("document", "query"):
        raise ValueError(f"input_type must be 'document' or 'query', got {input_type!r}")

    if not texts:
        return []

    client = _get_client()
    results: list[list[float]] = []

    for i in range(0, len(texts), _BATCH_SIZE):
        batch = texts[i : i + _BATCH_SIZE]
        try:
            response = await client.embed(
                batch,
                model=EMBEDDING_MODEL,
                input_type=input_type,
                truncation=True,
            )
        except (
            voyageai.error.RateLimitError,
            voyageai.error.ServiceUnavailableError,
            voyageai.error.Timeout,
            voyageai.error.APIConnectionError,
        ) as e:
            # Wrap SDK-specific errors so callers are decoupled from the voyageai SDK.
            raise EmbeddingError(str(e)) from e
        results.extend(response.embeddings)

    return results
