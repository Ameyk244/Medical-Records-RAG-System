"""Redis-backed query result cache — keyed by sha256(patient_id|query). Phase 5."""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any

import redis.asyncio as redis


class CacheError(Exception):
    """Raised when a cache operation fails. Wraps the underlying Redis error."""


DEFAULT_TTL_SECONDS = 300  # 5 minutes
_KEY_PREFIX = "query:"

_client: redis.Redis | None = None


def _get_client() -> redis.Redis:
    global _client
    if _client is None:
        url = os.environ.get("REDIS_URL")
        if not url:
            raise CacheError("REDIS_URL env var is required")
        _client = redis.from_url(url, decode_responses=True)
    return _client


def _cache_key(patient_id: str, query_text: str) -> str:
    # sha256(patient_id|query_text) — patient_id is part of the digest so cache hits
    # cannot cross patients even if query_text collides.
    digest = hashlib.sha256(f"{patient_id}|{query_text}".encode()).hexdigest()
    return f"{_KEY_PREFIX}{patient_id}:{digest}"


async def get_cached_query_result(patient_id: str, query_text: str) -> dict[str, Any] | None:
    if not patient_id or not query_text:
        raise ValueError("patient_id and query_text must be non-empty")
    key = _cache_key(patient_id, query_text)
    try:
        raw = await _get_client().get(key)
    except redis.RedisError as e:
        raise CacheError("failed to read cache") from e
    if raw is None:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Corrupted entry — drop it on next set, return miss for now.
        return None


async def set_cached_query_result(
    patient_id: str,
    query_text: str,
    result: dict[str, Any],
    ttl: int = DEFAULT_TTL_SECONDS,
) -> None:
    if not patient_id or not query_text:
        raise ValueError("patient_id and query_text must be non-empty")
    if ttl <= 0:
        raise ValueError(f"ttl must be > 0, got {ttl}")
    key = _cache_key(patient_id, query_text)
    try:
        await _get_client().set(key, json.dumps(result), ex=ttl)
    except redis.RedisError as e:
        raise CacheError("failed to write cache") from e
    except (TypeError, ValueError) as e:
        # `result` not JSON-serialisable. Surface clearly; programmer bug.
        raise ValueError(f"result is not JSON-serialisable: {e}") from e


async def invalidate_patient_cache(patient_id: str) -> int:
    if not patient_id:
        raise ValueError("patient_id must be non-empty")
    pattern = f"{_KEY_PREFIX}{patient_id}:*"
    client = _get_client()
    removed = 0
    try:
        # SCAN is non-blocking; KEYS would block the server. Iterate in batches.
        async for key in client.scan_iter(match=pattern, count=200):
            await client.delete(key)
            removed += 1
    except redis.RedisError as e:
        raise CacheError("failed to invalidate patient cache") from e
    return removed
