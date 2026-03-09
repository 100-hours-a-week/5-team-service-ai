from __future__ import annotations

import json
import logging
from functools import lru_cache
from typing import Any

import redis

from app.core.config import Settings, get_settings

logger = logging.getLogger(__name__)


class RedisClient:
    """
    Thin wrapper around redis-py with JSON helpers.
    """

    def __init__(self, client: redis.Redis):
        self._client = client

    def get_json(self, key: str) -> Any | None:
        try:
            raw = self._client.get(key)
            if raw is None:
                return None
            return json.loads(raw)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Redis GET failed for key=%s: %s", key, exc)
            raise

    def set_json(self, key: str, value: Any, ttl_seconds: int | None = None) -> None:
        try:
            payload = json.dumps(value, ensure_ascii=False)
            if ttl_seconds and ttl_seconds > 0:
                self._client.setex(key, ttl_seconds, payload)
            else:
                self._client.set(key, payload)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Redis SET failed for key=%s: %s", key, exc)
            raise


def _build_client(settings: Settings) -> RedisClient | None:
    if not settings.redis_url:
        logger.info("Redis disabled: REDIS_URL not set")
        return None

    try:
        client = redis.Redis.from_url(
            settings.redis_url,
            decode_responses=True,
            socket_timeout=3,
            socket_connect_timeout=3,
        )
        client.ping()
        return RedisClient(client)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Redis connection failed, cache disabled: %s", exc)
        return None


@lru_cache
def get_redis_client(settings: Settings | None = None) -> RedisClient | None:
    """
    Return a cached Redis client or None if not configured/reachable.
    """
    settings = settings or get_settings()
    return _build_client(settings)
