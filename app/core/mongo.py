from __future__ import annotations

"""MongoDB client helper used by behavior-aware recommendation pipeline.

The client is cached per-process and created only when Mongo settings are
provided. Defaults to secondaryPreferred reads to reduce primary load.
"""

from functools import lru_cache
from typing import Any

from pymongo import MongoClient
from pymongo.read_preferences import SecondaryPreferred

from app.core.config import get_settings


@lru_cache
def get_mongo_db() -> Any | None:
    settings = get_settings()
    if not settings.mongo_uri or not settings.mongo_db:
        return None

    client = MongoClient(
        settings.mongo_uri,
        serverSelectionTimeoutMS=1500,
        read_preference=SecondaryPreferred(),
    )
    return client[settings.mongo_db]
