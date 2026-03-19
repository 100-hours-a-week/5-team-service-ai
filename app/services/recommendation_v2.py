from __future__ import annotations

"""Online recommendation pipeline (diversified candidates + LightGBM rerank)."""

import logging
from functools import lru_cache
from typing import Iterable

from qdrant_client import QdrantClient

from app.core.config import get_settings
from app.db.repositories.recommendation_repo import RecommendationRepo
from app.services.behavior_profile import build_behavior_profile
from app.services.candidate_v2 import CandidateGeneratorV2
from app.services.embedder import Embedder
from app.services.lgbm_reranker import LGBMReranker
from app.services.recommender import build_user_query, normalize_meeting_row

logger = logging.getLogger(__name__)


_embedder: Embedder | None = None


def _get_embedder() -> Embedder:
    global _embedder
    if _embedder is None:
        _embedder = Embedder()
    return _embedder


@lru_cache
def _get_qdrant_client() -> QdrantClient:
    settings = get_settings()
    return QdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
        prefer_grpc=True,
    )


def generate_recommendations_v2(
    *,
    user_id: int,
    top_k: int = 4,
    search_k: int = 60,
    repo: RecommendationRepo,
    db,
) -> list[int]:
    settings = get_settings()
    user = repo.fetch_user(db, user_id)
    if not user:
        raise ValueError(f"user_id {user_id} not found")

    meetings_raw = repo.fetch_meetings(db)
    meetings = [normalize_meeting_row(m) for m in meetings_raw]
    if not meetings:
        raise RuntimeError("no meetings available")

    meeting_map = {int(m["id"]): m for m in meetings}

    behavior_profile = build_behavior_profile(user_id, meeting_meta=meeting_map)
    behavior_prompt = behavior_profile.to_prompt() if behavior_profile else None

    user_query_text = build_user_query(user)

    embedder = _get_embedder()
    qdrant = _get_qdrant_client()
    generator = CandidateGeneratorV2(
        qdrant=qdrant,
        embedder=embedder,
        meeting_meta=meetings,
    )

    candidates = generator.generate(
        user_query_text=user_query_text,
        behavior_prompt=behavior_prompt,
        behavior_profile=behavior_profile,
        search_k=search_k,
    )

    reranker = LGBMReranker(settings.lgbm_model_path)
    rec_ids = reranker.rerank(
        candidates,
        meeting_meta=meeting_map,
        behavior_profile=behavior_profile,
        user_genres=user.get("genre_codes") or [],
        top_k=top_k,
    )
    return rec_ids

