from __future__ import annotations

import logging
import time
from typing import List

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.db.repositories.recommendation_repo import RecommendationRepo
from app.services.embedder import Embedder
from app.services.faiss_store import FaissStore
from app.services.recommender import (
    build_meeting_text,
    build_user_query,
    rerank_recruiting_with_genre_bonus,
    normalize_user_row,
    normalize_meeting_row,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["recommendations"])
repo = RecommendationRepo()


class RecommendationRequest(BaseModel):
    top_k: int = Field(4, ge=1, le=10)
    search_k: int = Field(20, ge=1)


@router.post(
    "/recommendations",
    summary="Generate recommendations from DB",
)
def generate_recommendations_post(
    body: RecommendationRequest,
    db: Session = Depends(get_db),
) -> dict:
    return _generate_from_db(top_k=body.top_k, search_k=body.search_k, db=db)


def _week_start_iso() -> str:
    from datetime import date, timedelta

    today = date.today()
    monday = today - timedelta(days=today.weekday())
    return monday.isoformat()


def _generate_rows(
    *,
    top_k: int,
    search_k: int,
    meetings: List[dict],
    users: List[dict],
) -> tuple[List[dict], int, dict]:
    embedder = Embedder()

    meeting_texts = [build_meeting_text(m) for m in meetings]
    t0 = time.perf_counter()
    meeting_vecs = embedder.encode(meeting_texts)
    t1 = time.perf_counter()

    store = FaissStore()
    store.build(meeting_vecs, [{"meeting_id": m["id"], "status": m["status"]} for m in meetings])

    user_queries = [build_user_query(u) for u in users]
    t2 = time.perf_counter()
    user_vecs = embedder.encode(user_queries)
    t3 = time.perf_counter()

    rows: List[dict] = []
    week_start_date = _week_start_iso()
    for user, vec in zip(users, user_vecs):
        hits = store.search(vec, top_k=search_k)
        scores = {h["meeting_id"]: h["score"] for h in hits}
        meeting_ids = rerank_recruiting_with_genre_bonus(
            scores,
            meetings,
            user_genres=user.get("genre_codes") or user.get("genre_ids") or [],
            top_k=top_k,
            candidate_pool=search_k,
        )
        for rank, mid in enumerate(meeting_ids, start=1):
            rows.append(
                {
                    "user_id": user["user_id"],
                    "meeting_id": mid,
                    "week_start_date": week_start_date,
                    "rank": rank,
                }
            )

    timings = {
        "embed_meeting_ms": int((t1 - t0) * 1000),
        "embed_user_ms": int((t3 - t2) * 1000),
    }
    return rows, len(users), timings


def _generate_from_db(*, top_k: int, search_k: int, db: Session) -> dict:
    """
    Generate recommendations from MySQL using FAISS + genre-aware reranking,
    then persist them to the recommendation table.
    """
    if search_k < top_k:
        raise HTTPException(status_code=400, detail="search_k must be >= top_k")

    meetings_raw = repo.fetch_meetings(db)
    users_raw = repo.fetch_users(db)

    meetings = [normalize_meeting_row(m) for m in meetings_raw]
    users = [normalize_user_row(u) for u in users_raw]

    if not meetings or not users:
        raise HTTPException(status_code=503, detail="meetings or users not available")

    rows, users_count, timings = _generate_rows(
        top_k=top_k,
        search_k=search_k,
        meetings=meetings,
        users=users,
    )

    logger.info(
        "recommendations generated",
        extra={
            "rows": len(rows),
            "users": users_count,
            "embed_meeting_ms": timings["embed_meeting_ms"],
            "embed_user_ms": timings["embed_user_ms"],
        },
    )
    inserted = repo.upsert_recommendations(db, rows)
    return {
        "users": users_count,
        "rows_count": len(rows),
        "inserted": inserted,
    }
