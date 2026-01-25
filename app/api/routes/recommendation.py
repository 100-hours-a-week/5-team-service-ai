from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, Field

from app.clients.spring_client import post_recommendations as push_to_spring
from app.services.embedder import Embedder
from app.services.faiss_store import FaissStore
from app.services.recommender import (
    build_meeting_text,
    build_user_query,
    load_jsonl,
    select_recruiting_top_k,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["recommendations"])


class RecommendationRow(BaseModel):
    user_id: int = Field(..., description="User identifier")
    meeting_id: int = Field(..., description="Meeting identifier")
    week_start_date: str = Field(..., description="ISO week start date (YYYY-MM-DD)")
    rank: int = Field(..., ge=1, description="Rank within the week's recommendations")


class RecommendationsPayload(BaseModel):
    rows: List[RecommendationRow]


class DemoRequest(BaseModel):
    base_url: Optional[str] = Field(None, description="Spring base URL; required when push is true")
    push: bool = Field(False, description="If true, send rows to Spring after generation")
    top_k: int = Field(4, ge=1, le=10)
    search_k: int = Field(20, ge=1)


@router.post(
    "/recommendations",
    status_code=status.HTTP_202_ACCEPTED,
    summary="Accept weekly recommendations (test stub)",
)
def post_recommendations(payload: RecommendationsPayload) -> dict:
    """
    Test-only endpoint to accept recommendation rows and echo back counts.

    This is a lightweight stub so local tests (e.g., Postman) can hit
    `/api/recommendations` without Spring.
    """
    count = len(payload.rows)
    logger.info("Received %s recommendation rows", count, extra={"rows_count": count})
    return {"received": count}


@router.get(
    "/recommendations/demo",
    summary="Generate demo recommendations from JSONL fixtures (test only)",
)
def get_demo_recommendations(
    top_k: int = Query(4, ge=1, le=10, description="Final recommendations per user"),
    search_k: int = Query(20, ge=1, description="Initial search candidates (>= top_k)"),
) -> dict:
    """
    Generate recommendations using local JSONL fixtures with no request body.

    This is for local testing when Spring is not available.
    """
    if search_k < top_k:
        raise HTTPException(status_code=400, detail="search_k must be >= top_k")

    rows, users_count, timings = _generate_rows(top_k=top_k, search_k=search_k)

    logger.info(
        "demo recommendations generated",
        extra={
            "rows": len(rows),
            "users": users_count,
            "embed_meeting_ms": timings["embed_meeting_ms"],
            "embed_user_ms": timings["embed_user_ms"],
        },
    )
    return {"rows": rows, "users": users_count, "rows_count": len(rows)}


@router.post(
    "/recommendations/demo",
    summary="Generate and optionally push demo recommendations (test only)",
)
def post_demo_recommendations(body: DemoRequest) -> dict:
    """
    Generate recommendations from fixtures and optionally push to Spring.
    """
    if body.search_k < body.top_k:
        raise HTTPException(status_code=400, detail="search_k must be >= top_k")

    rows, users_count, timings = _generate_rows(top_k=body.top_k, search_k=body.search_k)

    pushed = False
    push_status = None
    if body.push:
        if not body.base_url:
            raise HTTPException(status_code=400, detail="base_url is required when push=true")
        try:
            resp = push_to_spring(body.base_url, rows)
            pushed = resp.get("ok", False)
            push_status = resp.get("status_code")
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=502, detail=f"Push failed: {exc}") from exc

    logger.info(
        "demo recommendations generated",
        extra={
            "rows": len(rows),
            "users": users_count,
            "embed_meeting_ms": timings["embed_meeting_ms"],
            "embed_user_ms": timings["embed_user_ms"],
            "pushed": pushed,
        },
    )
    return {
        "rows": rows,
        "users": users_count,
        "rows_count": len(rows),
        "pushed": pushed,
        "push_status": push_status,
    }


def _week_start_iso() -> str:
    from datetime import date, timedelta

    today = date.today()
    monday = today - timedelta(days=today.weekday())
    return monday.isoformat()


def _generate_rows(*, top_k: int, search_k: int) -> tuple[List[dict], int, dict]:
    fixtures_dir = Path(__file__).resolve().parents[3] / "tests" / "fixtures"
    meetings = load_jsonl(fixtures_dir / "meetings.jsonl")
    users = load_jsonl(fixtures_dir / "users.jsonl")

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
        meeting_ids = select_recruiting_top_k(scores, meetings, top_k=top_k, candidate_pool=search_k)
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
