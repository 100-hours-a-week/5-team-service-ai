"""Weekly recommendation batch runner (tracked in repo, DB-only)."""

from __future__ import annotations

import argparse
import logging
import time
from datetime import date, timedelta
from typing import Iterable, List, Mapping, Optional

from app.clients.spring_client import post_recommendations
from app.db.repositories.recommendation_repo import RecommendationRepo
from app.db.session import SessionLocal
from app.services.embedder import Embedder
from app.services.faiss_store import FaissStore
from app.services.recommender import (
    build_meeting_text,
    build_user_query,
    normalize_meeting_row,
    normalize_user_row,
    rerank_recruiting_with_genre_bonus,
)

logger = logging.getLogger(__name__)

# Lazily initialize and reuse a single Embedder instance per process to avoid
# repeated model downloads/loads on every batch invocation.
_embedder: Embedder | None = None


def get_embedder() -> Embedder:
    global _embedder
    if _embedder is None:
        _embedder = Embedder()
    return _embedder


def week_start_iso(today: Optional[date] = None) -> str:
    """Return ISO string for Monday of the given week (or this week)."""

    today = today or date.today()
    monday = today - timedelta(days=today.weekday())
    return monday.isoformat()


def embed_meetings(meetings: List[Mapping], embedder: Embedder) -> list:
    """
    Embed meeting texts into vectors.
    """
    meeting_texts = [build_meeting_text(m) for m in meetings]
    return embedder.encode(meeting_texts)


def build_index(meeting_vecs, meetings: List[Mapping]) -> FaissStore:
    """
    Build FAISS index from meeting vectors and metadata.
    """
    store = FaissStore()
    store.build(meeting_vecs, [{"meeting_id": m["id"], "status": m["status"]} for m in meetings])
    return store


def embed_users(users: List[Mapping], embedder: Embedder) -> list:
    """
    Embed user preference queries into vectors.
    """
    user_queries = [build_user_query(u) for u in users]
    return embedder.encode(user_queries)


def search_candidates(store: FaissStore, user_vec, search_k: int) -> dict[int, float]:
    """
    Search FAISS store and return meeting_id -> score mapping.
    """
    hits = store.search(user_vec, top_k=search_k)
    return {h["meeting_id"]: h["score"] for h in hits}


def generate_rows(
    *,
    top_k: int,
    search_k: int,
    meetings: List[Mapping],
    users: List[Mapping],
    embedder: Embedder | None = None,
) -> tuple[List[dict], dict]:
    """Generate recommendation rows using FAISS + genre-aware reranking."""

    embedder = embedder or get_embedder()

    t0 = time.perf_counter()
    meeting_vecs = embed_meetings(meetings, embedder)
    t1 = time.perf_counter()

    store = build_index(meeting_vecs, meetings)

    t2 = time.perf_counter()
    user_vecs = embed_users(users, embedder)
    t3 = time.perf_counter()

    owner_by_meeting = {m["id"]: m.get("leader_user_id") for m in meetings}

    logger.info(
        "reco batch data loaded",
        extra={"meetings": len(meetings), "users": len(users)},
    )
    logger.info(
        "reco batch embeddings ready",
        extra={
            "embed_meeting_ms": int((t1 - t0) * 1000),
            "embed_user_ms": int((t3 - t2) * 1000),
        },
    )

    rows: List[dict] = []
    week_start_date = week_start_iso()
    processed = 0
    for user, vec in zip(users, user_vecs):
        scores = search_candidates(store, vec, search_k)
        # 필터: 사용자가 만든 모임은 제외
        scores = {
            mid: score
            for mid, score in scores.items()
            if owner_by_meeting.get(mid) is None or owner_by_meeting.get(mid) != user["user_id"]
        }
        meeting_ids = rerank_recruiting_with_genre_bonus(
            scores,
            meetings,
            user_genres=user.get("genre_codes") or user.get("genre_ids") or [],
            user_id=user.get("user_id"),
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
        processed += 1
        if processed % 20 == 0:
            logger.info("reco batch progress: %s users processed", processed)

    timings = {
        "embed_meeting_ms": int((t1 - t0) * 1000),
        "embed_user_ms": int((t3 - t2) * 1000),
    }
    return rows, timings


def generate_from_db(
    *,
    top_k: int,
    search_k: int,
    db=None,
    repo: Optional[RecommendationRepo] = None,
    persist: bool = True,
) -> dict:
    """Fetch data from DB, generate rows, and optionally persist them."""

    if search_k < top_k:
        raise ValueError("search_k must be >= top_k")

    repo = repo or RecommendationRepo()
    owns_session = db is None
    db = db or SessionLocal()

    try:
        meetings_raw = repo.fetch_meetings(db)
        users_raw = repo.fetch_users(db)

        meetings = [normalize_meeting_row(m) for m in meetings_raw]
        users = [normalize_user_row(u) for u in users_raw]

        if not meetings or not users:
            raise RuntimeError("meetings or users not available")

        rows, timings = generate_rows(
            top_k=top_k,
            search_k=search_k,
            meetings=meetings,
            users=users,
        )

        inserted = repo.upsert_recommendations(db, rows) if persist else 0
        return {
            "rows": rows,
            "users": len(users),
            "inserted": inserted,
            "timings": timings,
        }
    finally:
        if owns_session:
            db.close()


def _push(base_url: str, rows: Iterable[dict]) -> dict:
    """POST rows to Spring service and return response metadata."""

    resp = post_recommendations(base_url, rows)
    logger.info("push response", extra={"status": resp.get("status_code"), "ok": resp.get("ok")})
    return resp


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run weekly recommendation batch (tracked)")
    parser.add_argument("--base-url", type=str, default=None, help="Spring service base URL for push")
    parser.add_argument("--push", action="store_true", help="POST rows to Spring service")
    parser.add_argument("--dry-run", action="store_true", help="Skip DB upsert and just print sample")
    parser.add_argument("--top-k", type=int, default=4, help="Final recommendations per user")
    parser.add_argument("--search-k", type=int, default=20, help="Initial search candidates before rerank")
    args = parser.parse_args(list(argv) if argv is not None else None)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s")

    try:
        result = generate_from_db(top_k=args.top_k, search_k=args.search_k, persist=not args.dry_run)
    except Exception as exc:  # noqa: BLE001
        logger.error("DB batch failed: %s", exc)
        return 1

    rows = result["rows"]
    print(
        f"rows={len(rows)} users={result['users']} inserted={result['inserted']} "
        f"embed_ms={result['timings']['embed_meeting_ms']} / {result['timings']['embed_user_ms']}"
    )

    if args.push:
        if not args.base_url:
            parser.error("--base-url is required when --push is set")
        resp = _push(args.base_url, rows)
        print(f"push status={resp.get('status_code')} ok={resp.get('ok')} text={resp.get('text')}")
    else:
        print(f"dry-run: sample rows -> {rows[:3]}")

    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())
