"""Weekly recommendation batch runner (tracked in repo, DB-only)."""

from __future__ import annotations

import argparse
import logging
import time
from collections import Counter
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo
from typing import Iterable, List, Mapping, Optional, Sequence

import faulthandler
import os

from app.core.ssm import load_ssm_parameters

# Load SSM-backed settings for one-shot batch execution before importing modules
# that initialize settings-dependent clients such as the DB session.
load_ssm_parameters()

from app.clients.spring_client import post_recommendations
from app.db.repositories.recommendation_repo import RecommendationRepo
from app.db.session import SessionLocal
from app.services.embedder import Embedder
from app.services.qdrant_store import QdrantStore
from app.services.recommender import (
    build_meeting_text,
    build_user_query,
    normalize_meeting_row,
    normalize_user_row,
    rerank_recruiting_with_genre_bonus,
)

logger = logging.getLogger(__name__)

# Enable crash tracebacks for native segfaults and tame thread explosion on macOS/OpenMP.
faulthandler.enable()
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

# Lazily initialize and reuse a single Embedder instance per process to avoid
# repeated model downloads/loads on every batch invocation.
# CI 배치 경로 검증 시 이 모듈 변경이 배치 배포 트리거로 잡히도록 유지한다.
_embedder: Embedder | None = None


def get_embedder() -> Embedder:
    global _embedder
    if _embedder is None:
        _embedder = Embedder()
    return _embedder


def week_start_iso(today: Optional[date] = None, tz_name: str | None = "Asia/Seoul") -> str:
    """Return ISO string for Monday of the given week (or this week).

    Note: The batch runs on servers configured to UTC. Without an explicit
    timezone this would use Sunday (UTC) when it's already Monday in KST,
    causing `week_start_date` to point to the previous week. Default to
    Asia/Seoul but allow override via `tz_name`.
    """

    if today is None:
        if tz_name:
            today = datetime.now(ZoneInfo(tz_name)).date()
        else:
            today = date.today()
    monday = today - timedelta(days=today.weekday())
    return monday.isoformat()


def embed_meetings(
    meetings: Sequence[Mapping], embedder: Embedder, *, batch_size: int | None = None
) -> list:
    """
    Embed meeting texts into vectors.
    """
    meeting_texts = [build_meeting_text(m) for m in meetings]
    return embedder.encode(meeting_texts, batch_size=batch_size)


def build_index_streaming(
    meetings: List[Mapping],
    embedder: Embedder,
    *,
    meeting_batch_size: int = 800,
    embed_batch_size: int | None = None,
    collection_name: str = "reco_meetings",
    qdrant_url: str | None = None,
    qdrant_api_key: str | None = None,
    qdrant_location: str | None = ":memory:",
) -> QdrantStore:
    """Build a Qdrant index without materializing all embeddings at once."""

    store = QdrantStore(
        collection=collection_name,
        url=qdrant_url,
        api_key=qdrant_api_key,
        location=qdrant_location,
    )
    for idx in range(0, len(meetings), meeting_batch_size):
        chunk = meetings[idx : idx + meeting_batch_size]
        meeting_texts = [build_meeting_text(m) for m in chunk]
        vecs = embedder.encode(
            meeting_texts, batch_size=embed_batch_size, show_progress_bar=False
        )
        store.add_batch(
            vecs,
            [{"meeting_id": m["id"], "status": m["status"]} for m in chunk],
        )
        logger.info(
            "qdrant index chunk built",
            extra={"start": idx, "count": len(chunk), "ntotal": store.ntotal},
        )
    return store


def embed_users(
    users: Sequence[Mapping], embedder: Embedder, *, batch_size: int | None = None
) -> list:
    """
    Embed user preference queries into vectors.
    """
    user_queries = [build_user_query(u) for u in users]
    return embedder.encode(
        user_queries, batch_size=batch_size, show_progress_bar=False
    )


def search_candidates(store: QdrantStore, user_vec, search_k: int) -> dict[int, float]:
    """
    Search Qdrant store and return meeting_id -> score mapping.
    """
    hits = store.search(user_vec, top_k=search_k)
    return {h["meeting_id"]: h["score"] for h in hits}


def generate_from_db(
    *,
    top_k: int,
    search_k: int,
    db=None,
    repo: Optional[RecommendationRepo] = None,
    persist: bool = True,
    meeting_batch_size: int = 50,
    user_batch_size: int = 500,
    embed_batch_size: int | None = 64,
    sample_rows: int = 3,
    collect_rows: bool = False,
) -> dict:
    """Fetch data from DB, generate rows, and optionally persist them."""

    if search_k < top_k:
        raise ValueError("search_k must be >= top_k")

    repo = repo or RecommendationRepo()
    owns_session = db is None
    db = db or SessionLocal()

    try:
        meetings_raw = repo.fetch_meetings(db)
        meetings = [normalize_meeting_row(m) for m in meetings_raw]

        if not meetings:
            raise RuntimeError("meetings not available")

        embedder = get_embedder()

        # Build index in streaming fashion to avoid double-buffering all vectors.
        t0 = time.perf_counter()
        store = build_index_streaming(
            meetings,
            embedder,
            meeting_batch_size=meeting_batch_size,
            embed_batch_size=embed_batch_size,
            collection_name=os.getenv("QDRANT_COLLECTION_RECO", "reco_meetings"),
            qdrant_url=os.getenv("QDRANT_URL"),
            qdrant_api_key=os.getenv("QDRANT_API_KEY"),
            qdrant_location=os.getenv("QDRANT_LOCATION", ":memory:"),
        )
        t1 = time.perf_counter()

        owner_by_meeting = {m["id"]: m.get("leader_user_id") for m in meetings}

        week_start_date = week_start_iso()
        total_users = 0
        total_rows = 0
        inserted = 0
        users_with_recs = 0
        users_zero_recs = 0
        recs_per_user: list[int] = []
        user_ids_in_rows: set[int] = set()
        sample: list[dict] = []
        log_rows_sample: list[dict] = []
        all_rows: list[dict] | None = [] if collect_rows else None
        user_embed_ms = 0

        for user_batch in repo.iter_users(db, chunk_size=user_batch_size):
            users = [normalize_user_row(u) for u in user_batch]
            if not users:
                continue
            total_users += len(users)

            t2 = time.perf_counter()
            user_vecs = embed_users(
                users, embedder, batch_size=embed_batch_size
            )
            user_embed_ms += int((time.perf_counter() - t2) * 1000)

            rows_batch: list[dict] = []
            for user, vec in zip(users, user_vecs):
                scores = search_candidates(store, vec, search_k)
                scores = {
                    mid: score
                    for mid, score in scores.items()
                    if owner_by_meeting.get(mid) is None
                    or owner_by_meeting.get(mid) != user["user_id"]
                }
                meeting_ids = rerank_recruiting_with_genre_bonus(
                    scores,
                    meetings,
                    user_genres=user.get("genre_codes") or user.get("genre_ids") or [],
                    user_id=user.get("user_id"),
                    top_k=top_k,
                    candidate_pool=search_k,
                )
                rec_count = len(meeting_ids)
                recs_per_user.append(rec_count)
                if rec_count == 0:
                    users_zero_recs += 1
                    continue

                users_with_recs += 1
                rows_for_user: list[dict] = []
                for rank, mid in enumerate(meeting_ids, start=1):
                    rows_for_user.append(
                        {
                            "user_id": user["user_id"],
                            "meeting_id": mid,
                            "week_start_date": week_start_date,
                            "rank": rank,
                        }
                    )

                rows_batch.extend(rows_for_user)
                user_ids_in_rows.add(user["user_id"])

                if collect_rows and all_rows is not None:
                    all_rows.extend(rows_for_user)
                elif len(sample) < sample_rows:
                    sample.extend(rows_for_user[: sample_rows - len(sample)])

                if len(log_rows_sample) < 10:
                    log_rows_sample.extend(rows_for_user[: 10 - len(log_rows_sample)])

                total_rows += rec_count

            if persist and rows_batch:
                inserted += repo.upsert_recommendations(db, rows_batch)

            if total_users % 100 == 0:
                logger.info(
                    "reco batch progress",
                    extra={"users": total_users, "rows": total_rows},
                )

        if total_users == 0:
            raise RuntimeError("users not available")

        rec_dist = Counter(recs_per_user)
        logger.info(
            "reco coverage summary",
            extra={
                "total_users": total_users,
                "users_with_recs": users_with_recs,
                "users_zero_recs": users_zero_recs,
                "unique_users_in_rows": len(user_ids_in_rows),
                "rows_generated": total_rows,
                "rec_count_distribution": dict(rec_dist.most_common()),
                "rows_sample": log_rows_sample,
            },
        )

        timings = {
            "embed_meeting_ms": int((t1 - t0) * 1000),
            "embed_user_ms": user_embed_ms,
        }

        return {
            "rows": all_rows if collect_rows else sample,
            "row_count": total_rows,
            "users": total_users,
            "inserted": inserted if persist else total_rows,
            "timings": timings,
        }
    finally:
        if owns_session:
            db.close()


def _push(base_url: str, rows: Iterable[dict]) -> dict:
    """POST rows to Spring service and return response metadata."""

    resp = post_recommendations(base_url, rows)
    logger.info(
        "push response", extra={"status": resp.get("status_code"), "ok": resp.get("ok")}
    )
    return resp


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run weekly recommendation batch (tracked)"
    )
    parser.add_argument(
        "--base-url", type=str, default=None, help="Spring service base URL for push"
    )
    parser.add_argument(
        "--push", action="store_true", help="POST rows to Spring service"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Skip DB upsert and just print sample"
    )
    parser.add_argument(
        "--top-k", type=int, default=4, help="Final recommendations per user"
    )
    parser.add_argument(
        "--search-k",
        type=int,
        default=20,
        help="Initial search candidates before rerank",
    )
    parser.add_argument(
        "--meeting-batch-size",
        type=int,
        default=50,
        help="Chunk size for meeting embeddings/index build",
    )
    parser.add_argument(
        "--user-batch-size",
        type=int,
        default=500,
        help="Chunk size for user fetch/embedding/recommendation",
    )
    parser.add_argument(
        "--embed-batch-size",
        type=int,
        default=64,
        help="Batch size for model.encode",
    )
    parser.add_argument(
        "--sample-rows",
        type=int,
        default=3,
        help="Number of sample rows to print in dry-run",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s"
    )

    try:
        result = generate_from_db(
            top_k=args.top_k,
            search_k=args.search_k,
            persist=not args.dry_run,
            meeting_batch_size=args.meeting_batch_size,
            user_batch_size=args.user_batch_size,
            embed_batch_size=args.embed_batch_size,
            sample_rows=args.sample_rows,
            collect_rows=args.push,
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("DB batch failed: %s", exc)
        return 1

    rows_output = result.get("rows", [])
    print(
        f"rows={result.get('row_count')} users={result.get('users')} inserted={result.get('inserted')} "
        f"embed_ms={result['timings']['embed_meeting_ms']} / {result['timings']['embed_user_ms']}"
    )

    if args.push:
        if not args.base_url:
            parser.error("--base-url is required when --push is set")
        resp = _push(args.base_url, rows_output)
        print(
            f"push status={resp.get('status_code')} ok={resp.get('ok')} text={resp.get('text')}"
        )
    else:
        print(f"dry-run: sample rows -> {rows_output[:3]}")

    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())
