"""
Offline weekly batch runner using JSONL fixtures and semantic retrieval.

Default behavior is dry-run (prints rows). Use --push with --base-url to POST.
"""

from __future__ import annotations

import argparse
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Iterable, List

from app.clients.spring_client import post_recommendations
from app.services.embedder import Embedder
from app.services.faiss_store import FaissStore
from app.services.recommender import (
    build_meeting_text,
    build_user_query,
    load_jsonl,
    select_recruiting_top_k,
)


def week_start(today: date | None = None) -> date:
    today = today or date.today()
    return today - timedelta(days=today.weekday())


def build_rows(fixtures_dir: Path, *, top_k: int = 4, search_k: int = 20) -> list[dict]:
    meetings = load_jsonl(fixtures_dir / "meetings.jsonl")
    users = load_jsonl(fixtures_dir / "users.jsonl")

    embedder = Embedder()

    # Embed meetings
    meeting_texts = [build_meeting_text(m) for m in meetings]
    t0 = time.perf_counter()
    meeting_vecs = embedder.encode(meeting_texts)
    t1 = time.perf_counter()

    store = FaissStore()
    store.build(meeting_vecs, [{"meeting_id": m["id"], "status": m["status"]} for m in meetings])

    # Embed users
    user_queries = [build_user_query(u) for u in users]
    t2 = time.perf_counter()
    user_vecs = embedder.encode(user_queries)
    t3 = time.perf_counter()

    rows: list[dict] = []
    start_date = week_start().isoformat()
    for user, vec in zip(users, user_vecs):
        hits = store.search(vec, top_k=search_k)
        scores = {h["meeting_id"]: h["score"] for h in hits}
        meeting_ids = select_recruiting_top_k(scores, meetings, top_k=top_k, candidate_pool=search_k)
        for rank, mid in enumerate(meeting_ids, start=1):
            rows.append(
                {
                    "user_id": user["user_id"],
                    "meeting_id": mid,
                    "week_start_date": start_date,
                    "rank": rank,
                }
            )

    print(
        f"Embed meetings: {t1 - t0:.2f}s, embed users: {t3 - t2:.2f}s, "
        f"users={len(users)}, rows={len(rows)}"
    )
    return rows


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run weekly recommendation batch.")
    parser.add_argument("--fixtures", type=Path, default=Path("tests/fixtures"), help="Path to fixture directory")
    parser.add_argument("--base-url", type=str, default=None, help="Spring service base URL (e.g., http://localhost:8080)")
    parser.add_argument("--push", action="store_true", help="Actually POST to Spring API")
    parser.add_argument("--dry-run", action="store_true", help="Print rows instead of pushing (default)")
    parser.add_argument("--top-k", type=int, default=4, help="Final recommendations per user")
    parser.add_argument("--search-k", type=int, default=20, help="Initial search candidates before recruiting filter")
    args = parser.parse_args(list(argv) if argv is not None else None)

    rows = build_rows(args.fixtures, top_k=args.top_k, search_k=args.search_k)

    if args.push and not args.dry_run:
        if not args.base_url:
            parser.error("--base-url is required when --push is set")
        resp = post_recommendations(args.base_url, rows)
        print(f"Push response: status={resp['status_code']} ok={resp.get('ok')}")
    else:
        print(f"Dry-run: prepared {len(rows)} rows (not pushed). Sample: {rows[:3]}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
