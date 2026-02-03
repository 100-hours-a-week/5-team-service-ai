"""
Utility helpers for offline/test-mode recommendation experiments.

These functions avoid DB access and operate directly on JSONL fixtures.
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Iterable, Mapping

logger = logging.getLogger(__name__)

__all__ = [
    "load_jsonl",
    "build_meeting_text",
    "build_user_query",
    "select_recruiting_top_k",
    "rerank_recruiting_with_genre_bonus",
    "normalize_user_row",
    "normalize_meeting_row",
]


def load_jsonl(path: str | Path) -> list[dict]:
    """
    Load a JSONL file into a list of dicts.

    Parameters
    ----------
    path : str | Path
        Path to a JSON Lines file.

    Returns
    -------
    list[dict]
        Parsed objects; blank lines are ignored.
    """
    file_path = Path(path)
    with file_path.open(encoding="utf-8") as fp:
        return [json.loads(line) for line in fp if line.strip()]


def build_meeting_text(meeting: Mapping) -> str:
    """
    Turn a meeting record into a natural language string for embedding/search.

    Uses only genre code/id, title, description, leader_intro as requested.
    """
    genre = (
        meeting.get("reading_genre_code")
        or meeting.get("reading_genre_id")
        or meeting.get("genre_code")
    )
    title = meeting.get("title") or ""
    desc = meeting.get("description") or ""
    leader_intro = meeting.get("leader_intro") or ""
    return (
        f"장르 {genre} 책모임. 제목: {title}. "
        f"소개: {desc} 리더: {leader_intro}"
    ).strip()


def build_user_query(user: Mapping) -> str:
    """
    Build a user preference query string from profile attributes.

    Combines reading volume, purposes, and preferred genres.
    """
    volume = (
        user.get("reading_volume_code")
        or user.get("reading_volume_id")
        or user.get("volume_code")
    )
    purposes = (
        user.get("purpose_codes")
        or user.get("purpose_ids")
        or user.get("reading_purpose_codes")
        or []
    )
    genres = (
        user.get("genre_codes")
        or user.get("genre_ids")
        or user.get("reading_genre_codes")
        or []
    )
    purpose_text = ", ".join(str(p) for p in sorted(purposes))
    genre_text = ", ".join(str(g) for g in sorted(genres))
    return (
        f"월 독서량 {volume} 수준. "
        f"목적: {purpose_text or '없음'}. "
        f"선호 장르: {genre_text or '없음'}."
    ).strip()


def normalize_user_row(row: Mapping) -> dict:
    """
    Normalize a DB user row to expected fields with codes.
    """
    volume = (
        row.get("reading_volume_code")
        or row.get("reading_volume_id")
        or row.get("volume_code")
    )
    purposes = row.get("purpose_codes") or row.get("purpose_ids") or []
    genres = row.get("genre_codes") or row.get("genre_ids") or []
    return {
        "user_id": row.get("user_id") or row.get("id"),
        "reading_volume_code": volume,
        "purpose_codes": purposes,
        "genre_codes": genres,
    }


def normalize_meeting_row(row: Mapping) -> dict:
    """
    Normalize a DB meeting row to expected fields with codes.
    """
    genre = (
        row.get("reading_genre_code")
        or row.get("reading_genre_id")
        or row.get("genre_code")
    )
    return {
        "id": row.get("id"),
        "reading_genre_code": genre,
        "title": row.get("title") or "",
        "description": row.get("description") or "",
        "status": row.get("status"),
        "capacity": row.get("capacity"),
        "current_count": row.get("current_count"),
        "leader_intro": row.get("leader_intro") or "",
        "leader_user_id": row.get("leader_user_id"),
    }


def select_recruiting_top_k(
    scores: Mapping[int, float],
    meetings: Iterable[Mapping],
    *,
    top_k: int = 4,
    candidate_pool: int = 12,
) -> list[int]:
    """
    Pick Top-K meeting_ids limited to RECRUITING status.

    Strategy:
    1) Sort scores desc and take a generous candidate_pool.
    2) Filter to meetings with status == "RECRUITING".
    3) If fewer than top_k remain, backfill with remaining RECRUITING meetings
       (sorted by score where available, otherwise by id) to reach top_k.
       If still short, return whatever is available.
    """
    # Map meeting_id -> status for quick lookup
    status_map = {int(m.get("id")): m.get("status") for m in meetings}

    # Step 1: candidate pool sorted by score
    sorted_candidates = sorted(
        ((mid, score) for mid, score in scores.items()),
        key=lambda item: item[1],
        reverse=True,
    )[:candidate_pool]

    # Step 2: filter recruiting
    recruiting_ids: list[int] = [
        mid for mid, _ in sorted_candidates if status_map.get(mid) == "RECRUITING"
    ]

    # Step 3: backfill if needed
    if len(recruiting_ids) < top_k:
        remaining = [
            mid
            for mid, _ in sorted_candidates
            if status_map.get(mid) == "RECRUITING" and mid not in recruiting_ids
        ]
        # If still short, consider other recruiting meetings not in scores
        if len(recruiting_ids) + len(remaining) < top_k:
            missing = top_k - len(recruiting_ids) - len(remaining)
            unscored = [
                mid
                for mid, status in status_map.items()
                if status == "RECRUITING" and mid not in recruiting_ids and mid not in remaining
            ]
            random.shuffle(unscored)
            remaining.extend(unscored[:missing])

        recruiting_ids.extend(remaining)

    if len(recruiting_ids) > top_k:
        recruiting_ids = recruiting_ids[:top_k]

    if len(recruiting_ids) < top_k:
        logger.info(
            "Insufficient recruiting meetings to fill top_k (got %s of %s)",
            len(recruiting_ids),
            top_k,
        )

    return recruiting_ids


def rerank_recruiting_with_genre_bonus(
    scores: Mapping[int, float],
    meetings: Iterable[Mapping],
    user_genres: Iterable[str],
    *,
    top_k: int = 4,
    candidate_pool: int = 20,
    genre_bonus: float = 0.05,
    duplicate_penalty: float = 0.07,
) -> list[int]:
    """
    Re-rank recruiting meetings using similarity + genre bonus - duplicate penalty.

    - Start from top similarity candidates (candidate_pool).
    - Bonus if meeting genre is in user's preferred genres.
    - Penalty grows per already-selected meeting of the same genre to diversify.
    """
    user_genre_set = {str(g) for g in user_genres if g is not None}
    meta_map = {int(m.get("id")): m for m in meetings}

    # initial recruiting candidates sorted by raw sim
    sorted_candidates = sorted(
        ((mid, score) for mid, score in scores.items()),
        key=lambda item: item[1],
        reverse=True,
    )[:candidate_pool]
    candidates = [
        (mid, score)
        for mid, score in sorted_candidates
        if meta_map.get(mid, {}).get("status") == "RECRUITING"
    ]

    selected: list[int] = []
    genre_counts: dict[str, int] = {}

    while candidates and len(selected) < top_k:
        best_idx = -1
        best_score = float("-inf")
        for idx, (mid, sim) in enumerate(candidates):
            meta = meta_map.get(mid, {})
            genre = (
                meta.get("reading_genre_code")
                or meta.get("reading_genre_id")
                or meta.get("genre_code")
            )
            base = float(sim)
            if genre in user_genre_set:
                base += genre_bonus
            penalty = genre_counts.get(str(genre), 0) * duplicate_penalty
            final = base - penalty
            if final > best_score:
                best_score = final
                best_idx = idx

        if best_idx == -1:
            break

        mid, _ = candidates.pop(best_idx)
        meta = meta_map.get(mid, {})
        genre = meta.get("reading_genre_code") or meta.get("reading_genre_id") or meta.get("genre_code")
        genre_counts[str(genre)] = genre_counts.get(str(genre), 0) + 1
        selected.append(mid)

    # Backfill if not enough
    if len(selected) < top_k and candidates:
        remaining = [mid for mid, _ in candidates]
        selected.extend(remaining[: top_k - len(selected)])

    return selected
