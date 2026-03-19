from __future__ import annotations

"""Behavior-based user preference extraction from Mongo logs."""

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Mapping, Sequence

from app.core.config import get_settings
from app.core.mongo import get_mongo_db


@dataclass
class BehaviorProfile:
    user_id: int
    genre_scores: dict[str, float]
    last_event_at: datetime | None

    def to_prompt(self) -> str:
        if not self.genre_scores:
            return "최근 행동 데이터가 없어 선호 장르를 알 수 없습니다."
        sorted_genres = sorted(self.genre_scores.items(), key=lambda x: x[1], reverse=True)
        genre_text = ", ".join(f"{g}: {score:.2f}" for g, score in sorted_genres[:5])
        return f"최근 행동 기반 선호 장르: {genre_text}."


def _compute_score(log: Mapping) -> float:
    # Weighted heuristic: clicks and join requests dominate, dwell adds nuance.
    click = int(log.get("detailClickCount", 0))
    dwell = float(log.get("detailDwellTimeMs", 0))
    join = 1 if log.get("hasJoinRequest") else 0
    return click * 2.0 + join * 3.0 + dwell / 10_000.0


def build_behavior_profile(
    user_id: int,
    meeting_meta: Mapping[int, Mapping] | None = None,
    *,
    lookback_days: int | None = None,
) -> BehaviorProfile | None:
    settings = get_settings()
    db = get_mongo_db()
    if db is None:
        return None

    coll = db[settings.mongo_interaction_collection]
    horizon = timedelta(days=lookback_days or settings.behavior_lookback_days)
    cutoff = datetime.now(timezone.utc) - horizon

    cursor = coll.find({"userId": user_id, "sentAt": {"$gte": cutoff}})
    genre_scores: dict[str, float] = {}
    last_event_at: datetime | None = None

    for log in cursor:
        meeting_id = int(log.get("meetingId", 0))
        if meeting_id == 0:
            continue
        score = _compute_score(log)

        genre = None
        if meeting_meta:
            meta = meeting_meta.get(meeting_id)
            if meta:
                genre = (
                    meta.get("reading_genre_code")
                    or meta.get("reading_genre_id")
                    or meta.get("genre_code")
                )
        if genre is None:
            # Fallback: bucket under unknown
            genre = "unknown"

        genre_scores[str(genre)] = genre_scores.get(str(genre), 0.0) + score

        sent_at = log.get("sentAt")
        if isinstance(sent_at, datetime):
            if last_event_at is None or sent_at > last_event_at:
                last_event_at = sent_at

    if not genre_scores:
        return None

    # Normalize scores to probabilities for easier feature use.
    total = sum(genre_scores.values())
    if total > 0:
        genre_scores = {g: s / total for g, s in genre_scores.items()}

    return BehaviorProfile(user_id=user_id, genre_scores=genre_scores, last_event_at=last_event_at)

