from __future__ import annotations

"""LightGBM-based reranker with safe fallback to rule-based ranking."""

import logging
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

from app.services.candidate_v2 import Candidate
from app.services.recommender import rerank_recruiting_with_genre_bonus

logger = logging.getLogger(__name__)

try:  # optional dependency
    import lightgbm as lgb  # type: ignore
except Exception:  # noqa: BLE001
    lgb = None


FEATURE_COLUMNS = [
    "source_onboard",
    "source_behavior",
    "source_popular",
    "source_new",
    "source_rank",
    "source_score",
    "pop_ratio",
    "is_new_flag",
    "recent_genre_prob",
    "onboard_genre_match",
]


def _vectorize(rows: list[dict]) -> np.ndarray:
    mat = []
    for row in rows:
        mat.append([float(row.get(col, 0.0)) for col in FEATURE_COLUMNS])
    return np.asarray(mat, dtype=np.float32)


class LGBMReranker:
    def __init__(self, model_path: str | None) -> None:
        self.model = None
        if model_path and lgb is not None:
            path = Path(model_path)
            if path.exists():
                try:
                    self.model = lgb.Booster(model_file=str(path))
                    logger.info("LightGBM reranker loaded", extra={"path": str(path)})
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "LightGBM load failed, fallback to rule-based",
                        extra={"error": str(exc), "path": str(path)},
                    )
            else:
                logger.info("LightGBM model path not found", extra={"path": str(path)})
        elif model_path and lgb is None:
            logger.info("lightgbm package not installed; using fallback reranker")

    # ------------------------------------------------------------------
    def rerank(
        self,
        candidates: Sequence[Candidate],
        *,
        meeting_meta: Mapping[int, Mapping],
        behavior_profile,
        user_genres: Iterable[str] | None = None,
        top_k: int = 4,
    ) -> list[int]:
        # Build feature rows
        feature_rows: list[dict] = []
        genre_cache = {}
        user_genre_set = {str(g) for g in (user_genres or [])}
        for cand in candidates:
            meta = meeting_meta.get(int(cand.meeting_id), {})
            genre = meta.get("reading_genre_code") or meta.get("reading_genre_id") or meta.get("genre_code")
            pop_ratio = 0.0
            cap = meta.get("capacity") or 0
            cur = meta.get("current_count") or 0
            if cap:
                pop_ratio = float(cur) / float(cap)

            recent_prob = 0.0
            if behavior_profile and genre is not None:
                recent_prob = float(behavior_profile.genre_scores.get(str(genre), 0.0))

            feature_rows.append(
                {
                    "source_onboard": 1.0 if cand.source == "onboard" else 0.0,
                    "source_behavior": 1.0 if cand.source == "behavior" else 0.0,
                    "source_popular": 1.0 if cand.source == "popular" else 0.0,
                    "source_new": 1.0 if cand.source == "new" else 0.0,
                    "source_rank": 1.0 / cand.source_rank,
                    "source_score": cand.score,
                    "pop_ratio": pop_ratio,
                    "is_new_flag": 1.0 if cand.source == "new" else 0.0,
                    "recent_genre_prob": recent_prob,
                    "onboard_genre_match": 1.0 if genre is not None and str(genre) in user_genre_set else 0.0,
                }
            )

            genre_cache[cand.meeting_id] = genre

        if self.model:
            data = _vectorize(feature_rows)
            scores = self.model.predict(data)
            scored = list(zip(candidates, scores))
            scored.sort(key=lambda x: x[1], reverse=True)
            logger.debug(
                "rerank using lightgbm",
                extra={"candidates": len(candidates), "top_k": top_k},
            )
            return [c.meeting_id for c, _ in scored[:top_k]]

        # Fallback: use existing genre-diversity heuristic
        # Convert to mapping for rerank_recruiting_with_genre_bonus
        logger.debug(
            "rerank fallback to rule-based",
            extra={"candidates": len(candidates), "top_k": top_k},
        )
        score_map = {c.meeting_id: c.score for c in candidates}
        meetings = [dict(meeting_meta.get(mid, {}), id=mid) for mid in score_map]
        user_genre_list = list(user_genre_set) if user_genre_set else []
        return rerank_recruiting_with_genre_bonus(
            score_map,
            meetings,
            user_genres=user_genre_list,
            top_k=top_k,
            candidate_pool=len(candidates),
        )
