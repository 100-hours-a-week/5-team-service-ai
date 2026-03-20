"""Behavior-aware recommendation batch with LightGBM training + serving.

This script does three things in one run (configurable via CLI flags):
1) Build/refresh a Qdrant collection with meeting embeddings.
2) Train a LightGBM reranker from Mongo interaction logs (optional).
3) Generate weekly recommendations using diversified candidates + LGBM rerank.

The goal is to keep batch and online pipelines consistent.
"""

from __future__ import annotations

import argparse
import logging
import os
import random
from collections import defaultdict
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Mapping, Sequence

# Env tuning to avoid OpenBLAS/torch fork segfaults on macOS + keep tokenizer quiet.
import os
import faulthandler

faulthandler.enable()
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import numpy as np
import pandas as pd
import lightgbm as lgb
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, SearchParams, VectorParams

from app.core.ssm import load_ssm_parameters

load_ssm_parameters()

from app.core.config import get_settings
from app.core.mongo import get_mongo_db
from app.db.repositories.recommendation_repo import RecommendationRepo
from app.db.session import SessionLocal
from app.services.behavior_profile import _compute_score, BehaviorProfile
from app.services.candidate_v2 import CandidateGeneratorV2
from app.services.embedder import Embedder
from app.services.lgbm_reranker import FEATURE_COLUMNS, LGBMReranker
from app.services.recommender import build_user_query, normalize_meeting_row, normalize_user_row

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Qdrant helpers


def build_reco_collection(
    *, meetings: Sequence[Mapping], embedder: Embedder, settings, batch_size: int = 256
) -> QdrantClient:
    """Recreate and populate the reco_meetings collection in Qdrant."""

    # Avoid passing both url and location; QdrantClient allows only one.
    # prefer_grpc=False to avoid hitting gRPC port (6334) when only HTTP is available.
    qdrant_kwargs = {"prefer_grpc": False}
    if settings.qdrant_url:
        qdrant_kwargs.update({"url": settings.qdrant_url, "api_key": settings.qdrant_api_key})
    elif settings.qdrant_location:
        qdrant_kwargs.update({"location": settings.qdrant_location})
    client = QdrantClient(**qdrant_kwargs)

    texts = [
        f"장르 {m.get('reading_genre_code')}. 제목: {m.get('title','')}. 소개: {m.get('description','')}"
        for m in meetings
    ]
    vecs = embedder.encode(texts, batch_size=batch_size, show_progress_bar=False)
    dim = vecs.shape[1]

    client.recreate_collection(
        collection_name=settings.qdrant_collection_reco,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
    )

    points = []
    for vec, meta in zip(vecs, meetings):
        points.append(
            PointStruct(
                id=int(meta["id"]),
                vector=vec.tolist(),
                payload={
                    "meeting_id": int(meta["id"]),
                    "reading_genre_code": meta.get("reading_genre_code"),
                    "capacity": meta.get("capacity"),
                    "current_count": meta.get("current_count"),
                },
            )
        )
    client.upsert(collection_name=settings.qdrant_collection_reco, points=points)
    logger.info("qdrant collection built", extra={"ntotal": len(points)})
    return client


# ---------------------------------------------------------------------------
# Mongo aggregation


def aggregate_behavior_profiles(
    meeting_meta: Mapping[int, Mapping],
) -> tuple[dict[int, BehaviorProfile], "pd.DataFrame"]:
    """Aggregate Mongo interaction logs into per-user behavior profiles and user-meeting interactions."""

    settings = get_settings()
    db = get_mongo_db()
    if db is None:
        logger.warning("Mongo not configured; behavior profiles empty")
        return {}, {}

    coll = db[settings.mongo_interaction_collection]
    horizon = datetime.now(timezone.utc) - timedelta(days=settings.behavior_lookback_days)

    try:
        cursor = coll.find(
            {"sentAt": {"$gte": horizon}},
            projection={
                "userId": 1,
                "meetingId": 1,
                "detailClickCount": 1,
                "detailDwellTimeMs": 1,
                "hasJoinRequest": 1,
                "sentAt": 1,
            },
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Mongo cursor creation failed; skipping behavior features",
            extra={"error": str(exc)},
        )
        return {}, {}

    per_user_genre: dict[int, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    per_user_last: dict[int, datetime] = {}

    # interaction accumulator keyed by (user, session, meeting)
    interactions = defaultdict(
        lambda: {
            "impressionCount": 0,
            "detailClickCount": 0,
            "detailDwellTimeMs": 0,
            "hasJoinRequest": False,
        }
    )

    try:
        for log in cursor:
            user_id = int(log.get("userId"))
            meeting_id = int(log.get("meetingId", 0))
            if meeting_id == 0:
                continue
            score = _compute_score(log)
            meta = meeting_meta.get(meeting_id, {})
            genre = meta.get("reading_genre_code") or meta.get("reading_genre_id") or meta.get("genre_code") or "unknown"
            per_user_genre[user_id][str(genre)] += score

            sent_at = log.get("sentAt")
            if isinstance(sent_at, datetime):
                prev = per_user_last.get(user_id)
                if prev is None or sent_at > prev:
                    per_user_last[user_id] = sent_at

            key = (user_id, str(log.get("sessionId", "")), meeting_id)
            acc = interactions[key]
            acc["impressionCount"] += int(log.get("impressionCount", 0) or 0)
            acc["detailClickCount"] += int(log.get("detailClickCount", 0) or 0)
            acc["detailDwellTimeMs"] += int(log.get("detailDwellTimeMs", 0) or 0)
            acc["hasJoinRequest"] = acc["hasJoinRequest"] or bool(log.get("hasJoinRequest"))
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Mongo cursor iteration failed; skipping behavior features",
            extra={"error": str(exc)},
        )
        return {}, pd.DataFrame()

    profiles: dict[int, BehaviorProfile] = {}
    for user_id, genre_scores in per_user_genre.items():
        total = sum(genre_scores.values())
        if total > 0:
            genre_scores = {g: s / total for g, s in genre_scores.items()}
        profiles[user_id] = BehaviorProfile(
            user_id=user_id,
            genre_scores=dict(genre_scores),
            last_event_at=per_user_last.get(user_id),
        )

    # Build interaction dataframe aggregated to user-meeting level
    rows = []
    for (user_id, _session, meeting_id), acc in interactions.items():
        rows.append(
            {
                "user_id": user_id,
                "meeting_id": meeting_id,
                "impressionCount": acc["impressionCount"],
                "detailClickCount": acc["detailClickCount"],
                "detailDwellTimeMs": acc["detailDwellTimeMs"],
                "hasJoinRequest": acc["hasJoinRequest"],
            }
        )
    inter_df = pd.DataFrame(rows) if rows else pd.DataFrame()
    if not inter_df.empty:
        inter_df = (
            inter_df.groupby(["user_id", "meeting_id"], as_index=False)
            .agg(
                {
                    "impressionCount": "sum",
                    "detailClickCount": "sum",
                    "detailDwellTimeMs": "sum",
                    "hasJoinRequest": "max",
                }
            )
        )

    return profiles, inter_df


# ---------------------------------------------------------------------------
# Training data builder


def build_training_rows(
    *,
    users: Sequence[Mapping],
    meetings: Sequence[Mapping],
    qdrant: QdrantClient,
    embedder: Embedder,
    behavior_profiles: Mapping[int, BehaviorProfile],
    interactions: pd.DataFrame,
    search_k: int,
) -> pd.DataFrame:
    """Generate candidate rows and labels for lambdarank training."""

    meeting_map = {int(m["id"]): m for m in meetings}
    generator = CandidateGeneratorV2(
        qdrant=qdrant,
        embedder=embedder,
        meeting_meta=meetings,
    )

    records = []
    group_sizes = []
    interactions_by_user: dict[int, pd.DataFrame] = {}
    if not interactions.empty:
        interactions_by_user = {uid: df for uid, df in interactions.groupby("user_id")}

    for user in users:
        uid = int(user.get("user_id") or user.get("id"))
        beh = behavior_profiles.get(uid)
        behavior_prompt = beh.to_prompt() if beh else None
        user_query = build_user_query(user)

        candidates = generator.generate(
            user_query_text=user_query,
            behavior_prompt=behavior_prompt,
            behavior_profile=beh,
            search_k=search_k,
            final_limit=50,
        )

        if not candidates:
            continue

        inter_df = interactions_by_user.get(uid)
        if inter_df is None:
            inter_lookup = {}
        else:
            inter_lookup = {
                int(row.meeting_id): {
                    "impressionCount": int(row.impressionCount),
                    "detailClickCount": int(row.detailClickCount),
                    "detailDwellTimeMs": int(row.detailDwellTimeMs),
                    "hasJoinRequest": bool(row.hasJoinRequest),
                }
                for row in inter_df.itertuples()
            }

        user_rows: list[dict] = []

        for cand in candidates:
            meta = meeting_map.get(int(cand.meeting_id), {})
            genre = meta.get("reading_genre_code") or meta.get("reading_genre_id") or meta.get("genre_code")
            cap = meta.get("capacity") or 0
            cur = meta.get("current_count") or 0
            pop_ratio = float(cur) / float(cap) if cap else 0.0
            recent_prob = beh.genre_scores.get(str(genre), 0.0) if beh and genre is not None else 0.0
            onboard_match = 1.0 if genre and genre in (user.get("genre_codes") or []) else 0.0
            inter = inter_lookup.get(int(cand.meeting_id), {})
            impression = inter.get("impressionCount", 0)
            click = inter.get("detailClickCount", 0)
            dwell = inter.get("detailDwellTimeMs", 0)
            join = inter.get("hasJoinRequest", False)
            inter_score = (
                min(impression, 10) * 0.1
                + min(click, 5) * 1.0
                + min(dwell / 5000.0, 6) * 0.8
                + (1.5 if join else 0.0)
            )

            user_rows.append(
                {
                    "user_id": uid,
                    "meeting_id": int(cand.meeting_id),
                    "label": 0,  # placeholder; set after scoring
                    "source_onboard": 1.0 if cand.source == "onboard" else 0.0,
                    "source_behavior": 1.0 if cand.source == "behavior" else 0.0,
                    "source_popular": 1.0 if cand.source == "popular" else 0.0,
                    "source_new": 1.0 if cand.source == "new" else 0.0,
                    "source_rank": 1.0 / cand.source_rank,
                    "source_score": cand.score,
                    "pop_ratio": pop_ratio,
                    "is_new_flag": 1.0 if cand.source == "new" else 0.0,
                    "recent_genre_prob": recent_prob,
                    "onboard_genre_match": onboard_match,
                    "interaction_score": inter_score,
                }
            )

        # relative labeling within user
        if user_rows:
            user_rows.sort(key=lambda r: r["interaction_score"], reverse=True)
            n = len(user_rows)
            pos_cut = max(1, int(math.ceil(n * 0.4)))
            if n >= 2 and pos_cut == n:
                pos_cut = n - 1
            for idx, row in enumerate(user_rows):
                row["label"] = 1 if idx < pos_cut else 0
            records.extend(user_rows)

        group_sizes.append(len(candidates))

    if not records:
        raise RuntimeError("No training records generated; check logs and meetings")

    df = pd.DataFrame.from_records(records)
    # group sizes need to align with lambdarank input; group by user_id
    group_counts = df.groupby("user_id").size().tolist()
    df["group"] = df["user_id"].map(df.groupby("user_id").size())
    return df, group_counts


# ---------------------------------------------------------------------------
# Training and save


def train_lgbm(df: pd.DataFrame, model_path: str) -> None:
    features = FEATURE_COLUMNS

    # Lambdarank은 한 그룹 안에 양·음 라벨이 모두 있어야 유효한 pairwise loss를 만든다.
    total_users = df["user_id"].nunique()
    user_stats = df.groupby("user_id")["label"].agg(size="size", pos="sum")
    user_stats["neg"] = user_stats["size"] - user_stats["pos"]
    single_label_users = int(((user_stats["pos"] == 0) | (user_stats["pos"] == user_stats["size"])).sum())
    logger.info(
        "label diversity per user before filter",
        extra={
            "total_users": total_users,
            "single_label_users": single_label_users,
            "examples": user_stats.head(5).to_dict("index"),
        },
    )

    multi_label_mask = df.groupby("user_id")["label"].transform("nunique") > 1
    has_multi_label_group = multi_label_mask.any()

    if has_multi_label_group:
        df = df[multi_label_mask]
        group_sizes = df.groupby("user_id").size().tolist()
        kept_users = df["user_id"].nunique()
        dropped_users = total_users - kept_users
        logger.info(
            "users kept after removing single-label groups",
            extra={"kept": kept_users, "dropped": dropped_users},
        )
    else:
        logger.warning(
            "모든 그룹이 단일 라벨입니다. LightGBM 학습을 건너뛰고 heuristic fallback을 사용합니다.",
            extra={"total_users": total_users},
        )
        return

    # 상수 피처 제거로 분할 불가 상황을 방지한다.
    nunique_per_feature = df[features].nunique()
    constant_features = [f for f, n in nunique_per_feature.items() if n <= 1]
    if constant_features:
        logger.info("dropping constant features", extra={"features": constant_features})
        features = [f for f in features if f not in constant_features]
    if not features:
        raise RuntimeError("모든 피처가 상수여서 학습할 수 없음")

    train_set_kwargs = {"free_raw_data": False}
    if group_sizes is not None:
        train_set_kwargs["group"] = group_sizes

    train_set = lgb.Dataset(
        df[features],
        label=df["label"],
        **train_set_kwargs,
    )

    params = {
        "objective": "lambdarank",
        "metric": "ndcg@10",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": -1,
        "min_data_in_leaf": 5,
        "min_sum_hessian_in_leaf": 1e-3,
        "min_gain_to_split": 0.0,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "feature_pre_filter": False,
        "verbosity": 1,
        "seed": 42,
    }

    booster = lgb.train(
        params,
        train_set,
        num_boost_round=400,
        valid_sets=[train_set],
        valid_names=["train"],
    )
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    booster.save_model(model_path)
    logger.info("LightGBM model saved", extra={"path": model_path, "objective": "lambdarank"})


# ---------------------------------------------------------------------------
# Recommendation generation (v2) with trained model


def generate_recommendations(
    *,
    users: Sequence[Mapping],
    meetings: Sequence[Mapping],
    qdrant: QdrantClient,
    embedder: Embedder,
    behavior_profiles: Mapping[int, BehaviorProfile],
    top_k: int,
    search_k: int,
    repo: RecommendationRepo,
    db,
    model_path: str,
):
    meeting_map = {int(m["id"]): m for m in meetings}
    generator = CandidateGeneratorV2(qdrant=qdrant, embedder=embedder, meeting_meta=meetings)
    reranker = LGBMReranker(model_path)

    rows = []
    for user in users:
        uid = int(user.get("user_id") or user.get("id"))
        beh = behavior_profiles.get(uid)
        behavior_prompt = beh.to_prompt() if beh else None
        user_query = build_user_query(user)

        candidates = generator.generate(
            user_query_text=user_query,
            behavior_prompt=behavior_prompt,
            behavior_profile=beh,
            search_k=search_k,
            final_limit=50,
        )
        if not candidates:
            continue

        rec_ids = reranker.rerank(
            candidates,
            meeting_meta=meeting_map,
            behavior_profile=beh,
            user_genres=user.get("genre_codes") or [],
            top_k=top_k,
        )

        for rank, mid in enumerate(rec_ids, 1):
            rows.append(
                {
                    "user_id": uid,
                    "meeting_id": mid,
                    "week_start_date": datetime.now().date().isoformat(),
                    "rank": rank,
                }
            )

    if rows:
        repo.upsert_recommendations(db, rows)
    logger.info("recommendations generated", extra={"rows": len(rows)})


# ---------------------------------------------------------------------------
# CLI


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Behavior-aware reco batch")
    parser.add_argument("--train", action="store_true", help="Train LightGBM before generating recs")
    parser.add_argument("--generate", action="store_true", help="Generate recommendations")
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--search-k", type=int, default=60)
    parser.add_argument("--sample-users", type=int, default=0, help="Optional user sample for faster test")
    args = parser.parse_args(list(argv) if argv is not None else None)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s")

    settings = get_settings()
    repo = RecommendationRepo()
    db = SessionLocal()
    try:
        meetings_raw = repo.fetch_meetings(db)
        meetings = [normalize_meeting_row(m) for m in meetings_raw]
        if not meetings:
            raise RuntimeError("no meetings available")

        users_raw = repo.fetch_users(db)
        users = [normalize_user_row(u) for u in users_raw]
        if args.sample_users > 0:
            users = random.sample(users, min(args.sample_users, len(users)))

        embedder = Embedder()
        qdrant = build_reco_collection(meetings=meetings, embedder=embedder, settings=settings)

        behavior_profiles, interactions = aggregate_behavior_profiles({int(m["id"]): m for m in meetings})

        model_path = settings.lgbm_model_path or "./models/lgbm_rerank.txt"

        if args.train:
            df, _ = build_training_rows(
                users=users,
                meetings=meetings,
                qdrant=qdrant,
                embedder=embedder,
                behavior_profiles=behavior_profiles,
                interactions=interactions,
                search_k=args.search_k,
            )
            train_lgbm(df, model_path)
        else:
            logger.info(
                "train step skipped (no --train)",
                extra={"model_path": model_path, "model_exists": Path(model_path).exists()},
            )

        if args.generate:
            generate_recommendations(
                users=users,
                meetings=meetings,
                qdrant=qdrant,
                embedder=embedder,
                behavior_profiles=behavior_profiles,
                top_k=args.top_k,
                search_k=args.search_k,
                repo=repo,
                db=db,
                model_path=model_path,
            )
        else:
            logger.info("generate step skipped (no --generate)")

    finally:
        db.close()

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
