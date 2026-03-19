from __future__ import annotations

"""Diversified candidate generation for recommendation v2."""

import logging
from dataclasses import dataclass
from typing import Iterable, List, Mapping, Sequence

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import SearchParams

from app.core.config import get_settings
from app.services.behavior_profile import BehaviorProfile

logger = logging.getLogger(__name__)


@dataclass
class Candidate:
    meeting_id: int
    source: str  # onboard|behavior|popular|new
    source_rank: int
    score: float


def _round_robin_merge(candidates: Mapping[str, List[Candidate]], limit: int) -> List[Candidate]:
    """Merge source buckets in round-robin order to enforce diversity."""

    merged: list[Candidate] = []
    idx = 0
    keys = list(candidates.keys())
    while len(merged) < limit:
        added = False
        for key in keys:
            bucket = candidates.get(key) or []
            if idx < len(bucket):
                cand = bucket[idx]
                if all(c.meeting_id != cand.meeting_id for c in merged):
                    merged.append(cand)
                added = True
            # If bucket shorter, skip
        if not added:
            break
        idx += 1
    return merged[:limit]


class CandidateGeneratorV2:
    def __init__(
        self,
        *,
        qdrant: QdrantClient,
        embedder,
        meeting_meta: Sequence[Mapping],
    ) -> None:
        self.qdrant = qdrant
        self.embedder = embedder
        self.settings = get_settings()
        self.collection = self.settings.qdrant_collection_reco
        # Build quick lookup for meeting metadata
        self.meeting_meta = {int(m["id"]): m for m in meeting_meta}

    # ------------------------------------------------------------------
    def _search(self, vector: np.ndarray, top_k: int) -> list[Candidate]:
        hits = self.qdrant.search(
            collection_name=self.collection,
            query_vector=vector.tolist(),
            limit=top_k,
            with_payload=False,
            search_params=SearchParams(hnsw_ef=64),
        )
        return [
            Candidate(
                meeting_id=int(hit.id),
                score=float(hit.score),
                source="onboard",
                source_rank=idx + 1,
            )
            for idx, hit in enumerate(hits)
        ]

    # ------------------------------------------------------------------
    def generate(
        self,
        *,
        user_query_text: str,
        behavior_prompt: str | None,
        behavior_profile: BehaviorProfile | None,
        search_k: int = 60,
        pop_n: int = 20,
        new_n: int = 10,
        final_limit: int = 50,
    ) -> list[Candidate]:
        # 1) Onboarding search
        onboard_vec = self.embedder.encode([user_query_text], batch_size=8)[0]
        onboard = self._search(onboard_vec, top_k=search_k)

        # 2) Behavior search (optional)
        behavior: list[Candidate] = []
        if behavior_prompt:
            bvec = self.embedder.encode([behavior_prompt], batch_size=8)[0]
            behavior_hits = self._search(bvec, top_k=search_k)
            for cand in behavior_hits:
                cand.source = "behavior"
            behavior = behavior_hits

        # 3) Popular candidates (capacity fill ratio)
        popular = self._build_popular(pop_n)

        # 4) New/recency candidates
        new = self._build_new(new_n)

        merged = _round_robin_merge(
            {
                "behavior": behavior,
                "onboard": onboard,
                "popular": popular,
                "new": new,
            },
            limit=final_limit,
        )
        return merged

    # ------------------------------------------------------------------
    def _build_popular(self, n: int) -> list[Candidate]:
        scored: list[tuple[int, float]] = []
        for mid, meta in self.meeting_meta.items():
            cap = meta.get("capacity") or 1
            cur = meta.get("current_count") or 0
            ratio = float(cur) / float(cap) if cap else 0.0
            scored.append((mid, ratio))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [
            Candidate(meeting_id=mid, source="popular", source_rank=i + 1, score=score)
            for i, (mid, score) in enumerate(scored[:n])
        ]

    def _build_new(self, n: int) -> list[Candidate]:
        # Fallback: highest id approximates recency when created_at not available
        sorted_ids = sorted(self.meeting_meta.keys(), reverse=True)
        return [
            Candidate(meeting_id=mid, source="new", source_rank=i + 1, score=0.0)
            for i, mid in enumerate(sorted_ids[:n])
        ]

