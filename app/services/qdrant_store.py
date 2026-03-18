"""
Lightweight Qdrant wrapper used by recommendation batch jobs and tests.

Defaults to an embedded (in-process) Qdrant instance so unit tests do not
require a running server. Provide ``url``/``api_key`` to talk to a remote
Qdrant deployment instead.
"""

from __future__ import annotations

from typing import List, Mapping, Optional

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, SearchParams, VectorParams

__all__ = ["QdrantStore"]


class QdrantStore:
    """Minimal Qdrant wrapper that stores metadata alongside vectors."""

    def __init__(
        self,
        *,
        collection: str = "reco_meetings",
        url: str | None = None,
        api_key: str | None = None,
        location: str | None = ":memory:",
        prefer_grpc: bool = False,
    ) -> None:
        # Remote takes precedence; otherwise fall back to embedded/local mode.
        if url:
            self.client = QdrantClient(url=url, api_key=api_key, prefer_grpc=prefer_grpc)
        else:
            self.client = QdrantClient(location=location)

        self.collection = collection
        self._vector_size: int | None = None
        self._id_to_meta: dict[int, Mapping] = {}
        self._ntotal: int = 0

    # Public API ---------------------------------------------------------
    def build(self, vectors: np.ndarray, metadatas: List[Mapping]) -> None:
        """Build a collection from scratch (idempotent).

        This mirrors the old ``FaissStore.build`` signature to minimize call-site
        changes. State is reset before adding the new batch.
        """

        self._reset()
        self.add_batch(vectors, metadatas)

    def add_batch(self, vectors: np.ndarray, metadatas: List[Mapping]) -> None:
        """Add a batch of vectors + metadata to the collection."""

        if vectors is None:
            raise ValueError("vectors must not be None")
        arr = np.asarray(vectors, dtype=np.float32)
        if arr.ndim != 2:
            raise ValueError(f"vectors must be 2D (got shape {arr.shape})")
        if len(arr) != len(metadatas):
            raise ValueError("vectors and metadatas length mismatch")

        self._ensure_collection(arr.shape[1])

        # Normalize to keep cosine similarity semantics consistent with FAISS.
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        arr = arr / np.maximum(norms, 1e-8)

        points = []
        for idx, (vec, meta) in enumerate(zip(arr, metadatas)):
            # Use meeting_id when available; fall back to a stable numeric id.
            pid = int(meta.get("meeting_id", self._ntotal + idx))
            payload = dict(meta)
            points.append(
                PointStruct(
                    id=pid,
                    vector=vec.tolist(),
                    payload=payload,
                )
            )
            self._id_to_meta[pid] = payload

        self.client.upsert(collection_name=self.collection, points=points)
        self._ntotal += len(points)

    def search(self, query_vec: np.ndarray, top_k: int) -> List[dict]:
        """Search Top-K results for a single query vector."""

        if self._vector_size is None:
            raise RuntimeError("Qdrant collection is not built")

        q = np.asarray(query_vec, dtype=np.float32)
        if q.ndim == 1:
            q = q[None, :]
        if q.ndim != 2 or q.shape[0] != 1:
            raise ValueError(f"query_vec must be shape (d,) or (1, d); got {q.shape}")
        if q.shape[1] != self._vector_size:
            raise ValueError(
                f"dimension mismatch: collection dim {self._vector_size} vs query dim {q.shape[1]}"
            )

        norms = np.linalg.norm(q, axis=1, keepdims=True)
        q = q / np.maximum(norms, 1e-8)

        hits = self.client.search(
            collection_name=self.collection,
            query_vector=q[0].tolist(),
            limit=top_k,
            with_payload=True,
            search_params=SearchParams(hnsw_ef=64),
        )

        results: List[dict] = []
        for hit in hits:
            payload = hit.payload or {}
            meeting_id = payload.get("meeting_id")
            if meeting_id is None:
                continue
            results.append({"meeting_id": int(meeting_id), "score": float(hit.score)})
        return results

    def get_metadata(self, meeting_id: int) -> Optional[Mapping]:
        return self._id_to_meta.get(int(meeting_id))

    @property
    def ntotal(self) -> int:
        return self._ntotal

    # Internal helpers ---------------------------------------------------
    def _reset(self) -> None:
        self._vector_size = None
        self._id_to_meta = {}
        self._ntotal = 0

    def _ensure_collection(self, dim: int) -> None:
        if self._vector_size is None:
            self._vector_size = dim
        elif self._vector_size != dim:
            raise ValueError(
                f"dimension mismatch: collection dim {self._vector_size} vs batch dim {dim}"
            )

        try:
            self.client.get_collection(self.collection)
        except Exception:
            self.client.recreate_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )

