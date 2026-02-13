"""
In-memory FAISS index wrapper for semantic retrieval in tests.
"""

from __future__ import annotations

from typing import Iterable, List, Mapping, Optional

import faiss
import numpy as np

__all__ = ["FaissStore"]


class FaissStore:
    """
    Minimal FAISS wrapper that stores metadata alongside vectors.
    """

    def __init__(self) -> None:
        self.index: Optional[faiss.IndexFlatIP] = None
        self.metadatas: List[Mapping] = []
        self._id_to_meta: dict[int, Mapping] = {}

    def load(self, index_path: str, meta_path: str) -> None:
        """
        Load index and metadata from disk.
        """
        self.index = faiss.read_index(index_path)
        import json

        with open(meta_path, encoding="utf-8") as f:
            self.metadatas = json.load(f)
        self._id_to_meta = {int(m["meeting_id"]): m for m in self.metadatas}

    def build(self, vectors: np.ndarray, metadatas: List[Mapping]) -> None:
        """
        Build an index from embedding matrix and metadata.

        Parameters
        ----------
        vectors : np.ndarray
            Shape (n, d) float32 embeddings. Will be L2-normalized.
        metadatas : list[Mapping]
            Metadata aligned with vectors; must include meeting_id.
        """
        if vectors is None:
            raise ValueError("vectors must not be None")
        arr = np.asarray(vectors, dtype=np.float32)
        if arr.ndim != 2:
            raise ValueError(f"vectors must be 2D (got shape {arr.shape})")
        if len(arr) != len(metadatas):
            raise ValueError("vectors and metadatas length mismatch")

        faiss.normalize_L2(arr)
        dim = arr.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(arr)

        self.metadatas = list(metadatas)
        self._id_to_meta = {int(meta["meeting_id"]): meta for meta in self.metadatas}

    def search(self, query_vec: np.ndarray, top_k: int) -> List[dict]:
        """
        Search Top-K results for a single query vector.

        Returns
        -------
        list[dict]
            Each dict: {\"meeting_id\": int, \"score\": float}
        """
        if self.index is None:
            raise RuntimeError("FAISS index is not built")

        q = np.asarray(query_vec, dtype=np.float32)
        if q.ndim == 1:
            q = q[None, :]
        if q.ndim != 2 or q.shape[0] != 1:
            raise ValueError(f"query_vec must be shape (d,) or (1, d); got {q.shape}")

        faiss.normalize_L2(q)
        scores, idxs = self.index.search(q, top_k)
        results: List[dict] = []
        for idx, score in zip(idxs[0], scores[0]):
            if idx < 0 or idx >= len(self.metadatas):
                continue
            meta = self.metadatas[idx]
            results.append({"meeting_id": int(meta["meeting_id"]), "score": float(score)})
        return results

    def search_with_meta(self, query_vec: np.ndarray, top_k: int) -> List[dict]:
        """
        Search Top-K results and return metadata.

        Each dict: {\"meeting_id\": int, \"score\": float, \"meta\": Mapping}
        """
        if self.index is None:
            raise RuntimeError("FAISS index is not built")

        q = np.asarray(query_vec, dtype=np.float32)
        if q.ndim == 1:
            q = q[None, :]
        if q.ndim != 2 or q.shape[0] != 1:
            raise ValueError(f"query_vec must be shape (d,) or (1, d); got {q.shape}")

        faiss.normalize_L2(q)
        scores, idxs = self.index.search(q, top_k)
        results: List[dict] = []
        for idx, score in zip(idxs[0], scores[0]):
            if idx < 0 or idx >= len(self.metadatas):
                continue
            meta = self.metadatas[idx]
            results.append(
                {"meeting_id": int(meta["meeting_id"]), "score": float(score), "meta": meta}
            )
        return results

    def get_metadata(self, meeting_id: int) -> Optional[Mapping]:
        """
        Return metadata for a given meeting_id if present.
        """
        return self._id_to_meta.get(int(meeting_id))

    def search_with_meta(self, query_vec: np.ndarray, top_k: int) -> List[dict]:
        """
        Search Top-K results and return metadata alongside scores.

        Each dict: {"meeting_id": int, "score": float, "meta": Mapping}
        """
        if self.index is None:
            raise RuntimeError("FAISS index is not built")

        q = np.asarray(query_vec, dtype=np.float32)
        if q.ndim == 1:
            q = q[None, :]
        if q.ndim != 2 or q.shape[0] != 1:
            raise ValueError(f"query_vec must be shape (d,) or (1, d); got {q.shape}")

        faiss.normalize_L2(q)
        scores, idxs = self.index.search(q, top_k)
        results: List[dict] = []
        for idx, score in zip(idxs[0], scores[0]):
            if idx < 0 or idx >= len(self.metadatas):
                continue
            meta = self.metadatas[idx]
            results.append({"meeting_id": int(meta["meeting_id"]), "score": float(score), "meta": meta})
        return results
