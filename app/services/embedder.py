"""
Lightweight sentence embedding helper for test/offline pipelines.

Uses KURE-v1 via sentence-transformers and runs on CPU by default.
"""

from __future__ import annotations

import time
from typing import Iterable, List

import numpy as np
from sentence_transformers import SentenceTransformer

from app.core.metrics import observe_model_load, observe_model_load_fail

__all__ = ["Embedder"]


class Embedder:
    """
    Sentence embedding wrapper around KURE-v1.

    Parameters
    ----------
    model_name : str
        Hugging Face model name or local path. Defaults to nlpai-lab/KURE-v1.
    device : str
        Device identifier; keep as \"cpu\" for deterministic tests.
    """

    def __init__(
        self, model_name: str = "nlpai-lab/KURE-v1", device: str = "cpu"
    ) -> None:
        started_at = time.perf_counter()
        try:
            self.model = SentenceTransformer(model_name, device=device)
            observe_model_load(
                stage="init", elapsed_seconds=time.perf_counter() - started_at
            )
        except Exception:
            observe_model_load_fail(stage="init")
            raise

    def encode(
        self,
        texts: Iterable[str],
        *,
        batch_size: int | None = None,
        show_progress_bar: bool = True,
    ) -> np.ndarray:
        """
        Encode a list of texts into float32 embeddings.

        Returns
        -------
        np.ndarray
            Shape (n, d) float32 embedding matrix.
        """
        # SentenceTransformer can handle generator input; we ensure list for length.
        text_list: List[str] = list(texts)
        # Prefer no worker pool to avoid leaked semaphore warnings; fall back if the
        # installed sentence-transformers version does not support num_workers.
        common_kwargs = {
            "convert_to_numpy": True,
            "device": self.model.device,
            "show_progress_bar": show_progress_bar,
        }
        encode_kwargs = {**common_kwargs, "num_workers": 0}
        if batch_size is not None:
            encode_kwargs["batch_size"] = batch_size

        try:
            embeddings = self.model.encode(text_list, **encode_kwargs)
        except TypeError:
            # Older sentence-transformers may not accept num_workers.
            encode_kwargs.pop("num_workers", None)
            embeddings = self.model.encode(text_list, **encode_kwargs)
        return embeddings.astype(np.float32, copy=False)
