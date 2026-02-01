"""
Lightweight sentence embedding helper for test/offline pipelines.

Uses KURE-v1 via sentence-transformers and runs on CPU by default.
"""

from __future__ import annotations

from typing import Iterable, List

import numpy as np
from sentence_transformers import SentenceTransformer

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

    def __init__(self, model_name: str = "nlpai-lab/KURE-v1", device: str = "cpu") -> None:
        self.model = SentenceTransformer(model_name, device=device)

    def encode(self, texts: Iterable[str]) -> np.ndarray:
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
        try:
            embeddings = self.model.encode(
                text_list,
                convert_to_numpy=True,
                device=self.model.device,
                num_workers=0,
            )
        except TypeError:
            embeddings = self.model.encode(
                text_list,
                convert_to_numpy=True,
                device=self.model.device,
            )
        return embeddings.astype(np.float32, copy=False)
