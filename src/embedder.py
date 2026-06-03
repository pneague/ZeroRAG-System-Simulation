from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np


DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"


@dataclass(slots=True)
class SentenceTransformerEmbedder:
    model_name: str = DEFAULT_EMBEDDING_MODEL
    batch_size: int = 64
    device: str | None = None
    normalize_embeddings: bool = True
    _model: Any = field(init=False, repr=False)

    def __post_init__(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "The sentence-transformers package is required for embedding computation. "
                "Install it with 'pip install sentence-transformers>=2.7'."
            ) from exc
        self._model = SentenceTransformer(self.model_name, device=self.device)

    def encode(self, texts: Sequence[str], *, show_progress_bar: bool = True) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        vectors = self._model.encode(
            list(texts),
            batch_size=self.batch_size,
            show_progress_bar=show_progress_bar,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize_embeddings,
        )
        return np.asarray(vectors, dtype=np.float32)

    @property
    def embedding_dim(self) -> int:
        return int(self._model.get_sentence_embedding_dimension())

