"""Feature extraction pipeline."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoModel, AutoTokenizer
import torch

from ..config import ModelConfig
from ..utils.cache import CacheManager
from ..utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class FeatureBundle:
    """Container for TF-IDF and transformer embedding features."""

    tfidf: np.ndarray
    transformer: np.ndarray


class FeatureExtractor:
    """Compute TF-IDF vectors and contextual embeddings."""

    def __init__(self, config: ModelConfig, cache: CacheManager | None = None) -> None:
        self.config = config
        self.cache = cache
        self._vectorizer: Optional[TfidfVectorizer] = None
        self._tokenizer = AutoTokenizer.from_pretrained(config.transformer_model)
        self._model = AutoModel.from_pretrained(config.transformer_model)
        self._model.eval()

    @property
    def vectorizer(self) -> TfidfVectorizer:
        if self._vectorizer is None:
            if self.cache:
                cached = self.cache.load_vectorizer()
                if cached is not None:
                    self._vectorizer = cached
                    return self._vectorizer
            self._vectorizer = TfidfVectorizer(
                max_features=self.config.tfidf_max_features,
                ngram_range=self.config.tfidf_ngram_range,
                stop_words="english",
            )
        return self._vectorizer

    def fit_vectorizer(self, corpus: Iterable[str]) -> None:
        self.vectorizer.fit(corpus)
        if self.cache:
            self.cache.save_vectorizer(self.vectorizer)

    def transform_vectorizer(self, texts: Iterable[str]) -> np.ndarray:
        vectorizer = self.vectorizer
        if not hasattr(vectorizer, "vocabulary_"):
            raise RuntimeError("TF-IDF vectorizer must be fitted before transforming text.")
        return vectorizer.transform(texts).toarray()

    @torch.no_grad()
    def transformer_embeddings(self, texts: Iterable[str], batch_size: int = 8) -> np.ndarray:
        embeddings: list[np.ndarray] = []
        buffer: list[str] = []
        for text in texts:
            buffer.append(text)
            if len(buffer) == batch_size:
                embeddings.append(self._batch_embeddings(buffer))
                buffer.clear()
        if buffer:
            embeddings.append(self._batch_embeddings(buffer))
        if not embeddings:
            return np.zeros((0, self._model.config.hidden_size))
        return np.vstack(embeddings)

    def _batch_embeddings(self, batch: list[str]) -> np.ndarray:
        encoded = self._tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt",
        )
        outputs = self._model(**encoded)
        hidden_state = outputs.last_hidden_state
        # Mean pooling
        attention_mask = encoded["attention_mask"].unsqueeze(-1)
        masked_hidden = hidden_state * attention_mask
        sums = masked_hidden.sum(dim=1)
        counts = attention_mask.sum(dim=1).clamp(min=1)
        mean_pooled = sums / counts
        return mean_pooled.cpu().numpy()

    def build_features(self, texts: Iterable[str]) -> FeatureBundle:
        tfidf_matrix = self.transform_vectorizer(texts)
        transformer_matrix = self.transformer_embeddings(texts)
        return FeatureBundle(tfidf=tfidf_matrix, transformer=transformer_matrix)


__all__ = ["FeatureExtractor", "FeatureBundle"]
