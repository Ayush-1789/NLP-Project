"""Email classification module."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from ..config import ClassificationConfig
from ..utils.cache import CacheManager
from ..utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ClassificationResult:
    label: str
    confidence: float
    scores: List[Tuple[str, float]]


class EmailClassifier:
    """Classify emails based on combined TF-IDF and transformer features."""

    def __init__(self, config: ClassificationConfig, cache: CacheManager | None = None) -> None:
        self.config = config
        self.cache = cache
        self.model: LogisticRegression | None = None
        self.scaler = StandardScaler(with_mean=False)
        if self.cache:
            cached_model = self.cache.load_classifier()
            if cached_model is not None:
                self.model = cached_model

    def _ensure_model(self) -> None:
        if self.model is None:
            raise RuntimeError("Classifier has not been trained. Call fit() first.")

    def fit(self, X_tfidf: np.ndarray, X_transformer: np.ndarray, labels: Iterable[str]) -> None:
        X = self._merge_features(X_tfidf, X_transformer)
        X = self.scaler.fit_transform(X)
        self.model = LogisticRegression(max_iter=200)
        self.model.fit(X, list(labels))
        if self.cache:
            self.cache.save_classifier(self.model)

    def _merge_features(self, X_tfidf: np.ndarray, X_transformer: np.ndarray) -> np.ndarray:
        if X_tfidf.shape[0] != X_transformer.shape[0]:
            raise ValueError("TF-IDF and transformer feature matrices must have matching rows")
        return np.hstack([X_tfidf, X_transformer])

    def predict(self, X_tfidf: np.ndarray, X_transformer: np.ndarray, texts: Iterable[str]) -> List[ClassificationResult]:
        if self.model is None:
            logger.warning("Classifier not trained; falling back to heuristic rules.")
            return [self._heuristic_classify(text) for text in texts]
        X = self._merge_features(X_tfidf, X_transformer)
        X = self.scaler.transform(X)
        proba = self.model.predict_proba(X)
        labels = self.model.classes_
        results: List[ClassificationResult] = []
        for row, text in zip(proba, texts):
            top_idx = int(np.argmax(row))
            scores = list(zip(labels, row.tolist()))
            heuristic = self._heuristic_classify(text)
            if row[top_idx] < 0.5:
                label = heuristic.label
                confidence = heuristic.confidence
            else:
                label = labels[top_idx]
                confidence = float(row[top_idx])
            results.append(
                ClassificationResult(
                    label=label,
                    confidence=confidence,
                    scores=scores,
                )
            )
        return results

    def _heuristic_classify(self, text: str) -> ClassificationResult:
        lowered = text.lower()
        for category, keywords in self.config.urgency_keywords.items():
            if any(keyword in lowered for keyword in keywords):
                confidence = 0.8 if category != "Spam" else 0.9
                return ClassificationResult(label=category, confidence=confidence, scores=[(category, confidence)])
        return ClassificationResult(label="Normal", confidence=0.55, scores=[("Normal", 0.55)])


__all__ = ["EmailClassifier", "ClassificationResult"]
