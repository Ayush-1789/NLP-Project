"""Sentiment analysis pipeline."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional

from transformers import pipeline

from ..utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class SentimentResult:
    label: str
    score: float


class SentimentAnalyzer:
    """Wrapper around transformer sentiment models with graceful fallbacks."""

    def __init__(
        self,
        model_name: str,
        *,
        revision: Optional[str] = None,
        device: Optional[int] = None,
    ) -> None:
        self.model_name = model_name
        self.revision = revision
        self.device = device
        try:
            pipeline_kwargs = {"model": model_name}
            if revision:
                pipeline_kwargs["revision"] = revision
            if device is not None:
                pipeline_kwargs["device"] = device
            self._pipeline = pipeline("sentiment-analysis", **pipeline_kwargs)
        except Exception as exc:  # pragma: no cover - hit if model download fails
            logger.error(
                "Falling back to rule-based sentiment analyzer for %s (%s)",
                model_name,
                exc,
            )
            self._pipeline = None

    def analyze(self, texts: Iterable[str]) -> List[SentimentResult]:
        results: List[SentimentResult] = []
        if self._pipeline is None:
            for text in texts:
                score = 0.5
                label = "Neutral"
                lowered = text.lower()
                if any(word in lowered for word in ["great", "thanks", "appreciate", "good job"]):
                    label, score = "Positive", 0.7
                elif any(word in lowered for word in ["unhappy", "bad", "frustrated", "issue"]):
                    label, score = "Negative", 0.7
                results.append(SentimentResult(label=label, score=score))
            return results

        raw = self._pipeline(list(texts))
        for record in raw:
            label = record["label"].capitalize()
            if label not in {"Positive", "Negative"}:
                label = "Neutral"
            results.append(SentimentResult(label=label, score=float(record["score"])))
        return results


__all__ = ["SentimentAnalyzer", "SentimentResult"]
