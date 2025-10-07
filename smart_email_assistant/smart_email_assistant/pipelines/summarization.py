"""Hybrid extractive summarization utilities."""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from ..config import SummarizationConfig


class HybridSummarizer:
    """Combine multiple heuristics for robust extractive summaries."""

    def __init__(self, config: SummarizationConfig) -> None:
        self.config = config

    def summarize(
        self,
        sentences: Sequence[str],
        entities: Sequence[Dict[str, str]] | None = None,
        action_items: Sequence[str] | None = None,
    ) -> Dict[str, List[str]]:
        if not sentences:
            return {"summary": [], "scored_sentences": []}

        entity_texts = {ent["text"].lower() for ent in entities or []}
        action_texts = {item.lower() for item in action_items or []}

        tfidf_vectorizer = TfidfVectorizer(stop_words="english")
        tfidf_matrix = tfidf_vectorizer.fit_transform(sentences)
        tfidf_scores = tfidf_matrix.max(axis=1).toarray().flatten()

        positional_scores = np.array([
            self.config.positional_weight / (idx + 1)
            for idx in range(len(sentences))
        ])

        entity_scores = np.array([
            self.config.entity_weight
            if any(entity in sentences[idx].lower() for entity in entity_texts)
            else 0.0
            for idx in range(len(sentences))
        ])

        action_scores = np.array([
            self.config.action_weight
            if any(action in sentences[idx].lower() for action in action_texts)
            else 0.0
            for idx in range(len(sentences))
        ])

        combined = (
            self.config.tfidf_weight * tfidf_scores
            + positional_scores
            + entity_scores
            + action_scores
        )

        ranked_indices = np.argsort(-combined)
        top_indices = sorted(ranked_indices[: self.config.max_sentences])
        summary_sentences = [sentences[idx] for idx in top_indices]

        scored = [
            {
                "sentence": sentences[idx],
                "score": float(combined[idx]),
                "position": idx,
            }
            for idx in ranked_indices
        ]

        return {
            "summary": summary_sentences,
            "scored_sentences": scored,
        }


__all__ = ["HybridSummarizer"]
