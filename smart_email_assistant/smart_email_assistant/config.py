"""Configuration settings for the Smart Email Assistant."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple


@dataclass
class ModelConfig:
    """Configuration for third-party NLP models."""

    transformer_model: str = "distilbert-base-uncased"
    spacy_model: str = "en_core_web_sm"
    sentiment_model: str = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
    sentiment_revision: str | None = "714eb0f"
    tfidf_max_features: int = 1000
    tfidf_ngram_range: Tuple[int, int] = (1, 2)


@dataclass
class ClassificationConfig:
    """Configuration for email classification categories and thresholds."""

    categories: List[str] = field(
        default_factory=lambda: ["Urgent", "Important", "Normal", "Spam"]
    )
    urgency_keywords: Dict[str, List[str]] = field(
        default_factory=lambda: {
            "Urgent": [
                "asap",
                "immediately",
                "urgent",
                "deadline",
                "action required",
                "important",
            ],
            "Important": [
                "follow up",
                "meeting",
                "schedule",
                "invoice",
                "proposal",
                "review",
            ],
            "Spam": ["lottery", "prince", "inheritance", "free money", "click"],
        }
    )
    spam_threshold: float = 0.75
    urgency_threshold: float = 0.65


@dataclass
class SummarizationConfig:
    """Configuration for the hybrid summarization pipeline."""

    max_sentences: int = 3
    action_weight: float = 1.2
    entity_weight: float = 1.1
    positional_weight: float = 1.05
    tfidf_weight: float = 0.95


@dataclass
class CacheConfig:
    """Configuration for on-disk caching of models and pipeline artefacts."""

    base_dir: Path = Path(".cache")
    persist_vectorizer: bool = True
    persist_classifier: bool = True


@dataclass
class AppConfig:
    """Top-level configuration for the Smart Email Assistant application."""

    model: ModelConfig = field(default_factory=ModelConfig)
    classification: ClassificationConfig = field(default_factory=ClassificationConfig)
    summarization: SummarizationConfig = field(default_factory=SummarizationConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)


DEFAULT_CONFIG = AppConfig()

__all__ = [
    "ModelConfig",
    "ClassificationConfig",
    "SummarizationConfig",
    "CacheConfig",
    "AppConfig",
    "DEFAULT_CONFIG",
]
