"""Cache utilities for persisting models and artefacts."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from joblib import dump, load

from ..config import CacheConfig
from .logger import get_logger

logger = get_logger(__name__)


class CacheManager:
    """Manage persistence of vectorizers, classifiers, and analytics caches."""

    def __init__(self, config: CacheConfig) -> None:
        self.config = config
        self.base_dir = config.base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, name: str, suffix: str) -> Path:
        return self.base_dir / f"{name}{suffix}"

    def save_vectorizer(self, vectorizer: Any, name: str = "tfidf_vectorizer") -> None:
        if not self.config.persist_vectorizer:
            return
        path = self._path(name, ".joblib")
        dump(vectorizer, path)
        logger.info("Persisted vectorizer to %s", path)

    def load_vectorizer(self, name: str = "tfidf_vectorizer") -> Optional[Any]:
        path = self._path(name, ".joblib")
        if path.exists():
            logger.info("Loaded cached vectorizer from %s", path)
            return load(path)
        return None

    def save_classifier(self, classifier: Any, name: str = "email_classifier") -> None:
        if not self.config.persist_classifier:
            return
        path = self._path(name, ".joblib")
        dump(classifier, path)
        logger.info("Persisted classifier to %s", path)

    def load_classifier(self, name: str = "email_classifier") -> Optional[Any]:
        path = self._path(name, ".joblib")
        if path.exists():
            logger.info("Loaded cached classifier from %s", path)
            return load(path)
        return None

    def save_json(self, payload: Any, name: str) -> None:
        path = self._path(name, ".json")
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        logger.info("Persisted JSON payload to %s", path)

    def load_json(self, name: str) -> Optional[Any]:
        path = self._path(name, ".json")
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        return None


__all__ = ["CacheManager"]
