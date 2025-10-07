"""Email preprocessing utilities."""
from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, Iterable, List

import spacy
from spacy.language import Language

from ..config import ModelConfig
from ..utils.logger import get_logger

logger = get_logger(__name__)

_URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
_EMAIL_PATTERN = re.compile(r"\b[\w.-]+@[\w.-]+\.[a-zA-Z]{2,}\b")
_HTML_TAG_PATTERN = re.compile(r"<[^>]+>")


@dataclass
class PreprocessResult:
    """Structured preprocessing output."""

    cleaned_text: str
    tokens: List[str]
    lemmas: List[str]
    pos_tags: List[str]
    dependencies: List[str]
    sentences: List[str]


@lru_cache(maxsize=1)
def _load_spacy_model(model_name: str) -> Language:
    try:
        return spacy.load(model_name)
    except OSError as exc:  # pragma: no cover - only hit on missing model
        logger.warning("spaCy model %s not found (%s); attempting to download...", model_name, exc)
        from spacy.cli import download

        download(model_name)
        return spacy.load(model_name)


class EmailPreprocessor:
    """Perform cleaning, tokenization, and syntactic analysis on emails."""

    def __init__(self, config: ModelConfig) -> None:
        self.config = config
        self.nlp = _load_spacy_model(config.spacy_model)

    def clean(self, text: str) -> str:
        text = text.lower()
        text = _HTML_TAG_PATTERN.sub(" ", text)
        text = _URL_PATTERN.sub(" ", text)
        text = _EMAIL_PATTERN.sub(" ", text)
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def process(self, text: str) -> PreprocessResult:
        cleaned = self.clean(text)
        doc = self.nlp(cleaned)
        tokens = [token.text for token in doc if not token.is_space]
        lemmas = [token.lemma_ for token in doc if not token.is_space]
        pos_tags = [token.pos_ for token in doc if not token.is_space]
        dependencies = [f"{token.dep_}:{token.head.text}" for token in doc if not token.is_space]
        sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
        return PreprocessResult(
            cleaned_text=cleaned,
            tokens=tokens,
            lemmas=lemmas,
            pos_tags=pos_tags,
            dependencies=dependencies,
            sentences=sentences,
        )

    def get_pos_distribution(self, pos_tags: Iterable[str]) -> Dict[str, int]:
        distribution: Dict[str, int] = {}
        for tag in pos_tags:
            distribution[tag] = distribution.get(tag, 0) + 1
        return distribution


__all__ = ["EmailPreprocessor", "PreprocessResult"]
