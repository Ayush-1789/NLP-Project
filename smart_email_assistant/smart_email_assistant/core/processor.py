"""Smart Email Assistant core orchestrator."""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from ..config import AppConfig, DEFAULT_CONFIG
from ..pipelines.actions import ActionItemExtractor
from ..pipelines.analytics import AnalyticsEngine, EmailAnalyticsRecord
from ..pipelines.classification import ClassificationResult, EmailClassifier
from ..pipelines.features import FeatureBundle, FeatureExtractor
from ..pipelines.ner_wsd import EntityAndWSDExtractor
from ..pipelines.preprocess import EmailPreprocessor, PreprocessResult
from ..pipelines.sentiment import SentimentAnalyzer, SentimentResult
from ..pipelines.summarization import HybridSummarizer
from ..utils.cache import CacheManager
from ..utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ProcessedEmail:
    email_id: str
    subject: str
    sender: str
    body: str
    timestamp: datetime
    classification: ClassificationResult
    sentiment: SentimentResult
    preprocess: PreprocessResult
    entities: Dict[str, List]
    action_items: List[Dict[str, object]]
    summary: Dict[str, List]

    def to_dict(self) -> Dict[str, object]:
        return {
            "email_id": self.email_id,
            "subject": self.subject,
            "sender": self.sender,
            "timestamp": self.timestamp.isoformat(),
            "body": self.body,
            "classification": {
                "label": self.classification.label,
                "confidence": self.classification.confidence,
                "scores": self.classification.scores,
            },
            "sentiment": {
                "label": self.sentiment.label,
                "score": self.sentiment.score,
            },
            "preprocess": {
                "cleaned_text": self.preprocess.cleaned_text,
                "tokens": self.preprocess.tokens,
                "lemmas": self.preprocess.lemmas,
                "pos_tags": self.preprocess.pos_tags,
                "dependencies": self.preprocess.dependencies,
                "sentences": self.preprocess.sentences,
            },
            "entities": self.entities,
            "action_items": self.action_items,
            "summary": self.summary,
        }


class SmartEmailProcessor:
    """Coordinate all processing stages."""

    def __init__(self, config: AppConfig | None = None, data_dir: Optional[Path] = None) -> None:
        self.config = config or DEFAULT_CONFIG
        self.data_dir = data_dir or Path(__file__).resolve().parent.parent / "data"
        self.cache = CacheManager(self.config.cache)

        self.preprocessor = EmailPreprocessor(self.config.model)
        self.feature_extractor = FeatureExtractor(self.config.model, cache=self.cache)
        self.classifier = EmailClassifier(self.config.classification, cache=self.cache)
        self.sentiment_analyzer = SentimentAnalyzer(
            model_name=self.config.model.sentiment_model,
            revision=self.config.model.sentiment_revision,
        )
        self.entity_extractor = EntityAndWSDExtractor(self.config.model)
        self.action_extractor = ActionItemExtractor(self.config.model)
        self.summarizer = HybridSummarizer(self.config.summarization)
        self.analytics_engine = AnalyticsEngine()

        self._ensure_training_data()

    def _ensure_training_data(self) -> None:
        try:
            samples_path = self.data_dir / "sample_emails.json"
            if not samples_path.exists():
                logger.warning("Sample training data not found at %s", samples_path)
                return
            samples = json.loads(samples_path.read_text(encoding="utf-8"))
            bodies = [sample["body"] for sample in samples]
            labels = [sample["category"] for sample in samples]
            self.feature_extractor.fit_vectorizer(sample["body"] for sample in samples)
            features = self.feature_extractor.build_features(bodies)
            self.classifier.fit(features.tfidf, features.transformer, labels)
            logger.info("Bootstrapped classifier using %d sample emails", len(samples))
        except Exception as exc:  # pragma: no cover - training should succeed
            logger.error("Failed to bootstrap classifier: %s", exc)

    def _build_features(self, cleaned_texts: Iterable[str]) -> FeatureBundle:
        texts = list(cleaned_texts)
        try:
            return self.feature_extractor.build_features(texts)
        except RuntimeError:
            if texts:
                self.feature_extractor.fit_vectorizer(texts)
                return self.feature_extractor.build_features(texts)
            raise

    def process_email(self, email: Dict[str, object]) -> ProcessedEmail:
        email_id = str(email.get("id", email.get("email_id", "unknown")))
        subject = str(email.get("subject", ""))
        sender = str(email.get("sender", "unknown"))
        body = str(email.get("body", ""))
        timestamp_value = email.get("timestamp")
        if isinstance(timestamp_value, str):
            timestamp = datetime.fromisoformat(timestamp_value)
        elif isinstance(timestamp_value, datetime):
            timestamp = timestamp_value
        else:
            timestamp = datetime.utcnow()

        preprocess = self.preprocessor.process(body)
        features = self._build_features([preprocess.cleaned_text])
        classification = self.classifier.predict(features.tfidf, features.transformer, [body])[0]
        sentiment = self.sentiment_analyzer.analyze([body])[0]
        entity_payload = self.entity_extractor.analyze(body)
        action_items_records = self.action_extractor.extract(body)
        action_items = [record.__dict__ for record in action_items_records]
        summary = self.summarizer.summarize(
            preprocess.sentences,
            entity_payload.get("entities", []),
            [item["text"] for item in action_items],
        )

        processed = ProcessedEmail(
            email_id=email_id,
            subject=subject,
            sender=sender,
            body=body,
            timestamp=timestamp,
            classification=classification,
            sentiment=sentiment,
            preprocess=preprocess,
            entities=entity_payload,
            action_items=action_items,
            summary=summary,
        )
        return processed

    def process_inbox(self, emails: Iterable[Dict[str, object]]) -> Dict[str, object]:
        processed_emails: List[ProcessedEmail] = []
        for email in emails:
            processed_emails.append(self.process_email(email))

        analytics_records = [
            EmailAnalyticsRecord(
                email_id=processed.email_id,
                category=processed.classification.label,
                sentiment=processed.sentiment.label,
                timestamp=processed.timestamp,
                action_items=len(processed.action_items),
                entities=len(processed.entities.get("entities", [])),
            )
            for processed in processed_emails
        ]
        analytics_raw = self.analytics_engine.compute(analytics_records)
        analytics = {
            "totals": analytics_raw.get("totals", {}),
            "categories": dict(analytics_raw.get("categories", {})),
            "sentiments": dict(analytics_raw.get("sentiments", {})),
            "timeline": analytics_raw.get("timeline", {}),
        }
        return {
            "emails": [email.to_dict() for email in processed_emails],
            "analytics": analytics,
        }


__all__ = ["SmartEmailProcessor", "ProcessedEmail"]
