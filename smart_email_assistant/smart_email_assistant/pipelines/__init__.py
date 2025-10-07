"""Processing pipelines for the Smart Email Assistant."""

from .actions import ActionItemExtractor
from .analytics import AnalyticsEngine
from .classification import EmailClassifier
from .features import FeatureExtractor
from .ner_wsd import EntityAndWSDExtractor
from .preprocess import EmailPreprocessor
from .sentiment import SentimentAnalyzer
from .summarization import HybridSummarizer

__all__ = [
    "ActionItemExtractor",
    "AnalyticsEngine",
    "EmailClassifier",
    "FeatureExtractor",
    "EntityAndWSDExtractor",
    "EmailPreprocessor",
    "SentimentAnalyzer",
    "HybridSummarizer",
]
