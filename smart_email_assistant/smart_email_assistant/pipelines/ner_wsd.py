"""Named entity recognition, word sense disambiguation, and relationship extraction."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Optional

import spacy

try:
    from nltk.corpus import wordnet as wn
    from nltk.wsd import lesk
except LookupError:  # pragma: no cover - executed on first import without corpora
    import nltk

    nltk.download("wordnet")
    nltk.download("omw-1.4")
    from nltk.corpus import wordnet as wn
    from nltk.wsd import lesk
except ImportError:  # pragma: no cover - fallback if nltk missing
    wn = None
    lesk = None

from spacy.language import Language
from spacy.tokens import Doc

from ..config import ModelConfig
from ..utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class EntityRecord:
    text: str
    label: str
    start_char: int
    end_char: int


@dataclass
class RelationshipRecord:
    subject: str
    relation: str
    obj: str


@dataclass
class WSDRecord:
    word: str
    sense: Optional[str]
    definition: Optional[str]


@lru_cache(maxsize=1)
def _load_spacy(model_name: str) -> Language:
    return spacy.load(model_name)


class EntityAndWSDExtractor:
    """Extract entities, determine relationships, and perform WSD."""

    def __init__(self, config: ModelConfig) -> None:
        self.config = config
        self.nlp = _load_spacy(config.spacy_model)

    def _ensure_doc(self, text: str, doc: Optional[Doc] = None) -> Doc:
        return doc if doc is not None else self.nlp(text)

    def extract_entities(self, text: str, doc: Optional[Doc] = None) -> List[EntityRecord]:
        doc = self._ensure_doc(text, doc)
        return [
            EntityRecord(ent.text, ent.label_, ent.start_char, ent.end_char)
            for ent in doc.ents
        ]

    def extract_relationships(self, text: str, doc: Optional[Doc] = None) -> List[RelationshipRecord]:
        doc = self._ensure_doc(text, doc)
        relationships: List[RelationshipRecord] = []
        for token in doc:
            if token.dep_ == "ROOT" and token.pos_ == "VERB":
                subject = None
                obj = None
                for child in token.children:
                    if child.dep_ in {"nsubj", "nsubjpass"}:
                        subject = child.text
                    if child.dep_ in {"dobj", "pobj", "attr", "dative"}:
                        obj = child.text
                if subject and obj:
                    relationships.append(RelationshipRecord(subject=subject, relation=token.lemma_, obj=obj))
        return relationships

    def disambiguate(self, text: str, doc: Optional[Doc] = None) -> List[WSDRecord]:
        doc = self._ensure_doc(text, doc)
        if wn is None or lesk is None:
            logger.warning("NLTK WordNet resources not available; skipping WSD.")
            return []
        sentences = [sent.text for sent in doc.sents]
        results: List[WSDRecord] = []
        for sent in sentences:
            sent_doc = self.nlp(sent)
            context_words = [token.text for token in sent_doc]
            for token in sent_doc:
                if token.pos_ in {"NOUN", "VERB", "ADJ"}:
                    sense = lesk(context_words, token.text)
                    if sense is not None:
                        results.append(
                            WSDRecord(
                                word=token.text,
                                sense=sense.name(),
                                definition=sense.definition(),
                            )
                        )
        return results

    def analyze(self, text: str, doc: Optional[Doc] = None) -> Dict[str, List]:
        doc = self._ensure_doc(text, doc)
        entities = self.extract_entities(text, doc)
        relationships = self.extract_relationships(text, doc)
        wsd = self.disambiguate(text, doc)
        return {
            "entities": [entity.__dict__ for entity in entities],
            "relationships": [rel.__dict__ for rel in relationships],
            "word_senses": [sense.__dict__ for sense in wsd],
        }


__all__ = ["EntityAndWSDExtractor", "EntityRecord", "RelationshipRecord", "WSDRecord"]
