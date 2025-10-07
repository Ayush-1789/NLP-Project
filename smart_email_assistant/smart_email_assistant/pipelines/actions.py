"""Action item extraction."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import spacy
from spacy.language import Language

from ..config import ModelConfig


@dataclass
class ActionItem:
    text: str
    verb: str
    owner: str | None
    due_date: str | None


class ActionItemExtractor:
    """Identify actionable sentences within emails."""

    ACTION_VERBS = {
        "schedule",
        "review",
        "approve",
        "send",
        "call",
        "follow",
        "complete",
        "finish",
        "submit",
        "prepare",
        "update",
    }

    def __init__(self, config: ModelConfig) -> None:
        self.nlp: Language = spacy.load(config.spacy_model)

    def extract(self, text: str) -> List[ActionItem]:
        doc = self.nlp(text)
        items: List[ActionItem] = []
        for sent in doc.sents:
            if any(token.lemma_.lower() in self.ACTION_VERBS for token in sent if token.pos_ == "VERB"):
                verb = next(
                    (token.lemma_ for token in sent if token.pos_ == "VERB" and token.lemma_.lower() in self.ACTION_VERBS),
                    sent.root.lemma_,
                )
                owner = None
                due_date = None
                for ent in sent.ents:
                    if ent.label_ in {"PERSON", "ORG"}:
                        owner = ent.text
                    if ent.label_ in {"DATE", "TIME"}:
                        due_date = ent.text
                items.append(ActionItem(text=sent.text.strip(), verb=verb, owner=owner, due_date=due_date))
        return items


__all__ = ["ActionItemExtractor", "ActionItem"]
