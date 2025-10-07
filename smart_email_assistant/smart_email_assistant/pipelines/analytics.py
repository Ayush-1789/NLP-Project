"""Analytics aggregation for processed emails."""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Iterable, List


@dataclass
class EmailAnalyticsRecord:
    email_id: str
    category: str
    sentiment: str
    timestamp: datetime
    action_items: int
    entities: int


class AnalyticsEngine:
    """Compute aggregate metrics for dashboards."""

    def compute(self, records: Iterable[EmailAnalyticsRecord]) -> Dict[str, object]:
        category_counter: Counter[str] = Counter()
        sentiment_counter: Counter[str] = Counter()
        timeline: Dict[str, int] = defaultdict(int)
        total_actions = 0
        total_entities = 0
        total_emails = 0

        for record in records:
            total_emails += 1
            category_counter[record.category] += 1
            sentiment_counter[record.sentiment] += 1
            timeline_key = record.timestamp.strftime("%Y-%m-%d")
            timeline[timeline_key] += 1
            total_actions += record.action_items
            total_entities += record.entities

        return {
            "totals": {
                "emails": total_emails,
                "action_items": total_actions,
                "entities": total_entities,
            },
            "categories": category_counter,
            "sentiments": sentiment_counter,
            "timeline": dict(sorted(timeline.items())),
        }


__all__ = ["AnalyticsEngine", "EmailAnalyticsRecord"]
