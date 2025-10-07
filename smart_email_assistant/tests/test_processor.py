from __future__ import annotations

from datetime import datetime

from smart_email_assistant import SmartEmailProcessor


def test_process_single_email():
    processor = SmartEmailProcessor()
    email = {
        "id": "test-1",
        "subject": "Team Standup",
        "sender": "manager@example.com",
        "body": "Please prepare the status update and send the report by tomorrow.",
        "timestamp": datetime.utcnow().isoformat(),
    }
    result = processor.process_inbox([email])
    assert "emails" in result
    assert result["emails"][0]["classification"]["label"] in {"Urgent", "Important", "Normal", "Spam"}
    assert result["emails"][0]["sentiment"]["label"] in {"Positive", "Negative", "Neutral"}
    assert isinstance(result["analytics"], dict)
