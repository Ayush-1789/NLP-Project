"""Flask web interface for the Smart Email Assistant."""
from __future__ import annotations

import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from flask import Flask, jsonify, redirect, render_template, request, url_for

from ..config import DEFAULT_CONFIG
from ..core.processor import SmartEmailProcessor
from ..utils.logger import get_logger

logger = get_logger(__name__)


def create_app(data_dir: Path | None = None) -> Flask:
    app = Flask(__name__)
    processor = SmartEmailProcessor(config=DEFAULT_CONFIG, data_dir=data_dir)
    raw_emails: List[Dict[str, Any]] = _load_initial_emails(processor)
    inbox_state: Dict[str, Any] = processor.process_inbox(raw_emails)

    @app.get("/")
    def index() -> str:
        return render_template(
            "index.html",
            emails=inbox_state["emails"],
            analytics=inbox_state["analytics"],
        )

    @app.get("/email/<email_id>")
    def email_detail(email_id: str) -> str:
        email = next((email for email in inbox_state["emails"] if email["email_id"] == email_id), None)
        if email is None:
            return redirect(url_for("index"))
        return render_template("email_detail.html", email=email)

    @app.get("/api/emails")
    def api_emails():
        return jsonify(inbox_state["emails"])

    @app.get("/api/analytics")
    def api_analytics():
        return jsonify(_serialize_analytics(inbox_state["analytics"]))

    @app.get("/api/summary")
    def api_summary():
        return jsonify(
            {
                "total_emails": inbox_state["analytics"]["totals"]["emails"],
                "categories": inbox_state["analytics"]["categories"],
                "sentiments": inbox_state["analytics"]["sentiments"],
            }
        )

    @app.post("/process")
    def process_single():
        payload = request.get_json(silent=True)
        if not payload:
            payload = {
                "subject": request.form.get("subject", ""),
                "sender": request.form.get("sender", "anonymous@example.com"),
                "body": request.form.get("body", ""),
                "timestamp": datetime.utcnow().isoformat(),
            }
        payload.setdefault("id", str(uuid.uuid4()))
        raw_emails.append(payload)
        updated = processor.process_inbox(raw_emails)
        inbox_state["emails"] = updated["emails"]
        inbox_state["analytics"] = updated["analytics"]
        return jsonify(updated)

    @app.post("/process_inbox")
    def process_bulk():
        payload = request.get_json(force=True)
        emails = payload.get("emails", [])
        for email in emails:
            email.setdefault("id", str(uuid.uuid4()))
            email.setdefault("timestamp", datetime.utcnow().isoformat())
        raw_emails.extend(emails)
        updated = processor.process_inbox(raw_emails)
        inbox_state["emails"] = updated["emails"]
        inbox_state["analytics"] = updated["analytics"]
        return jsonify(updated)

    return app


def _load_initial_emails(processor: SmartEmailProcessor) -> List[Dict[str, Any]]:
    data_path = processor.data_dir / "sample_emails.json"
    if data_path.exists():
        import json

        return json.loads(data_path.read_text(encoding="utf-8"))
    logger.warning("No sample emails found for initial load")
    return []


def _serialize_analytics(analytics: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "totals": analytics.get("totals", {}),
        "categories": dict(analytics.get("categories", {})),
        "sentiments": dict(analytics.get("sentiments", {})),
        "timeline": analytics.get("timeline", {}),
    }


__all__ = ["create_app"]
