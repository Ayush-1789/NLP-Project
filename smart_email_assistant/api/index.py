"""Vercel serverless entrypoint for the Smart Email Assistant."""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure the project root is importable when running on Vercel
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
	sys.path.append(str(PROJECT_ROOT))

from smart_email_assistant.web.server import create_app

app = create_app()
handler = app  # Vercel expects a WSGI-compatible callable named "handler"
