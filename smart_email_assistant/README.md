# Smart Email Assistant

A comprehensive NLP-powered email processing system that automatically categorizes emails, analyzes sentiment, extracts named entities and relationships, performs word sense disambiguation, generates summaries, and highlights actionable items. The project is built with Python, spaCy, transformers, scikit-learn, and Flask for an interactive web interface.

## Features

- **Core NLP Pipeline**: preprocessing, TF-IDF vectorization, transformer embeddings, classification, sentiment analysis, NER, dependency parsing, word sense disambiguation, action item extraction, hybrid extractive summarization.
- **Advanced Analytics**: entity relationship extraction, POS distribution, timeline aggregation, actionable insight tracking.
- **Web Interface**: Bootstrap dashboard for categorized inbox, analytics charts, detailed email view, and REST API endpoints (`/api/emails`, `/api/analytics`, `/api/summary`, `/process`, `/process_inbox`).

## Requirements

- Python 3.10+
- spaCy model `en_core_web_sm`
- Transformers model `distilbert-base-uncased`

Install dependencies:

```bash
pip install -r requirements.txt  # installs dependencies and the spaCy model
```

## Quick Start

```bash
export FLASK_APP=smart_email_assistant.web.server:create_app
flask run --reload
```

Visit http://127.0.0.1:5000 to access the dashboard.

## Tests

```bash
pytest
```

## Customization

- Modify `smart_email_assistant/config.py` to adjust model choices, classification categories, or summarization weights.
- Extend `smart_email_assistant/pipelines/ner_wsd.py` to recognize new entity patterns or relationships.
- Customize templates under `smart_email_assistant/web/templates` for UI changes.
