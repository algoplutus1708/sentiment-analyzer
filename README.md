# Sentiment Analyzer Web App

This project includes:
- `LLM.ipynb` for training/exporting the model
- Python backend API for inference (`app/server.py`)
- Browser frontend (`templates/` + `static/`)

## Prerequisites
- Python 3.10+
- Trained checkpoint in project root: `tinyllm_complete.pt`

## Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run the app
```bash
python -m app.server
```

Open: `http://127.0.0.1:8000`

## API
- `GET /health`
- `POST /api/predict`

Example request body:
```json
{
  "text": "This movie was amazing and emotional."
}
```

## Notes
- If `tinyllm_complete.pt` is missing, run notebook training/saving first.
- Frontend calls backend endpoint `/api/predict`.
