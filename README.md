# Sentiment Analyzer Web App

Sentiment analysis web app built with a custom TinyLLM-style transformer, a Python HTTP backend, and a React frontend.

The project covers the full workflow:
- training and experimentation in `LLM.ipynb`
- script-based model training in `scripts/train_tinyllm_80.py`
- checkpoint loading and inference in `app/model.py`
- backend API in `app/server.py`
- frontend UI in `templates/` and `static/`

## Features
- binary sentiment classification: `Positive` or `Negative`
- confidence score and per-class probabilities
- lightweight local HTTP server
- browser UI for interactive predictions
- training pipeline for the IMDB movie review dataset

## Dataset

This project uses the **Large Movie Review Dataset (IMDB)** by Andrew Maas et al.

Dataset link:
- https://ai.stanford.edu/~amaas/data/sentiment/

Direct download:
- https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz

After downloading, extract it so the project has this folder structure:

```text
sentiment-analyzer/
├── aclImdb/
│   ├── train/
│   │   ├── pos/
│   │   └── neg/
│   └── test/
│       ├── pos/
│       └── neg/
├── app/
├── scripts/
├── static/
├── templates/
└── README.md
```

By default, the training script expects the dataset directory to be named `aclImdb` in the project root.

## Requirements
- Python 3.10+
- `pip`
- IMDB dataset extracted to `aclImdb/` if you want to train a model
- trained checkpoint `tinyllm_complete.pt` in the project root if you only want to run inference

## Installation
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Train the model

You can train the model either from the notebook or from the script.

### Option 1: Notebook
- Open `LLM.ipynb`
- run the cells to train and export the checkpoint
- make sure the final checkpoint is saved as `tinyllm_complete.pt`

### Option 2: Training script
```bash
python scripts/train_tinyllm_80.py
```

Common options:
```bash
python scripts/train_tinyllm_80.py \
  --data-dir aclImdb \
  --output tinyllm_complete.pt \
  --epochs 8 \
  --batch-size 32 \
  --max-len 256
```

The script:
- reads IMDB reviews from `aclImdb/train` and `aclImdb/test`
- builds a vocabulary from the training split
- trains the model on CPU
- saves the best checkpoint to `tinyllm_complete.pt`

## Run the app

Start the local server:

```bash
python -m app.server
```

Open:
- `http://127.0.0.1:8000`

The server loads `tinyllm_complete.pt` from the project root on startup.

## API

### `GET /health`
Returns server and model status.

Example response fields:
- `status`
- `model_loaded`
- `labels`
- `inference_version`

### `POST /api/predict`
Runs sentiment prediction on input text.

Example request:
```json
{
  "text": "This movie was amazing and emotional."
}
```

Example response fields:
- `label`
- `confidence`
- `probabilities`
- `positive_reply`
- `negative_reply`
- `inference_version`

## Project structure
```text
app/
  __init__.py
  model.py          # tokenizer, transformer model, checkpoint loading, inference
  server.py         # HTTP server and API endpoints
scripts/
  train_tinyllm_80.py  # script-based training entry point
static/
  app.jsx           # React frontend logic
  style.css         # frontend styles
templates/
  index.html        # app shell
LLM.ipynb           # notebook for training and experiments
plot_efficiency.py  # plotting utility
plot_metrics.py     # plotting utility
```

## Notes
- If `tinyllm_complete.pt` is missing, the app will not be able to serve predictions until a checkpoint is trained or added.
- The frontend sends prediction requests to `/api/predict`.
- React is loaded from a CDN in `templates/index.html`, so the browser needs internet access to fetch React assets.
- The provided training script uses CPU by default.
