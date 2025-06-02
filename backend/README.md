# Backend – Flask API

This is the backend for the Emotion Classification & Rewriting App. It provides two main endpoints:

## Features

- `/predict`: Returns multi-label emotion classification from text
- `/revise`: Returns a rewritten version of the text using Gemini via LangChain

## Setup

1. Rename `.env.example` to `.env` and insert your Google API key:

```
GOOGLE_API_KEY=your_key_here
```

2. Install dependencies:

```bash
python3 -m venv venv
source venv/bin/activate  # or venv\Scripts\activate
pip install -r requirements.txt
```

3. Run the app:

```bash
python app.py
```

## Optional: Train RF+BERT Model

Use `rf.py` to train a more accurate Random Forest classifier using sentence embeddings.

## Output

- `model/model.pkl` – TF-IDF logistic regression model (used in deployment)
- `final_model/` – BERT + RF model output (not deployed)

## Endpoints

- POST `/predict`: `{ text: "your text" }`
- POST `/revise`: `{ text: "...", tone: "...", audience: "..." }`
