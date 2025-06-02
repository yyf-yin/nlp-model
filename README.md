# nlp-model

This project builds an NLP system to classify emotions in English text and rewrite it according to a specified tone and audience.

## Features

- Multi-label emotion classification (TF-IDF baseline deployed, RF+BERT optional)
- Text rewriting powered by Google Gemini via LangChain
- React frontend with live feedback
- GCP + Firebase deployment

## Live Demo

Access the app at: https://nlp-model-83d1c.web.app

## Structure

- `backend/` — Flask API for prediction and rewriting
- `frontend/` — React app (Ant Design)
- `rf.py` — Optional script to train a better Random Forest model with BERT embeddings
- `.env.example` — Rename and populate as `.env` to use Gemini API

## Model Info

| Model       | Accuracy | Micro F1 | Macro F1 | Weighted F1 |
| ----------- | -------- | -------- | -------- | ----------- |
| TF-IDF + LR | 0.1425   | 0.2505   | 0.1657   | 0.2211      |
| BERT + RF   | –        | 0.3775   | 0.3164   | 0.3814      |

> Note: Only the TF-IDF model is deployed due to resource constraints.

## Dependencies

- Python (Flask, joblib, scikit-learn, pandas, dotenv)
- LangChain + Google Generative AI
- React + Ant Design (frontend)
