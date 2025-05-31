# nlp_model

A full-stack NLP project that classifies emotional tone from English sentences and rewrites them according to a specified tone and target audience.

- Emotion classifier: BERT + RandomForest
- Rewriting powered by Google Gemini (LangChain)
- Fullstack: Flask backend + React frontend with Ant Design
- Ready for deployment (GCP, GitHub)

---

## Features

- Detects multiple emotions from text (multi-label)
- Automatically rewrites text based on selected tone & audience
- Frontend styled using [Ant Design](https://ant.design/)
- Trained model saved and loaded from local files
- Google API Key handled via `.env` file

---

## Project Structure

```
nlp-model/
├── backend/
│   ├── app.py                    # Flask API with /predict and /revise endpoints
│   ├── revise_chain.py           # LangChain + Gemini rewriting logic
│   ├── model/
│   │   └── rf.py                 # RandomForest training script (BERT embeddings)
│   └── .env                      # API key file (ignored by git)
│
├── frontend/
│   ├── src/
│   │   └── App.jsx               # Main React app with Ant Design components
│   └── public/
│
├── goemotions_data/             # Original raw & cleaned datasets (ignored by git)
│
├── mlp.py                       # Optional MLP-based classifier
├── .env.example                 # Template for backend env file
├── .gitignore                   # Git ignore rules
└── README.md                    # You are here
```

---

## Getting Started

### Prerequisites

- Python 3.10+
- Node.js (v18 recommended)
- OpenAI or Google Gemini API key

---

### Backend Setup

```bash
cd backend
python -m venv venv
source venv/bin/activate      # On Windows: .\venv\Scripts\activate
pip install -r requirements.txt
```

Create a `.env` file inside the `backend/` directory:

```
GOOGLE_API_KEY=your_google_genai_key_here
```

Then start the Flask server:

```bash
python app.py
# Server running on http://localhost:8080
```

---

### Frontend Setup

```bash
cd frontend
npm install
npm start
# App will open at http://localhost:3000
```

---

## Example Usage

1. Type a sentence like:

   > “I am frustrated and disappointed.”

2. View **detected emotions** immediately

3. Fill in:

   - Desired tone: `polite`
   - Audience: `my professor`

4. Click **Revise** and see rewritten version below

---

## Environment Variables

Create your `.env` in `backend/` using this template:

```bash
GOOGLE_API_KEY=your_google_genai_api_key_here
```

Make sure this file is **NOT** committed by git:

```
# In .gitignore
backend/.env
```

---

## Git Ignore Summary

Your `.gitignore` should include:

```bash
# Python
venv/
*.pyc
__pycache__/

# Node
frontend/node_modules/

# Env/API keys
.env
backend/.env

# Dataset files
goemotions_data/*.csv
```

## Model Information

- **Classifier**: RandomForest trained on GoEmotions + BERT embeddings
- **Embedding**: `all-MiniLM-L6-v2` from `sentence-transformers`
- **Rewrite**: Google Gemini 1.5 Flash via LangChain

Trained model files are already included in:

```
backend/model/final_model/
```

---

## Testing API (example via Postman)

**Emotion Classification**

```json
POST http://localhost:8080/predict
{
  "text": "I’m extremely happy and excited today!"
}
```

**Rewrite Suggestion**

```json
POST http://localhost:8080/revise
{
  "text": "Need help with my job!",
  "tone": "polite",
  "audience": "a recruiter"
}
```

## Acknowledgements

- Dataset: [Google Research – GoEmotions](https://github.com/google-research/goemotions)
- Embedding: [Sentence Transformers](https://www.sbert.net/)
- Rewrite: [LangChain](https://www.langchain.com/) + [Google Gemini API](https://ai.google.dev/)
