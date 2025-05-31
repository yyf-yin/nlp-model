import os
from flask import Flask, request, jsonify
from flask_cors import CORS 
import pandas as pd
import joblib
import numpy as np
from dotenv import load_dotenv
from revise_chain import revise_chain
from sentence_transformers import SentenceTransformer

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

MODEL_DIR = os.path.join(os.path.dirname(__file__), "model", "final_model")
rf_model = joblib.load(os.path.join(MODEL_DIR, "randomforest_bert_model.pkl"))
mlb = joblib.load(os.path.join(MODEL_DIR, "mlb.pkl"))
encoder = SentenceTransformer(os.path.join(MODEL_DIR, "bert_encoder"))

app = Flask(__name__)
CORS(app)  

@app.route('/')
def home():
    return "Emotion classifier and text revision API is running."

THRESHOLD = 0.3  

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        text = data.get("text", "")
        if not text:
            return jsonify({"error": "No text provided."}), 400

        X = encoder.encode([text])
        preds = rf_model.predict_proba(X)
        preds_array = np.array([p[:, 1] for p in preds])  
        preds_array = preds_array.reshape(1, -1)  

        binary_preds = (preds_array > THRESHOLD).astype(int)

        try:
            predicted_labels = mlb.inverse_transform(binary_preds)
        except Exception as e:
            return jsonify({"error": str(e)}), 500

        if len(predicted_labels[0]) == 0:
            top_indices = np.argsort(preds_array[0])[::-1] 
            top1 = mlb.classes_[top_indices[0]]
            if top1 != "neutral":
                predicted_labels = [[top1]]
            else:
                top2 = mlb.classes_[top_indices[1]]
                predicted_labels = [[top2]]

        return jsonify({"labels": predicted_labels[0]})

    except Exception as e:
        return jsonify({"error": str(e)}), 500



@app.route('/revise', methods=['POST'])
def revise():
    try:
        data = request.get_json()
        original = data.get("text", "")
        emotion = data.get("tone", "")  
        audience = data.get("audience", "")

        if not original or not emotion or not audience:
            return jsonify({"error": "Missing required field(s)."}), 400

        suggestion = revise_chain.invoke({
            "text": original,
            "emotion": emotion,  
            "audience": audience
        })

        return jsonify({"suggestion": suggestion})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
