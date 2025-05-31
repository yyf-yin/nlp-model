import pandas as pd
import numpy as np
import os
import joblib
import time
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.multiclass import OneVsRestClassifier
from sentence_transformers import SentenceTransformer
from pathlib import Path
import torch

# === Config ===
MODEL_NAME = "all-mpnet-base-v2"  # Try: all-MiniLM-L6-v2, paraphrase-MiniLM-L6-v2
THRESHOLDS = [0.3, 0.4, 0.5]
CLASSIFIERS = {
    "Logistic": LogisticRegression(max_iter=1000),
    "RandomForest": RandomForestClassifier(n_estimators=150, n_jobs=-1),  # ✅ 并行训练
    "MLP": MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=300)
}

# === Load Data ===
data_path = Path(__file__).parent.parent / "goemotions_data" / "goemotions_clean.csv"
df = pd.read_csv(data_path)
df["labels"] = df["labels"].apply(eval)
texts = df["text"].tolist()

mlb = MultiLabelBinarizer()
y = mlb.fit_transform(df["labels"])

# === Embed Text ===
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
print(f"Loading sentence embedding model: {MODEL_NAME}")
encoder = SentenceTransformer(MODEL_NAME).to(device)

start_embed = time.time()
X = encoder.encode(texts, show_progress_bar=True)
X = np.array(X, dtype=np.float32)  
print(f"Sentence embedding complete in {time.time() - start_embed:.2f} seconds")

# === Train/Test split ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# === Run all classifier + threshold combos ===
results = []

for clf_name, clf_model in CLASSIFIERS.items():
    print(f"\n--- Training classifier: {clf_name} ---")
    start_time = time.time()

    model = OneVsRestClassifier(clf_model)
    model.fit(X_train, y_train)

    elapsed = time.time() - start_time
    print(f"{clf_name} training completed in {elapsed:.2f} seconds")

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_test)
    else:
        probs = model.decision_function(X_test)
        probs = 1 / (1 + np.exp(-probs))  

    for threshold in THRESHOLDS:
        preds = (probs >= threshold).astype(int)
        micro = f1_score(y_test, preds, average="micro")
        macro = f1_score(y_test, preds, average="macro")
        results.append((clf_name, threshold, micro, macro))
        print(f"→ {clf_name} @ threshold {threshold:.2f}: micro-F1={micro:.4f}, macro-F1={macro:.4f}")

    
    preds = (probs >= 0.3).astype(int)
    f1 = f1_score(y_test, preds, average="micro")
    print(f"Saving {clf_name} model with threshold=0.3 (F1={f1:.4f})")
    base = os.path.dirname(__file__)
    joblib.dump(model, os.path.join(base, f"{clf_name.lower()}_bert_model.pkl"))


joblib.dump(encoder, os.path.join(base, "bert_encoder.pkl"))
joblib.dump(mlb, os.path.join(base, "mlb.pkl"))


print("\n==== Model Performance Summary ====")
for clf_name, threshold, micro, macro in results:
    print(f"{clf_name} @ {threshold:.2f} → micro: {micro:.4f}, macro: {macro:.4f}")
