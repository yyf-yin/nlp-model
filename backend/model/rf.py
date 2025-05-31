import pandas as pd
import numpy as np
import ast
import os
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sentence_transformers import SentenceTransformer

# load data
base_dir = os.path.dirname(__file__)
data_path = os.path.join(base_dir, "..", "goemotions_data", "goemotions_clean.csv")
output_dir = os.path.join(base_dir, "final_model")
df = pd.read_csv(data_path)

# process data
df["labels"] = df["labels"].apply(ast.literal_eval)
encoder = SentenceTransformer("all-MiniLM-L6-v2")
X = encoder.encode(df["text"].tolist(), show_progress_bar=True)
mlb = MultiLabelBinarizer()
Y = mlb.fit_transform(df["labels"])

# train
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
model.fit(X_train, Y_train)

# evaluation
from sklearn.metrics import f1_score
threshold = 0.3
probs = model.predict_proba(X_test)
probs_matrix = np.array([p[:, 1] if p.ndim == 2 else p for p in probs]).T
preds = (probs_matrix >= threshold).astype(int)
f1_micro = f1_score(Y_test, preds, average='micro')
f1_macro = f1_score(Y_test, preds, average='macro')
f1_weighted = f1_score(Y_test, preds, average='weighted')

print("\n F1 Score Evaluation:")
print(f"Micro F1:    {f1_micro:.4f}")
print(f"Macro F1:    {f1_macro:.4f}")
print(f"Weighted F1: {f1_weighted:.4f}")

# save
os.makedirs(output_dir, exist_ok=True)
joblib.dump(model, os.path.join(output_dir, "randomforest_bert_model.pkl"))
joblib.dump(mlb, os.path.join(output_dir, "mlb.pkl"))
from sentence_transformers import SentenceTransformer
encoder = SentenceTransformer("all-MiniLM-L6-v2")
encoder.save(os.path.join(output_dir, "bert_encoder"))

print("Model, encoder, and label binarizer saved to:", output_dir)

# test
threshold = 0.3
probs = model.predict_proba(X_test)
probs_matrix = np.array([p[:, 1] if p.ndim == 2 else p for p in probs]).T
preds = (probs_matrix >= threshold).astype(int)
decoded = mlb.inverse_transform(preds)
print("\nSample Predictions:")
for i in range(3):
    print(f"Text: {df['text'].iloc[i]}")
    print(f"Predicted Labels: {decoded[i]}")
