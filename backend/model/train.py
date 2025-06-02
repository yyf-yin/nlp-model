import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
import joblib

data_path = Path(__file__).parent.parent / "goemotions_data" / "goemotions_clean.csv"
df = pd.read_csv(data_path)
df["labels"] = df["labels"].apply(eval)

X_train, X_test, y_train_raw, y_test_raw = train_test_split(df["text"], df["labels"], test_size=0.2, random_state=42)
mlb = MultiLabelBinarizer()
y_train = mlb.fit_transform(y_train_raw)
y_test = mlb.transform(y_test_raw)

tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

clf = OneVsRestClassifier(LogisticRegression(max_iter=300, solver="liblinear"))
clf.fit(X_train_tfidf, y_train)

out_dir = Path(__file__).parent.parent / "model"
out_dir.mkdir(exist_ok=True)
joblib.dump(clf, out_dir / "model.pkl")
joblib.dump(tfidf, out_dir / "tfidf.pkl")
joblib.dump(mlb, out_dir / "mlb.pkl")

print("model saved")
print(f"model performance: {clf.score(X_test_tfidf, y_test):.4f}")

from sklearn.metrics import f1_score

y_pred = clf.predict(X_test_tfidf)
print("Micro F1:", f1_score(y_test, y_pred, average="micro"))
print("Macro F1:", f1_score(y_test, y_pred, average="macro"))
print("Weighted F1:", f1_score(y_test, y_pred, average="weighted"))
