import pandas as pd
from pathlib import Path

data_dir = Path(__file__).parent.parent / "goemotions_data"
csv_files = ["goemotions_1.csv", "goemotions_2.csv", "goemotions_3.csv"]

dfs = [pd.read_csv(data_dir / f) for f in csv_files]
df = pd.concat(dfs, ignore_index=True)

non_label_cols = [
    'text', 'id', 'author', 'subreddit', 'link_id',
    'parent_id', 'created_utc', 'rater_id', 'example_very_unclear'
]
emotion_cols = [col for col in df.columns if col not in non_label_cols]

df["labels"] = df[emotion_cols].apply(
    lambda row: [emotion for emotion in emotion_cols if row[emotion] == 1],
    axis=1
)

df = df[df["labels"].map(len) > 0].reset_index(drop=True)

df_clean = df[["text", "labels"]].copy()
output_path = data_dir / "goemotions_clean.csv"
df_clean.to_csv(output_path, index=False)

print(f"cleaned data: {output_path}")
