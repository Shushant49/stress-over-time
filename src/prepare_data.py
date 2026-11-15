"""Build the single cleaned dataset that both Split A and Split B are cut from.

Reads data/raw/dreaddit_{train,test}.csv, writes data/processed/dreaddit_clean.csv.
Run from the repo root:  python src/prepare_data.py
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed" / "dreaddit_clean.csv"

# Exists only in Jan 2018, so it would mix a topic shift into the time shift.
# See notebooks/01_inspect_data.ipynb, step 2.
DROP_SUBREDDITS = ["relationships"]

KEEP_COLS = ["post_id", "subreddit", "text", "label", "confidence", "social_timestamp"]


def build_clean() -> pd.DataFrame:
    parts = []
    for split in ["train", "test"]:
        df = pd.read_csv(RAW_DIR / f"dreaddit_{split}.csv", usecols=KEEP_COLS)
        df["orig_split"] = split  # Split A reuses the dataset's own train/test division
        parts.append(df)
    df = pd.concat(parts, ignore_index=True)

    df = df[~df["subreddit"].isin(DROP_SUBREDDITS)]  # ~ means NOT: keep rows whose subreddit is not in the list
    df["date"] = pd.to_datetime(df["social_timestamp"], unit="s", utc=True)
    # Sort oldest first; post_id breaks ties so the order is identical on every run.
    return df.sort_values(["social_timestamp", "post_id"]).reset_index(drop=True)


def summarize(df: pd.DataFrame) -> None:
    print(f"rows: {len(df)}   unique posts: {df['post_id'].nunique()}")
    print(f"dates: {df['date'].min()}  to  {df['date'].max()}")
    print(f"overall stress rate: {df['label'].mean():.3f}")
    print("\nby original split:")
    print(df.groupby("orig_split").agg(rows=("label", "size"), stress_rate=("label", "mean")))
    print("\nsubreddits:", sorted(df["subreddit"].unique()))


def main() -> None:
    df = build_clean()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"wrote {OUT.relative_to(ROOT)}\n")
    summarize(df)


if __name__ == "__main__":
    main()
