"""TF-IDF + logistic regression baseline on Split A and Split B.

Split A = 10 grouped random splits (seeds in splits.RANDOM_SEEDS), reported as mean +/- std.
Also scores the dataset's original train/test split as a side note.
Writes results/baseline_runs.csv (one row per run), results/baseline_summary.csv and
results/baseline_predictions.csv (TF-IDF + LR test predictions, one row per test row per run).
Run from the repo root:  python src/train_baseline.py
"""
from pathlib import Path

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from evaluate import compute_metrics
from splits import RANDOM_SEEDS, load_clean, split_random, split_random_grouped, split_temporal

SEED = 42
RESULTS = Path(__file__).resolve().parent.parent / "results"


def make_models() -> dict:
    return {
        # Sanity floor: always predicts the most common label in train. Any real model must beat this.
        "majority": DummyClassifier(strategy="most_frequent"),
        # A pipeline runs its steps in order. fit() learns the TF-IDF vocabulary from train only;
        # predict() reuses that vocabulary on test, so test data never shapes the features.
        "tfidf_logreg": make_pipeline(
            TfidfVectorizer(
                ngram_range=(1, 2),  # single words and two-word phrases ("not happy")
                min_df=2,            # ignore terms seen in only one train post (typos, names)
                sublinear_tf=True,   # a word used 10 times isn't 10x more important than once
            ),
            LogisticRegression(max_iter=1000, random_state=SEED),
        ),
    }


def all_splits(df: pd.DataFrame):
    """Yield (split_name, seed, train, test) for every run. seed is None where there is no randomness."""
    for seed in RANDOM_SEEDS:
        yield ("A_random", seed, *split_random_grouped(df, seed))
    yield ("B_temporal", None, *split_temporal(df))
    yield ("A_original", None, *split_random(df))  # side note: the dataset's own split


def main() -> None:
    df = load_clean()
    rows, pred_parts = [], []
    for split_name, seed, train, test in all_splits(df):
        for model_name, model in make_models().items():  # fresh, untrained models for each run
            model.fit(train["text"], train["label"])
            preds = model.predict(test["text"])
            metrics = compute_metrics(test["label"], preds)
            rows.append({"model": model_name, "split": split_name, "seed": seed,
                         "n_train": len(train), "n_test": len(test), **metrics})
            if model_name == "tfidf_logreg":  # one row per test row, for breakdowns (no text, to keep it small)
                pred_parts.append(test[["post_id", "subreddit", "social_timestamp", "label"]]
                                  .assign(model=model_name, split=split_name, seed=seed, pred=preds))
    runs = pd.DataFrame(rows)
    pd.concat(pred_parts).to_csv(RESULTS / "baseline_predictions.csv", index=False)

    # One line per (model, split). std is blank for splits with a single run.
    summary = (runs.groupby(["model", "split"], sort=False)
                   .agg(n_runs=("macro_f1", "size"),
                        macro_f1_mean=("macro_f1", "mean"), macro_f1_std=("macro_f1", "std"),
                        accuracy_mean=("accuracy", "mean"), accuracy_std=("accuracy", "std"))
                   .reset_index())

    RESULTS.mkdir(exist_ok=True)
    runs.to_csv(RESULTS / "baseline_runs.csv", index=False)
    summary.to_csv(RESULTS / "baseline_summary.csv", index=False)
    print(summary.to_string(index=False, float_format="%.4f"))
    print("\nwrote baseline_runs.csv, baseline_summary.csv")


if __name__ == "__main__":
    main()
