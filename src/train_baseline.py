"""TF-IDF + logistic regression baseline on Split A and Split B.

Writes results/baseline_metrics.csv.
Run from the repo root:  python src/train_baseline.py
"""
from pathlib import Path

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from evaluate import compute_metrics
from splits import load_clean, split_random, split_temporal

SEED = 42
OUT = Path(__file__).resolve().parent.parent / "results" / "baseline_metrics.csv"


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


def main() -> None:
    df = load_clean()
    rows = []
    for split_name, (train, test) in [("A_random", split_random(df)), ("B_temporal", split_temporal(df))]:
        for model_name, model in make_models().items():  # fresh, untrained models for each split
            model.fit(train["text"], train["label"])
            metrics = compute_metrics(test["label"], model.predict(test["text"]))
            rows.append({"model": model_name, "split": split_name,
                         "n_train": len(train), "n_test": len(test), **metrics})

    results = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUT, index=False)
    print(results.to_string(index=False, float_format="%.4f"))
    print(f"\nwrote {OUT.name}")


if __name__ == "__main__":
    main()
