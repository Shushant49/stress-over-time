"""Sanity check (a): how much does the baseline score vary across random splits?

Trains TF-IDF + LR on 10 grouped random splits (seeds 42-51) and compares the spread
with the single Split B score. Writes results/sanity_random_seeds.csv.
Run from the repo root:  python src/sanity_random_seeds.py
"""
from pathlib import Path

import pandas as pd

from evaluate import compute_metrics
from splits import load_clean, split_random_grouped, split_temporal
from train_baseline import make_models

SEEDS = range(42, 52)  # 10 fixed seeds -> repeatable
OUT = Path(__file__).resolve().parent.parent / "results" / "sanity_random_seeds.csv"


def score(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    model = make_models()["tfidf_logreg"]  # same model and settings as the main baseline
    model.fit(train["text"], train["label"])
    return compute_metrics(test["label"], model.predict(test["text"]))


def main() -> None:
    df = load_clean()
    rows = []
    for seed in SEEDS:
        train, test = split_random_grouped(df, seed)
        rows.append({"seed": seed, "n_test": len(test),
                     "test_stress_rate": test["label"].mean(), **score(train, test)})
    runs = pd.DataFrame(rows)
    runs.to_csv(OUT, index=False)

    print(runs.to_string(index=False, float_format="%.4f"))
    f1 = runs["macro_f1"]
    print(f"\nrandom splits macro-F1: min {f1.min():.4f}  max {f1.max():.4f}  "
          f"mean {f1.mean():.4f}  std {f1.std():.4f}")

    b = score(*split_temporal(df))  # * unpacks the (train, test) pair into two arguments
    print(f"Split B macro-F1:       {b['macro_f1']:.4f}")
    print(f"random splits scoring at or above Split B: {(f1 >= b['macro_f1']).sum()} of {len(f1)}")
    print(f"\nwrote {OUT.name}")


if __name__ == "__main__":
    main()
