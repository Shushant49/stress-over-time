"""Why does Split B score higher? Break test predictions down by label, date period and subreddit.

Uses results/metrics/{baseline,transformer}_predictions.csv. Split A is restricted to seeds 42-46 for both
models so they are compared on the same splits. Scores are computed per run, then averaged over runs.
Writes results/metrics/breakdown_<view>.csv for each view.
Run from the repo root:  python src/analyze_predictions.py
"""

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, recall_score

from paths import METRICS
from splits import CUTOFF

SEEDS = range(42, 47)
FILES = {"tfidf_logreg": "baseline_predictions.csv", "distilroberta-base": "transformer_predictions.csv"}

# Three periods; Split B's test set is exactly the last one.
PERIOD_EDGES = [0, int(pd.Timestamp("2018-01-01", tz="UTC").timestamp()), int(CUTOFF.timestamp()), float("inf")]
PERIOD_NAMES = ["2017", "Jan-Aug 2018", "Sep-Nov 2018"]


def load() -> pd.DataFrame:
    parts = []
    for model, fname in FILES.items():
        path = METRICS / fname
        if not path.exists():
            print(f"skip {model}: {fname} not found")
            continue
        p = pd.read_csv(path)
        keep = (p["split"] == "B_temporal") | ((p["split"] == "A_random") & p["seed"].isin(SEEDS))
        parts.append(p[keep].assign(model=model))
    df = pd.concat(parts, ignore_index=True)
    df["seed"] = df["seed"].fillna(-1)  # baseline Split B has no seed; -1 keeps it in groupby
    df["period"] = pd.cut(df["social_timestamp"], PERIOD_EDGES, labels=PERIOD_NAMES, right=False)
    return df


def scores(g: pd.DataFrame) -> pd.Series:
    y, p = g["label"], g["pred"]
    # labels=[0, 1] keeps both classes in the average even if a small group has only one;
    # zero_division=0 avoids warnings when a class is never predicted.
    rec0, rec1 = recall_score(y, p, labels=[0, 1], average=None, zero_division=0)
    return pd.Series({"n": len(g), "stress_rate": y.mean(),
                      "macro_f1": f1_score(y, p, labels=[0, 1], average="macro", zero_division=0),
                      "accuracy": accuracy_score(y, p),
                      "recall_not_stressed": rec0, "recall_stressed": rec1})


def per_run(df: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """One row per (model, split, run, group) with that run's scores."""
    return (df.groupby(["model", "split", "seed", *by], observed=True)[["label", "pred"]]
              .apply(scores).reset_index())


def breakdown(df: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    # Score each run separately, then average over runs (n = rows per run).
    return per_run(df, by).groupby(["model", "split", *by], observed=True).mean(numeric_only=True) \
                          .drop(columns="seed").round(4).reset_index()


def main() -> None:
    df = load()
    views = {"overall": [], "period": ["period"], "subreddit": ["subreddit"]}
    for name, by in views.items():
        table = breakdown(df, by)
        table.to_csv(METRICS / f"breakdown_{name}.csv", index=False)
        print(f"\n=== {name} ===")
        print(table.to_string(index=False))


if __name__ == "__main__":
    main()
