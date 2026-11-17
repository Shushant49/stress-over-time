"""Sanity check (c): does a time gap between train and test lower the score?

Same test set as Split B (Sep-Nov 2018). Two training sets of (nearly) equal size:
  old    = all of 2017                         -> ~8-month gap before test
  recent = the same number of rows just before the cutoff -> no gap
Writes results/metrics/sanity_time_gap.csv.
Run from the repo root:  python src/sanity_time_gap.py
"""

import pandas as pd

from evaluate import compute_metrics
from paths import METRICS
from splits import load_clean, split_temporal
from train_baseline import make_models

OLD_END = pd.Timestamp("2018-01-01", tz="UTC")
OUT = METRICS / "sanity_time_gap.csv"


def date_range(part: pd.DataFrame) -> str:
    d = pd.to_datetime(part["social_timestamp"], unit="s", utc=True)
    return f"{d.min().date()} to {d.max().date()}"


def main() -> None:
    df = load_clean()
    pre, test = split_temporal(df)  # pre = everything before the Split B cutoff, oldest first

    old = pre[pre["social_timestamp"] < int(OLD_END.timestamp())]
    # Take the newest len(old) rows, but cut at a TIMESTAMP so no post is split across the boundary.
    start_ts = pre["social_timestamp"].iloc[len(pre) - len(old)]
    recent = pre[pre["social_timestamp"] >= start_ts]

    rows = []
    for name, train in [("old_2017", old), ("recent", recent)]:
        model = make_models()["tfidf_logreg"]
        model.fit(train["text"], train["label"])
        metrics = compute_metrics(test["label"], model.predict(test["text"]))
        rows.append({"train_set": name, "train_dates": date_range(train), "n_train": len(train),
                     "train_stress_rate": train["label"].mean(), "n_test": len(test), **metrics})

    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)
    print(f"test: {date_range(test)}  ({len(test)} rows)\n")
    print(out.to_string(index=False, float_format="%.4f"))

    # Subreddit mix of each training set, to see if 'old vs recent' also means 'different topics'.
    mix = pd.DataFrame({name: t["subreddit"].value_counts(normalize=True)
                        for name, t in [("old_2017", old), ("recent", recent), ("test", test)]}).round(3)
    print("\nsubreddit share:\n", mix.fillna(0).to_string())
    print(f"\nwrote {OUT.name}")


if __name__ == "__main__":
    main()
