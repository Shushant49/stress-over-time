"""Split A (random) and Split B (temporal) of the cleaned Dreaddit data.

Both splits come from the same cleaned file, so they differ only in HOW posts are divided.
Run from the repo root to print a summary:  python src/splits.py
"""
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parent.parent
CLEAN = ROOT / "data" / "processed" / "dreaddit_clean.csv"

# Train on posts before this moment, test on posts at/after it.
# Chosen near the 80% point; see notebooks/01_inspect_data.ipynb, step 4.
CUTOFF = pd.Timestamp("2018-09-01", tz="UTC")


def load_clean(path: Path = CLEAN) -> pd.DataFrame:
    return pd.read_csv(path)


def _check_no_shared_posts(train: pd.DataFrame, test: pd.DataFrame) -> None:
    shared = set(train["post_id"]) & set(test["post_id"])
    if shared:  # fail loudly: a shared post means leakage
        raise ValueError(f"{len(shared)} posts appear in both train and test")


def split_random(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split A: the dataset's own train/test division (random over time, grouped by post)."""
    train = df[df["orig_split"] == "train"].reset_index(drop=True)
    test = df[df["orig_split"] == "test"].reset_index(drop=True)
    _check_no_shared_posts(train, test)
    return train, test


def split_random_grouped(df: pd.DataFrame, seed: int, test_size: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    """A fresh random split for a given seed. All rows of a post (same post_id) go to the same side."""
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    # test_size is a share of POSTS, not rows, so the row share can differ slightly from 0.2
    train_idx, test_idx = next(gss.split(df, groups=df["post_id"]))
    train = df.iloc[train_idx].reset_index(drop=True)
    test = df.iloc[test_idx].reset_index(drop=True)
    _check_no_shared_posts(train, test)
    return train, test


def split_temporal(df: pd.DataFrame, cutoff: pd.Timestamp = CUTOFF) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split B: older posts train, newer posts test."""
    is_train = df["social_timestamp"] < int(cutoff.timestamp())  # compare in Unix seconds, like the column
    train = df[is_train].reset_index(drop=True)
    test = df[~is_train].reset_index(drop=True)
    _check_no_shared_posts(train, test)
    return train, test


def main() -> None:
    df = load_clean()
    for name, (train, test) in [("A (random)", split_random(df)), ("B (temporal)", split_temporal(df))]:
        share = len(test) / (len(train) + len(test))
        print(f"Split {name}: train {len(train)}  test {len(test)}  test share {share:.3f}  "
              f"stress rate {train['label'].mean():.3f} / {test['label'].mean():.3f}")


if __name__ == "__main__":
    main()
