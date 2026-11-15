"""Download Dreaddit from Hugging Face and save it, untouched, to data/raw/.

Run from the repo root:  python src/download_data.py
"""
from pathlib import Path

from datasets import load_dataset

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    ds = load_dataset("andreagasparini/dreaddit")  # DatasetDict with "train" and "test"

    for split_name, split in ds.items():
        out = RAW_DIR / f"dreaddit_{split_name}.csv"
        # Raw files are never edited, so don't overwrite one that already exists.
        if out.exists():
            print(f"skip  {out.name} (already exists)")
            continue
        split.to_pandas().to_csv(out, index=False)
        print(f"wrote {out.name}: {len(split)} rows")


if __name__ == "__main__":
    main()
