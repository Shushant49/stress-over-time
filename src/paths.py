"""Where outputs go. Every script imports these, so the layout is defined in one place."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
METRICS = ROOT / "results" / "metrics"  # CSVs: per-run scores, summaries, predictions, breakdowns
FIGURES = ROOT / "results" / "figures"  # PNGs

for _d in (METRICS, FIGURES):
    _d.mkdir(parents=True, exist_ok=True)
