"""Figures from data and results that already exist. Saves PNGs to results/.

  fig1_rows_per_month.png   - raw data: the Jan 2018 r/relationships spike
  fig2_stress_over_time.png - cleaned data: stress rate and rows per month, Split B cutoff marked
  fig3_baseline_splits.png  - TF-IDF + LR macro-F1: 10 random splits vs Split B
  fig4_time_gap.png         - check (c): old (2017) vs recent training data

Run from the repo root:  python src/make_figures.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # draw to files only, no window
import matplotlib.pyplot as plt
import pandas as pd

from splits import CUTOFF, load_clean

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

# Colors (validated categorical pair + neutral chart chrome).
BLUE, ORANGE = "#2a78d6", "#eb6834"
GRAY = "#c3c2b7"  # de-emphasized marks ("everything else")
SURFACE, INK, INK_2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 10,
    "text.color": INK, "axes.labelcolor": INK_2, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.edgecolor": GRAY, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "legend.frameon": False,
})


def titles(fig, title: str, subtitle: str) -> None:
    fig.text(0.01, 0.98, title, fontsize=13, weight="semibold", va="top")
    fig.text(0.01, 0.92, subtitle, fontsize=9.5, color=INK_2, va="top")


def to_month(ts: pd.Series) -> pd.Series:
    return pd.to_datetime(ts, unit="s", utc=True).dt.strftime("%Y-%m")


def month_ticks(ax, months: list[str]) -> None:
    # Label every 3rd month so the axis stays readable.
    idx = list(range(0, len(months), 3))
    ax.set_xticks(idx, [months[i] for i in idx], rotation=0)


def fig1_rows_per_month() -> None:
    raw = pd.concat([pd.read_csv(ROOT / "data" / "raw" / f"dreaddit_{s}.csv",
                                 usecols=["subreddit", "social_timestamp"]) for s in ["train", "test"]])
    raw["month"] = to_month(raw["social_timestamp"])
    raw["group"] = raw["subreddit"].where(raw["subreddit"] == "relationships", "other")
    counts = raw.groupby(["month", "group"]).size().unstack(fill_value=0)  # rows: month, cols: group
    months = list(counts.index)
    x = range(len(months))

    fig, ax = plt.subplots(figsize=(10, 4.6))
    fig.subplots_adjust(top=0.84, left=0.07, right=0.98, bottom=0.12)
    ax.bar(x, counts["other"], width=0.8, color=GRAY, label="9 other subreddits")
    # bottom= stacks r/relationships on top of the other subreddits
    ax.bar(x, counts["relationships"], width=0.8, bottom=counts["other"], color=BLUE,
           label="r/relationships", edgecolor=SURFACE, linewidth=1)
    jan = months.index("2018-01")
    ax.annotate(f"r/relationships: {counts.loc['2018-01', 'relationships']} rows,\nall posted in Jan 2018",
                xy=(jan, counts.loc["2018-01"].sum()), xytext=(jan + 1.2, counts.loc["2018-01"].sum() - 60),
                fontsize=9, color=INK_2, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    ax.grid(axis="x", visible=False)
    month_ticks(ax, months)
    ax.set_ylabel("rows (train + test)")
    ax.legend(loc="upper left")
    titles(fig, "One subreddit causes the January 2018 spike",
           "Raw Dreaddit rows per month. r/relationships appears only in Jan 2018, so it was dropped from both splits.")
    fig.savefig(RESULTS / "fig1_rows_per_month.png", dpi=200)
    plt.close(fig)


def fig2_stress_over_time(clean: pd.DataFrame) -> None:
    clean = clean.assign(month=to_month(clean["social_timestamp"]))
    monthly = clean.groupby("month").agg(rows=("label", "size"), stress_rate=("label", "mean"))
    months = list(monthly.index)
    x = list(range(len(months)))
    cut_x = months.index(CUTOFF.strftime("%Y-%m")) - 0.5  # boundary between Aug and Sep 2018

    # Two panels sharing the x-axis (not two y-scales on one plot): rate on top, sample size below.
    fig, (top, bot) = plt.subplots(2, 1, figsize=(10, 6), sharex=True,
                                   gridspec_kw={"height_ratios": [2, 1], "hspace": 0.12})
    fig.subplots_adjust(top=0.87, left=0.08, right=0.98, bottom=0.08)

    overall = clean["label"].mean()
    top.axhline(overall, color=MUTED, lw=0.8)
    top.text(-0.4, 0.72, f"horizontal line: overall rate {overall:.3f}", ha="left", fontsize=8.5, color=INK_2)
    top.grid(axis="x", visible=False)
    top.plot(x, monthly["stress_rate"], color=BLUE, lw=2, marker="o", markersize=5,
             markeredgecolor=SURFACE, markeredgewidth=1.5)
    top.set_ylim(0.3, 0.8)
    top.set_ylabel("share of rows labelled stressed")

    bot.bar(x, monthly["rows"], width=0.8, color=GRAY)
    bot.set_ylabel("rows")
    bot.grid(axis="x", visible=False)
    month_ticks(bot, months)

    for ax in (top, bot):
        ax.axvline(cut_x, color=INK_2, lw=1)
    top.text(cut_x - 0.15, 0.77, "Split B: train", ha="right", fontsize=8.5, color=INK_2)
    top.text(cut_x + 0.15, 0.77, "test", ha="left", fontsize=8.5, color=INK_2)

    titles(fig, f"Stress rate stays near {overall:.2f}; bigger swings in the smaller 2017 months",
           f"Cleaned data (train + test, r/relationships removed), {len(clean)} rows. "
           "Line marks the Split B cutoff, 2018-09-01.")
    fig.savefig(RESULTS / "fig2_stress_over_time.png", dpi=200)
    plt.close(fig)


def fig3_baseline_splits() -> None:
    all_runs = pd.read_csv(RESULTS / "baseline_runs.csv")
    floor = all_runs.query("model == 'majority' and split == 'A_random'")["macro_f1"].mean()
    runs = all_runs[all_runs["model"] == "tfidf_logreg"]
    a = runs.loc[runs["split"] == "A_random", "macro_f1"]
    b = runs.loc[runs["split"] == "B_temporal", "macro_f1"].iloc[0]
    orig = runs.loc[runs["split"] == "A_original", "macro_f1"].iloc[0]

    fig, ax = plt.subplots(figsize=(10, 3.8))
    fig.subplots_adjust(top=0.80, left=0.25, right=0.97, bottom=0.16)
    rows = {"Split A: 10 random splits": 2, "Split B: temporal": 1, "Original split (side note)": 0}

    # Shaded band = mean +/- 1 std of the random splits: the "normal variation" range.
    ax.axvspan(a.mean() - a.std(), a.mean() + a.std(), color=BLUE, alpha=0.08, lw=0)
    # Fixed small up/down offsets so dots with near-equal scores don't hide each other.
    offsets = [0.07 if i % 2 else -0.07 for i in range(len(a))]
    a_sorted = a.sort_values()
    ax.scatter(a_sorted, [2 + o for o in offsets], s=60, color=BLUE, edgecolor=SURFACE, linewidth=1.5, zorder=3)
    ax.plot([a.mean()] * 2, [1.75, 2.25], color=INK, lw=1.5, zorder=4)
    ax.text(a.mean(), 2.32, f"mean {a.mean():.3f} ± {a.std():.3f}", ha="center", fontsize=9, color=INK_2)
    ax.scatter([b], [1], s=80, color=ORANGE, edgecolor=SURFACE, linewidth=1.5, zorder=3)
    ax.text(b + 0.002, 1.16, f"{b:.3f}", fontsize=9, color=INK_2)
    ax.scatter([orig], [0], s=60, color=MUTED, edgecolor=SURFACE, linewidth=1.5, zorder=3)
    ax.text(orig + 0.002, 0.16, f"{orig:.3f}", fontsize=9, color=INK_2)

    ax.set_yticks(list(rows.values()), list(rows.keys()))
    ax.tick_params(axis="y", colors=INK_2, length=0)
    ax.set_ylim(-0.6, 2.7)
    ax.set_xlim(0.70, 0.79)
    ax.set_xlabel("macro-F1 on the test set (TF-IDF + logistic regression)")
    ax.grid(axis="y", visible=False)
    titles(fig, "Split B scores within the normal range of random splits",
           "Each dot is one train/test run. Shaded band = random-split mean ± 1 std. "
           f"Always-'stressed' floor scores {floor:.2f} (not shown).")
    fig.savefig(RESULTS / "fig3_baseline_splits.png", dpi=200)
    plt.close(fig)


def fig4_time_gap() -> None:
    gap = pd.read_csv(RESULTS / "sanity_time_gap.csv")
    dates = dict(zip(gap["train_set"], gap["train_dates"]))  # e.g. "2018-01-26 to 2018-08-31"
    labels = {"old_2017": f"Old: {dates['old_2017']}\n(about 8-month gap)",
              "recent": f"Recent: {dates['recent']}\n(no gap)"}
    y = [1, 0]

    fig, ax = plt.subplots(figsize=(10, 3.2))
    fig.subplots_adjust(top=0.76, left=0.25, right=0.97, bottom=0.18)
    ax.barh(y, gap["macro_f1"], height=0.5, color=BLUE)
    for yi, v in zip(y, gap["macro_f1"]):
        ax.text(v + 0.01, yi, f"{v:.3f}", va="center", fontsize=9.5, color=INK)
    ax.set_yticks(y, [labels[s] for s in gap["train_set"]])
    ax.tick_params(axis="y", colors=INK_2, length=0)
    ax.set_xlim(0, 1)
    ax.set_xlabel("macro-F1 on Split B's test set (Sep–Nov 2018)")
    ax.grid(axis="y", visible=False)
    n = int(gap["n_train"].iloc[0])
    diff = 100 * gap["macro_f1"].diff().abs().iloc[1]  # in points (1 point = 0.01 macro-F1)
    runs = pd.read_csv(RESULTS / "baseline_runs.csv")
    spread = 100 * runs.query("model == 'tfidf_logreg' and split == 'A_random'")["macro_f1"].std()
    titles(fig, "Older training data did not score worse",
           f"Both training sets have {n} rows; one run each. The {diff:.1f}-point difference is smaller than "
           f"the {spread:.1f}-point std across the 10 random splits.")
    fig.savefig(RESULTS / "fig4_time_gap.png", dpi=200)
    plt.close(fig)


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    fig1_rows_per_month()
    fig2_stress_over_time(load_clean())
    fig3_baseline_splits()
    fig4_time_gap()
    print("wrote", ", ".join(p.name for p in sorted(RESULTS.glob("fig*.png"))))


if __name__ == "__main__":
    main()
