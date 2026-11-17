"""Figures from data and results that already exist. Saves PNGs to results/.

  fig1_rows_per_month.png   - raw data: the Jan 2018 r/relationships spike
  fig2_stress_over_time.png - cleaned data: stress rate and rows per month, Split B cutoff marked
  fig3_main_results.png     - headline: macro-F1 per run, both models, Split A vs Split B
  fig4_time_gap.png         - check (c): old (2017) vs recent training data
  fig5_by_period.png        - Split A test posts by period vs Split B (why B scores higher)

Run from the repo root:  python src/make_figures.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # draw to files only, no window
import matplotlib.pyplot as plt
import pandas as pd

from analyze_predictions import load as load_predictions, per_run
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


def dot_row(ax, values: pd.Series, y: float, color: str) -> None:
    """One row of a dot plot: a dot per run, plus a mean tick and label when there are several runs."""
    v = values.sort_values().reset_index(drop=True)
    # Fixed small up/down offsets so near-equal scores don't hide each other.
    ys = [y + (0.09 if i % 2 else -0.09) for i in range(len(v))] if len(v) > 1 else [y]
    ax.scatter(v, ys, s=55, color=color, edgecolor=SURFACE, linewidth=1.5, zorder=3)
    if len(v) > 1:
        ax.plot([v.mean()] * 2, [y - 0.25, y + 0.25], color=INK, lw=1.5, zorder=4)
        label = f"{v.mean():.3f} ± {v.std():.3f}  ({len(v)} runs)"
    else:
        label = f"{v.iloc[0]:.3f}  (1 run)"
    ax.text(v.max() + 0.004, y, label, va="center", fontsize=9, color=INK_2)


def fig3_main_results() -> None:
    base = pd.read_csv(RESULTS / "baseline_runs.csv")
    trans = pd.read_csv(RESULTS / "transformer_runs.csv")
    floor = base.query("model == 'majority' and split == 'A_random'")["macro_f1"].mean()

    def f1(df, model, split):
        return df.query("model == @model and split == @split")["macro_f1"]

    rows = [  # (label, scores, color, y) top to bottom
        ("TF-IDF + LR  ·  Split A (random)", f1(base, "tfidf_logreg", "A_random"), BLUE, 4.6),
        ("TF-IDF + LR  ·  Split B (temporal)", f1(base, "tfidf_logreg", "B_temporal"), ORANGE, 3.6),
        ("distilroberta  ·  Split A (random)", f1(trans, "distilroberta-base", "A_random"), BLUE, 1.6),
        ("distilroberta  ·  Split B (temporal)", f1(trans, "distilroberta-base", "B_temporal"), ORANGE, 0.6),
    ]
    fig, ax = plt.subplots(figsize=(10, 4.4))
    fig.subplots_adjust(top=0.82, left=0.27, right=0.97, bottom=0.14)
    for _, vals, color, y in rows:
        dot_row(ax, vals, y, color)
    ax.axhline(2.6, color=GRID, lw=1)  # separates the two models
    ax.set_yticks([r[3] for r in rows], [r[0] for r in rows])
    ax.tick_params(axis="y", colors=INK_2, length=0)
    ax.set_ylim(0, 5.2)
    ax.set_xlim(0.70, 0.90)
    ax.set_xlabel("macro-F1 on the test set")
    ax.grid(axis="y", visible=False)
    titles(fig, "No drop on newer posts for either model",
           "Each dot is one run; black tick = mean. Split B tests on Sep–Nov 2018 after training on older posts. "
           f"Always-'stressed' floor: {floor:.2f}.")
    fig.savefig(RESULTS / "fig3_main_results.png", dpi=200)
    plt.close(fig)


def fig5_by_period() -> None:
    preds = load_predictions()
    runs = per_run(preds, ["period"])
    groups = [  # (label, split, period, color)
        ("Split A test posts from 2017", "A_random", "2017", BLUE),
        ("Split A test posts from Jan–Aug 2018", "A_random", "Jan-Aug 2018", BLUE),
        ("Split A test posts from Sep–Nov 2018", "A_random", "Sep-Nov 2018", BLUE),
        ("Split B (all Sep–Nov 2018)", "B_temporal", "Sep-Nov 2018", ORANGE),
    ]
    models = [("distilroberta-base", "distilroberta-base"), ("tfidf_logreg", "TF-IDF + logistic regression")]

    fig, axes = plt.subplots(2, 1, figsize=(10, 6.4), sharex=True, gridspec_kw={"hspace": 0.35})
    fig.subplots_adjust(top=0.86, left=0.30, right=0.97, bottom=0.09)
    for ax, (model, model_label) in zip(axes, models):
        for i, (label, split, period, color) in enumerate(groups):
            sel = runs.query("model == @model and split == @split and period == @period")["macro_f1"]
            dot_row(ax, sel, len(groups) - 1 - i, color)
        n_sep = runs.query("model == @model and split == 'A_random' and period == 'Sep-Nov 2018'")["n"].mean()
        ax.set_yticks(range(len(groups) - 1, -1, -1), [g[0] for g in groups])
        ax.tick_params(axis="y", colors=INK_2, length=0)
        ax.set_ylim(-0.7, len(groups) - 0.3)
        ax.grid(axis="y", visible=False)
        ax.set_title(f"{model_label}   (Split A has ~{n_sep:.0f} Sep–Nov rows per run, so those dots are noisy)",
                     loc="left", fontsize=9.5, color=INK_2)
    lo = runs["macro_f1"].min()
    axes[-1].set_xlim(min(0.66, lo - 0.02), 0.96)  # never cut off a run
    axes[-1].set_xlabel("macro-F1 (seeds 42–46)")
    titles(fig, "On the same test months, training only on older posts scores about the same",
           "Compare the last two rows in each panel: same test months, only the training data differs. "
           "Wide spread = small groups.")
    fig.savefig(RESULTS / "fig5_by_period.png", dpi=200)
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
    fig3_main_results()
    fig4_time_gap()
    fig5_by_period()
    print("wrote", ", ".join(p.name for p in sorted(RESULTS.glob("fig*.png"))))


if __name__ == "__main__":
    main()
