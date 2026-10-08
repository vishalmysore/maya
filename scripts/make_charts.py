"""Charts for the article (docs/images/chart-*.png) from the results files.

    python scripts/make_charts.py

Palette: categorical slots 1-3 (#2a78d6, #eb6834, #1baf7a), validated with the dataviz palette checker
(CVD dE >= 9.2, normal-vision dE >= 27.6); the third slot is below 3:1 contrast on the surface, so every
bar carries a visible value label.
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "images"
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
NEUTRAL = "#b9b8b2"
SETS = [("v3", "v3: judgment questions (192)"), ("v2", "v2: unseen domains (256)"), ("v1", "v1: familiar domains (120)")]

plt.rcParams.update({"font.family": "Segoe UI", "font.size": 11, "text.color": INK, "axes.labelcolor": MUTED,
                     "xtick.color": MUTED, "ytick.color": INK, "axes.edgecolor": GRID})


def load():
    b = json.loads((ROOT / "results" / "baselines.json").read_text(encoding="utf-8"))

    def ev(name):
        d = json.loads((ROOT / "results" / f"eval_{name}.json").read_text(encoding="utf-8"))
        return {s: d["sets"][s]["summary"] for s in ("v1", "v2", "v3")}

    laya = {s: b[s]["models"]["laya-typed-decisions"]["summary"] for s in ("v1", "v2", "v3")}
    return {
        "laya-typed-decisions (421M)": laya,
        "DeBERTa-v3-large zero-shot (435M)": ev("deberta-large-zeroshot"),
        "Maya v0.1 (150M)": ev("maya-v0.1"),
        "Maya v0.2 (435M)": ev("maya-v2-large"),
    }


def style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.xaxis.grid(True, color=GRID, linewidth=1)
    ax.set_axisbelow(True)


def bold_maya(ax):
    for t in ax.get_yticklabels():
        if t.get_text().startswith("Maya v0.2"):
            t.set_fontweight("bold")


def accuracy_chart(data):
    labels = list(data)
    fig, ax = plt.subplots(figsize=(9.5, 5.2), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    style(ax)
    h = 0.25
    for i, lab in enumerate(labels):
        y = len(labels) - 1 - i
        for j, (s, _) in enumerate(SETS):
            val = data[lab][s]["accuracy"] * 100
            yy = y + (1 - j) * (h + 0.02)
            ax.barh(yy, val, height=h, color=SERIES[j], edgecolor=SURFACE, linewidth=2)
            ax.text(val + 0.8, yy, f"{val:.1f}%", va="center", fontsize=9.5, color=INK)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(list(reversed(labels)))
    bold_maya(ax)
    ax.set_xlim(0, 108)
    ax.set_xlabel("Accuracy on hand-labeled yes/no answers (%)")
    ax.set_title("Maya v0.2 is the most accurate model on all three test sets", loc="left", fontsize=12.5, pad=26)
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c) for c in SERIES], labels=[n for _, n in SETS],
              loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3, frameon=False, fontsize=9.5, handlelength=1.2)
    fig.tight_layout()
    fig.savefig(OUT / "chart-accuracy.png", facecolor=SURFACE)
    plt.close(fig)


def contradiction_chart(data):
    labels = list(data)
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.9), dpi=200, sharey=True)
    fig.patch.set_facecolor(SURFACE)
    for ax, (s, title) in zip(axes, SETS):
        style(ax)
        for i, lab in enumerate(labels):
            y = len(labels) - 1 - i
            v = data[lab][s]["negation_contradictions"] * 100
            ax.barh(y, v, height=0.6, color=SERIES[0] if lab.startswith("Maya v0.2") else NEUTRAL)
            ax.text(v + 2, y, f"{v:.0f}%", va="center", fontsize=9.5, color=INK)
        ax.set_xlim(0, 120)
        ax.set_xticks([0, 25, 50, 75, 100])
        n = data[labels[0]][s].get("negation_pairs", 12)
        ax.set_title(f"{title.split(':')[0]} ({n} negation pairs)", loc="left", fontsize=11, color=MUTED)
    axes[0].set_yticks(range(len(labels)))
    axes[0].set_yticklabels(list(reversed(labels)))
    bold_maya(axes[0])
    fig.suptitle("Same answer to a statement and its negation (lower is better)", x=0.01, ha="left", fontsize=12.5)
    fig.supxlabel("Share of negation pairs answered inconsistently (%)", fontsize=10, color=MUTED)
    fig.tight_layout()
    fig.savefig(OUT / "chart-contradictions.png", facecolor=SURFACE)
    plt.close(fig)


def pairs_chart(data):
    """Minimal pairs both right, v2 and v3 (single measure, Maya v0.2 highlighted)."""
    labels = list(data)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.9), dpi=200, sharey=True)
    fig.patch.set_facecolor(SURFACE)
    for ax, (s, title) in zip(axes, SETS[:2]):
        style(ax)
        for i, lab in enumerate(labels):
            y = len(labels) - 1 - i
            v = data[lab][s]["minimal_pairs_both_right"] * 100
            ax.barh(y, v, height=0.6, color=SERIES[0] if lab.startswith("Maya v0.2") else NEUTRAL)
            ax.text(v + 2, y, f"{v:.0f}%", va="center", fontsize=9.5, color=INK)
        ax.set_xlim(0, 115)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_title(f"{title.split(':')[0]} ({data[labels[0]][s]['minimal_pairs']} pairs)", loc="left", fontsize=11, color=MUTED)
    axes[0].set_yticks(range(len(labels)))
    axes[0].set_yticklabels(list(reversed(labels)))
    bold_maya(axes[0])
    fig.suptitle("Minimal pairs: both texts answered correctly (higher is better)", x=0.01, ha="left", fontsize=12.5)
    fig.supxlabel("Share of pairs where one changed detail flips the answer and the model gets both right (%)", fontsize=10, color=MUTED)
    fig.tight_layout()
    fig.savefig(OUT / "chart-minimal-pairs.png", facecolor=SURFACE)
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    d = load()
    accuracy_chart(d)
    contradiction_chart(d)
    pairs_chart(d)
    print("charts written to", OUT)
