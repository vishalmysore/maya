"""Charts for the article (docs/images/chart-*.png) from the results files.

    python scripts/make_charts.py
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "images"
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0"
V1, V2 = "#2a78d6", "#eb6834"  # categorical slots 1-2 (validated: CVD dE 24.7, normal 33.6)
NEUTRAL = "#b9b8b2"

plt.rcParams.update({"font.family": "Segoe UI", "font.size": 11, "text.color": INK, "axes.labelcolor": MUTED,
                     "xtick.color": MUTED, "ytick.color": INK, "axes.edgecolor": GRID})


def load():
    b = json.loads((ROOT / "results" / "baselines.json").read_text(encoding="utf-8"))
    m = json.loads((ROOT / "results" / "eval_maya.json").read_text(encoding="utf-8"))
    rows = {
        "laya-typed-decisions (421M)": ("laya-typed-decisions", "laya-typed-decisions"),
        "layaMOE (421M + heads)": ("layaMOE (prompted router)", "layaMOE (trained router)"),
        "ModernBERT NLI, Maya's start (150M)": ("MoritzLaurer/ModernBERT-base-zeroshot-v2.0",) * 2,
        "deberta-v3-large NLI (435M)": ("MoritzLaurer/deberta-v3-large-zeroshot-v2.0",) * 2,
    }
    data = {}
    for label, (k1, k2) in rows.items():
        data[label] = (b["v1"]["models"][k1]["summary"], b["v2"]["models"][k2]["summary"])
    data["Maya (150M)"] = (m["sets"]["v1"]["summary"], m["sets"]["v2"]["summary"])
    return data


def style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.xaxis.grid(True, color=GRID, linewidth=1)
    ax.set_axisbelow(True)


def accuracy_chart(data):
    labels = list(data)
    fig, ax = plt.subplots(figsize=(9, 4.6), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    style(ax)
    h = 0.36
    for i, lab in enumerate(labels):
        y = len(labels) - 1 - i
        for off, val, col in ((h / 2 + 0.02, data[lab][0]["accuracy"], V1), (-h / 2 - 0.02, data[lab][1]["accuracy"], V2)):
            ax.barh(y + off, val * 100, height=h, color=col, edgecolor=SURFACE, linewidth=2)
            ax.text(val * 100 + 0.6, y + off, f"{val * 100:.1f}%", va="center", fontsize=9.5, color=INK)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(list(reversed(labels)), fontweight="normal")
    for t in ax.get_yticklabels():
        if t.get_text().startswith("Maya"):
            t.set_fontweight("bold")
    ax.set_xlim(0, 105)
    ax.set_xlabel("Accuracy on hand-labeled yes/no answers (%)")
    ax.set_title("Maya leads on familiar domains; a 3x larger NLI model leads on unseen ones", loc="left", fontsize=12.5, pad=24)
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=V1), plt.Rectangle((0, 0), 1, 1, color=V2)],
              labels=["v1: 120 answers, mostly familiar domains", "v2: 256 answers, 8 unseen domains"],
              loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, frameon=False, fontsize=9.5, handlelength=1.2)
    fig.tight_layout()
    fig.savefig(OUT / "chart-accuracy.png", facecolor=SURFACE)
    plt.close(fig)


def contradiction_chart(data):
    labels = list(data)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.0), dpi=200, sharey=True)
    fig.patch.set_facecolor(SURFACE)
    for ax, idx, title in ((axes[0], 0, "v1 (12 negation pairs)"), (axes[1], 1, "v2 (64 negation pairs)")):
        style(ax)
        for i, lab in enumerate(labels):
            y = len(labels) - 1 - i
            v = data[lab][idx]["negation_contradictions"] * 100
            col = V1 if lab.startswith("Maya") else NEUTRAL
            ax.barh(y, v, height=0.6, color=col)
            ax.text(v + 1.5, y, f"{v:.0f}%", va="center", fontsize=9.5, color=INK)
        ax.set_xlim(0, 115)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_title(title, loc="left", fontsize=11, color=MUTED)
    axes[0].set_yticks(range(len(labels)))
    axes[0].set_yticklabels(list(reversed(labels)))
    for t in axes[0].get_yticklabels():
        if t.get_text().startswith("Maya"):
            t.set_fontweight("bold")
    fig.suptitle("Same answer to a statement and its negation (lower is better)", x=0.01, ha="left", fontsize=12.5)
    fig.supxlabel("Share of negation pairs answered inconsistently (%)", fontsize=10, color=MUTED)
    fig.tight_layout()
    fig.savefig(OUT / "chart-contradictions.png", facecolor=SURFACE)
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    d = load()
    accuracy_chart(d)
    contradiction_chart(d)
    print("charts written to", OUT)
