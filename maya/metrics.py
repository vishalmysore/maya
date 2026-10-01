"""Metrics for yes/no gates: accuracy, ranking, calibration, consistency, abstention."""
import numpy as np
from sklearn.metrics import roc_auc_score

from .data import NEGATION_PAIRS


def ece(p, y, bins=10):
    p, y = np.asarray(p, float), np.asarray(y, float)
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & ((p < hi) if hi < 1 else (p <= hi))
        if m.any():
            total += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(total)


def coverage_at_risk(p, y, max_error):
    """Largest share of items that can be answered (most confident first) with error <= max_error.

    Fitted and measured on the same items, so it is an optimistic upper bound; Maya's
    conformal version will fit the threshold on held-out data.
    """
    p, y = np.asarray(p, float), np.asarray(y, bool)
    conf = np.abs(p - 0.5)
    order = np.argsort(-conf, kind="stable")
    wrong = ((p[order] >= 0.5) != y[order]).astype(float)
    err = np.cumsum(wrong) / np.arange(1, len(wrong) + 1)
    ok = np.where(err <= max_error)[0]
    return float((ok.max() + 1) / len(p)) if len(ok) else 0.0


def contradiction_rate(items, probs):
    """Share of negation pairs (same case) where both answers are yes or both are no."""
    by_key = {(it["domain"], it["id"], it["question"]): pr for it, pr in zip(items, probs)}
    bad = n = 0
    for domain, qa, qb in NEGATION_PAIRS:
        for (d, cid, q), pa in by_key.items():
            if d != domain or q != qa:
                continue
            pb = by_key.get((d, cid, qb))
            if pb is None:
                continue
            n += 1
            bad += (pa >= 0.5) == (pb >= 0.5)
    return (bad / n if n else float("nan")), n


def summarize(items, probs):
    y = np.array([it["label"] for it in items], bool)
    p = np.asarray(probs, float)
    pred = p >= 0.5
    contra, pairs = contradiction_rate(items, p)
    return {
        "n": int(len(y)),
        "accuracy": float((pred == y).mean()),
        "balanced_accuracy": float(0.5 * (pred[y].mean() + (~pred[~y]).mean())),
        "auroc": float(roc_auc_score(y, p)),
        "ece": ece(p, y),
        "brier": float(((p - y) ** 2).mean()),
        "yes_rate": float(pred.mean()),
        "coverage_at_10pct_error": coverage_at_risk(p, y, 0.10),
        "coverage_at_5pct_error": coverage_at_risk(p, y, 0.05),
        "contradiction_rate": contra,
        "contradiction_pairs": pairs,
    }


def per_domain_accuracy(items, probs):
    out = {}
    for it, pr in zip(items, probs):
        d = out.setdefault(it["domain"], [0, 0])
        d[0] += (pr >= 0.5) == it["label"]
        d[1] += 1
    return {k: round(a / n, 3) for k, (a, n) in out.items()}
