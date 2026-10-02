"""Metrics for yes/no gates: accuracy, ranking, calibration, consistency, abstention."""
import numpy as np
from sklearn.metrics import roc_auc_score


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

    Fitted and measured on the same items, so it is an optimistic upper bound; see
    maya/conformal.py for a threshold fitted on separate data.
    """
    p, y = np.asarray(p, float), np.asarray(y, bool)
    conf = np.abs(p - 0.5)
    order = np.argsort(-conf, kind="stable")
    wrong = ((p[order] >= 0.5) != y[order]).astype(float)
    err = np.cumsum(wrong) / np.arange(1, len(wrong) + 1)
    ok = np.where(err <= max_error)[0]
    return float((ok.max() + 1) / len(p)) if len(ok) else 0.0


def consistency(items, probs, relations):
    """Negation pairs answered the same way, implications A=yes/B=no, minimal pairs not flipped."""
    p = {(it["domain"], it["id"], it["question"]): pr >= 0.5 for it, pr in zip(items, probs)}
    cases = {(it["domain"], it["id"]) for it in items}
    neg = [0, 0]
    imp = [0, 0]
    for d, cid in cases:
        rel = relations.get(d, {})
        for qa, qb in rel.get("negations", []):
            if (d, cid, qa) in p and (d, cid, qb) in p:
                neg[1] += 1
                neg[0] += p[(d, cid, qa)] == p[(d, cid, qb)]
        for qa, qb in rel.get("implications", []):
            if (d, cid, qa) in p and (d, cid, qb) in p:
                imp[1] += 1
                imp[0] += p[(d, cid, qa)] and not p[(d, cid, qb)]
    # minimal pairs: both texts right on the flipped question
    pairs = {}
    for it, pr in zip(items, probs):
        flip = relations.get(it["domain"], {}).get("minimal_pairs_flip")
        if it.get("pair") and it["question"] == flip:
            pairs.setdefault((it["domain"], it["pair"]), []).append((pr >= 0.5) == it["label"])
    mp = [sum(all(v) for v in pairs.values()), len(pairs)]

    def rate(a):
        return a[0] / a[1] if a[1] else float("nan")

    return {"negation_contradictions": rate(neg), "negation_pairs": neg[1],
            "implication_violations": rate(imp), "implication_pairs": imp[1],
            "minimal_pairs_both_right": rate(mp), "minimal_pairs": mp[1]}


def summarize(items, probs, relations=None):
    y = np.array([it["label"] for it in items], bool)
    p = np.asarray(probs, float)
    pred = p >= 0.5
    out = {
        "n": int(len(y)),
        "accuracy": float((pred == y).mean()),
        "balanced_accuracy": float(0.5 * (pred[y].mean() + (~pred[~y]).mean())),
        "auroc": float(roc_auc_score(y, p)),
        "ece": ece(p, y),
        "brier": float(((p - y) ** 2).mean()),
        "yes_rate": float(pred.mean()),
        "coverage_at_10pct_error": coverage_at_risk(p, y, 0.10),
        "coverage_at_5pct_error": coverage_at_risk(p, y, 0.05),
    }
    if relations is not None:
        out.update(consistency(items, p, relations))
    return out


def per_domain_accuracy(items, probs):
    out = {}
    for it, pr in zip(items, probs):
        d = out.setdefault(it["domain"], [0, 0])
        d[0] += (pr >= 0.5) == it["label"]
        d[1] += 1
    return {k: round(a / n, 3) for k, (a, n) in out.items()}
