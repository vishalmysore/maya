"""Abstention with a bounded error rate (Learn-then-Test style threshold selection).

Maya answers
    yes       if p >= t_yes
    no        if p <= t_no
    not sure  otherwise
and the thresholds are fitted on labeled calibration data so that, with probability >= 1 - delta
over the draw of that data, the error rate among "yes" answers is <= alpha (and likewise for "no").

Each side tests a fixed grid of thresholds with a Clopper-Pearson upper bound at delta / (grid
size) (Learn-then-Test with a Bonferroni correction) and keeps the least conservative threshold
that passes. delta is also split between the two sides.

The guarantee holds when new inputs come from the same distribution as the calibration data. If
they don't (a new domain), the bound can fail; scripts/eval_maya.py measures this.
"""
import json

import numpy as np
from scipy.stats import beta


def cp_upper(k, n, delta):
    """One-sided Clopper-Pearson upper bound on a binomial rate."""
    if n == 0:
        return 1.0
    if k >= n:
        return 1.0
    return float(beta.ppf(1 - delta, k + 1, n - k))


# Candidate thresholds are fixed in advance (data-independent), so a Bonferroni correction over
# the grid gives a valid family-wise guarantee. (Plain fixed-sequence testing that starts at the
# single most confident item stops at once: one answer can never pass a 10% bound.)
YES_GRID = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.98, 0.99, 0.995, 0.999]
NO_GRID = [round(1 - t, 4) for t in YES_GRID]


def _fit_side(p, y, alpha, delta, side):
    """side="yes": answer yes when p >= t. Returns the least conservative valid t (or None)."""
    grid = YES_GRID if side == "yes" else NO_GRID
    d = delta / len(grid)
    for t in grid:  # least conservative first; the first one that passes wins
        ans = (p >= t) if side == "yes" else (p <= t)
        n = int(ans.sum())
        if n == 0:
            continue
        wrong = int((~y[ans]).sum()) if side == "yes" else int(y[ans].sum())
        if cp_upper(wrong, n, d) <= alpha:
            return t
    return None


class Abstainer:
    def __init__(self, t_yes=None, t_no=None, alpha=None, delta=None):
        self.t_yes, self.t_no, self.alpha, self.delta = t_yes, t_no, alpha, delta

    @classmethod
    def fit(cls, p, y, alpha=0.1, delta=0.1):
        p, y = np.asarray(p, float), np.asarray(y, bool)
        return cls(_fit_side(p, y, alpha, delta / 2, "yes"), _fit_side(p, y, alpha, delta / 2, "no"), alpha, delta)

    def decide(self, p):
        """1 = yes, 0 = no, -1 = not sure."""
        p = np.asarray(p, float)
        out = np.full(p.shape, -1)
        if self.t_yes is not None:
            out[p >= self.t_yes] = 1
        if self.t_no is not None:
            out[p <= self.t_no] = 0
        return out

    def report(self, p, y):
        d = self.decide(p)
        y = np.asarray(y, bool)
        yes, no = d == 1, d == 0
        ans = yes | no
        return {
            "coverage": float(ans.mean()),
            "error_when_answered": float(((d[ans] == 1) != y[ans]).mean()) if ans.any() else float("nan"),
            "yes_answers": int(yes.sum()), "yes_error": float((~y[yes]).mean()) if yes.any() else float("nan"),
            "no_answers": int(no.sum()), "no_error": float(y[no].mean()) if no.any() else float("nan"),
        }

    def to_dict(self):
        return {"t_yes": self.t_yes, "t_no": self.t_no, "alpha": self.alpha, "delta": self.delta}

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=1)


def fit_temperature(z, y, grid=np.linspace(0.3, 5, 95)):
    """Temperature T for p = sigmoid(z / T), by grid search on log loss."""
    z, y = np.asarray(z, float), np.asarray(y, float)
    best, best_t = None, 1.0
    for t in grid:
        p = 1 / (1 + np.exp(-z / t))
        p = np.clip(p, 1e-6, 1 - 1e-6)
        nll = -(y * np.log(p) + (1 - y) * np.log(1 - p)).mean()
        if best is None or nll < best:
            best, best_t = nll, float(t)
    return best_t
