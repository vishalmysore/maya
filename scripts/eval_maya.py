"""Evaluate a Maya checkpoint (or any NLI model) on the test sets, with calibration and abstention.

Test sets: v1 (data/eval), v2 (data/eval_v2, unseen domains), v3 (data/eval_v3, judgment questions,
written before the v0.2 training data). Calibration data: data/dev (hand-labeled, never a test set).

  - plain metrics at threshold 0.5 (accuracy, AUROC, ECE, consistency, minimal pairs, traps)
  - "asked both ways" on v2/v3: p = (p(key) + 1 - p(not_key)) / 2 for the key statement
  - temperature fitted on dev -> ECE on every test set
  - abstention (maya/conformal.py, alpha=0.20, delta=0.10) fitted on dev -> coverage / error on the test sets

    python scripts/eval_maya.py checkpoints/maya-v2-large --name maya-v2-large
    python scripts/eval_maya.py MoritzLaurer/deberta-v3-large-zeroshot-v2.0 --name deberta-large
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from maya.conformal import Abstainer, fit_temperature  # noqa: E402
from maya.data import DEV_DIR, EVAL_SETS, load_relations, load_yes_no_items  # noqa: E402
from maya.metrics import ece, per_domain_accuracy, summarize  # noqa: E402
from maya.nli import NLIScorer  # noqa: E402

ALPHA, DELTA = 0.20, 0.10


def logit(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def both_ways(items, p):
    """Accuracy on the key statements when also asking the negation."""
    P = {(it["domain"], it["id"], it["question"]): x for it, x in zip(items, p)}
    plain = sym = n = 0
    for it in items:
        if it["question"] != "key" or (it["domain"], it["id"], "not_key") not in P:
            continue
        pk, pn = P[(it["domain"], it["id"], "key")], P[(it["domain"], it["id"], "not_key")]
        n += 1
        plain += (pk >= 0.5) == it["label"]
        sym += ((pk + 1 - pn) / 2 >= 0.5) == it["label"]
    return {"key_plain": plain / n, "key_both_ways": sym / n, "n": n} if n else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--name", default=None)
    ap.add_argument("--cached", action="store_true", help="reuse probabilities from a previous run's results file")
    args = ap.parse_args()
    name = args.name or Path(args.model).name
    res = ROOT / "results" / f"eval_{name.replace('/', '_')}.json"

    dirs = {**EVAL_SETS, "dev": DEV_DIR}
    sets = {s: load_yes_no_items(d) for s, d in dirs.items()}
    rels = {s: load_relations(d) for s, d in dirs.items()}
    if args.cached:
        old = json.loads(res.read_text(encoding="utf-8"))
        probs = {s: np.array(old["sets"][s]["probs"]) for s in sets}
    else:
        scorer = NLIScorer(args.model)
        probs = {s: np.array(scorer.score([(it["text"], it["statement"]) for it in items])) for s, items in sets.items()}
    ys = {s: np.array([it["label"] for it in items], bool) for s, items in sets.items()}

    T = fit_temperature(logit(probs["dev"]), ys["dev"])
    out = {"model": args.model, "temperature_fit_on_dev": T, "alpha": ALPHA, "delta": DELTA, "sets": {}}
    for s, items in sets.items():
        summ = summarize(items, probs[s], rels[s])
        summ["ece_after_dev_temperature"] = ece(1 / (1 + np.exp(-logit(probs[s]) / T)), ys[s])
        summ["per_domain"] = per_domain_accuracy(items, probs[s])
        bw = both_ways(items, probs[s])
        if bw:
            summ["both_ways"] = bw
        out["sets"][s] = {"summary": summ, "probs": [round(float(x), 4) for x in probs[s]]}

    ab = Abstainer.fit(probs["dev"], ys["dev"], ALPHA, DELTA)
    out["abstention_fit_on_dev"] = {"thresholds": ab.to_dict(),
                                   **{s: ab.report(probs[s], ys[s]) for s in ("v1", "v2", "v3")}}
    res.write_text(json.dumps(out, indent=1), encoding="utf-8")

    print(f"== {name}  (temperature fitted on dev: {T:.2f})")
    for s in ("dev", "v1", "v2", "v3"):
        m = out["sets"][s]["summary"]
        line = (f"[{s}] acc {m['accuracy']:.3f}  auroc {m['auroc']:.3f}  ece {m['ece']:.3f}->{m['ece_after_dev_temperature']:.3f}  "
                f"neg-contra {m['negation_contradictions']:.2f}")
        if m.get("implication_pairs"):
            line += f"  impl-viol {m['implication_violations']:.2f}  minpairs {m['minimal_pairs_both_right']:.2f}"
        if m.get("trap_n"):
            line += f"  traps {m['trap_accuracy']:.2f}"
        if m.get("both_ways"):
            line += f"  key {m['both_ways']['key_plain']:.2f} -> both ways {m['both_ways']['key_both_ways']:.2f}"
        print(line)
    a = out["abstention_fit_on_dev"]
    print(f"abstain (alpha {ALPHA}) thresholds {a['thresholds']['t_no']} / {a['thresholds']['t_yes']}: " + "  ".join(
        f"{s} coverage {a[s]['coverage']:.2f} error {a[s]['error_when_answered']:.3f}" for s in ("v1", "v2", "v3")))


if __name__ == "__main__":
    main()
