"""Evaluate a Maya checkpoint (or any NLI model) on eval v1 and v2, with calibration and abstention.

  - plain metrics at threshold 0.5 (accuracy, AUROC, ECE, consistency) on v1 and v2
  - temperature fitted on the synthetic validation split (never on eval data) -> ECE after scaling,
    and a temperature fitted on v1 -> ECE on v2
  - abstention (maya/conformal.py, alpha=0.10, delta=0.10) fitted three ways:
      synthetic-val -> v1 / v2    calibrate on generated data only
      v1 -> v2                    calibrate on one hand-labeled set, test on unseen domains
      v2 cross-fit                calibrate on a random half of v2, test on the other half
                                  (200 splits, split by case so a text never sits on both sides)

    python scripts/eval_maya.py checkpoints/maya
    python scripts/eval_maya.py MoritzLaurer/ModernBERT-base-zeroshot-v2.0 --name base
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from maya.conformal import Abstainer, fit_temperature  # noqa: E402
from maya.data import EVAL_SETS, load_relations, load_yes_no_items  # noqa: E402
from maya.metrics import ece, per_domain_accuracy, summarize  # noqa: E402
from maya.nli import NLIScorer  # noqa: E402

ALPHA, DELTA = 0.10, 0.10


def logit(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def cross_fit(items, p, splits=200, seed=0):
    rng = np.random.default_rng(seed)
    cases = sorted({(it["domain"], it["id"]) for it in items})
    y = np.array([it["label"] for it in items], bool)
    key = [(it["domain"], it["id"]) for it in items]
    reps = []
    for _ in range(splits):
        cal = {cases[i] for i in rng.permutation(len(cases))[:len(cases) // 2]}
        m = np.array([k in cal for k in key])
        a = Abstainer.fit(p[m], y[m], ALPHA, DELTA)
        reps.append(a.report(p[~m], y[~m]))
    cov = np.array([r["coverage"] for r in reps])
    err = np.array([r["error_when_answered"] for r in reps if r["coverage"] > 0])
    return {"coverage_mean": float(cov.mean()), "error_mean": float(err.mean()) if len(err) else None,
            "share_of_splits_error_above_alpha": float((err > ALPHA).mean()) if len(err) else None,
            "splits": splits}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--name", default=None)
    ap.add_argument("--cached", action="store_true", help="reuse probabilities from a previous run's results file")
    args = ap.parse_args()
    name = args.name or Path(args.model).name

    res = ROOT / "results" / f"eval_{name.replace('/', '_')}.json"
    sets = {s: load_yes_no_items(d) for s, d in EVAL_SETS.items()}
    rels = {s: load_relations(d) for s, d in EVAL_SETS.items()}
    val_rows = [json.loads(l) for l in open(ROOT / "data" / "train" / "val.jsonl", encoding="utf-8")]
    val = [{"text": r["text"], "statement": it["statement"], "label": it["label"]} for r in val_rows for it in r["items"]]

    if args.cached:
        old = json.loads(res.read_text(encoding="utf-8"))
        probs = {s: np.array(old["sets"][s]["probs"]) for s in sets}
        probs["val"] = np.array(old["val"]["probs"])
    else:
        scorer = NLIScorer(args.model)
        probs = {s: np.array(scorer.score([(it["text"], it["statement"]) for it in items])) for s, items in sets.items()}
        probs["val"] = np.array(scorer.score([(it["text"], it["statement"]) for it in val]))
    ys = {s: np.array([it["label"] for it in items], bool) for s, items in (*sets.items(), ("val", val))}

    T = fit_temperature(logit(probs["val"]), ys["val"])
    out = {"model": args.model, "temperature": T, "alpha": ALPHA, "delta": DELTA, "sets": {}}
    for s, items in sets.items():
        summ = summarize(items, probs[s], rels[s])
        p_t = 1 / (1 + np.exp(-logit(probs[s]) / T))
        summ["ece_after_temperature"] = ece(p_t, ys[s])
        summ["per_domain"] = per_domain_accuracy(items, probs[s])
        out["sets"][s] = {"summary": summ, "probs": [round(float(x), 4) for x in probs[s]]}

    T1 = fit_temperature(logit(probs["v1"]), ys["v1"])
    out["temperature_fit_on_v1"] = T1
    out["sets"]["v2"]["summary"]["ece_after_v1_temperature"] = ece(1 / (1 + np.exp(-logit(probs["v2"]) / T1)), ys["v2"])
    a_val = Abstainer.fit(probs["val"], ys["val"], ALPHA, DELTA)
    a_v1 = Abstainer.fit(probs["v1"], ys["v1"], ALPHA, DELTA)
    out["abstention"] = {
        "fit_on_synthetic_val": {"thresholds": a_val.to_dict(), "v1": a_val.report(probs["v1"], ys["v1"]),
                                 "v2": a_val.report(probs["v2"], ys["v2"])},
        "fit_on_v1": {"thresholds": a_v1.to_dict(), "v2": a_v1.report(probs["v2"], ys["v2"])},
        "v2_cross_fit": cross_fit(sets["v2"], probs["v2"]),
    }
    out["val"] = {"accuracy": float(((probs["val"] >= 0.5) == ys["val"]).mean()), "n": int(len(val)),
                  "probs": [round(float(x), 4) for x in probs["val"]]}

    res.write_text(json.dumps(out, indent=1), encoding="utf-8")

    print(f"== {name}  (temperature {T:.2f}, synthetic val acc {out['val']['accuracy']:.3f})")
    for s in sets:
        m = out["sets"][s]["summary"]
        line = (f"[{s}] acc {m['accuracy']:.3f}  bal {m['balanced_accuracy']:.3f}  auroc {m['auroc']:.3f}  "
                f"ece {m['ece']:.3f}->{m['ece_after_temperature']:.3f}  neg-contra {m['negation_contradictions']:.2f}")
        if m.get("implication_pairs"):
            line += f"  impl-viol {m['implication_violations']:.2f}  minpairs {m['minimal_pairs_both_right']:.2f}"
        print(line)
    print(f"v2 ECE with temperature fitted on v1 ({T1:.2f}): {out['sets']['v2']['summary']['ece_after_v1_temperature']:.3f}")
    ab = out["abstention"]
    for k, v in (("val->v1", ab["fit_on_synthetic_val"]["v1"]), ("val->v2", ab["fit_on_synthetic_val"]["v2"]),
                 ("v1->v2", ab["fit_on_v1"]["v2"])):
        print(f"abstain {k:8s} coverage {v['coverage']:.2f}  error when answered {v['error_when_answered']:.3f}")
    cf = ab["v2_cross_fit"]
    print(f"abstain v2 cross-fit coverage {cf['coverage_mean']:.2f}  error {cf['error_mean']}  "
          f"splits over alpha {cf['share_of_splits_error_above_alpha']}")


if __name__ == "__main__":
    main()
