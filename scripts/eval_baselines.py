"""How far do existing models get on the yes/no eval sets?

Baselines:
  - always "no" (the majority answer)
  - laya-typed-decisions and layaMOE (probabilities recorded with layaMOE's code:
    results/laya_moe_eval_reference.json for v1, results/laya_v2_reference.json for v2)
  - off-the-shelf zero-shot NLI models: premise = the text, hypothesis = the yes/no statement

Writes results/baselines.json (summary + every probability, per eval set).
Latency is measured separately by scripts/bench_latency.py.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from maya.data import EVAL_SETS, load_relations, load_yes_no_items  # noqa: E402
from maya.metrics import per_domain_accuracy, summarize  # noqa: E402
from maya.nli import NLIScorer  # noqa: E402

NLI_MODELS = [
    "cross-encoder/nli-deberta-v3-xsmall",
    "MoritzLaurer/deberta-v3-xsmall-zeroshot-v1.1-all-33",
    "MoritzLaurer/deberta-v3-base-zeroshot-v2.0",
    "MoritzLaurer/ModernBERT-base-zeroshot-v2.0",
    "MoritzLaurer/deberta-v3-large-zeroshot-v2.0",
]

REFERENCES = {
    "v1": ("laya_moe_eval_reference.json", {"general": "laya-typed-decisions", "moe": "layaMOE (prompted router)"}),
    "v2": ("laya_v2_reference.json", {"general": "laya-typed-decisions", "moe": "layaMOE (trained router)"}),
    "v3": ("laya_v3_reference.json", {"general": "laya-typed-decisions", "moe": "layaMOE (trained router)"}),
}


def laya_reference(split, items):
    fname, names = REFERENCES[split]
    path = ROOT / "results" / fname
    if not path.exists():
        return {}
    rows = json.loads(path.read_text(encoding="utf-8"))
    by_key = {(r["domain"], r["id"], r["question"]): r for r in rows if r["answers"]["general"]["type"] == "noul"}
    return {label: [by_key[(it["domain"], it["id"], it["question"])]["answers"][k]["noul"] for it in items]
            for k, label in names.items()}


def fmt(name, s):
    line = (f"{name:52s} acc {s['accuracy']:.3f}  bal {s['balanced_accuracy']:.3f}  auroc {s['auroc']:.3f}  "
            f"ece {s['ece']:.3f}  cov@10% {s['coverage_at_10pct_error']:.2f}  "
            f"neg-contra {s['negation_contradictions']:.2f}")
    if s.get("implication_pairs"):
        line += f"  impl-viol {s['implication_violations']:.2f}  minpairs {s['minimal_pairs_both_right']:.2f}"
    return line


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=NLI_MODELS)
    ap.add_argument("--sets", nargs="*", default=list(EVAL_SETS))
    args = ap.parse_args()

    data = {s: load_yes_no_items(EVAL_SETS[s]) for s in args.sets}
    rels = {s: load_relations(EVAL_SETS[s]) for s in args.sets}
    out_path = ROOT / "results" / "baselines.json"
    old = json.loads(out_path.read_text(encoding="utf-8")) if out_path.exists() else {}
    out = {s: old.get(s, {}) for s in EVAL_SETS}  # keep results of models not re-run

    def add(split, name, probs):
        items = data[split]
        s = summarize(items, probs, rels[split])
        s["per_domain"] = per_domain_accuracy(items, probs)
        out[split].setdefault("models", {})[name] = {"summary": s, "probs": [round(float(p), 4) for p in probs]}
        print(f"[{split}] " + fmt(name, s))

    for split, items in data.items():
        out[split]["items"] = [{k: it[k] for k in ("domain", "id", "question", "label")} for it in items]
        print(f"[{split}] {len(items)} yes/no answers, {sum(i['label'] for i in items)} yes")
        add(split, "always-no", [0.0] * len(items))
        for name, probs in laya_reference(split, items).items():
            add(split, name, probs)

    for name in args.models:
        try:
            scorer = NLIScorer(name)
        except Exception as e:  # keep going if one model fails to load
            print(f"{name}: FAILED {type(e).__name__}: {e}")
            continue
        for split, items in data.items():
            add(split, name, scorer.score([(it["text"], it["statement"]) for it in items]))

    out_path.write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
