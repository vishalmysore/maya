"""Step 0: how far do existing models get on the yes/no eval items?

Baselines:
  - always "no" (the majority answer)
  - laya-typed-decisions and layaMOE (probabilities recorded by layaMOE's eval_moe.py)
  - off-the-shelf zero-shot NLI models: premise = the text, hypothesis = the yes/no statement

Writes results/baselines.json (summary + every probability).
"""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from maya.data import load_yes_no_items  # noqa: E402
from maya.metrics import per_domain_accuracy, summarize  # noqa: E402

NLI_MODELS = [
    "cross-encoder/nli-deberta-v3-xsmall",
    "MoritzLaurer/deberta-v3-xsmall-zeroshot-v1.1-all-33",
    "MoritzLaurer/deberta-v3-base-zeroshot-v2.0",
    "MoritzLaurer/ModernBERT-base-zeroshot-v2.0",
    "MoritzLaurer/deberta-v3-large-zeroshot-v2.0",
]


def laya_reference(items):
    rows = json.loads((ROOT / "results" / "laya_moe_eval_reference.json").read_text(encoding="utf-8"))
    by_key = {(r["domain"], r["id"], r["question"]): r for r in rows if r["answers"]["general"]["type"] == "noul"}
    out = {}
    for name in ("general", "moe"):
        out[name] = [by_key[(it["domain"], it["id"], it["question"])]["answers"][name]["noul"] for it in items]
    return out


def entail_index(model):
    labels = {i: l.lower() for i, l in model.config.id2label.items()}
    for i, l in labels.items():
        if l.startswith("entail"):
            return i, labels
    raise ValueError(f"no entailment label in {labels}")


@torch.inference_mode()
def run_nli(name, items, threads):
    torch.set_num_threads(threads)
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForSequenceClassification.from_pretrained(name).eval()
    ent, labels = entail_index(model)
    params = sum(p.numel() for p in model.parameters())
    # warm-up
    enc = tok(items[0]["text"], items[0]["statement"], return_tensors="pt", truncation=True, max_length=512)
    model(**enc)
    probs, times = [], []
    for it in items:
        enc = tok(it["text"], it["statement"], return_tensors="pt", truncation=True, max_length=512)
        t0 = time.perf_counter()
        logits = model(**enc).logits[0]
        times.append((time.perf_counter() - t0) * 1000)
        # 2-way (entailment / not) or 3-way NLI: P(yes) = P(entailment)
        probs.append(float(torch.softmax(logits, -1)[ent]))
    return probs, {"params_m": round(params / 1e6, 1), "ms_per_answer": round(statistics.median(times), 1),
                   "labels": labels}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=NLI_MODELS)
    ap.add_argument("--threads", type=int, default=2)
    args = ap.parse_args()

    items = load_yes_no_items()
    print(f"{len(items)} yes/no answers, {sum(i['label'] for i in items)} yes")
    results = {}

    def add(name, probs, extra=None):
        s = summarize(items, probs)
        s["per_domain"] = per_domain_accuracy(items, probs)
        s.update(extra or {})
        results[name] = {"summary": s, "probs": [round(p, 4) for p in probs]}
        print(f"{name:55s} acc {s['accuracy']:.3f}  bal {s['balanced_accuracy']:.3f}  auroc {s['auroc']:.3f}  "
              f"ece {s['ece']:.3f}  cov@10% {s['coverage_at_10pct_error']:.2f}  contra {s['contradiction_rate']:.2f}"
              + (f"  {extra['ms_per_answer']} ms  {extra['params_m']}M" if extra else ""))

    add("always-no", [0.0] * len(items))
    for name, probs in laya_reference(items).items():
        add(f"laya-typed-decisions ({name})" if name == "general" else "layaMOE (prompted router)", probs)
    for name in args.models:
        try:
            probs, extra = run_nli(name, items, args.threads)
        except Exception as e:  # keep going if one model fails to load
            print(f"{name}: FAILED {type(e).__name__}: {e}")
            continue
        add(name, probs, extra)

    out = {"items": [{k: it[k] for k in ("domain", "id", "question", "label")} for it in items], "models": results}
    (ROOT / "results" / "baselines.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
