"""Latency per request on CPU (PyTorch fp32), for 1, 5 and 20 yes/no questions about one text.

A cross-encoder needs one forward pass per (text, question) pair; the questions are batched
together, so the cost grows with the number of questions. Run on an otherwise idle machine.

    python scripts/bench_latency.py checkpoints/maya MoritzLaurer/deberta-v3-large-zeroshot-v2.0
"""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from maya.data import EVAL_V2_DIR, load_yes_no_items  # noqa: E402
from maya.nli import NLIScorer  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--reps", type=int, default=15)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    items = load_yes_no_items(EVAL_V2_DIR)
    texts = sorted({it["text"] for it in items})[:10]
    statements = sorted({it["statement"] for it in items})[:20]
    out = {"threads": args.threads, "models": {}}
    for name in args.models:
        sc = NLIScorer(name, batch_size=32)
        params = sum(p.numel() for p in sc.model.parameters()) / 1e6
        res = {"params_m": round(params, 1)}
        for nq in (1, 5, 20):
            sc.score([(texts[0], s) for s in statements[:nq]])  # warm-up
            times = []
            for r in range(args.reps):
                t = texts[r % len(texts)]
                t0 = time.perf_counter()
                sc.score([(t, s) for s in statements[:nq]])
                times.append((time.perf_counter() - t0) * 1000)
            res[f"ms_{nq}q"] = round(statistics.median(times), 1)
        out["models"][name] = res
        print(name, res, flush=True)
    (ROOT / "results" / "latency.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
