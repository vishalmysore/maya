"""Evaluate an ensemble of already-scored models by averaging their logits (no model runs needed).

    python scripts/eval_ensemble.py maya-v2-base maya-v2-large --name maya-v2-ensemble

Reads results/eval_<name>.json written by scripts/eval_maya.py and writes results/eval_<ensemble>.json
in the same format (so --cached re-runs and the charts work on it).
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def logit(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("members", nargs="+")
    ap.add_argument("--name", required=True)
    args = ap.parse_args()
    runs = [json.loads((ROOT / "results" / f"eval_{m}.json").read_text(encoding="utf-8")) for m in args.members]
    out = {"model": "ensemble of " + ", ".join(args.members), "sets": {}}
    for s in runs[0]["sets"]:
        z = np.mean([logit(r["sets"][s]["probs"]) for r in runs], axis=0)
        out["sets"][s] = {"probs": [round(float(x), 4) for x in 1 / (1 + np.exp(-z))]}
    (ROOT / "results" / f"eval_{args.name}.json").write_text(json.dumps(out), encoding="utf-8")
    # reuse eval_maya.py for every metric, calibration and abstention
    subprocess.run([sys.executable, str(ROOT / "scripts" / "eval_maya.py"), "ensemble", "--name", args.name, "--cached"], check=True)


if __name__ == "__main__":
    main()
