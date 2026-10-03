"""Record laya-typed-decisions (general head) and layaMOE (trained router) yes/no answers on an eval set.

Needs a layaMOE checkout with trained experts (set LAYA_MOE_DIR; default C:/work/jev/layaMOE); run with its venv:
    <layaMOE>/.venv/Scripts/python scripts/record_laya.py data/eval_v3 results/laya_v3_reference.json
"""
import glob, json, os, sys
import torch
ROOT = os.environ.get("LAYA_MOE_DIR", r"C:\work\jev\layaMOE")
sys.path.insert(0, ROOT)
from laya_moe.moe import MoEAgent
src, out = sys.argv[1], sys.argv[2]
torch.set_grad_enabled(False)
moe = MoEAgent(experts=[os.path.join(ROOT, "checkpoints", e) for e in ("safety", "customer_ops")], threshold=0.3,
               router=os.path.join(ROOT, "checkpoints", "router"))
rows = []
for p in sorted(glob.glob(os.path.join(src, "*.json"))):
    if p.endswith("index.json"): continue
    d = json.load(open(p, encoding="utf-8"))
    for c in d["cases"]:
        g = moe.answer_with("general", c["state"], d["questions"])
        m = moe.system_one(c["state"], d["questions"])
        for q in d["questions"]:
            rows.append({"domain": d["domain"], "id": c["id"], "question": q, "expected": c["expected"][q],
                         "routed_to": m["expert"], "answers": {"general": g[q], "moe": m["answers"][q]}})
    print(d["domain"], flush=True)
json.dump(rows, open(out, "w", encoding="utf-8"), indent=1)
