"""Load the yes/no (noul) items from a hand-labeled eval set.

Eval files follow the layaMOE format (domain, questions, cases). eval_v2 files also carry
`relations` (negation and implication pairs between questions) and a `pair` id per case
(minimal pairs: two texts that differ in one detail and flip the `minimal_pairs_flip` answer).
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = ROOT / "data" / "eval"
EVAL_V2_DIR = ROOT / "data" / "eval_v2"
EVAL_V3_DIR = ROOT / "data" / "eval_v3"   # judgment-heavy test set, written before the v0.2 training data
DEV_DIR = ROOT / "data" / "dev"           # small dev set: model selection and calibration only
EVAL_SETS = {"v1": EVAL_DIR, "v2": EVAL_V2_DIR, "v3": EVAL_V3_DIR}

# v1 files predate `relations`; its one natural negation pair is listed here.
V1_RELATIONS = {"agent-guardrails": {"negations": [["needs_human", "safe_without_approval"]]}}


def state_text(state):
    """States are plain text or a JSON object (support tickets)."""
    if isinstance(state, str):
        return state
    return json.dumps(state, ensure_ascii=False)


def load_domains(eval_dir=EVAL_DIR):
    index = json.loads((eval_dir / "index.json").read_text(encoding="utf-8"))
    for entry in index["domains"]:
        domain = json.loads((eval_dir / entry["file"]).read_text(encoding="utf-8"))
        domain.setdefault("relations", V1_RELATIONS.get(domain["domain"], {}))
        yield domain


def load_yes_no_items(eval_dir=EVAL_DIR):
    items = []
    for domain in load_domains(eval_dir):
        for key, q in domain["questions"].items():
            if q["type"] != "noul":
                continue
            for case in domain["cases"]:
                label = case["expected"].get(key)
                if label is None:
                    continue
                items.append({
                    "domain": domain["domain"],
                    "id": case["id"],
                    "question": key,
                    "statement": q["instructions"],
                    "text": state_text(case["state"]),
                    "label": bool(label),
                    "pair": case.get("pair"),
                    "trap": bool(case.get("trap")),
                })
    return items


def load_relations(eval_dir=EVAL_DIR):
    """{domain: {"negations": [[qa, qb]], "implications": [[qa, qb]], "minimal_pairs_flip": q}}"""
    return {d["domain"]: d["relations"] for d in load_domains(eval_dir)}
