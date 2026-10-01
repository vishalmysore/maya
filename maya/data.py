"""Load the yes/no (noul) items from the hand-labeled eval set."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = ROOT / "data" / "eval"

# Question pairs whose answers must disagree (one is the negation of the other).
NEGATION_PAIRS = [("agent-guardrails", "needs_human", "safe_without_approval")]


def state_text(state):
    """States are plain text or a JSON object (support tickets)."""
    if isinstance(state, str):
        return state
    return json.dumps(state, ensure_ascii=False)


def load_yes_no_items(eval_dir=EVAL_DIR):
    index = json.loads((eval_dir / "index.json").read_text(encoding="utf-8"))
    items = []
    for entry in index["domains"]:
        domain = json.loads((eval_dir / entry["file"]).read_text(encoding="utf-8"))
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
                })
    return items
