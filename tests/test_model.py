"""Model checks (slow): the gate API and the int8 ONNX build.

Uses checkpoints/maya-final-copy if present, otherwise downloads VishalMysore/maya.
Run with:  pytest -m slow
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
pytestmark = pytest.mark.slow

LOCAL = ROOT / "checkpoints" / "maya-final-copy"
MODEL = str(LOCAL) if LOCAL.exists() else "VishalMysore/maya"


@pytest.fixture(scope="module")
def maya():
    from maya.gate import Maya
    return Maya.load(MODEL)


def test_answers_are_only_yes_no_or_not_sure(maya):
    stmts = ["Is the customer angry?", "What colour is the sky?", "The customer sounds calm", ""]
    out = maya.ask("Hi team, how do I export the dashboard to Excel? No rush.", stmts)
    assert [o["answer"] for o in out] and all(o["answer"] in {"yes", "no"} for o in out)
    maya.t_yes, maya.t_no = 0.9, 0.1
    try:
        out = maya.ask("Hi team, how do I export the dashboard to Excel? No rush.", stmts)
        assert all(o["answer"] in {"yes", "no", "not sure"} for o in out)
    finally:
        maya.t_yes, maya.t_no = 0.5, 0.5


def test_batch_equals_single(maya):
    text = "Pull request: Fix typos in the README. Tests added, CI is green. Approved by two reviewers."
    stmts = ["Can this be merged now?", "Is this a breaking change?", "Has someone approved this pull request?"]
    batch = [o["p_yes"] for o in maya.ask(text, stmts)]
    single = [maya.ask(text, [s])[0]["p_yes"] for s in stmts]
    assert np.allclose(batch, single, atol=1e-3)


def test_matches_recorded_eval_probabilities(maya):
    """Re-score a slice of eval_v2 and compare with results/eval_maya.json (raw, temperature 1)."""
    from maya.data import EVAL_V2_DIR, load_yes_no_items
    items = load_yes_no_items(EVAL_V2_DIR)[::8]
    rec = json.loads((ROOT / "results" / "eval_maya.json").read_text(encoding="utf-8"))["sets"]["v2"]["probs"][::8]
    t = maya.temperature
    maya.temperature = 1.0
    try:
        got = [maya.ask(it["text"], [it["statement"]])[0]["p_yes"] for it in items]
    finally:
        maya.temperature = t
    assert np.abs(np.array(got) - np.array(rec)).max() < 2e-3


def test_int8_onnx_matches_pytorch():
    web = ROOT / "build" / "onnx" / "model.onnx"
    if not web.exists():
        pytest.skip("run scripts/export_onnx.py first")
    import onnxruntime as ort
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from maya.data import EVAL_V2_DIR, load_yes_no_items
    items = load_yes_no_items(EVAL_V2_DIR)[::4]
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL, dtype=torch.float32).eval()
    sess = ort.InferenceSession(str(web), providers=["CPUExecutionProvider"])
    enc = tok([i["text"] for i in items], [i["statement"] for i in items], return_tensors="pt", padding=True)
    with torch.inference_mode():
        lg = model(**enc).logits
    ref = torch.sigmoid(lg[:, 0] - lg[:, 1]).numpy()
    lo = sess.run(None, {"input_ids": enc["input_ids"].numpy(), "attention_mask": enc["attention_mask"].numpy()})[0]
    got = 1 / (1 + np.exp(-(lo[:, 0] - lo[:, 1])))
    assert np.abs(ref - got).max() < 0.15
    assert ((ref >= 0.5) != (got >= 0.5)).sum() <= 2
