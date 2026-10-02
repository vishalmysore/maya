"""Fast checks: eval sets, metrics and abstention (no model needed)."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from maya.conformal import Abstainer, cp_upper, fit_temperature  # noqa: E402
from maya.data import EVAL_DIR, EVAL_V2_DIR, load_domains, load_relations, load_yes_no_items  # noqa: E402
from maya.metrics import consistency, coverage_at_risk, ece, summarize  # noqa: E402


# ------------------------------------------------------------------ eval sets
def test_v1_counts():
    items = load_yes_no_items(EVAL_DIR)
    assert len(items) == 120
    assert sum(i["label"] for i in items) == 45


def test_v2_counts_and_balance():
    items = load_yes_no_items(EVAL_V2_DIR)
    assert len(items) == 256
    assert len({(i["domain"], i["id"]) for i in items}) == 64
    assert len({i["statement"] for i in items}) == 32


def test_v2_relations_hold_in_labels():
    for dom in load_domains(EVAL_V2_DIR):
        rel = dom["relations"]
        for c in dom["cases"]:
            e = c["expected"]
            for a, b in rel["negations"]:
                assert e[a] != e[b], (dom["domain"], c["id"])
            for a, b in rel["implications"]:
                assert not (e[a] and not e[b]), (dom["domain"], c["id"])


def test_v2_minimal_pairs_flip_key():
    for dom in load_domains(EVAL_V2_DIR):
        pairs = {}
        for c in dom["cases"]:
            pairs.setdefault(c["pair"], []).append(c["expected"][dom["relations"]["minimal_pairs_flip"]])
        assert all(len(v) == 2 and v[0] != v[1] for v in pairs.values()), dom["domain"]


def test_v1_negation_pair_is_consistent_in_labels():
    rel = load_relations(EVAL_DIR)["agent-guardrails"]["negations"][0]
    for dom in load_domains(EVAL_DIR):
        if dom["domain"] == "agent-guardrails":
            for c in dom["cases"]:
                assert c["expected"][rel[0]] != c["expected"][rel[1]]


# ------------------------------------------------------------------ metrics
def test_ece_perfect_and_worst():
    assert ece([1.0, 0.0], [1, 0]) == 0.0
    assert ece([1.0, 0.0], [0, 1]) == pytest.approx(1.0)


def test_coverage_at_risk():
    p = np.array([0.99, 0.98, 0.9, 0.6, 0.4])
    y = np.array([True, True, True, False, True])
    assert coverage_at_risk(p, y, 0.0) == pytest.approx(0.6)


def test_consistency_counts_contradictions():
    items = [
        {"domain": "d", "id": "a", "question": "q", "label": True, "pair": "p1"},
        {"domain": "d", "id": "a", "question": "nq", "label": False, "pair": "p1"},
        {"domain": "d", "id": "b", "question": "q", "label": False, "pair": "p1"},
        {"domain": "d", "id": "b", "question": "nq", "label": True, "pair": "p1"},
    ]
    rel = {"d": {"negations": [["q", "nq"]], "implications": [["q", "nq"]], "minimal_pairs_flip": "q"}}
    c = consistency(items, [0.9, 0.8, 0.1, 0.9], rel)  # case a: yes/yes = contradiction
    assert c["negation_contradictions"] == pytest.approx(0.5)
    assert c["implication_violations"] == pytest.approx(0.0)  # a: q yes, nq yes -> implication holds
    assert c["minimal_pairs_both_right"] == pytest.approx(1.0)


def test_summarize_runs_on_v2_shape():
    items = load_yes_no_items(EVAL_V2_DIR)
    s = summarize(items, [0.5] * len(items), load_relations(EVAL_V2_DIR))
    assert s["n"] == 256 and s["negation_pairs"] == 64 and s["implication_pairs"] == 64 and s["minimal_pairs"] == 32


# ------------------------------------------------------------------ abstention
def test_cp_upper_bounds():
    assert cp_upper(0, 0, 0.1) == 1.0
    assert cp_upper(0, 100, 0.05) == pytest.approx(1 - 0.05 ** (1 / 100), rel=1e-6)


def test_abstainer_guarantee_on_in_distribution_data():
    """Fit on one sample, test on another from the same distribution: error on answered <= alpha."""
    rng = np.random.default_rng(0)

    def draw(n):
        y = rng.random(n) < 0.5
        p = np.clip(np.where(y, rng.beta(5, 2, n), rng.beta(2, 5, n)), 0, 1)
        return p, y

    p_cal, y_cal = draw(4000)
    ab = Abstainer.fit(p_cal, y_cal, alpha=0.1, delta=0.1)
    assert ab.t_yes is not None and ab.t_no is not None
    p_te, y_te = draw(20000)
    r = ab.report(p_te, y_te)
    assert 0 < r["coverage"] < 1
    assert r["yes_error"] <= 0.1 and r["no_error"] <= 0.1


def test_abstainer_decides_only_three_answers():
    ab = Abstainer(t_yes=0.8, t_no=0.2)
    assert set(ab.decide([0.9, 0.5, 0.1]).tolist()) == {1, -1, 0}


def test_fit_temperature_recovers_scale():
    rng = np.random.default_rng(1)
    z = rng.normal(0, 3, 20000)
    y = rng.random(20000) < 1 / (1 + np.exp(-z / 2.0))
    assert fit_temperature(z, y) == pytest.approx(2.0, abs=0.15)


# ------------------------------------------------------------------ recorded results are complete
@pytest.mark.parametrize("name", ["maya", "base", "maya-noconsist"])
def test_result_files_have_every_probability(name):
    d = json.loads((ROOT / "results" / f"eval_{name}.json").read_text(encoding="utf-8"))
    assert len(d["sets"]["v1"]["probs"]) == 120 and len(d["sets"]["v2"]["probs"]) == 256
    assert all(0.0 <= p <= 1.0 for p in d["sets"]["v2"]["probs"])
