"""Training data generators: rules hold, minimal pairs really differ, no leakage into the eval sets."""
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from gen import laya_domains  # noqa: E402
from gen.facts import LAYA_FACTS  # noqa: E402
from gen.new_domains import NEW_DOMAINS  # noqa: E402
from maya.data import EVAL_SETS, EVAL_V2_DIR, load_domains, state_text  # noqa: E402


def fact_value(facts, ref):
    neg = ref.startswith("not:")
    v = facts[ref[4:] if neg else ref].value
    return (not v) if neg else v


def test_laya_domain_implications_hold():
    laya_domains.R.seed(3)
    for dom, (_, gen) in laya_domains.GENERATORS.items():
        to_facts, impl = LAYA_FACTS[dom]
        for _ in range(500):
            _, lab = gen()
            facts = {f.name: f for f in to_facts(lab)}
            for a, b in impl:
                assert not facts[a].value or fact_value(facts, b), (dom, a, b)


def test_new_domain_implications_and_wordings():
    rng = random.Random(3)
    for D in NEW_DOMAINS:
        for _ in range(500):
            s = D.sample(rng)
            facts = {f.name: f for f in D.facts(s)}
            for a, b in D.implications:
                assert not facts[a].value or fact_value(facts, b), (D.name, a, b, s)
            for f in facts.values():
                assert f.pos, (D.name, f.name)


def test_render_is_deterministic_and_flips_change_text():
    rng = random.Random(5)
    changed = 0
    for D in NEW_DOMAINS:
        for _ in range(100):
            s = D.sample(rng)
            assert D.render(s) == D.render(dict(s))
            slot = rng.choice(list(D.flips))
            alt = [v for v in D.flips[slot] if v != s[slot]][0]
            changed += D.render({**s, slot: alt}) != D.render(s)
    assert changed > 0.7 * 100 * len(NEW_DOMAINS)


def test_no_training_wording_equals_a_v2_statement():
    v2 = {q["instructions"].lower() for d in load_domains(EVAL_V2_DIR) for q in d["questions"].values()}
    wordings = set()
    laya_domains.R.seed(4)
    for dom, (_, gen) in laya_domains.GENERATORS.items():
        _, lab = gen()
        for f in LAYA_FACTS[dom][0](lab):
            wordings |= {w.lower() for w in f.pos + f.neg}
    rng = random.Random(4)
    for D in NEW_DOMAINS:
        for f in D.facts(D.sample(rng)):
            wordings |= {w.lower() for w in f.pos + f.neg}
    assert not (wordings & v2)


def test_generated_training_file_has_no_eval_overlap():
    path = ROOT / "data" / "train" / "train.jsonl"
    if not path.exists():
        import pytest
        pytest.skip("run scripts/gen_train_data.py first")
    import json

    def grams(t):
        w = re.findall(r"[a-z0-9]+", t.lower())
        return {tuple(w[i:i + 5]) for i in range(max(0, len(w) - 4))}

    ev = set()
    for name, d in EVAL_SETS.items():
        if name == "v3":  # eval_v3 was written after the v0.1 data; v0.2 data is checked against it below
            continue
        for dom in load_domains(d):
            for c in dom["cases"]:
                ev |= grams(state_text(c["state"]))
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        assert not (grams(r["text"]) & ev), r["text"]


# ------------------------------------------------------------------ v0.2 generators and data
def test_v2_generators_rules_hold():
    from gen.guardrails_v2 import GUARDRAILS
    from gen.tone_v2 import TONE
    rng = random.Random(9)
    for D in (GUARDRAILS, TONE):
        for _ in range(2000):
            s = D.sample(rng)
            facts = {f.name: f for f in D.facts(s)}
            for a, b in D.implications:
                if a in facts and (b[4:] if b.startswith("not:") else b) in facts:
                    assert not facts[a].value or fact_value(facts, b), (D.name, a, b, s)
            assert D.render(s) == D.render(dict(s))


def test_train_v2_has_no_held_out_statements_or_texts():
    import json
    path = ROOT / "data" / "train_v2" / "train.jsonl"
    if not path.exists():
        import pytest
        pytest.skip("run scripts/gen_train_data_v2.py first")
    from maya.data import DEV_DIR

    def grams(t):
        w = re.findall(r"[a-z0-9]+", t.lower())
        return {tuple(w[i:i + 5]) for i in range(max(0, len(w) - 4))}

    held, ev = set(), set()
    for name, d in {**EVAL_SETS, "dev": DEV_DIR}.items():
        for dom in load_domains(d):
            if name != "v1":
                held |= {q["instructions"].lower() for q in dom["questions"].values()}
            for c in dom["cases"]:
                ev |= grams(state_text(c["state"]))
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        assert not (grams(r["text"]) & ev), r["text"]
        assert not ({it["statement"].lower() for it in r["items"]} & held)
