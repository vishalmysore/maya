"""Generate Maya v0.2 training data -> data/train_v2/{train,val}.jsonl

Mix (texts):
  - agent guardrails with varied wording, safety nets and traps (gen/guardrails_v2.py), ~45% minimal pairs
  - customer messages with varied tone and traps (gen/tone_v2.py), ~45% minimal pairs
  - a smaller share of the v0.1 domains (gen/laya_domains.py via gen/facts.py, gen/new_domains.py)
BoolQ and MNLI pairs are mixed in at training time (scripts/train.py --boolq / --mnli-per-step).

Leakage guards: texts sharing a word 5-gram with any eval or dev text are dropped, and no training
wording may equal a statement of eval_v2, eval_v3 or dev (case-insensitive).
"""
import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from gen import laya_domains  # noqa: E402
from gen.facts import LAYA_FACTS  # noqa: E402
from gen.guardrails_v2 import GUARDRAILS  # noqa: E402
from gen.new_domains import NEW_DOMAINS  # noqa: E402
from gen.tone_v2 import TONE  # noqa: E402
from gen_train_data import fact_value, make_items, ngrams  # noqa: E402
from maya.data import DEV_DIR, EVAL_SETS, load_domains, state_text  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--guardrails", type=int, default=1600)
    ap.add_argument("--tone", type=int, default=1400)
    ap.add_argument("--laya-per-domain", type=int, default=120)
    ap.add_argument("--new-per-domain", type=int, default=90)
    ap.add_argument("--pair-rate", type=float, default=0.6)
    ap.add_argument("--facts-per-text", type=int, default=3)
    ap.add_argument("--val-frac", type=float, default=0.06)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--out", default=str(ROOT / "data" / "train_v2"))
    args = ap.parse_args()
    rng = random.Random(args.seed)
    laya_domains.R.seed(args.seed)

    eval_grams, held_out_statements = set(), set()
    for split, d in {**EVAL_SETS, "dev": DEV_DIR}.items():
        for dom in load_domains(d):
            for c in dom["cases"]:
                eval_grams |= ngrams(state_text(c["state"]))
            if split != "v1":
                held_out_statements |= {q["instructions"].lower() for q in dom["questions"].values()}

    rows, seen, dropped = [], set(), 0

    def add(row):
        nonlocal dropped
        if row["text"] in seen:
            return False
        if ngrams(row["text"]) & eval_grams:
            dropped += 1
            return False
        seen.add(row["text"])
        rows.append(row)
        return True

    def slot_domain(D, n_texts):
        n = tries = pairs = 0
        while n < n_texts and tries < n_texts * 200:
            tries += 1
            s = D.sample(rng)
            facts = {f.name: f for f in D.facts(s)}
            for a, b in D.implications:
                if a in facts and (b[4:] if b.startswith("not:") else b) in facts:
                    assert not facts[a].value or fact_value(facts, b), (D.name, a, b, s)
            partner = None
            if rng.random() < args.pair_rate:
                slots = [k for k in D.flips if getattr(D, "valid_flip", lambda s, k: True)(s, k)]
                if slots:
                    slot = rng.choice(slots)
                    alt = rng.choice([v for v in D.flips[slot] if v != s[slot]])
                    s2 = {**s, slot: alt}
                    f2 = {f.name: f for f in D.facts(s2)}
                    common = [k for k in facts if k in f2]
                    flipped = [k for k in common if facts[k].value != f2[k].value]
                    if flipped and D.render(s2) != D.render(s):
                        partner = (s2, f2, rng.choice(flipped), common)
            items = make_items(rng, list(facts.values()), args.facts_per_text)
            row = {"domain": D.name, "text": D.render(s), "implications": D.implications, "items": items}
            if partner:
                s2, f2, key, common = partner
                if key not in {it["fact"] for it in items}:
                    items.append({"statement": rng.choice(facts[key].pos), "label": facts[key].value, "fact": key, "pol": 1})
                row["items"] = [it for it in items if it["fact"] in common]
                row["pair"] = f"{D.name}-{len(rows)}"
            if not add(row):
                continue
            n += 1
            if partner:
                items2 = [{**it, "label": (f2[it["fact"]].value if it["pol"] == 1 else not f2[it["fact"]].value)}
                          for it in row["items"]]
                if add({"domain": D.name, "text": D.render(s2), "implications": D.implications, "items": items2,
                        "pair": row["pair"]}):
                    pairs += 1
        print(f"{D.name:22s} {n} texts + {pairs} minimal-pair partners", flush=True)

    slot_domain(GUARDRAILS, args.guardrails)
    slot_domain(TONE, args.tone)
    for D in NEW_DOMAINS:
        slot_domain(D, args.new_per_domain)
    for dom, (_, gen) in laya_domains.GENERATORS.items():
        to_facts, impl = LAYA_FACTS[dom]
        n = tries = 0
        while n < args.laya_per_domain and tries < args.laya_per_domain * 100:
            tries += 1
            state, lab = gen()
            facts = {f.name: f for f in to_facts(lab)}
            n += add({"domain": dom, "text": state_text(state), "implications": impl,
                      "items": make_items(rng, list(facts.values()), args.facts_per_text)})
        print(f"{dom:22s} {n} texts", flush=True)

    for r in rows:
        for it in r["items"]:
            assert it["statement"].lower() not in held_out_statements, it["statement"]

    groups = {}
    for r in rows:
        groups.setdefault(r.get("pair") or r["text"], []).append(r)
    keys = sorted(groups)
    rng.shuffle(keys)
    n_val = int(len(keys) * args.val_frac)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, ks in (("val", keys[:n_val]), ("train", keys[n_val:])):
        with open(out / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for k in ks:
                for r in groups[k]:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
    n_items = sum(len(r["items"]) for r in rows)
    yes = sum(it["label"] for r in rows for it in r["items"])
    print(f"{len(rows)} texts, {n_items} items ({yes / n_items:.0%} yes), dropped {dropped} close to eval/dev")


if __name__ == "__main__":
    main()
