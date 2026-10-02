"""Generate Maya's rule-labeled yes/no training data.

Sources:
  - the five layaMOE domains (gen/laya_domains.py), turned into yes/no facts by gen/facts.py
  - seven slot-based domains (gen/new_domains.py); about half of their texts get a minimal-pair
    partner that differs in one slot and flips at least one fact

Each output row is one text with several (statement, label) items. An item carries its fact name
and polarity (+1 for a wording that is true when the fact holds, -1 for a negated wording), so the
trainer can tie together items that talk about the same fact (negation consistency) and apply the
domain's implications. About a third of the statements are phrased as questions.

Leakage guards: texts sharing a word 5-gram with any eval text (v1 or v2) are dropped, and no
training wording may equal an eval_v2 statement (v2 tests unseen domains and unseen statements).

    python scripts/gen_train_data.py          # -> data/train/train.jsonl, data/train/val.jsonl
"""
import argparse
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from gen import laya_domains  # noqa: E402
from gen.facts import LAYA_FACTS  # noqa: E402
from gen.new_domains import NEW_DOMAINS  # noqa: E402
from maya.data import EVAL_SETS, load_domains, state_text  # noqa: E402


def ngrams(text, n=5):
    w = re.findall(r"[a-z0-9]+", text.lower())
    return {tuple(w[i:i + n]) for i in range(max(0, len(w) - n + 1))}


def fact_value(facts, ref):
    neg = ref.startswith("not:")
    v = facts[ref[4:] if neg else ref].value
    return (not v) if neg else v


def make_items(rng, facts, k):
    """Pick k facts; one wording each, and often the opposite-polarity wording too."""
    items = []
    for f in rng.sample(facts, min(k, len(facts))):
        pols = [1, -1] if f.neg else [1]
        first = rng.choice(pols)
        chosen = [first] + ([-first] if f.neg and rng.random() < 0.5 else [])
        for pol in chosen:
            text = rng.choice(f.pos if pol == 1 else f.neg)
            items.append({"statement": text, "label": f.value if pol == 1 else not f.value,
                          "fact": f.name, "pol": pol})
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--laya-per-domain", type=int, default=400)
    ap.add_argument("--new-per-domain", type=int, default=300)
    ap.add_argument("--facts-per-text", type=int, default=3)
    ap.add_argument("--pair-rate", type=float, default=0.5)
    ap.add_argument("--val-frac", type=float, default=0.08)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    laya_domains.R.seed(args.seed)

    eval_grams, v2_statements = set(), set()
    for split, d in EVAL_SETS.items():
        for dom in load_domains(d):
            for c in dom["cases"]:
                eval_grams |= ngrams(state_text(c["state"]))
            if split == "v2":
                v2_statements |= {q["instructions"].lower() for q in dom["questions"].values()}

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

    # layaMOE domains
    for dom, (_, gen) in laya_domains.GENERATORS.items():
        to_facts, impl = LAYA_FACTS[dom]
        n = tries = 0
        while n < args.laya_per_domain and tries < args.laya_per_domain * 100:
            tries += 1
            state, lab = gen()
            facts = {f.name: f for f in to_facts(lab)}
            for a, b in impl:
                assert not facts[a].value or fact_value(facts, b), (dom, a, b, state)
            row = {"domain": dom, "text": state_text(state), "implications": impl,
                   "items": make_items(rng, list(facts.values()), args.facts_per_text)}
            n += add(row)

    # slot-based domains with minimal pairs
    for D in NEW_DOMAINS:
        n = tries = pairs = 0
        while n < args.new_per_domain and tries < args.new_per_domain * 100:
            tries += 1
            s = D.sample(rng)
            facts = {f.name: f for f in D.facts(s)}
            for a, b in D.implications:
                assert not facts[a].value or fact_value(facts, b), (D.name, a, b, s)
            pid = None
            partner = None
            if rng.random() < args.pair_rate:
                slot = rng.choice(list(D.flips))
                alt = rng.choice([v for v in D.flips[slot] if v != s[slot]])
                s2 = {**s, slot: alt}
                f2 = {f.name: f for f in D.facts(s2)}
                flipped = [k for k in facts if facts[k].value != f2[k].value]
                if flipped and D.render(s2) != D.render(s):
                    pid = f"{D.name}-{len(rows)}"
                    # the pair shares its flipped fact so the trainer sees the contrast
                    key = rng.choice(flipped)
                    partner = (s2, f2, key)
            items = make_items(rng, list(facts.values()), args.facts_per_text)
            row = {"domain": D.name, "text": D.render(s), "implications": D.implications, "items": items}
            if partner:
                s2, f2, key = partner
                if key not in {it["fact"] for it in items}:
                    f = facts[key]
                    items.append({"statement": rng.choice(f.pos), "label": f.value, "fact": key, "pol": 1})
                row["pair"] = pid
            if not add(row):
                continue
            n += 1
            if partner:
                s2, f2, key = partner
                # same wordings for the partner, labels recomputed from its slots
                items2 = [{**it, "label": (f2[it["fact"]].value if it["pol"] == 1 else not f2[it["fact"]].value)}
                          for it in items]
                if add({"domain": D.name, "text": D.render(s2), "implications": D.implications,
                        "items": items2, "pair": pid}):
                    pairs += 1
        print(f"{D.name:20s} {n} texts + {pairs} minimal-pair partners")

    for r in rows:
        for it in r["items"]:
            assert it["statement"].lower() not in v2_statements, it["statement"]

    # split by text (and keep minimal pairs together)
    groups = {}
    for r in rows:
        groups.setdefault(r.get("pair") or r["text"], []).append(r)
    keys = sorted(groups)
    rng.shuffle(keys)
    n_val = int(len(keys) * args.val_frac)
    out = ROOT / "data" / "train"
    out.mkdir(parents=True, exist_ok=True)
    for name, ks in (("val", keys[:n_val]), ("train", keys[n_val:])):
        with open(out / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for k in ks:
                for r in groups[k]:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
    n_items = sum(len(r["items"]) for r in rows)
    yes = sum(it["label"] for r in rows for it in r["items"])
    print(f"{len(rows)} texts, {n_items} items ({yes / n_items:.0%} yes), dropped {dropped} close to eval; "
          f"val groups {n_val}")


if __name__ == "__main__":
    main()
