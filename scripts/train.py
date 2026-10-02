"""Fine-tune a zero-shot NLI cross-encoder into Maya, a yes/no gate.

P(yes) = sigmoid(z), z = logit(entailment) - logit(not_entailment), for the pair (text, statement).

Loss per step (a step = several texts with all their statements, plus a few MNLI pairs):
  - BCE on every statement's yes/no label
  - consistency (--consistency weight):
      * statements about the same fact on the same text must agree once polarity is applied:
        (t_i - t_j)^2 with t = p for a plain wording and 1 - p for a negated one
      * domain implications A => B: relu(t_A - t_B) (B may be "not:x", then t_B = 1 - t_x)
  - BCE on MNLI pairs (entailment = yes) so the model keeps its general NLI ability

Model selection uses only the synthetic validation split; data/eval and data/eval_v2 are never
looked at during training.

    python scripts/train.py --out checkpoints/maya
    python scripts/train.py --out checkpoints/maya-noconsist --consistency 0
"""
import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from maya.metrics import ece  # noqa: E402

BASE = "MoritzLaurer/ModernBERT-base-zeroshot-v2.0"


def load_rows(path):
    return [json.loads(l) for l in open(path, encoding="utf-8")]


def load_mnli(n, seed):
    from datasets import load_dataset
    d = load_dataset("nyu-mll/glue", "mnli", split="train[:60000]").shuffle(seed=seed).select(range(n))
    return [(r["premise"], r["hypothesis"], r["label"] == 0) for r in d]


def yes_logit(model, tok, texts, statements, max_length):
    enc = tok(texts, statements, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
    logits = model(**enc).logits
    return logits[:, 0] - logits[:, 1]  # entailment - not_entailment


def consistency_loss(rows, p):
    """rows: list of (row, item_offset). Returns mean penalty over agreement and implication terms."""
    terms = []
    for row, off in rows:
        by_fact = {}
        for j, it in enumerate(row["items"]):
            t = p[off + j] if it["pol"] == 1 else 1 - p[off + j]
            by_fact.setdefault(it["fact"], []).append(t)
        for ts in by_fact.values():
            for a in range(len(ts)):
                for b in range(a + 1, len(ts)):
                    terms.append((ts[a] - ts[b]) ** 2)
        for a, b in row.get("implications", []):
            neg = b.startswith("not:")
            b = b[4:] if neg else b
            if a in by_fact and b in by_fact:
                ta = torch.stack(by_fact[a]).mean()
                tb = torch.stack(by_fact[b]).mean()
                terms.append(F.relu(ta - ((1 - tb) if neg else tb)))
    return torch.stack(terms).mean() if terms else p.sum() * 0


@torch.inference_mode()
def evaluate(model, tok, rows, max_length, bs=32):
    model.eval()
    pairs = [(r["text"], it["statement"], it["label"]) for r in rows for it in r["items"]]
    ps = []
    for i in range(0, len(pairs), bs):
        t, s, _ = zip(*pairs[i:i + bs])
        ps.extend(torch.sigmoid(yes_logit(model, tok, list(t), list(s), max_length)).tolist())
    p = np.array(ps)
    y = np.array([l for _, _, l in pairs], bool)
    # negation agreement on val
    off, dis, n = 0, 0, 0
    for r in rows:
        by = {}
        for j, it in enumerate(r["items"]):
            by.setdefault(it["fact"], []).append((p[off + j] >= 0.5) == (it["pol"] == 1))
        for v in by.values():
            if len(v) > 1:
                n += 1
                dis += len(set(v)) > 1
        off += len(r["items"])
    model.train()
    return {"accuracy": float(((p >= 0.5) == y).mean()), "ece": ece(p, y),
            "negation_disagreement": dis / max(n, 1), "n": len(y)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--out", default=str(ROOT / "checkpoints" / "maya"))
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--texts-per-step", type=int, default=6)
    ap.add_argument("--mnli-per-step", type=int, default=4)
    ap.add_argument("--consistency", type=float, default=1.0)
    ap.add_argument("--max-length", type=int, default=160)
    ap.add_argument("--eval-every", type=int, default=230)
    ap.add_argument("--val-texts", type=int, default=160, help="validation texts used for checkpoint selection")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    train = load_rows(ROOT / "data" / "train" / "train.jsonl")
    val = load_rows(ROOT / "data" / "train" / "val.jsonl")
    val = random.Random(1).sample(val, min(args.val_texts, len(val)))
    steps = math.ceil(len(train) * args.epochs / args.texts_per_step)
    mnli = load_mnli(steps * args.mnli_per_step, args.seed) if args.mnli_per_step else []

    tok = AutoTokenizer.from_pretrained(args.base)
    model = AutoModelForSequenceClassification.from_pretrained(args.base, dtype=torch.float32)  # bf16 default is very slow on CPU
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    warm = max(1, int(0.06 * steps))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * max(0.0, (steps - s) / max(1, steps - warm)))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log = {"args": vars(args), "steps": steps, "evals": []}
    v0 = evaluate(model, tok, val, args.max_length)
    log["evals"].append({"step": 0, **v0})
    print(f"step 0 val {v0}", flush=True)
    best = None

    order = []
    while len(order) < steps * args.texts_per_step:
        idx = list(range(len(train)))
        random.shuffle(idx)
        order += idx
    t0 = time.time()
    for step in range(steps):
        batch = [train[i] for i in order[step * args.texts_per_step:(step + 1) * args.texts_per_step]]
        texts, stmts, labels, offs = [], [], [], []
        for r in batch:
            offs.append((r, len(texts)))
            for it in r["items"]:
                texts.append(r["text"])
                stmts.append(it["statement"])
                labels.append(float(it["label"]))
        n_task = len(texts)
        for prem, hyp, lab in mnli[step * args.mnli_per_step:(step + 1) * args.mnli_per_step]:
            texts.append(prem)
            stmts.append(hyp)
            labels.append(float(lab))
        z = yes_logit(model, tok, texts, stmts, args.max_length)
        y = torch.tensor(labels)
        bce = F.binary_cross_entropy_with_logits(z[:n_task], y[:n_task])
        loss = bce
        if len(texts) > n_task:
            loss = loss + 0.5 * F.binary_cross_entropy_with_logits(z[n_task:], y[n_task:])
        cons = consistency_loss(offs, torch.sigmoid(z[:n_task])) if args.consistency else torch.tensor(0.0)
        loss = loss + args.consistency * cons
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        opt.zero_grad()
        if step % 20 == 0:
            el = time.time() - t0
            print(f"step {step}/{steps} bce {bce.item():.3f} cons {cons.item():.4f} "
                  f"{el / (step + 1):.1f}s/step eta {el / (step + 1) * (steps - step - 1) / 60:.0f}m", flush=True)
        if (step + 1) % args.eval_every == 0 or step + 1 == steps:
            v = evaluate(model, tok, val, args.max_length)
            log["evals"].append({"step": step + 1, **v})
            print(f"step {step + 1} val {v}", flush=True)
            if best is None or v["accuracy"] >= best:
                best = v["accuracy"]
                model.config.id2label = {0: "yes", 1: "no"}
                model.config.label2id = {"yes": 0, "no": 1}
                model.save_pretrained(out)
                tok.save_pretrained(out)
                log["best_step"] = step + 1
            (out / "train_log.json").write_text(json.dumps(log, indent=1), encoding="utf-8")
    print("best val accuracy", best, "at step", log.get("best_step"))


if __name__ == "__main__":
    main()
