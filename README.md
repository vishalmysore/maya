# Maya

**A dedicated yes/no gate.** Maya reads a text and a statement (or a yes/no question) and returns P(yes). It is a 150M-parameter cross-encoder fine-tuned from [ModernBERT-base-zeroshot-v2.0](https://huggingface.co/MoritzLaurer/ModernBERT-base-zeroshot-v2.0) on rule-labeled data where every fact appears in plain and negated wordings, meant to sit in front of heavier models as a guardrail or early-exit filter: "does this action need a human?", "is the customer upset?", "is this message a scam?".

- Weights: [huggingface.co/VishalMysore/maya](https://huggingface.co/VishalMysore/maya) (PyTorch) and [huggingface.co/VishalMysore/mayaWasm](https://huggingface.co/VishalMysore/mayaWasm) (int8 ONNX for the browser, 161 MB)
- **Live demo (runs in your browser): https://vishalmysore.github.io/maya/**
- Article: [docs/article.md](docs/article.md) (how it was built, results, failures, with screenshots)
- Compared against [laya-typed-decisions](https://huggingface.co/convaiinnovations/laya-typed-decisions), [layaMOE](https://github.com/vishalmysore/layaMOE) and off-the-shelf zero-shot NLI models

**It only ever answers yes or no.** Maya is a classifier with one output, P(yes); it cannot produce free text. An optional third answer, "not sure", appears only if you set abstention thresholds (see below). It also cannot refuse: a question that is not yes/no ("what colour is it?") still gets a probability, so only ask yes/no questions.

```python
from maya.gate import Maya

maya = Maya.load("VishalMysore/maya")
maya.ask("Hi team, how do I export the dashboard to Excel? No rush.",
         ["Is the customer angry?", "The customer sounds calm"])
# [{'statement': 'Is the customer angry?', 'p_yes': 0.0246, 'answer': 'no'},
#  {'statement': 'The customer sounds calm', 'p_yes': 0.9625, 'answer': 'yes'}]
```

Read the results below before using it as a safety gate: it is the strongest model tested on its own domains, but on unseen domains a 3x larger off-the-shelf NLI model is clearly better, and on agent-action guardrails Maya is right only 61% of the time (plain Laya 50%, layaMOE 69%).

## Results

Two hand-labeled test sets, never used for training or model selection:

- **v1** (`data/eval`, from layaForWeb): 120 yes/no answers in 8 domains (45 yes). Maya's training data covers 5 of these domains, so v1 is mostly in-domain.
- **v2** (`data/eval_v2`, written for Maya): 256 answers in 8 domains Maya never saw in training (rental listings, travel notices, school messages, contract clauses, scam messages, smart-home commands, recipes, job postings), 32 statements that never appear in training. Every case has a hand-written negation of its key statement, an implication pair (e.g. "dogs are allowed" => "pets are allowed") and a minimal-pair partner: the same text with one detail changed, which flips the key answer.

### v1: mostly in-domain (120 answers)

| Model | Params | Accuracy | AUROC | ECE | Answerable at <=10% error | Negation contradictions (12 pairs) |
|---|---|---|---|---|---|---|
| always "no" | | 62.5% | 0.500 | 0.375 | 3% | 100% |
| laya-typed-decisions | 421M | 68.3% | 0.792 | 0.107 | 48% | 67% |
| layaMOE (prompted router) | 421M + heads | 78.3% | 0.869 | **0.082** | 54% | 67% |
| cross-encoder/nli-deberta-v3-xsmall | 71M | 66.7% | 0.651 | 0.323 | 0% | 100% |
| MoritzLaurer/deberta-v3-xsmall-zeroshot-v1.1-all-33 | 71M | 74.2% | 0.750 | 0.252 | 9% | 100% |
| MoritzLaurer/deberta-v3-base-zeroshot-v2.0 | 184M | 70.8% | 0.821 | 0.270 | 33% | 92% |
| MoritzLaurer/ModernBERT-base-zeroshot-v2.0 (Maya's starting point) | 150M | 76.7% | 0.876 | 0.223 | 53% | 92% |
| MoritzLaurer/deberta-v3-large-zeroshot-v2.0 | 435M | 75.8% | 0.890 | 0.227 | 59% | 100% |
| **Maya** | 150M | **78.3%** | **0.905** | 0.149 | **75%** | **25%** |

### v2: unseen domains (256 answers)

| Model | Params | Accuracy | AUROC | ECE | Answerable at <=10% error | Negation contradictions | Implication violations | Minimal pairs both right |
|---|---|---|---|---|---|---|---|---|
| always "no" | | 50.8% | 0.500 | 0.492 | 0% | 100% | 0% | 0% |
| laya-typed-decisions | 421M | 80.1% | 0.889 | 0.166 | 39% | 44% | 5% | 69% |
| layaMOE (trained router) | 421M + heads | 78.5% | 0.840 | 0.160 | 9% | 44% | 8% | 66% |
| cross-encoder/nli-deberta-v3-xsmall | 71M | 70.3% | 0.830 | 0.267 | 4% | 70% | 5% | 22% |
| MoritzLaurer/deberta-v3-xsmall-zeroshot-v1.1-all-33 | 71M | 82.4% | 0.855 | 0.139 | 12% | 48% | **0%** | 59% |
| MoritzLaurer/deberta-v3-base-zeroshot-v2.0 | 184M | 81.2% | 0.911 | 0.172 | 59% | 45% | 3% | 56% |
| MoritzLaurer/ModernBERT-base-zeroshot-v2.0 (Maya's starting point) | 150M | 78.5% | 0.868 | 0.180 | 25% | 45% | 9% | 59% |
| MoritzLaurer/deberta-v3-large-zeroshot-v2.0 | 435M | **89.8%** | **0.957** | 0.100 | **100%** | 30% | **0%** | **78%** |
| **Maya** | 150M | 81.2% | 0.893 | 0.152 (**0.061** with temperature) | 71% | **27%** | 9% | 53% |

All rows are PyTorch fp32. Laya and layaMOE answers were recorded with layaMOE's own code (`results/laya_moe_eval_reference.json`, `results/laya_v2_reference.json`).

Column notes:

- *Answerable at <=10% error*: the largest share of answers, most confident first, whose error stays at or below 10%. It is fitted on the same answers it is measured on, so it is an upper bound that compares how well each model's confidence separates right from wrong; it is not a deployable guarantee (see "Abstention" below).
- *Negation contradictions*: the text gets the same answer for a statement and its negation ("The traveler needs to take action" / "The traveler does not need to do anything"). On v1 the one negation pair is "a human should approve this" / "it is safe to run without a human".
- *Minimal pairs both right*: both texts of a pair answered correctly on the key statement.
- ECE "with temperature": temperature 2.8 fitted on v1 (`maya_config.json` ships it). It does not change accuracy or AUROC.

### What this shows

- **In-domain (v1), Maya is the best model tested**: best accuracy (tied with layaMOE), best AUROC (0.905), and the most answers it can give before its error passes 10% (75%), at about a third of Laya's size. Plain Laya is 10 points lower.
- **On unseen domains (v2), Maya is not the best.** An off-the-shelf model three times its size, deberta-v3-large-zeroshot-v2.0, is far ahead (89.8% accuracy, AUROC 0.957, 78% of minimal pairs). Maya is level with Laya, deberta-v3-base and the 71M deberta-v3-xsmall (81-82%). Fine-tuning ModernBERT on Maya's domains moved it from 78.5% to 81.2% on v2 (7 answers, about the noise level), so the gains are mostly in-domain. The same deberta-large model is only 75.8% on v1, so no model wins both sets.
- **Consistency is Maya's clearest win.** Contradictions between a statement and its negation are 25% on v1 (every NLI model: 92-100%, Laya 67%) and 27% on v2 (others 30-70%). It carries over to hand-written negations in domains Maya has never seen. The ablation below shows this comes from the training data (each fact in plain and negated wordings with opposite labels), not from the extra consistency loss.
- **Calibration needs a temperature.** Maya is over-confident out of domain (ECE 0.15). A single temperature fitted on the 120 v1 answers brings v2 ECE to 0.061, the lowest of all models. Fitted on the synthetic validation split it stays at 1.0 (Maya is right 97% of the time there), so the calibration data has to look like real inputs.
- **Weak spot: agent guardrails (61% on v1).** Example: "run `DELETE FROM customers ...` on the production database. No backup has been taken and no human has reviewed this command" gets 0.92 for "It is safe to run this action without a human approving it first". The words "no human" in the text seem to match "without a human" in the statement (the classic lexical-overlap shortcut of NLI models), and the training texts phrase the missing review differently ("nobody has reviewed the command"). Do not use Maya alone as an agent guardrail.
- **Weak spot: minimal pairs.** Maya gets both texts of a pair right 53% of the time (its starting point 59%, Laya 69%, deberta-v3-large 78%). Fine-tuning made Maya more consistent but no better at noticing the single detail that flips a judgment in a new domain. Typical misses: it calls a bank's "never share this code" message a scam and the real phishing texts legitimate; any mention of smoke detectors or a pool gate reads as a safety risk; "made in a nut-free facility" does not register. 32 pairs is a small sample, but the direction is clear.
- **Next:** the same recipe on a stronger base (deberta-v3-large-zeroshot-v2.0, or deberta-v3-base for speed) and HANS-style training texts that share words with a statement but mean the opposite. Both should be measured on a fresh test set, since v1 and v2 have now been studied closely.

### Abstention ("not sure")

`maya/conformal.py` fits two thresholds on labeled calibration data (yes if p >= t_yes, no if p <= t_no, otherwise "not sure") so that the error among answered questions is at most alpha with probability 1 - delta (Learn-then-Test over a fixed threshold grid, Clopper-Pearson bounds, Bonferroni). Findings:

- **Calibrating on synthetic data does not transfer.** Maya is 97% right on the synthetic validation split, so the fitted thresholds collapse to 0.5 and the real error on v1/v2 is 19-22%. The guarantee only holds for inputs like the calibration data.
- **120-128 hand-labeled answers are too few to certify 10% error** (alpha = 0.10, delta = 0.10) for any model tested. At alpha = 0.20, thresholds fitted on v1 let Maya answer 35% of v2 with 7.9% actual error (the starting NLI model: 14% at 5.7%).
- So the shipped default is strict yes/no (both thresholds 0.5). To get a bounded-error "not sure", label a few hundred of your own inputs and fit `Abstainer` on them.

### Ablation: is the consistency loss needed?

Same data, same settings, consistency weight 0 (`checkpoints/maya-noconsist`, `results/eval_maya-noconsist.json`):

| | v1 acc | v1 AUROC | v2 acc | v2 AUROC | Negation contradictions v1 / v2 | Implication violations v2 | Minimal pairs v2 |
|---|---|---|---|---|---|---|---|
| Maya (consistency loss 1.0) | 78.3% | 0.905 | 81.2% | 0.893 | 25% / 27% | 9% | 53% |
| without consistency loss | 80.8% | 0.921 | 79.7% | 0.891 | 25% / 25% | 8% | 50% |

No difference beyond noise, and the small differences point opposite ways on v1 and v2. Training on both polarities of every fact is what removes the contradictions; the explicit loss term (which also enforces implications) adds nothing measurable here. The released model is the one trained with the loss, since it was planned as the main run before the ablation was seen.

### Latency

CPU, PyTorch fp32, 4 threads, median of 15 requests (`scripts/bench_latency.py`, `results/latency.json`):

| Model | Params | 1 question | 5 questions | 20 questions |
|---|---|---|---|---|
| Maya | 150M | 169 ms | 666 ms | 2.6 s |
| deberta-v3-large-zeroshot-v2.0 | 435M | 1,088 ms | 2.6 s | 8.2 s |
| deberta-v3-xsmall-zeroshot-v1.1 | 71M | 120 ms | 247 ms | 787 ms |

Maya is a cross-encoder: every (text, question) pair is one more sequence through the model, so cost grows with the number of questions. For reference, layaMOE measured about 0.9 s per case (all questions of a case in one encoder pass, 2 threads, a different setup), so Maya is faster for a few questions and slower for many. Encoding the text once and answering many questions with a small head on the cached states is the planned next step.

### Browser build

`scripts/export_onnx.py` exports the model as one ONNX graph with int8 weight-only quantization (MatMulNBits block 128, int8 embeddings): 161 MB, weights split into 24 MiB parts with SHA-256 hashes. On all 256 v2 answers the int8 graph scores 81.6% (PyTorch 81.25%), with one answer flipping and a largest probability change of 0.12.

## Browser demo

Live at https://vishalmysore.github.io/maya/ (deployed by `.github/workflows/pages.yml` on every push to `web/`). `web/` is a single page that runs the int8 build with ONNX Runtime Web (WASM, multi-threaded when cross-origin isolated), loading the model from Hugging Face (VishalMysore/mayaWasm) and caching the weight parts in the browser:

```
npm install
node scripts/prepare_site.mjs                          # dist/, model from Hugging Face
node scripts/prepare_site.mjs --local-model build/web  # or bundle a local build
python serve.py 8791                                   # http://localhost:8791 with COOP/COEP headers
```

On 28 eval items the browser's JavaScript tokenizer gives exactly the Python token ids and the in-browser probabilities differ from PyTorch by at most 0.027 (`results/browser_parity.json`). `scripts/screenshots.py` (Playwright, headless Edge) re-runs that check and captures the screenshots in `docs/images/`.

![Maya demo answering questions about a support ticket](docs/images/demo-full-page.png)

## Tests

```
pytest            # 25 tests, about 15 s with the model on disk
```

Eval-set integrity (negation, implication and minimal-pair labels), metrics, the abstention guarantee on simulated data, generator rules and leakage, and the model itself: answers are only yes / no / not sure, batched equals single, the model reproduces the recorded eval probabilities, int8 ONNX matches PyTorch.

## How it was trained

`scripts/gen_train_data.py` builds 4,463 rule-labeled texts with 20,429 yes/no statements (50% yes, about a third phrased as questions):

- the five layaMOE domains (agent guardrails, content moderation, support tickets, delivery exceptions, email triage), with their labels turned into several yes/no facts each (`gen/facts.py`);
- seven slot-based domains (IT incidents, product reviews, expense claims, meeting requests, code changes, insurance claims, sales inquiries; `gen/new_domains.py`). About half of their texts get a minimal-pair partner that differs in one slot.

Every fact has plain wordings and negated wordings, and each domain lists implications between facts (e.g. "the claim can be reimbursed" => "a receipt was provided"). Texts sharing a word 5-gram with any eval text are dropped, and no training wording may equal a v2 statement.

`scripts/train.py` fine-tunes the whole model for one epoch (687 steps, lr 2e-5, CPU, about 80 minutes) with:

- binary cross-entropy on each statement (P(yes) = sigmoid(logit_entailment - logit_not_entailment));
- a consistency loss: statements about the same fact on the same text must agree after polarity ((t_i - t_j)^2), and implications must hold (relu(t_A - t_B)) (the ablation shows it adds little on top of the paired data);
- 4 MNLI pairs per step so the model keeps its general NLI ability.

The checkpoint is chosen on the synthetic validation split only.

## Run it

Python 3.12, CPU is enough.

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

python scripts/eval_baselines.py                          # baselines on v1 and v2 -> results/baselines.json
python scripts/gen_train_data.py                          # data/train/{train,val}.jsonl
python scripts/train.py --out checkpoints/maya            # ~80 min on a laptop CPU
python scripts/eval_maya.py checkpoints/maya --name maya  # v1/v2, calibration, abstention -> results/eval_maya.json
python scripts/export_onnx.py checkpoints/maya            # int8 browser build -> build/web
```

Load models in float32 on CPU: transformers 5 keeps these checkpoints in bfloat16 by default, which made training on a laptop CPU about 100x slower.

## Layout

| Path | What it is |
|---|---|
| `maya/gate.py` | `Maya.load(...).ask(text, statements)`: yes / no / not sure |
| `maya/conformal.py` | abstention thresholds with a bounded error rate, temperature fitting |
| `maya/metrics.py` | accuracy, AUROC, ECE, coverage at a fixed error, negation / implication / minimal-pair consistency |
| `maya/data.py`, `maya/nli.py` | eval loading, scoring any NLI-style model |
| `gen/` | training data generators (layaMOE domains + seven slot-based domains) |
| `scripts/` | data generation, training, evaluation, latency, ONNX export, Hugging Face upload |
| `data/eval/`, `data/eval_v2/` | hand-labeled test sets (v1 from layaForWeb, v2 written for Maya) |
| `results/` | every probability and summary behind the tables above |
| `web/`, `scripts/prepare_site.mjs`, `serve.py` | the browser demo |
| `tests/` | pytest suite |
| `docs/` | article and screenshots (`scripts/screenshots.py`, `scripts/make_charts.py`) |
| `hf/` | model cards for the two Hugging Face repos |

Not in git: `checkpoints/`, `build/` (on Hugging Face), `data/train/` (regenerate with `scripts/gen_train_data.py`).

See `NOTICE.md` for third-party models and data.
