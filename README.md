# Maya

**A dedicated yes/no gate.** Maya reads a text and a statement (or a yes/no question) and returns P(yes). It is meant to sit in front of heavier models as a guardrail or early-exit filter: "can this change be undone?", "is the customer upset?", "is this message a scam?".

Maya v0.2 is a 435M-parameter cross-encoder fine-tuned from [DeBERTa-v3-large-zeroshot-v2.0](https://huggingface.co/MoritzLaurer/deberta-v3-large-zeroshot-v2.0).

- **Live demo (runs in your browser): https://vishalmysore.github.io/maya/**
- One-minute overview video: [docs/video/maya-overview.mp4](docs/video/maya-overview.mp4) (built by `scripts/record_demo.py` and `scripts/make_video.py`)
- Article: [docs/article.md](docs/article.md): what Jev, Laya, NLI encoders and LLM guards are, how Maya differs, the v0.1 failure, the v0.2 fix, results and remaining failures, with screenshots.
- Weights: [huggingface.co/VishalMysore/maya](https://huggingface.co/VishalMysore/maya) (PyTorch) and [huggingface.co/VishalMysore/mayaWasm](https://huggingface.co/VishalMysore/mayaWasm) (int8 ONNX for the browser, 600 MB). Maya v0.1 (ModernBERT-base, 150M) is kept under the `v0.1` tag of both repos.

**It only ever answers yes or no.** Maya is a classifier with one output, P(yes); it cannot produce free text. An optional third answer, "not sure", appears only if you set abstention thresholds. It also cannot refuse: a question that is not yes/no ("what colour is it?") still gets a probability, so only ask yes/no questions.

```python
from maya.gate import Maya

maya = Maya.load("VishalMysore/maya")
maya.ask("Agent plan: delete the `sessions` table on the production database. A verified backup was "
         "taken ten minutes ago and the on-call engineer has reviewed the plan.",
         ["The action is destructive and cannot be undone", "This action can be undone if needed"])
# [{'statement': 'The action is destructive and cannot be undone', 'p_yes': 0.148, 'answer': 'no'},
#  {'statement': 'This action can be undone if needed', 'p_yes': 0.9, 'answer': 'yes'}]
```

It still makes mistakes (about one in eight answers on judgment questions). Do not use it as the only safety check; see "Where it is still wrong".

## Results

Three hand-labeled test sets, never used for training or model selection, plus a small dev set used only to choose the checkpoint and fit the temperature:

- **v3** (`data/eval_v3`): 192 answers on *judgment* questions in 8 domains. Two are familiar kinds of text with new wording (agent actions, customer tone); six never appear in training (access requests, refunds, account security, contractor invoices, leave requests, landlord notices). Includes 10 word-overlap **trap** cases ("I'm not angry, just curious"). It was written and committed (`05dc39e`) before the v0.2 training data existed.
- **v2** (`data/eval_v2`): 256 answers in 8 domains never used in training (rental listings, travel notices, school messages, contract clauses, scam messages, smart-home commands, recipes, job postings).
- **v1** (`data/eval`, from layaForWeb): 120 answers in 8 domains, mostly familiar kinds of text.
- **dev** (`data/dev`): 64 answers, same domains as v3 but different cases.

v2 and v3 cases come with a hand-written negation of the key statement, an implication pair, and a minimal-pair partner (the same text with one detail changed, which flips the key answer).

![Accuracy of Laya, zero-shot DeBERTa-v3-large and Maya on the three test sets](docs/images/chart-accuracy.png)

### Accuracy

| Model | Params | v3: judgment (192) | v2: unseen domains (256) | v1: familiar (120) |
|---|---|---|---|---|
| always "no" | | 43.2% | 50.8% | 62.5% |
| laya-typed-decisions | 421M | 62.5% | 80.1% | 68.3% |
| layaMOE | 421M + heads | 63.5% | 78.5% | 78.3% |
| ModernBERT-base-zeroshot-v2.0 | 150M | 66.7% | 78.5% | 76.7% |
| DeBERTa-v3-base-zeroshot-v2.0 | 184M | 66.7% | 81.2% | 70.8% |
| DeBERTa-v3-large-zeroshot-v2.0 (Maya v0.2's starting point) | 435M | 72.9% | 89.8% | 75.8% |
| Maya v0.1 | 150M | 71.9% | 81.2% | 78.3% |
| **Maya v0.2** | 435M | **87.5%** | **95.3%** | **87.5%** |

### Maya v0.2 in detail

| | v3 | v2 | v1 |
|---|---|---|---|
| Accuracy | 87.5% | 95.3% | 87.5% |
| AUROC | 0.933 | 0.988 | 0.971 |
| Calibration error (ECE), with the shipped temperature 1.65 | 0.064 | 0.025 | 0.110 |
| Answerable at <=10% error (upper bound) | 95% | 100% | 88% |
| Statement and its negation get the same answer | 12% | 8% | 33% (4 of 12) |
| Implication violated | 6% | 2% | |
| Minimal pairs: both texts right | 71% | 91% | |
| Answers on the word-overlap trap cases | 90% | | |

For comparison, the zero-shot starting point contradicts itself on 73% of v3 negation pairs, 30% on v2 and 100% on v1, and gets both texts of a minimal pair right 33% (v3) and 78% (v2) of the time.

![Share of negation pairs answered inconsistently](docs/images/chart-contradictions.png)

Notes:

- *Answerable at <=10% error* is the largest share of answers, most confident first, whose error stays at or below 10%. It is fitted on the same answers it is measured on, so it only shows how well confidence separates right from wrong.
- All numbers are PyTorch fp32 at a threshold of 0.5. Laya and layaMOE answers were recorded with layaMOE's own code (`results/laya_*_reference.json`).
- The test sets are small: on v3 one answer is 0.5 points, and differences of under about 5 points between models are not reliable.

### How the checkpoint was chosen

Three v0.2 candidates were trained or assembled; the choice was made on the dev set, not on the test sets:

| Candidate | Dev (64) | v3 | v2 | v1 |
|---|---|---|---|---|
| DeBERTa-v3-base fine-tune (184M) | 78.1% | 83.3% | 87.1% | 91.7% |
| **DeBERTa-v3-large fine-tune (435M), shipped** | **87.5%** | 87.5% | 95.3% | 87.5% |
| Ensemble of the two (average of logits) | 85.9% | 88.5% | 93.0% | 93.3% |

The large model won on dev by one answer over the ensemble. On the three test sets together the ensemble is 3 answers better out of 568 (91.5% vs 91.0%): a tie. The single model was shipped; the ensemble is steadier on familiar domains (see below), at the cost of running two models.

Asking every question both ways (p = (p(X) + 1 - p(not X)) / 2) helped the zero-shot models but made v0.2 worse on dev (88% -> 75% on key statements), so it is not used.

### Where it is still wrong

- **Code changes.** For "Pull request: migration that drops the `legacy_status` column from `orders`. CI is green. Nobody has reviewed it yet." v0.2 says it can be merged (0.86) and is not a breaking change (0.10). Both are wrong; v0.1 and the base candidate get them right. v0.2's training mix has only 90 code-change texts (v0.1 had 300), and the chosen checkpoint is from step 234.
- **IT incidents and patient messages** on v1 (67% and 75%), smart-home commands on v2 (81%), access requests, account security and leave requests on v3 (75-79%).
- **One of the v0.1 failures is only half fixed.** For a production `DELETE` that "no human has reviewed", v0.2 now correctly says it is not safe to run without a human (0.10), but still answers "no" (0.32) to "A human should approve this action before it runs".
- **About one in eight judgment answers is wrong** (24 of 192 on v3), and 29% of v3 minimal pairs have at least one text wrong.

### What changed from v0.1

v0.1 (ModernBERT-base, 150M) scored 78.3% / 81.2% on v1 / v2 but failed in the demo: an obviously angry ticket ("STILL BROKEN ... fix it or I cancel") read as calm, and a delete with a verified backup read as irreversible. It had learned its training generator's phrases. v0.2 changes two things:

1. **Training data** (`scripts/gen_train_data_v2.py`, 5,687 texts, 25,717 statements). Agent actions and customer messages are generated with many phrasings per situation (`gen/guardrails_v2.py`, `gen/tone_v2.py`): safety nets described a dozen ways and present in about two thirds of the destructive plans (v0.1: 22 statements out of 20,429), anger through capitals, sarcasm, polite-but-firm and blunt styles, calm messages that still report problems, and word-overlap traps ("No human has checked this step" vs "No human approval is needed"). About 40% of those texts have a minimal-pair partner that differs only in the deciding detail. A smaller share of the v0.1 domains is kept, and real yes/no questions (BoolQ) and MNLI pairs are mixed in so the model keeps its general skill.
2. **Base model.** DeBERTa-v3-large-zeroshot-v2.0 was the strongest off-the-shelf model on unseen domains. Zero-shot NLI models read literally (they say "no" to "the customer is angry" unless the text says so), which is why they contradict themselves; fine-tuning teaches the judgment reading.

No training text shares a word 5-gram with any test or dev text, and no training statement equals a v2, v3 or dev statement (checked by unit tests).

### Abstention ("not sure")

`maya/conformal.py` fits two thresholds on labeled calibration data (yes if p >= t_yes, no if p <= t_no, otherwise "not sure") so that the error among answered questions is at most alpha with probability 1 - delta (Learn-then-Test over a fixed threshold grid, Clopper-Pearson bounds, Bonferroni). The 64 dev answers are too few to certify even a 20% bound for any model, and calibrating on synthetic data does not transfer (v0.1: thresholds collapse to 0.5 while real error is about 20%). So the shipped default is strict yes/no. To get a bounded-error "not sure", label a few hundred of your own inputs and fit `Abstainer` on them; the demo's sliders set a band by hand, without a guarantee.

### Latency

CPU, PyTorch fp32, 4 threads, median of 15 requests (`scripts/bench_latency.py`):

| Model | Params | 1 question | 5 questions | 20 questions |
|---|---|---|---|---|
| Maya v0.2 | 435M | 0.55 s | 1.3 s | 4.4 s |
| v0.2 base candidate | 184M | 0.17 s | 0.42 s | 1.4 s |
| Maya v0.1 | 150M | 0.10 s | 0.43 s | 1.4 s |

Maya is a cross-encoder: every (text, question) pair is one more sequence through the model. v0.2 trades speed and size for accuracy.

## Browser demo

Live at https://vishalmysore.github.io/maya/ (deployed by `.github/workflows/pages.yml`). `web/` is a single page that runs the int8 build with ONNX Runtime Web (WASM, multi-threaded when cross-origin isolated), loading the model from Hugging Face (VishalMysore/mayaWasm, 600 MB on the first visit, then cached in the browser).

```
npm install
node scripts/prepare_site.mjs                          # dist/, model from Hugging Face
node scripts/prepare_site.mjs --local-model build/web  # or bundle a local build
python serve.py 8791                                   # http://localhost:8791 with COOP/COEP headers
```

`scripts/export_onnx.py` exports one ONNX graph with int8 weight-only quantization (MatMulNBits block 128, int8 embeddings), split into 24 MiB parts with SHA-256 hashes. On all 256 v2 answers the int8 graph scores 94.9% (PyTorch 95.3%) with one answer flipping. In the browser, the JavaScript tokenizer gives exactly the Python token ids on 33 test items and probabilities differ from PyTorch by at most 0.03 (`results/browser_parity.json`).

![Maya demo answering questions about a support ticket](docs/images/demo-full-page.png)

## Tests

```
pytest            # 35 tests, about 90 s with the model on disk
```

Test-set integrity (negation, implication and minimal-pair labels on v2, v3 and dev), metrics, the abstention guarantee on simulated data, generator rules, no leakage from any test or dev set into the training data, and the model itself: answers are only yes / no / not sure, batched equals single, the model reproduces the recorded test probabilities, two v0.1 failures stay fixed, int8 ONNX matches PyTorch.

## Reproduce

Python 3.12, CPU is enough (v0.2 trained in about 2.5 hours on a laptop).

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

python scripts/eval_baselines.py                          # zero-shot NLI, Laya references -> results/baselines.json
python scripts/gen_train_data_v2.py                       # data/train_v2/{train,val}.jsonl
python scripts/train.py --base MoritzLaurer/deberta-v3-large-zeroshot-v2.0 --data data/train_v2 ^
    --epochs 0.35 --lr 1e-5 --texts-per-step 4 --boolq-per-step 1 --mnli-per-step 1 --max-length 128 ^
    --freeze-embeddings --select dev --eval-every 117 --val-texts 60 --out checkpoints/maya-v2-large
python scripts/eval_maya.py checkpoints/maya-v2-large --name maya-v2-large
python scripts/export_onnx.py checkpoints/maya-v2-large   # int8 browser build -> build/web
```

Load models in float32 on CPU: transformers 5 keeps these checkpoints in bfloat16 by default, which made training on a laptop CPU about 100x slower. Long runs are best started detached (`scripts/train_v2_large.cmd`).

## Layout

| Path | What it is |
|---|---|
| `maya/gate.py` | `Maya.load(...).ask(text, statements)`: yes / no / not sure |
| `maya/conformal.py` | abstention thresholds with a bounded error rate, temperature fitting |
| `maya/metrics.py` | accuracy, AUROC, ECE, coverage at a fixed error, negation / implication / minimal-pair / trap metrics |
| `maya/data.py`, `maya/nli.py` | test-set loading, scoring any NLI-style model |
| `gen/` | training data generators (v0.2: `guardrails_v2.py`, `tone_v2.py`; v0.1: `facts.py`, `new_domains.py`, `laya_domains.py`) |
| `scripts/` | data generation, training, evaluation, ensembles, latency, ONNX export, Hugging Face upload, screenshots, charts |
| `data/eval/`, `data/eval_v2/`, `data/eval_v3/`, `data/dev/` | hand-labeled test sets and the dev set |
| `results/` | every probability and summary behind the tables above |
| `web/`, `scripts/prepare_site.mjs`, `serve.py` | the browser demo |
| `tests/` | pytest suite |
| `docs/` | articles and screenshots |
| `hf/` | model cards for the two Hugging Face repos |

Not in git: `checkpoints/`, `build/` (on Hugging Face), `data/train*/` (regenerate with the scripts).

**License note:** the base model's card says its versions without "-c" in the name were trained on data that includes non-commercially licensed datasets. Maya v0.2 inherits that; check the base model's card before commercial use (a commercially friendly alternative would be to repeat the recipe on `deberta-v3-large-zeroshot-v2.0-c`).

See `NOTICE.md` for third-party models and data.
