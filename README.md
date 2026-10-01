# Maya

**A small, dedicated yes/no gate.** Maya answers one kind of question: does this statement hold for this text? It returns P(yes), and it is meant to run fast enough to sit in front of every request (in the browser or on the edge) as a guardrail or an early-exit filter.

Model weights (once trained): [huggingface.co/VishalMysore/maya](https://huggingface.co/VishalMysore/maya).

Work in progress. Planned features, each measured against [laya-typed-decisions](https://huggingface.co/convaiinnovations/laya-typed-decisions), [layaMOE](https://github.com/vishalmysore/layaMOE) and off-the-shelf zero-shot NLI models:

1. **Abstention with a bounded error rate.** Yes / no / not sure, with a conformal guarantee such as "when Maya says yes, it is wrong at most 5% of the time".
2. **Many questions, one encoder pass.** Encode the text once, then answer each question with a small head over the cached states.
3. **Consistent answers.** Negations, paraphrases and implications must agree (P(Q) + P(not Q) = 1).
4. **Trained on minimal pairs.** Cases that differ in one detail and flip the answer, so the model learns causes rather than templates.

## Step 0: baselines

`scripts/eval_baselines.py` scores existing models on the 120 yes/no answers in `data/eval` (10 questions x 12 cases in 8 domains, 45 yes / 75 no). NLI models get the text as premise and the yes/no statement as hypothesis; P(yes) = P(entailment).

| Model | Params | Accuracy | AUROC | ECE | Answerable at <=10% error | Contradictions |
|---|---|---|---|---|---|---|
| always "no" | | 62.5% | 0.500 | 0.375 | 3% | 12/12 |
| laya-typed-decisions | 421M | 68.3% | 0.792 | 0.107 | 48% | 8/12 |
| layaMOE (prompted router) | 421M + heads | 78.3% | 0.869 | 0.082 | 54% | 8/12 |
| cross-encoder/nli-deberta-v3-xsmall | 71M | 66.7% | 0.651 | 0.323 | 0% | 12/12 |
| MoritzLaurer/deberta-v3-xsmall-zeroshot-v1.1-all-33 | 71M | 73.3% | 0.750 | 0.252 | 9% | 12/12 |
| MoritzLaurer/deberta-v3-base-zeroshot-v2.0 | 184M | 70.8% | 0.821 | 0.270 | 33% | 11/12 |
| MoritzLaurer/ModernBERT-base-zeroshot-v2.0 | 150M | 76.7% | 0.876 | 0.224 | 53% | 11/12 |
| MoritzLaurer/deberta-v3-large-zeroshot-v2.0 | 435M | 75.8% | 0.890 | 0.227 | 59% | 12/12 |

- *Answerable at <=10% error*: the largest share of answers, most confident first, that keeps the error at or below 10%. Fitted on the same items, so it is an optimistic upper bound.
- *Contradictions*: of the 12 agent-guardrails cases, how often "a human should approve this" and "it is safe to run without a human" got the same answer. Every model mostly says "no" to both.

What this says:

- An off-the-shelf NLI model (ModernBERT-base, 150M) already ranks as well as layaMOE with no training, so Maya starts from it rather than from a raw encoder.
- The gaps are calibration (ECE 0.22 vs Laya's 0.08-0.11) and consistency. Those are what Maya is for.
- 120 answers means differences of under ~8 points are noise, and there are only 12 consistency pairs. A larger eval set (unseen questions, negation and implication pairs, minimal pairs) comes before any training.
- Latency is not reported yet: this run was on a busy laptop (two models with the same architecture measured 34 ms and 500 ms). It will be re-measured under controlled conditions.

## Run it

Python 3.12, CPU is enough.

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python scripts/eval_baselines.py        # -> results/baselines.json
```

## Layout

| Path | What it is |
|---|---|
| `maya/data.py` | load the yes/no items from `data/eval`, negation pairs |
| `maya/metrics.py` | accuracy, AUROC, ECE, coverage at a fixed error, contradiction rate |
| `scripts/eval_baselines.py` | always-no, Laya / layaMOE (recorded answers) and zero-shot NLI baselines |
| `results/baselines.json` | summary and every probability for each baseline |
| `data/eval/` | 108 hand-labeled cases in 9 domains (from layaForWeb) |

See `NOTICE.md` for third-party models and data.
