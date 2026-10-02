---
title: "Maya: Building a Small Yes/No AI Guardrail Model That Runs in the Browser (and What the Benchmarks Really Say)"
description: "How we fine-tuned a 150M-parameter ModernBERT NLI model into Maya, a yes/no classifier for AI guardrails and early-exit filters, tested it against Laya and five zero-shot NLI models on 376 hand-labeled answers, and shipped it to the browser with int8 ONNX Runtime Web."
keywords: [yes/no classifier, AI guardrail model, zero-shot classification, natural language inference, NLI, ModernBERT, DeBERTa, small language model, ONNX Runtime Web, WebAssembly, browser AI, model calibration, conformal prediction, LLM guardrails, Laya]
author: Vishal Mysore
date: 2026-10-02
slug: maya-yes-no-ai-guardrail-model
image: images/chart-accuracy.png
---

# Maya: Building a Small Yes/No AI Guardrail Model That Runs in the Browser

**TL;DR:** Maya is a 150M-parameter **yes/no classifier**. You give it a text and a statement such as "the action is destructive" or "is the customer angry?", and it returns a probability that the answer is yes. We fine-tuned it from a zero-shot **natural language inference (NLI)** model, tested it on 376 hand-labeled answers against Laya, layaMOE and five open NLI models, and exported it as a 161 MB int8 ONNX model that runs **entirely in the browser**.

- Maya is the most accurate model we tested on familiar text: 78.3%, AUROC 0.905.
- It contradicts itself far less than any other model: 25-27% of statement/negation pairs, against 30-100% for the others.
- **It is not the best model on text from new domains.** An off-the-shelf DeBERTa-v3-large NLI model three times its size is clearly ahead there.

This article covers how we built it, what the numbers say, and where Maya fails.

- Code, data and every result: [github.com/vishalmysore/maya](https://github.com/vishalmysore/maya)
- Model weights: [huggingface.co/VishalMysore/maya](https://huggingface.co/VishalMysore/maya)
- Browser build: [huggingface.co/VishalMysore/mayaWasm](https://huggingface.co/VishalMysore/mayaWasm)

![Maya running in the browser: the yes/no question form with a support ticket, five statements and the answers](images/demo-full-page.png)

## Contents

1. [Why build a dedicated yes/no model?](#why-build-a-dedicated-yesno-model)
2. [How Maya works](#how-maya-works)
3. [Building a test set that is hard to game](#building-a-test-set-that-is-hard-to-game)
4. [Step zero: off-the-shelf NLI is already strong](#step-zero-off-the-shelf-nli-is-already-strong)
5. [Training data with negations and minimal pairs](#training-data-with-negations-and-minimal-pairs)
6. [Results](#results)
7. [What the ablation taught us](#what-the-ablation-taught-us)
8. ["Not sure": abstention with an error guarantee](#not-sure-abstention-with-an-error-guarantee)
9. [Running Maya in the browser with ONNX Runtime Web](#running-maya-in-the-browser-with-onnx-runtime-web)
10. [Testing everything](#testing-everything)
11. [Where Maya fails](#where-maya-fails)
12. [Lessons learned](#lessons-learned)
13. [FAQ](#faq)

## Why build a dedicated yes/no model?

Many decisions in an AI system are binary:
- **Agents:** should the agent run this command without asking a person?
- **Support:** is the customer angry, and does the ticket need a reply today?
- **Moderation:** is this comment spam?

Large language models can answer these questions, but they are slow and expensive to call on every request, and their answers are hard to calibrate. Decision models like [Laya](https://huggingface.co/convaiinnovations/laya-typed-decisions) answer typed questions (yes/no, multiple choice, score) from one encoder, but they weigh in at over 400M parameters.

A dedicated yes/no model has three potential advantages:
- **Small and fast enough** to sit in front of every request as a **guardrail** or early-exit filter, including in a browser tab.
- **Easier to calibrate:** a single sigmoid output is simpler to calibrate than per-option temperatures.
- **Easier to make consistent:** you can train it so that "X" and "not X" never both get a yes.

Maya is the experiment that tests whether those advantages are real.

## How Maya works

Maya is a **cross-encoder**. The text and the statement go through the model together as one sequence, `[CLS] text [SEP] statement [SEP]`, and the classifier head produces two logits. The probability of yes is

```
P(yes) = sigmoid((logit_yes - logit_no) / temperature)
```

That has a useful consequence: **Maya can only ever answer yes or no.** It has no text generator, so it cannot ramble or produce a third label by accident. If you configure an abstention band, a third answer, "not sure", appears between two thresholds. The flip side is that Maya cannot refuse. A question that is not yes/no ("what colour is it?") still gets a probability, so only ask yes/no questions.

Statements can be written as claims ("The customer sounds angry") or as questions ("Is the customer angry?"). About a third of the training statements are phrased as questions.

```python
from maya.gate import Maya

maya = Maya.load("VishalMysore/maya")
maya.ask("Hi team, how do I export the dashboard to Excel? No rush.",
         ["Is the customer angry?", "The customer sounds calm"])
# [{'statement': 'Is the customer angry?', 'p_yes': 0.0246, 'answer': 'no'},
#  {'statement': 'The customer sounds calm', 'p_yes': 0.9625, 'answer': 'yes'}]
```

## Building a test set that is hard to game

Accuracy alone flatters yes/no models. On our first test set, answering "no" to everything scores 62.5%. So we measured four extra things on two hand-labeled test sets that are never used for training or model selection.

- **v1:** 120 yes/no answers in 8 domains, from the earlier [layaMOE](https://github.com/vishalmysore/layaMOE) project. Maya's training data covers five of these domains, so v1 measures *familiar* text.
- **v2:** 256 answers in 8 domains Maya never sees in training: rental listings, travel notices, school messages, contract clauses, scam messages, smart-home commands, recipes and job postings. Its 32 statements never appear in training either.

Every v2 case comes with three built-in checks:

| Check | Example | What a good model does |
|---|---|---|
| Negation pair | "The traveler needs to take action" / "The traveler does not need to do anything" | gives opposite answers |
| Implication pair | "Dogs are allowed" => "Pets are allowed" | never says yes to the first and no to the second |
| Minimal pair | the same recipe with "crushed peanuts" vs "toasted sesame seeds" | flips the answer to "safe for a peanut allergy" |

We also report:
- **AUROC:** how well the confidence ranks right answers above wrong ones.
- **ECE:** calibration error.
- **Answerable at ≤10% error:** the share of questions a model can answer, most confident first, before its error passes 10%.

## Step zero: off-the-shelf NLI is already strong

Before training anything, we scored existing models. Zero-shot NLI models already answer "does this text entail this statement?". The surprise was how good they are:
- **ModernBERT-base-zeroshot-v2.0** (150M parameters) matched layaMOE's ranking quality on v1 *without any training*.
- **Every model contradicted itself constantly.** On the v1 negation pair ("a human should approve this" / "it is safe to run without a human"), the NLI models said "no" to both in 11-12 of 12 cases.

That settled the design: start from the ModernBERT NLI model rather than from a raw encoder, and make consistency a training goal.

One practical lesson from this step: **transformers 5 loads these checkpoints in bfloat16 by default.** On a laptop CPU without bfloat16 matrix units, that made one training step take 82 s instead of 0.65 s. On CPU, always load with `dtype=torch.float32`.

## Training data with negations and minimal pairs

Maya's training data is generated by rules, so every label is consistent by construction. The generator builds 4,463 texts with 20,429 yes/no statements, 50% of them yes.

**Domains:**
- **Five domains from layaMOE:** agent guardrails, content moderation, support tickets, delivery exceptions and email triage. Their labels are turned into several yes/no "facts" each.
- **Seven new slot-based domains:** IT incidents, product reviews, expense claims, meeting requests, code changes, insurance claims and sales inquiries.

**Wordings and rules:**
- Every fact has plain wordings ("The customer sounds angry") and negated wordings ("The customer sounds calm") with opposite labels.
- Each domain lists implications, for example "the claim can be reimbursed" => "a receipt was provided".

**Minimal pairs:** about half of the slot-based texts get a partner that differs in exactly one slot, such as a backup present or absent, or CI green or red.

**Leakage guards:**
- Any generated text that shares a five-word sequence with a test text is dropped.
- No training wording may equal a v2 statement.
- Unit tests check both rules.

**Training:** one epoch of full fine-tuning on CPU (687 steps, about 80 minutes). The loss has three parts:
- binary cross-entropy on each statement;
- a **consistency loss**: statements about the same fact must agree, and implications must hold;
- a few MNLI pairs per step, so the model keeps its general NLI skill.

## Results

![Bar chart of accuracy on two hand-labeled test sets. Maya 78.3% on familiar domains and 81.2% on unseen domains; DeBERTa-v3-large NLI 75.8% and 89.8%; Laya 68.3% and 80.1%](images/chart-accuracy.png)

### Familiar domains (v1, 120 answers)

| Model | Params | Accuracy | AUROC | Answerable at ≤10% error | Negation contradictions |
|---|---|---|---|---|---|
| laya-typed-decisions | 421M | 68.3% | 0.792 | 48% | 67% |
| layaMOE | 421M + heads | 78.3% | 0.869 | 54% | 67% |
| ModernBERT-base-zeroshot-v2.0 (Maya's start) | 150M | 76.7% | 0.876 | 53% | 92% |
| DeBERTa-v3-large-zeroshot-v2.0 | 435M | 75.8% | 0.890 | 59% | 100% |
| **Maya** | 150M | **78.3%** | **0.905** | **75%** | **25%** |

### Unseen domains (v2, 256 answers)

| Model | Params | Accuracy | AUROC | Negation contradictions | Minimal pairs both right |
|---|---|---|---|---|---|
| laya-typed-decisions | 421M | 80.1% | 0.889 | 44% | 69% |
| DeBERTa-v3-xsmall-zeroshot-v1.1 | 71M | 82.4% | 0.855 | 48% | 59% |
| DeBERTa-v3-base-zeroshot-v2.0 | 184M | 81.2% | 0.911 | 45% | 56% |
| DeBERTa-v3-large-zeroshot-v2.0 | 435M | **89.8%** | **0.957** | 30% | **78%** |
| **Maya** | 150M | 81.2% | 0.893 | **27%** | 53% |

The full tables, with every model and metric, are in the [README](https://github.com/vishalmysore/maya#results).

What the two tables show:

- **On familiar text, Maya is the best model tested.** It is ten points above plain Laya at about a third of the size.
- **On new domains, Maya is in the pack, not at the top.** It is level with Laya and the smaller DeBERTa NLI models. DeBERTa-v3-large is far ahead, yet on familiar text that same model is three points *below* Maya. No model wins both sets.
- **Consistency is Maya's clearest advantage,** and it holds on hand-written negations in domains Maya never saw.

![Small multiples showing how often each model gives the same answer to a statement and its negation. Maya 25% and 27%, other models 30% to 100%](images/chart-contradictions.png)

- **Calibration needs one more step.** Out of domain, Maya is over-confident (ECE 0.15). One temperature fitted on the 120 v1 answers brings v2 calibration error down to 0.061, the lowest of all models, and Maya ships with that temperature (2.8).

## What the ablation taught us

We expected the consistency loss to be what made Maya consistent, so we trained a second model with the same data and the loss switched off:

| | v1 accuracy | v2 accuracy | Contradictions v1 / v2 |
|---|---|---|---|
| Maya (with consistency loss) | 78.3% | 81.2% | 25% / 27% |
| Without consistency loss | 80.8% | 79.7% | 25% / 25% |

There is no difference beyond noise. **The consistency comes from the data, not the loss.** Training on both polarities of every fact, with opposite labels, already teaches the model that "X" and "not X" cannot both be true. That makes the recipe cheaper to reuse: generate negated wordings and the extra loss term can be skipped.

## "Not sure": abstention with an error guarantee

A guardrail should be able to say "I don't know". Maya's `conformal` module fits two thresholds on labeled data, so that among the questions Maya does answer, the error is at most a target α with high probability. It uses a Learn-then-Test procedure over a fixed threshold grid, with Clopper-Pearson bounds and a Bonferroni correction.

![Maya answering "not sure" to four statements about an agent plan when the abstention band is widened to 0.3-0.8](images/demo-guardrail-not-sure.png)

The honest findings:

1. **Calibrating on synthetic data does not transfer.** Maya is 97% right on its synthetic validation data, so the fitted thresholds collapse to 0.5, and the real error on the hand-labeled sets is 19-22%. The guarantee holds only for inputs that look like the calibration data.
2. **120 hand-labeled answers are not enough to certify 10% error**, for Maya or for any other model. At a 20% target, thresholds fitted on v1 let Maya answer 35% of v2 with 7.9% actual error.

So Maya ships with strict yes/no as the default. If you want a bounded-error "not sure", label a few hundred of your own inputs and fit the thresholds on them.

## Running Maya in the browser with ONNX Runtime Web

**The browser build:**
- One ONNX graph with **int8 weight-only quantization**: MatMulNBits with block size 128, plus int8 embeddings.
- 161 MB, split into 24 MiB parts with SHA-256 hashes. The demo caches the parts in Cache Storage, so the second visit loads from disk.
- On all 256 v2 answers, the int8 graph scores 81.6% against 81.25% for PyTorch, with one answer flipping.

![The Maya demo after loading: WASM with 4 threads, int8 155 MB, temperature 2.8](images/demo-model-loaded.png)

**The demo page:**
- Uses **ONNX Runtime Web** (WebAssembly, multi-threaded when the page is cross-origin isolated) and the Hugging Face `tokenizers` JavaScript library.
- Everything runs on your device: no text leaves the browser.
- On a laptop, four questions about one text take about 0.3-0.4 s in a single batch.

![Maya's answers to a pull request that drops a column and has not been reviewed: not mergeable, breaking change, not approved](images/demo-code-change.png)

## Testing everything

**25 automated tests (pytest), all passing:**
- **Test sets:** answer counts, every negation pair has opposite labels, every implication holds, and every minimal pair flips its key answer.
- **Metrics and abstention:** ECE, coverage, consistency counting, Clopper-Pearson bounds, and a simulation showing the abstention guarantee holds on in-distribution data.
- **Generators:** rules hold on thousands of samples, minimal pairs really change the text, and nothing leaks into the test sets.
- **Model:**
  - the answers are only ever yes, no or not sure;
  - batched and single answers agree;
  - the published model reproduces the recorded test probabilities;
  - the int8 ONNX graph matches PyTorch.

**Browser parity:** on 28 test items, the JavaScript tokenizer produces exactly the same token ids as Python, and the in-browser probabilities differ from PyTorch by at most 0.027, with no answer flipping.

## Where Maya fails

The test suite says the pipeline is correct. It does not say the model is right. Trying the demo exposed real failures, and they are worth showing.

![Maya answering "no" to "Is the customer angry?" for a ticket that says STILL BROKEN, I'm paying for this, fix it or I cancel](images/demo-support-ticket-miss.png)

- **Template overfitting.** The ticket above is obviously angry, but Maya says "no" (P = 0.23) and "sounds calm" (0.72). The same complaint with the word "Unacceptable" or "furious" scores 0.90-0.96. Maya learned the training generator's phrases more than the idea of anger.
- **Agent guardrails: only 61% right on v1.** A production `DELETE` that "no human has reviewed" gets 0.92 for "it is safe to run without a human approving it". The words "no human" in the text seem to match "without a human" in the statement. This is the classic lexical-overlap shortcut of NLI models.
- **Contradictions on hard cases.** For "delete the sessions table, backup taken", Maya says yes to both "destructive and cannot be undone" (0.75) and "can be undone if needed" (0.67).
- **Minimal pairs in new domains: 53%,** against Laya's 69% and DeBERTa-v3-large's 78%. Maya often misses the one detail that changes a judgment:
  - "made in a nut-free facility" does not register;
  - a bank's "never share this code" message reads as a scam, and real phishing texts read as legitimate.

**Do not use Maya v0.1 as the only safety check.**

## Lessons learned

1. **Measure consistency, not just accuracy.** Every NLI model we tested contradicted itself on most negation pairs. Accuracy hides that completely.
2. **Test on domains you did not train on.** Maya's gains are mostly in-domain. Without v2 we would have claimed it beats everything.
3. **Paired data beats clever losses.** Training on both polarities of every fact gave all the consistency; the extra loss term added nothing.
4. **Bounded-error abstention needs real calibration data.** Synthetic data makes the model look perfectly calibrated and the guarantee meaningless.
5. **Check the defaults.** A bfloat16 default turned a 1-hour CPU training run into a multi-day one until we found it with the profiler.
6. **Look at the demo.** The template-overfitting failure showed up in the first five minutes of clicking presets, not in the aggregate numbers.

**What's next for Maya v0.2:**
- The same recipe on a stronger base (DeBERTa-v3-large, which already scores 89.8% on unseen domains).
- Training texts that share words with a statement but mean the opposite, to break the lexical-overlap shortcut.
- More varied phrasings per fact.
- A fresh hand-labeled test set, because v1 and v2 have now been studied closely.

## FAQ

**What is Maya?**
Maya is a 150M-parameter yes/no classifier fine-tuned from ModernBERT-base-zeroshot-v2.0. Given a text and a statement, it returns the probability that the statement is true for that text.

**Can Maya give answers other than yes or no?**
No. It is a classifier with one probability output, so every answer is yes or no. A third answer, "not sure", appears only if you set abstention thresholds.

**Does Maya run without a server?**
Yes. The int8 ONNX build (161 MB) runs in the browser with ONNX Runtime Web and WebAssembly, and no text leaves the device. It also runs in Python with PyTorch on a CPU.

**Is Maya better than Laya?**
On familiar domains, yes: 78.3% against 68.3% accuracy, and AUROC 0.905 against 0.792, at about a third of the size. On unseen domains the two are level, at 81.2% and 80.1%. Maya contradicts itself far less often than Laya.

**Is Maya better than zero-shot NLI models like DeBERTa?**
On familiar domains, yes. On unseen domains, DeBERTa-v3-large-zeroshot-v2.0, which is three times Maya's size, is clearly better (89.8% against 81.2%). The smaller DeBERTa models are level with Maya on unseen domains, but they are much weaker on familiar ones and contradict themselves far more.

**Can I use Maya as an AI agent guardrail?**
Not on its own. It is right only 61% of the time on our agent-action test cases and can be fooled by word overlap. Use it as one signal next to rules and human review.

**How do I get calibrated "not sure" answers?**
Label a few hundred examples of your own inputs, fit `maya.conformal.Abstainer` on Maya's probabilities, and pass the result to `Maya.with_abstention`. With fewer examples, a tight error bound cannot be certified.

**Is the code open source?**
Yes. Code, data, results and weights are released under Apache-2.0 at [github.com/vishalmysore/maya](https://github.com/vishalmysore/maya), with weights at [VishalMysore/maya](https://huggingface.co/VishalMysore/maya) and [VishalMysore/mayaWasm](https://huggingface.co/VishalMysore/mayaWasm).

---

*Maya is an unofficial experiment and is not affiliated with ConvAI Innovations (Laya), Answer.AI and LightOn (ModernBERT), or the authors of the NLI models it is compared against.*
