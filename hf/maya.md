---
license: apache-2.0
base_model: MoritzLaurer/deberta-v3-large-zeroshot-v2.0
pipeline_tag: text-classification
tags: [yes-no, guardrail, zero-shot, nli, deberta-v3, calibration]
language: [en]
---

# Maya v0.2: a dedicated yes/no gate

Maya reads a text and a statement (or a yes/no question) and returns P(yes). Version 0.2 has 435M parameters and is fine-tuned from [MoritzLaurer/deberta-v3-large-zeroshot-v2.0](https://huggingface.co/MoritzLaurer/deberta-v3-large-zeroshot-v2.0) on rule-labeled texts with varied wording, minimal pairs and word-overlap traps, plus BoolQ and MNLI pairs to keep its general skill.

- Code, data, tests and every result: https://github.com/vishalmysore/maya
- Live demo in the browser: https://vishalmysore.github.io/maya/
- Browser (int8 ONNX) build: [VishalMysore/mayaWasm](https://huggingface.co/VishalMysore/mayaWasm)
- The previous model (v0.1, ModernBERT-base) is kept under the `v0.1` tag of this repo.

```python
# pip install torch transformers sentencepiece ; then from the GitHub repo:
from maya.gate import Maya
maya = Maya.load("VishalMysore/maya")
maya.ask("Agent plan: delete the `sessions` table on the production database. A verified backup was taken ten minutes ago and the on-call engineer has reviewed the plan.",
         ["The action is destructive and cannot be undone", "This action can be undone if needed"])
# -> no (0.15), yes (0.90)
```

Without the repo: tokenize the pair `(text, statement)`, run `AutoModelForSequenceClassification` (load with `dtype=torch.float32` on CPU), and take `sigmoid((logits[:, 0] - logits[:, 1]) / 1.65)`; label 0 is "yes". The temperature is in `maya_config.json`.

## Results (hand-labeled test sets, never used for training or model selection)

| Model | v3: judgment questions (192 answers) | v2: 8 unseen domains (256) | v1: familiar domains (120) |
|---|---|---|---|
| laya-typed-decisions (421M) | 62.5% | 80.1% | 68.3% |
| deberta-v3-large-zeroshot-v2.0 (Maya's starting point) | 72.9% | 89.8% | 75.8% |
| Maya v0.1 (ModernBERT-base, 150M) | 71.9% | 81.2% | 78.3% |
| **Maya v0.2** | **87.5%** | **95.3%** | **87.5%** |

- v3 was written and committed before the v0.2 training data existed; it includes word-overlap traps (Maya v0.2: 90%) and minimal pairs (71% of pairs fully right; starting point 33%).
- Contradictions between a statement and its negation: 12% on v3 and 8% on v2 (starting point: 73% and 30%).
- Calibration error (ECE) with the shipped temperature: 0.064 on v3, 0.025 on v2.
- The checkpoint was chosen on a separate 64-answer dev set (87.5%).

## Limits

- It still makes mistakes: about one in eight answers on judgment questions is wrong. Example: for a production `DELETE` that "no human has reviewed", it correctly says the action is not safe to run without a human, but answers "no" to "A human should approve this action before it runs". Do not use it as the only safety check.
- Code changes are a known weak spot: it says an unreviewed pull request that drops a database column can be merged. Maya v0.1 got that right.
- It only outputs P(yes): every answer is yes or no (or "not sure" if you set abstention thresholds). It cannot refuse a question that is not yes/no.
- A "not sure" band with a guaranteed error rate has to be fitted on a few hundred labeled examples of your own inputs (`maya.conformal.Abstainer`).
- English only; 435M parameters, about 0.6 s per question on a laptop CPU (4 threads).

**License note:** the base model's card says its versions without "-c" in the name were trained on data that includes non-commercially licensed datasets. Maya v0.2 inherits that; check the base model's card before commercial use (a commercially friendly alternative would be to repeat the recipe on `deberta-v3-large-zeroshot-v2.0-c`).

Unofficial experiment, not affiliated with the authors of the base model or of Laya. Apache-2.0; see NOTICE.md.
