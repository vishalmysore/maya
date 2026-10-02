---
license: apache-2.0
base_model: MoritzLaurer/ModernBERT-base-zeroshot-v2.0
pipeline_tag: text-classification
tags: [yes-no, guardrail, zero-shot, nli, modernbert, calibration]
language: [en]
---

# Maya: a dedicated yes/no gate

Maya reads a text and a statement (or a yes/no question) and returns P(yes). 150M parameters, fine-tuned from [MoritzLaurer/ModernBERT-base-zeroshot-v2.0](https://huggingface.co/MoritzLaurer/ModernBERT-base-zeroshot-v2.0) on rule-labeled synthetic data in which every fact appears in plain and negated wordings with opposite labels (plus a consistency loss that, in an ablation, added nothing measurable).

Code, data, evaluation and full results: https://github.com/vishalmysore/maya. Browser (int8 ONNX) build: [VishalMysore/mayaWasm](https://huggingface.co/VishalMysore/mayaWasm).

```python
# pip install torch transformers ; then from the GitHub repo:
from maya.gate import Maya
maya = Maya.load("VishalMysore/maya")
maya.ask("Hi team, how do I export the dashboard to Excel? No rush.", ["Is the customer angry?"])
```

Without the repo: tokenize the pair `(text, statement)`, run `AutoModelForSequenceClassification` (load with `dtype=torch.float32` on CPU), and take `sigmoid((logits[:, 0] - logits[:, 1]) / 2.8)`; label 0 is "yes". The temperature 2.8 is in `maya_config.json`.

## Results (hand-labeled, never used for training)

| | v1: 120 answers, mostly in-domain | v2: 256 answers, 8 unseen domains |
|---|---|---|
| laya-typed-decisions (421M) | 68.3% acc, AUROC 0.792 | 80.1% acc, AUROC 0.889 |
| ModernBERT-base-zeroshot-v2.0 (starting point) | 76.7%, 0.875 | 78.5%, 0.868 |
| deberta-v3-large-zeroshot-v2.0 (435M) | 75.8%, 0.890 | **89.8%, 0.957** |
| **Maya** | **78.3%, 0.905** | 81.2%, 0.893 |

Maya is the strongest model tested on its own domains (v1). On unseen domains (v2) it is level with Laya and smaller NLI models, and a 3x larger off-the-shelf NLI model is clearly better.

- Contradictions between a statement and its negation: 25% (v1) and 27% (v2), against 44-92% for the other models.
- With the shipped temperature, v2 calibration error (ECE) is 0.061.
- Weak spots: agent guardrails (61% on v1, misled by lexical overlap such as "no human has reviewed" vs "without a human"), and minimal pairs in unseen domains (53% vs Laya's 69%). Do not use it as the only safety check.
- It only outputs P(yes): every answer is yes or no (or "not sure" if you set abstention thresholds). It cannot refuse non-yes/no questions.

Unofficial experiment, not affiliated with the authors of the base model or of Laya. Apache-2.0; see NOTICE.md.
