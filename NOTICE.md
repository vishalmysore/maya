# Notice

Maya is an unofficial experiment. It is not affiliated with or endorsed by ConvAI Innovations, Answer.AI, LightOn or the authors of the NLI models it is compared against.

## This project

The scripts, evaluation code and results in this repository are Copyright 2026 vishalmysore and licensed under the Apache License, Version 2.0 (see `LICENSE`). The evaluation cases in `data/eval` are copied from the author's layaForWeb repository (same license).

`results/laya_moe_eval_reference.json` holds the laya-typed-decisions and layaMOE answers recorded by the author's layaMOE repository (https://github.com/vishalmysore/layaMOE).

## Maya's weights

- Maya v0.2 is fine-tuned from MoritzLaurer/deberta-v3-large-zeroshot-v2.0 (MIT license, https://huggingface.co/MoritzLaurer/deberta-v3-large-zeroshot-v2.0), itself fine-tuned from microsoft/deberta-v3-large (MIT). Changes: fine-tuned on Maya's synthetic yes/no data plus BoolQ and MNLI pairs, labels renamed to yes/no.
- Maya v0.1 (tag `v0.1` on Hugging Face) is fine-tuned from MoritzLaurer/ModernBERT-base-zeroshot-v2.0 (Apache-2.0), itself fine-tuned from answerdotai/ModernBERT-base (Apache-2.0).

The released weights are modified derivatives of those models. The card of deberta-v3-large-zeroshot-v2.0 states that its versions without "-c" in the name were trained on data that includes non-commercially licensed datasets; check it before commercial use of Maya v0.2.

## Training data

- `gen/laya_domains.py` is copied from the author's layaMOE repository (same license).
- During training, MNLI pairs (GLUE `mnli`, https://huggingface.co/datasets/nyu-mll/glue) and BoolQ questions (https://huggingface.co/datasets/google/boolq, CC BY-SA 3.0) are mixed in. They are downloaded at run time and not redistributed.

## Models used as baselines

Downloaded from Hugging Face at run time, not redistributed here. Check each model card for its license and training data before reusing a model.

- convaiinnovations/laya-typed-decisions (Apache-2.0)
- cross-encoder/nli-deberta-v3-xsmall
- MoritzLaurer/deberta-v3-xsmall-zeroshot-v1.1-all-33
- MoritzLaurer/deberta-v3-base-zeroshot-v2.0
- MoritzLaurer/ModernBERT-base-zeroshot-v2.0
- MoritzLaurer/deberta-v3-large-zeroshot-v2.0
checkpoints/ (model weights) are not in git; they go to https://huggingface.co/VishalMysore/maya.
