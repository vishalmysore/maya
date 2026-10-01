# Notice

Maya is an unofficial experiment. It is not affiliated with or endorsed by ConvAI Innovations, Answer.AI, LightOn or the authors of the NLI models it is compared against.

## This project

The scripts, evaluation code and results in this repository are Copyright 2026 vishalmysore and licensed under the Apache License, Version 2.0 (see `LICENSE`). The evaluation cases in `data/eval` are copied from the author's layaForWeb repository (same license).

`results/laya_moe_eval_reference.json` holds the laya-typed-decisions and layaMOE answers recorded by the author's layaMOE repository (https://github.com/vishalmysore/layaMOE).

## Models used as baselines

Downloaded from Hugging Face at run time, not redistributed here. Check each model card for its license and training data before reusing a model.

- convaiinnovations/laya-typed-decisions (Apache-2.0)
- cross-encoder/nli-deberta-v3-xsmall
- MoritzLaurer/deberta-v3-xsmall-zeroshot-v1.1-all-33
- MoritzLaurer/deberta-v3-base-zeroshot-v2.0
- MoritzLaurer/ModernBERT-base-zeroshot-v2.0
- MoritzLaurer/deberta-v3-large-zeroshot-v2.0
checkpoints/ (model weights) are not in git; they go to https://huggingface.co/VishalMysore/maya.
