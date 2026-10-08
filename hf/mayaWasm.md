---
license: apache-2.0
base_model: VishalMysore/maya
tags: [onnx, onnxruntime-web, browser, wasm, yes-no, guardrail, quantized]
language: [en]
---

# Maya v0.2 for the browser (int8 ONNX)

[Maya v0.2](https://huggingface.co/VishalMysore/maya), a 435M-parameter yes/no gate (DeBERTa-v3-large fine-tune), exported for ONNX Runtime Web: one graph, int8 weight-only quantization (MatMulNBits block 128, int8 embeddings), about 600 MB. Live demo: https://vishalmysore.github.io/maya/

- `model.onnx` + `model.onnx.data` split into 24 MiB parts (`model.onnx.data.partNNN`); `manifest.json` lists the parts with the total size and SHA-256. Concatenate the parts in order to get `model.onnx.data`.
- Inputs `input_ids`, `attention_mask` (int64, [batch, seq]) from `tokenizer.json`, encoding the pair `(text, statement)`; pad with `pad_token_id` from `maya_config.json` (0). Output `logits` [batch, 2].
- P(yes) = sigmoid((logits[:, 0] - logits[:, 1]) / temperature), temperature in `maya_config.json` (1.65).
- Checked against PyTorch on 256 hand-labeled answers: 94.9% accuracy (PyTorch 95.3%), 1 answer flips, largest probability change 0.06. In the browser the JavaScript tokenizer gives the same token ids as Python on 33 test items and probabilities differ from PyTorch by at most 0.03.
- The previous model (v0.1, 161 MB) is kept under the `v0.1` tag.

**License note:** the base model's card says its versions without "-c" in the name were trained on data that includes non-commercially licensed datasets. Maya v0.2 inherits that; check the base model's card before commercial use (a commercially friendly alternative would be to repeat the recipe on `deberta-v3-large-zeroshot-v2.0-c`).

Code, tests and evaluation: https://github.com/vishalmysore/maya. It still makes mistakes (about one in eight judgment answers); do not use it as the only safety check.
