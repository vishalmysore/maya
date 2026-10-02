---
license: apache-2.0
base_model: VishalMysore/maya
tags: [onnx, onnxruntime-web, browser, wasm, yes-no, guardrail, quantized]
language: [en]
---

# Maya for the browser (int8 ONNX)

[Maya](https://huggingface.co/VishalMysore/maya), a 150M-parameter yes/no gate, exported for ONNX Runtime Web: one graph, int8 weight-only quantization (MatMulNBits block 128, int8 embeddings), 161 MB.

- `model.onnx` + `model.onnx.data` split into 24 MiB parts (`model.onnx.data.partNNN`); `manifest.json` lists the parts with the total size and SHA-256. Concatenate the parts in order to get `model.onnx.data`.
- Inputs `input_ids`, `attention_mask` (int64, [batch, seq]) from `tokenizer.json`, encoding the pair `(text, statement)`; output `logits` [batch, 2].
- P(yes) = sigmoid((logits[:, 0] - logits[:, 1]) / temperature), temperature in `maya_config.json` (2.8).
- Checked against PyTorch on 256 hand-labeled answers: 81.6% accuracy (PyTorch 81.25%), 1 answer flips, largest probability change 0.12.

Code and evaluation: https://github.com/vishalmysore/maya. Same weak spots as the PyTorch model (agent guardrails, minimal pairs in unseen domains); do not use it as the only safety check.
