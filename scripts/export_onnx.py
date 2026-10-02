"""Export Maya to ONNX for the browser (ONNX Runtime Web / WASM), int8 weight-only.

    python scripts/export_onnx.py checkpoints/maya        # -> build/web/

Graph: input_ids, attention_mask [B, L] (tokenized as the pair text, statement) -> logits [B, 2];
P(yes) = sigmoid(logits[:, 0] - logits[:, 1]).

Quantization follows layaMOE's browser build: int8 block-128 MatMulNBits for matrix weights and
int8 per-row embeddings. The quantized graph is checked against PyTorch on every eval_v2 answer
(max |dP|, answers that flip at 0.5) before packaging.

build/web/ holds model.onnx, its weights split into 24 MiB parts with SHA-256 hashes
(manifest.json), the tokenizer, maya_config.json (abstention thresholds) and a model card.
"""
import argparse
import gc
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from maya.data import EVAL_V2_DIR, load_yes_no_items  # noqa: E402

CHUNK = 24 * 1024 * 1024


def log(m):
    print(f"[export] {m}", flush=True)


def export_fp32(model, tok, out):
    from torch.export import Dim

    class Wrap(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, input_ids, attention_mask):
            return self.m(input_ids=input_ids, attention_mask=attention_mask).logits

    enc = tok(["Agent plan: run `DROP TABLE invoices` on production.", "The parcel is late."],
              ["The action is destructive", "Is the customer upset?"], return_tensors="pt", padding=True)
    batch, seq = Dim("batch", min=1, max=64), Dim("seq", min=4, max=512)
    prog = torch.onnx.export(Wrap(model).eval(), (enc["input_ids"], enc["attention_mask"]), dynamo=True,
                             dynamic_shapes={"input_ids": {0: batch, 1: seq}, "attention_mask": {0: batch, 1: seq}},
                             input_names=["input_ids", "attention_mask"], output_names=["logits"],
                             opset_version=18, optimize=True)
    raw = out / "model_fp32.onnx"
    prog.save(str(raw), external_data=True)
    del prog
    gc.collect()
    return raw


def _embedding_to_int8(m):
    from onnx import TensorProto, helper, numpy_helper
    g = m.graph
    inits = {i.name: i for i in g.initializer}
    new_nodes, drop, added = [], set(), []
    for node in g.node:
        if node.op_type == "Gather" and node.input[0] in inits:
            t = inits[node.input[0]]
            if t.data_type == TensorProto.FLOAT and len(t.dims) == 2 and t.dims[0] > 1000:
                w = numpy_helper.to_array(t).astype(np.float32)
                scale = np.maximum(np.abs(w).max(axis=1, keepdims=True) / 127.0, 1e-8).astype(np.float32)
                q = np.clip(np.round(w / scale), -127, 127).astype(np.int8)
                n = t.name
                added += [numpy_helper.from_array(q, n + "_q8"), numpy_helper.from_array(scale.reshape(-1), n + "_scale")]
                new_nodes.append(helper.make_node("DequantizeLinear", [n + "_q8", n + "_scale"], [n + "_dq"], axis=0, name=n + "_DQ"))
                drop.add(n)
                node.input[0] = n + "_dq"
    keep = [i for i in g.initializer if i.name not in drop]
    del g.initializer[:]
    g.initializer.extend(keep + added)
    nodes = list(g.node)
    del g.node[:]
    g.node.extend(new_nodes + nodes)


def quantize_q8(raw, dst):
    import onnx
    from onnxruntime.quantization.matmul_nbits_quantizer import DefaultWeightOnlyQuantConfig, MatMulNBitsQuantizer
    m = onnx.load(str(raw), load_external_data=True)
    del m.graph.value_info[:]  # stale shape annotations from the exporter trip the quantizer
    q = MatMulNBitsQuantizer(m, algo_config=DefaultWeightOnlyQuantConfig(block_size=128, is_symmetric=True, bits=8))
    q.process()
    mq = q.model.model
    _embedding_to_int8(mq)
    onnx.save(mq, str(dst), save_as_external_data=True, all_tensors_to_one_file=True, location=dst.name + ".data",
              size_threshold=1024)
    del q, m, mq
    gc.collect()
    log(f"{dst.name}: {(dst.parent / (dst.name + '.data')).stat().st_size / 1e6:.0f} MB")


def verify(model, tok, onnx_path):
    import onnxruntime as ort
    items = load_yes_no_items(EVAL_V2_DIR)
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    ref, got = [], []
    for i in range(0, len(items), 16):
        b = items[i:i + 16]
        enc = tok([it["text"] for it in b], [it["statement"] for it in b], return_tensors="pt", padding=True)
        with torch.inference_mode():
            lg = model(**enc).logits
        ref += torch.sigmoid(lg[:, 0] - lg[:, 1]).tolist()
        lo = sess.run(None, {"input_ids": enc["input_ids"].numpy(), "attention_mask": enc["attention_mask"].numpy()})[0]
        got += (1 / (1 + np.exp(-(lo[:, 0] - lo[:, 1])))).tolist()
    ref, got = np.array(ref), np.array(got)
    y = np.array([it["label"] for it in items])
    res = {"max_abs_dp": float(np.abs(ref - got).max()), "flips_at_0.5": int(((ref >= 0.5) != (got >= 0.5)).sum()),
           "acc_pytorch": float(((ref >= 0.5) == y).mean()), "acc_int8": float(((got >= 0.5) == y).mean()), "n": len(y)}
    log(f"int8 vs PyTorch on eval_v2: {res}")
    if res["max_abs_dp"] > 0.15:
        sys.exit("quantized graph drifts too far from PyTorch")
    return res


def split(data_path, dst_dir):
    parts, h, size = [], hashlib.sha256(), data_path.stat().st_size
    with open(data_path, "rb") as f:
        i = 0
        while buf := f.read(CHUNK):
            h.update(buf)
            name = f"{data_path.name}.part{i:03d}"
            (dst_dir / name).write_bytes(buf)
            parts.append(name)
            i += 1
    return {"name": data_path.name, "size": size, "sha256": h.hexdigest(), "parts": parts}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--out", default=str(ROOT / "build"))
    args = ap.parse_args()
    ckpt = Path(args.checkpoint)
    out = Path(args.out)
    onnx_dir, web = out / "onnx", out / "web"
    onnx_dir.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(ckpt)
    model = AutoModelForSequenceClassification.from_pretrained(ckpt, dtype=torch.float32).eval()
    raw = export_fp32(model, tok, onnx_dir)
    quantize_q8(raw, onnx_dir / "model.onnx")
    check = verify(model, tok, onnx_dir / "model.onnx")

    if web.exists():
        shutil.rmtree(web)
    web.mkdir(parents=True)
    shutil.copy(onnx_dir / "model.onnx", web / "model.onnx")
    manifest = {"version": 1, "model": {"onnx": "model.onnx", "data": split(onnx_dir / "model.onnx.data", web)},
                "inputs": ["input_ids", "attention_mask"], "output": "logits",
                "p_yes": "sigmoid(logits[:, 0] - logits[:, 1])", "pair_order": ["text", "statement"],
                "verification": check}
    for f in ("tokenizer.json", "tokenizer_config.json"):
        shutil.copy(ckpt / f, web / f)
    if (ckpt / "maya_config.json").exists():
        shutil.copy(ckpt / "maya_config.json", web / "maya_config.json")
    (web / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    for f in ("LICENSE", "NOTICE.md"):
        shutil.copy(ROOT / f, web / f)
    total = sum(p.stat().st_size for p in web.iterdir())
    log(f"browser build ready: {web} ({total / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
