"""Maya as a yes/no gate.

    from maya.gate import Maya
    maya = Maya.load("VishalMysore/maya")          # or a local checkpoint folder
    maya.ask("Agent plan: run `DROP TABLE invoices` on production. No backup.",
             ["The action is destructive and cannot be undone", "Is it safe to run without a human?"])
    # -> [{"statement": ..., "p_yes": 0.97, "answer": "yes"}, {"statement": ..., "p_yes": 0.04, "answer": "no"}]

Every answer is "yes", "no" or (only when abstention thresholds are set) "not sure"; the model is a
classifier with a single P(yes) output, so it cannot produce anything else. All questions about one
text are scored in one batch.
"""
import json
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class Maya:
    def __init__(self, model, tok, temperature=1.0, t_yes=0.5, t_no=0.5, max_length=256):
        self.model, self.tok = model.eval(), tok
        self.temperature, self.t_yes, self.t_no, self.max_length = temperature, t_yes, t_no, max_length

    @classmethod
    def load(cls, path):
        model = AutoModelForSequenceClassification.from_pretrained(path, dtype=torch.float32)  # bf16 is slow on CPU
        tok = AutoTokenizer.from_pretrained(path)
        cfg = {}
        local = Path(path) / "maya_config.json"
        if local.exists():
            cfg = json.loads(local.read_text(encoding="utf-8"))
        else:
            try:
                from huggingface_hub import hf_hub_download
                cfg = json.loads(Path(hf_hub_download(path, "maya_config.json")).read_text(encoding="utf-8"))
            except Exception:
                pass
        th = cfg.get("thresholds", {})
        return cls(model, tok, cfg.get("temperature", 1.0), th.get("t_yes", 0.5), th.get("t_no", 0.5))

    def with_abstention(self, abstainer):
        """Use thresholds fitted by maya.conformal.Abstainer on your own labeled data."""
        self.t_yes = abstainer.t_yes if abstainer.t_yes is not None else 1.01
        self.t_no = abstainer.t_no if abstainer.t_no is not None else -0.01
        return self

    @torch.inference_mode()
    def p_yes(self, text, statements):
        enc = self.tok([text] * len(statements), list(statements), return_tensors="pt", padding=True,
                       truncation=True, max_length=self.max_length)
        lg = self.model(**enc).logits.float()
        return torch.sigmoid((lg[:, 0] - lg[:, 1]) / self.temperature).tolist()

    def ask(self, text, statements):
        if isinstance(statements, str):
            statements = [statements]
        out = []
        for s, p in zip(statements, self.p_yes(text, statements)):
            ans = "yes" if p >= self.t_yes else "no" if p <= self.t_no else "not sure"
            out.append({"statement": s, "p_yes": round(p, 4), "answer": ans})
        return out
