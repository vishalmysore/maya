"""Score (text, statement) pairs with a sequence-classification NLI model: P(yes) = P(entailment)."""
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def entail_index(model):
    labels = {i: l.lower() for i, l in model.config.id2label.items()}
    for i, l in labels.items():
        if l.startswith("entail") or l == "yes":
            return i
    raise ValueError(f"no entailment label in {labels}")


class NLIScorer:
    def __init__(self, name_or_path, max_length=256, batch_size=16):
        self.tok = AutoTokenizer.from_pretrained(name_or_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(name_or_path, dtype=torch.float32).eval()
        self.ent = entail_index(self.model)
        self.max_length = max_length
        self.batch_size = batch_size

    @torch.inference_mode()
    def score(self, pairs):
        probs = []
        for i in range(0, len(pairs), self.batch_size):
            texts, statements = zip(*pairs[i:i + self.batch_size])
            enc = self.tok(list(texts), list(statements), return_tensors="pt", padding=True,
                           truncation=True, max_length=self.max_length)
            logits = self.model(**enc).logits
            probs.extend(torch.softmax(logits.float(), -1)[:, self.ent].tolist())
        return probs
