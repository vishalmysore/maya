// Maya in the browser: tokenize (text, statement) pairs, run the int8 ONNX graph, turn logits into P(yes).
// P(yes) = sigmoid((logit_yes - logit_no) / temperature). Answers are only "yes", "no" or "not sure".

export class Maya {
  constructor(ort, session, tokenizer, config) {
    this.ort = ort;
    this.session = session;
    this.tok = tokenizer;
    this.temperature = config.temperature ?? 1.0;
    const th = config.thresholds || {};
    this.tYes = th.t_yes ?? 0.5;
    this.tNo = th.t_no ?? 0.5;
    this.padId = 50283; // ModernBERT [PAD]
    this.maxLen = 256;
  }

  encodePair(text, statement) {
    const ids = Array.from(this.tok.encode(text, { text_pair: statement, add_special_tokens: true }).ids);
    return ids.length > this.maxLen ? ids.slice(0, this.maxLen - 1).concat(ids[ids.length - 1]) : ids;
  }

  answer(p) {
    return p >= this.tYes ? "yes" : p <= this.tNo ? "no" : "not sure";
  }

  // All statements about one text go through the model as one batch.
  async ask(text, statements, { raw = false } = {}) {
    const rows = statements.map((s) => this.encodePair(text, s));
    const L = Math.max(...rows.map((r) => r.length));
    const ids = new BigInt64Array(rows.length * L).fill(BigInt(this.padId));
    const mask = new BigInt64Array(rows.length * L);
    rows.forEach((r, i) => r.forEach((t, j) => { ids[i * L + j] = BigInt(t); mask[i * L + j] = 1n; }));
    const t0 = performance.now();
    const out = await this.session.run({
      input_ids: new this.ort.Tensor("int64", ids, [rows.length, L]),
      attention_mask: new this.ort.Tensor("int64", mask, [rows.length, L]),
    });
    const ms = performance.now() - t0;
    const lg = out.logits.data;
    const T = raw ? 1.0 : this.temperature;
    const results = statements.map((s, i) => {
      const z = (lg[i * 2] - lg[i * 2 + 1]) / T;
      const p = 1 / (1 + Math.exp(-z));
      return { statement: s, p_yes: p, answer: this.answer(p) };
    });
    return { results, ms, tokens: rows.map((r) => r.length) };
  }
}
