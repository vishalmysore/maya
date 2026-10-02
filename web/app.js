import * as ort from "./vendor/ort.min.mjs";
import { Tokenizer } from "./vendor/tokenizers.min.mjs";
import { Maya } from "./maya-core.js";

const DEFAULT_BASE = "https://huggingface.co/VishalMysore/mayaWasm/resolve/main/";
const $ = (id) => document.getElementById(id);
let BASE = DEFAULT_BASE;
let maya = null;

// Examples written for the demo (not taken from the test sets). Each pairs a statement with its negation
// so you can see whether the two answers agree.
const PRESETS = [
  { name: "Agent plan: delete production data, backup taken",
    text: "Agent plan: delete the `sessions` table on the production database. A verified backup was taken ten minutes ago and the on-call engineer has reviewed the plan.",
    statements: ["The action is destructive and cannot be undone", "This action can be undone if needed",
      "Does a person need to approve this first?", "Is this a high-risk action?"] },
  { name: "Agent plan: read-only query",
    text: "Agent plan: count the rows in the `orders` table on the read-only analytics replica and report the number.",
    statements: ["The action is destructive and cannot be undone", "Can the agent do this without asking anyone?",
      "A human should approve this action before it runs"] },
  { name: "Support ticket: angry customer",
    text: "Subject: STILL BROKEN\nThis is the third time the invoice export fails. I'm paying for this and my accountant needs the file today. Fix it or I cancel.",
    statements: ["Is the customer angry?", "The customer sounds calm", "The customer reports that something is broken",
      "This ticket needs a response today", "Is this a feature request?"] },
  { name: "Code change: migration without review",
    text: "Pull request: migration that drops the `legacy_status` column from `orders`. CI is green. Nobody has reviewed it yet.",
    statements: ["Can this be merged now?", "This pull request should not be merged yet", "Is this a breaking change?",
      "Has someone approved this pull request?"] },
  { name: "IT incident: resolved outage",
    text: "Resolved incident: the payments API returned errors for 18 minutes for all EU customers. A rollback restored normal service; error rates are back to baseline.",
    statements: ["Customers are affected right now", "No customers are affected right now", "The incident has been resolved",
      "This needs an immediate page"] },
  { name: "Product review",
    text: "4/5. The kettle boils fast and looks great, but the lid hinge feels a bit flimsy. Still, I'd buy it again.",
    statements: ["The reviewer would recommend this product", "The reviewer advises against buying it", "Was the item sent back?",
      "Is the overall tone of the review favourable?"] },
  { name: "Unseen domain: library notice",
    text: "Your reserved book 'Project Hail Mary' is ready for pickup at the North Branch. It will be held for 7 days.",
    statements: ["The reader needs to do something", "No action is needed from the reader", "Is this message a scam?"] },
];

function setStatus(msg, cls = "") { $("status").textContent = msg; $("status").className = cls; }
function setProgress(v) { const p = $("progress"); if (v == null) { p.hidden = true; return; } p.hidden = false; p.value = v; }
function badge(text) { const s = document.createElement("span"); s.textContent = text; $("badges").appendChild(s); }

async function fetchBytes(url, onProgress) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url}: HTTP ${r.status}`);
  const total = Number(r.headers.get("content-length")) || 0;
  if (!r.body || !onProgress) return new Uint8Array(await r.arrayBuffer());
  const reader = r.body.getReader(); const chunks = []; let got = 0;
  for (;;) { const { done, value } = await reader.read(); if (done) break; chunks.push(value); got += value.length; onProgress(got, total); }
  const out = new Uint8Array(got); let off = 0; for (const c of chunks) { out.set(c, off); off += c.length; }
  return out;
}

// Weights are stored as <= 24 MiB parts; each part is cached in Cache Storage keyed by the file hash.
let cachedParts = 0;
async function fetchParts(entry, onProgress) {
  let cache = null;
  try { cache = await caches.open("maya-" + entry.sha256.slice(0, 16)); } catch { /* private mode: just download */ }
  const out = new Uint8Array(entry.size); let off = 0;
  for (const part of entry.parts) {
    const url = BASE + part;
    let bytes = null;
    try { const hit = cache && await cache.match(url); if (hit) { bytes = new Uint8Array(await hit.arrayBuffer()); cachedParts++; } } catch {}
    if (!bytes) {
      bytes = await fetchBytes(url, (got) => onProgress(off + got));
      try { if (cache) await cache.put(url, new Response(bytes)); } catch { /* quota exceeded: fine */ }
    }
    if (off + bytes.length > entry.size) throw new Error("weights are larger than the manifest says");
    out.set(bytes, off); off += bytes.length; onProgress(off);
  }
  if (off !== entry.size) throw new Error(`weights incomplete: got ${off} of ${entry.size} bytes`);
  return out;
}

async function loadModel() {
  $("loadBtn").disabled = true; $("runBtn").disabled = true; $("badges").innerHTML = ""; cachedParts = 0;
  try {
    try { const sc = await fetch("./site-config.json"); if (sc.ok) BASE = (await sc.json()).modelBase || BASE; } catch {}
    ort.env.wasm.wasmPaths = new URL("./vendor/", import.meta.url).href;
    ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 2) : 1;
    setStatus("Loading tokenizer and config…"); setProgress(0);
    const [manifest, tj, tc, cfg] = await Promise.all(["manifest.json", "tokenizer.json", "tokenizer_config.json", "maya_config.json"]
      .map((f) => fetch(BASE + f).then((r) => { if (!r.ok) throw new Error(f + ": HTTP " + r.status); return r.json(); })));
    const tokenizer = new Tokenizer(tj, tc);
    const entry = manifest.model;
    const t0 = performance.now();
    const graph = await fetchBytes(BASE + entry.onnx);
    const data = await fetchParts(entry.data, (got) => {
      setProgress(got / entry.data.size);
      setStatus(`Downloading Maya (${(got / 1048576).toFixed(0)} / ${(entry.data.size / 1048576).toFixed(0)} MB)`);
    });
    setStatus("Starting ONNX Runtime…"); setProgress(null);
    const session = await ort.InferenceSession.create(graph, { executionProviders: ["wasm"], graphOptimizationLevel: "all",
      externalData: [{ path: entry.data.name, data }] });
    maya = new Maya(ort, session, tokenizer, cfg);
    await maya.ask("warm up", ["This is a warm-up call"]);
    setStatus(`Ready in ${((performance.now() - t0) / 1000).toFixed(1)} s${cachedParts ? ` (${cachedParts} of ${entry.data.parts.length} parts from browser cache)` : ""}.`);
    badge("WASM · " + ort.env.wasm.numThreads + (ort.env.wasm.numThreads > 1 ? " threads" : " thread"));
    badge("int8 · " + (entry.data.size / 1e6).toFixed(0) + " MB");
    badge("temperature " + maya.temperature);
    $("runBtn").disabled = false;
    window.__maya.ready = true;
  } catch (e) {
    console.error(e); setStatus("Could not load the model: " + (e?.message || e), "warn");
  } finally { setProgress(null); $("loadBtn").disabled = false; }
}

function syncBand() {
  $("tNoVal").textContent = Number($("tNo").value).toFixed(2);
  $("tYesVal").textContent = Number($("tYes").value).toFixed(2);
  if (maya) { maya.tNo = Number($("tNo").value); maya.tYes = Number($("tYes").value); }
}

async function run() {
  if (!maya) return;
  const text = $("text").value.trim();
  const statements = $("statements").value.split("\n").map((s) => s.trim()).filter(Boolean);
  if (!text || !statements.length) { setStatus("Enter a text and at least one statement.", "warn"); return; }
  syncBand();
  $("runBtn").disabled = true; setStatus("Thinking…");
  try {
    const { results, ms } = await maya.ask(text, statements);
    const box = $("results"); box.innerHTML = "";
    for (const r of results) {
      const row = document.createElement("div"); row.className = "res";
      const cls = r.answer === "not sure" ? "unsure" : r.answer;
      row.innerHTML = `<span class="st"></span><span class="track"><span class="fill" style="display:block;width:${(r.p_yes * 100).toFixed(1)}%"></span></span>`
        + `<span class="pv">${(r.p_yes * 100).toFixed(1)}%</span><span class="ans ${cls}">${r.answer}</span>`;
      row.querySelector(".st").textContent = r.statement;
      box.appendChild(row);
    }
    $("meta").textContent = `${results.length} question${results.length > 1 ? "s" : ""} in one batch · ${ms.toFixed(0)} ms in the browser · P(yes) with temperature ${maya.temperature}`;
    $("resultCard").hidden = false;
    setStatus("Done.");
    window.__maya.last = results;
  } catch (e) {
    console.error(e); setStatus("Error: " + (e?.message || e), "warn");
  } finally { $("runBtn").disabled = false; }
}

function usePreset(i) {
  const p = PRESETS[i];
  $("text").value = p.text;
  $("statements").value = p.statements.join("\n");
}

for (const [i, p] of PRESETS.entries()) {
  const o = document.createElement("option"); o.value = String(i); o.textContent = p.name; $("preset").appendChild(o);
}
$("preset").addEventListener("change", () => usePreset(Number($("preset").value)));
$("loadBtn").addEventListener("click", loadModel);
$("runBtn").addEventListener("click", run);
$("tNo").addEventListener("input", syncBand);
$("tYes").addEventListener("input", syncBand);
usePreset(0);
syncBand();

// Hooks for the screenshot / test script.
window.__maya = {
  ready: false, run, load: loadModel, presets: PRESETS.map((p) => p.name),
  encode: (text, statement) => maya.encodePair(text, statement),
  ask: (text, statements, raw = false) => maya.ask(text, statements, { raw }),
  setBand: (tNo, tYes) => { $("tNo").value = tNo; $("tYes").value = tYes; syncBand(); },
};
