// Assemble the static demo into dist/: page + ONNX Runtime Web + tokenizer library.
// The model is NOT copied by default; the page loads it from Hugging Face (VishalMysore/mayaWasm).
//   npm install && node scripts/prepare_site.mjs                      # model from Hugging Face
//   node scripts/prepare_site.mjs --local-model build/web             # bundle a local build (testing only)
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const arg = (n) => { const i = process.argv.indexOf(n); return i > 0 ? process.argv[i + 1] : undefined; };
const dist = path.resolve(root, arg("--out") || "dist");
const need = (p, hint) => { if (!fs.existsSync(p)) { console.error(`Missing ${path.relative(root, p)}. ${hint}`); process.exit(1); } };
need(path.join(root, "node_modules", "onnxruntime-web"), "Install dependencies first: npm install");

fs.rmSync(dist, { recursive: true, force: true });
fs.mkdirSync(path.join(dist, "vendor"), { recursive: true });
fs.cpSync(path.join(root, "web"), dist, { recursive: true });
const ort = path.join(root, "node_modules", "onnxruntime-web", "dist");
for (const f of ["ort.min.mjs", "ort-wasm-simd-threaded.jsep.mjs", "ort-wasm-simd-threaded.jsep.wasm"]) {
  need(path.join(ort, f), "Unexpected onnxruntime-web layout; check the pinned version in package.json.");
  fs.copyFileSync(path.join(ort, f), path.join(dist, "vendor", f));
}
fs.copyFileSync(path.join(root, "node_modules", "@huggingface", "tokenizers", "dist", "tokenizers.min.mjs"),
  path.join(dist, "vendor", "tokenizers.min.mjs"));

const local = arg("--local-model");
if (local) {
  fs.cpSync(path.resolve(root, local), path.join(dist, "model"), { recursive: true });
  fs.writeFileSync(path.join(dist, "site-config.json"), JSON.stringify({ modelBase: "./model/" }, null, 2));
  console.log(`bundled model from ${local} (local testing only)`);
}
for (const f of ["NOTICE.md", "LICENSE"]) fs.copyFileSync(path.join(root, f), path.join(dist, f));
fs.writeFileSync(path.join(dist, ".nojekyll"), "");
console.log("dist/ ready");
