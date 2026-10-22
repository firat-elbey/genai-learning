// Checks that the JavaScript forward pass reproduces the logits the Python
// trainer exported (verify.ids → verify.logits) for every model in ../models.
// Usage: node test/verify.mjs [models-dir]
import { readFileSync, readdirSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { GPT } from "../js/gpt.js";

const dir = process.argv[2] || join(dirname(fileURLToPath(import.meta.url)), "..", "models");
let failed = 0;
for (const f of readdirSync(dir).filter((f) => f.endsWith(".json"))) {
  const meta = JSON.parse(readFileSync(join(dir, f), "utf8"));
  const buf = readFileSync(join(dir, meta.weights_file));
  const w = new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4);
  const model = new GPT(meta, w);
  const t0 = performance.now();
  const trace = model.forward(meta.verify.ids);
  const ms = performance.now() - t0;
  let maxDiff = 0;
  const exp = meta.verify.logits;
  for (let t = 0; t < exp.length; t++)
    for (let v = 0; v < exp[t].length; v++)
      maxDiff = Math.max(maxDiff, Math.abs(exp[t][v] - trace.logits.get(t, v)));
  // explain helpers must reproduce the traced numbers exactly
  const L = trace.layers[0];
  const eq = model.explainLinear(L.ln1, model.layerT(0, "wq"), model.layerT(0, "bq"), trace.T - 1, 3).sum;
  const dq = Math.abs(eq - L.q.get(trace.T - 1, 3));
  const es = model.explainSoftmax(trace, 0, 0, trace.T - 1, 0).value;
  const ds = Math.abs(es - L.attn[0].get(trace.T - 1, 0));
  const el = model.explainLogit(trace, trace.T - 1, 0).sum;
  const dl = Math.abs(el - trace.logits.get(trace.T - 1, 0));
  const ok = maxDiff < 2e-3 && dq < 1e-4 && ds < 1e-6 && dl < 1e-4;
  const top = model.topK(trace.probs.row(trace.T - 1), 3).map((x) => `${JSON.stringify(x.token)}:${x.p.toFixed(3)}`).join(" ");
  console.log(`${ok ? "PASS" : "FAIL"} ${meta.name}: T=${trace.T} params=${meta.n_params} forward=${ms.toFixed(1)}ms  max|Δlogit|=${maxDiff.toExponential(2)}  explain Δ q=${dq.toExponential(1)} softmax=${ds.toExponential(1)} logit=${dl.toExponential(1)}  next: ${top}`);
  if (!ok) failed++;
}
process.exit(failed ? 1 : 0);
