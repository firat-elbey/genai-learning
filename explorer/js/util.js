export function esc(s) { return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;"); }
export function prettyToken(t) { return t === "\n" ? "⏎" : t === " " ? "␣" : t === "\t" ? "⇥" : t; }
export function tokHtml(t) { return `<span class="tk">${esc(prettyToken(t))}</span>`; }
export function fmt(v, digits = 3) {
  if (v === undefined || v === null || Number.isNaN(v)) return "–";
  if (v === -Infinity) return "−∞";
  if (v === Infinity) return "∞";
  const a = Math.abs(v);
  if (a !== 0 && (a < 1e-3 || a >= 1e5)) return v.toExponential(2);
  return (v < 0 ? "−" : "") + a.toFixed(digits);
}
export function pct(p) { return p >= 0.995 ? ">99%" : p < 0.0005 ? "<0.1%" : (p * 100).toFixed(p < 0.01 ? 2 : 1) + "%"; }
export function big(n) { return n.toLocaleString("en-US"); }
export function human(n) { return n >= 1e9 ? (n / 1e9).toFixed(n >= 1e11 ? 0 : 1) + " B" : n >= 1e6 ? (n / 1e6).toFixed(1) + " M" : n >= 1e3 ? (n / 1e3).toFixed(1) + " k" : String(n); }
export const sign = (v) => (v < 0 ? "−" : "+");
