// Colour scales. Signed values (weights, residual stream, logits…) use a
// diverging red ← neutral → blue ramp; probabilities (attention weights,
// softmax outputs) use a single-hue sequential blue ramp. Same palette as the
// main "Inside the Transformer" app so both read as one system.

function hex2rgb(h) { return [parseInt(h.slice(1, 3), 16) / 255, parseInt(h.slice(3, 5), 16) / 255, parseInt(h.slice(5, 7), 16) / 255]; }
function ramp(stops) {
  const rgb = stops.map(hex2rgb);
  return (t) => {
    t = Math.max(0, Math.min(1, t));
    const x = t * (rgb.length - 1), i = Math.min(Math.floor(x), rgb.length - 2), f = x - i;
    return [rgb[i][0] + (rgb[i + 1][0] - rgb[i][0]) * f, rgb[i][1] + (rgb[i + 1][1] - rgb[i][1]) * f, rgb[i][2] + (rgb[i + 1][2] - rgb[i][2]) * f];
  };
}
const NEG = ramp(["#2b2b29", "#8a3b3a", "#c95352", "#f08a7a"]);
const POS = ramp(["#2b2b29", "#20507f", "#3987e5", "#9cc5f5"]);
const SEQ = ramp(["#15181d", "#133a6b", "#2566b8", "#3987e5", "#86b6ef", "#dbeafd"]);

export const MASK_RGB = [0.07, 0.07, 0.08];
export const PLATE = { act: "#3b2d14", weight: "#15283d", attn: "#2a1f3d", prob: "#123a2c" };

// t ∈ [-1, 1] → rgb floats
export function diverging(t) { return t < 0 ? NEG(Math.min(1, -t)) : POS(Math.min(1, t)); }
// t ∈ [0, 1] → rgb floats
export function sequential(t) { return SEQ(t); }

export function rgbCss(c) { return `rgb(${Math.round(c[0] * 255)},${Math.round(c[1] * 255)},${Math.round(c[2] * 255)})`; }

// Robust scale for a block: the 98th percentile of |v| (sampled), so one
// outlier doesn't wash out the rest of the matrix.
export function robustMax(data, n = 4096) {
  const len = data.length;
  if (len === 0) return 1;
  const step = Math.max(1, Math.floor(len / n));
  const vals = [];
  for (let i = 0; i < len; i += step) { const v = Math.abs(data[i]); if (Number.isFinite(v)) vals.push(v); }
  vals.sort((a, b) => a - b);
  const p = vals[Math.min(vals.length - 1, Math.floor(vals.length * 0.98))];
  return p > 0 ? p : (vals[vals.length - 1] || 1);
}
