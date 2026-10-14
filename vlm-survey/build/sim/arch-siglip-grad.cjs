// Per-logit gradients of CLIP's softmax loss and SigLIP's sigmoid loss on a 5x5 toy table, before and after one wrong cell rises.
const N = 5, T = 10, Bb = -10;
const base = [
  [.80, .10, -.20, .05, -.10],
  [.15, .80, .00, -.25, .10],
  [-.10, .05, .80, .20, -.15],
  [.00, -.20, .10, .80, .05],
  [-.05, .15, -.10, .00, .80]];
const sig = u => 1 / (1 + Math.exp(-u));
function soft(c) { const L = c.map(r => r.map(v => T * v));
  const p = L.map(r => { const m = Math.max(...r), e = r.map(v => Math.exp(v - m)), s = e.reduce((a, b) => a + b); return e.map(v => v / s); });
  const q = L.map(r => r.slice()); for (let j = 0; j < N; j++) { const col = L.map(r => r[j]), m = Math.max(...col), e = col.map(v => Math.exp(v - m)), s = e.reduce((a, b) => a + b); for (let i = 0; i < N; i++) q[i][j] = e[i] / s; }
  return L.map((r, i) => r.map((_, j) => ((p[i][j] + q[i][j]) / 2 - (i === j)) / N)); }
function sg(c) { return c.map((r, i) => r.map((v, j) => (sig(T * v + Bb) - (i === j)) / N)); }
const after = base.map(r => r.slice()); after[0][1] = .85;
const pr = (n, G) => { console.log(n); G.forEach(r => console.log(r.map(v => v.toFixed(4).padStart(8)).join(''))); };
for (const [n, f] of [['softmax', soft], ['sigmoid', sg]]) {
  const A = f(base), Bm = f(after); pr(n + ' before', A); pr(n + ' after', Bm);
  let ch = 0; for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) if (Math.abs(A[i][j] - Bm[i][j]) > 1e-3) ch++;
  console.log(n, 'cells whose gradient changed by > 0.001:', ch);
}
