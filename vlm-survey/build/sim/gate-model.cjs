// Toy gated xattn-dense block in 2D; same code as parts/viz-gate.js model().
const words = ['dog', 'cat', 'car', 'beach'];
const U = [[0.95, 0.31], [0.2, 0.98], [-0.9, 0.44], [-0.1, -1]];   // LM-head directions
const y0 = [0.25, 0.55];                                            // text token "a" after the frozen layers so far
const X = [[1.0, 0.1], [0.8, -0.35], [0.9, 0.4]];                  // three visual tokens (resampler outputs)
const Wv0 = [[-1.2, 0.3], [-0.4, -1.0]], Wv1 = [[1.1, 0.2], [-0.5, 0.3]];   // value maps: random init / toy trained
const F0 = [[-0.8, 0.5], [0.6, -0.9]], F1 = [[0.7, -0.2], [0.2, -0.6]];    // FFW weights (2 hidden, sq-ReLU)
const lerpM = (A, B, u) => A.map((r, i) => r.map((v, j) => v + (B[i][j] - v) * u));
const mv = (M, v) => [M[0][0] * v[0] + M[0][1] * v[1], M[1][0] * v[0] + M[1][1] * v[1]];
const add = (a, b, s = 1) => [a[0] + s * b[0], a[1] + s * b[1]];
const sm = v => { const m = Math.max(...v), e = v.map(x => Math.exp(x - m)), Z = e.reduce((s, x) => s + x, 0); return e.map(x => x / Z); };
function block(p, gx, gd) {
  const w = 1 - Math.exp(-5 * p), Wv = lerpM(Wv0, Wv1, w), Fw = lerpM(F0, F1, w);
  const att = sm(X.map(x => 2 * (x[0] * y0[0] + x[1] * y0[1])));
  const A = X.map(x => mv(Wv, x)).reduce((s, v, i) => add(s, v, att[i]), [0, 0]);
  const y1 = add(y0, A, gx);
  const h = mv(Fw, y1).map(z => Math.max(0, z) ** 2);
  const Fv = [h[0] - 0.3 * h[1], 0.4 * h[0] + h[1]];
  const y2 = add(y1, Fv, gd);
  const g = y => [0.25 * Math.tanh(y[1]), 0.25 * Math.tanh(-y[0] + y[1])];
  const y3 = add(y2, g(y2));
  const pr = sm(U.map(u => 3 * (u[0] * y3[0] + u[1] * y3[1])));
  return { A, Fv, y1, y2, y3, gF: g(y2), pr };
}
const ax = p => 1.4 * (1 - Math.exp(-6 * p)), ad = p => 0.9 * (1 - Math.exp(-4 * p));
const ref = block(0, 0, 0);
const kl = (a, b) => a.reduce((s, x, i) => s + x * Math.log(x / b[i]), 0);
module.exports = { block, ax, ad, ref, kl, words };
if (require.main === module) {
  const f = pr => pr.map((x, i) => words[i] + ' ' + (100 * x).toFixed(0)).join(', ');
  console.log('frozen      ', f(ref.pr));
  for (const p of [0, .05, .1, .25, .5, 1]) {
    const r = block(p, Math.tanh(ax(p)), Math.tanh(ad(p)));
    console.log('p=' + p, 'ax', ax(p).toFixed(2), 'ad', ad(p).toFixed(2), f(r.pr), 'KL', kl(r.pr, ref.pr).toFixed(4), 'dist', Math.hypot(r.y3[0] - ref.y3[0], r.y3[1] - ref.y3[1]).toFixed(3));
  }
  const ng = block(0, 1, 1); console.log('no gate p=0', f(ng.pr), 'KL', kl(ng.pr, ref.pr).toFixed(3));
  const ng1 = block(1, 1, 1); console.log('no gate p=1', f(ng1.pr));
}
