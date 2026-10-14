// Gradient of a next-word loss through one toy gated xattn-dense block (same forward as gate-model.cjs / viz-gate.js),
// per parameter group, by central finite differences. Checks: at alpha = 0 the grads into Wv, FFW and the visual tokens X are 0,
// the grads into alpha are not, and the grad passed down to y0 equals the frozen-LM-only grad.
const U = [[0.95, 0.31], [0.2, 0.98], [-0.9, 0.44], [-0.1, -1]];
const y0 = [0.25, 0.55];
const X0 = [[1.0, 0.1], [0.8, -0.35], [0.9, 0.4]];
const Wv0 = [[-1.2, 0.3], [-0.4, -1.0]], F0 = [[0.9, 0.8], [-0.6, 0.9]];
const mv = (M, v) => [M[0][0] * v[0] + M[0][1] * v[1], M[1][0] * v[0] + M[1][1] * v[1]];
const add = (a, b, s = 1) => [a[0] + s * b[0], a[1] + s * b[1]];
const sm = v => { const m = Math.max(...v), e = v.map(x => Math.exp(x - m)), Z = e.reduce((s, x) => s + x, 0); return e.map(x => x / Z); };
function loss(P) {
  const { ax, ad, Wv, F, X, y } = P;
  const att = sm(X.map(x => 2 * (x[0] * y[0] + x[1] * y[1])));
  const A = X.map(x => mv(Wv, x)).reduce((s, v, i) => add(s, v, att[i]), [0, 0]);
  const y1 = add(y, A, Math.tanh(ax));
  const h = mv(F, y1).map(z => Math.max(0, z) ** 2);
  const y2 = add(y1, [h[0] - 0.3 * h[1], 0.4 * h[0] + h[1]], Math.tanh(ad));
  const y3 = add(y2, [0.25 * Math.tanh(y2[1]), 0.25 * Math.tanh(-y2[0] + y2[1])]);
  const pr = sm(U.map(u => 3 * (u[0] * y3[0] + u[1] * y3[1])));
  return -Math.log(pr[0]);
}
const clone = P => JSON.parse(JSON.stringify(P));
function grads(P) {
  const e = 1e-6, out = {};
  const paths = { ax: [['ax']], ad: [['ad']], Wv: [], F: [], X: [], y: [['y', 0], ['y', 1]] };
  for (let i = 0; i < 2; i++) for (let j = 0; j < 2; j++) { paths.Wv.push(['Wv', i, j]); paths.F.push(['F', i, j]); }
  for (let i = 0; i < 3; i++) for (let j = 0; j < 2; j++) paths.X.push(['X', i, j]);
  const get = (o, p) => p.slice(0, -1).reduce((a, k) => a[k], o);
  for (const [g, list] of Object.entries(paths)) {
    out[g] = list.map(p => { const a = clone(P), b = clone(P); get(a, p)[p.at(-1)] += e; get(b, p)[p.at(-1)] -= e; return (loss(a) - loss(b)) / (2 * e); });
  }
  return out;
}
const norm = v => Math.hypot(...v);
const base = a => ({ ax: a, ad: a, Wv: Wv0, F: F0, X: X0, y: y0 });
module.exports = { loss, grads, base, norm };
if (require.main === module) {
  for (const a of [0, 0.05, 0.1, 0.25, 0.5, 1, 2]) {
    const g = grads(base(a));
    console.log('alpha', a, 'L', loss(base(a)).toFixed(4), Object.entries(g).map(([k, v]) => k + ' ' + norm(v).toFixed(4)).join('  '));
  }
  const g0 = grads(base(0)); console.log('dL/dalpha at 0:', g0.ax[0].toFixed(5), g0.ad[0].toFixed(5));
  // zero-initialised Wv as well: gate gradient on xattn should vanish
  const z = base(0); z.Wv = [[0, 0], [0, 0]]; const gz = grads(z);
  console.log('Wv=0 & alpha=0:', Object.entries(gz).map(([k, v]) => k + ' ' + norm(v).toFixed(5)).join('  '));
  // plain gradient descent from alpha=0, lr .5: which group moves first?
  let P = base(0); P.Wv = clone(Wv0); P.F = clone(F0); P.X = clone(X0);
  for (let s = 0; s <= 30; s++) {
    const g = grads(P);
    if (s < 5 || s % 5 === 0) console.log('step', s, 'L', loss(P).toFixed(4), 'ax', P.ax.toFixed(3), 'ad', P.ad.toFixed(3), '|gWv|', norm(g.Wv).toFixed(4), '|gF|', norm(g.F).toFixed(4), '|gax|', Math.abs(g.ax[0]).toFixed(4));
    const lr = .5; P.ax -= lr * g.ax[0]; P.ad -= lr * g.ad[0];
    for (let i = 0; i < 2; i++) for (let j = 0; j < 2; j++) { P.Wv[i][j] -= lr * g.Wv[i * 2 + j]; P.F[i][j] -= lr * g.F[i * 2 + j]; }
  }
}
if (require.main === module) {
  // gradient into y with the new layers removed entirely (frozen layer + head only)
  const frozenOnly = y => { const y3 = add(y, [0.25 * Math.tanh(y[1]), 0.25 * Math.tanh(-y[0] + y[1])]); const pr = sm(U.map(u => 3 * (u[0] * y3[0] + u[1] * y3[1]))); return -Math.log(pr[0]); };
  const e = 1e-6, gy = [0, 1].map(k => { const a = [...y0], b = [...y0]; a[k] += e; b[k] -= e; return (frozenOnly(a) - frozenOnly(b)) / (2 * e); });
  console.log('frozen-only dL/dy', gy.map(v => v.toFixed(5)), ' block at alpha 0:', grads(base(0)).y.map(v => v.toFixed(5)));
  for (const a of [0, .25, .5, 1, 1.5]) { const g = grads(base(a)); console.log('a', a, 'L', loss(base(a)).toFixed(3), 'gax', g.ax[0].toFixed(3), 'gad', g.ad[0].toFixed(3), 'Wv', norm(g.Wv).toFixed(3), 'F', norm(g.F).toFixed(3), 'X', norm(g.X).toFixed(3), 'y', norm(g.y).toFixed(3)); }
}
