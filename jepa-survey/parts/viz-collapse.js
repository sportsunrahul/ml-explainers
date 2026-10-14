<script>
(() => {
  // Deep linear self-distillation toy (Littwin et al. 2024, Sec. 2): encoder W = W^L...W^1, predictor V,
  // loss (1/2n) sum ||V W x - W_tgt y||^2 with the target branch blocked. Data: three axes with
  // (gamma, noise std) = slow (0.99, 0.1), noisy (0, 1), mixed (0.6, 0.6); the "fixed background" preset
  // swaps the mixed axis for a constant one (1, 0). Toggles beyond stop-grad + predictor are ours.
  const fig = document.getElementById('viz-collapse'), el = document.getElementById('collapse-canvas'), out = document.getElementById('collapse-readout');
  const d = 3, n = 256, eta = .1, eps = .3, STEPS = 600;  // eta .05 with init .05 sits on the saddle for over 2000 steps; nothing visible happens
  const S = { sg: 1, pred: 1, ema: 0, L: 2, bg: 0, step: 0 };
  let W, V, Wt, X, Y, busy = false, stopRun = () => {};
  const axes = () => S.bg ? [[.99, .1], [0, 1], [1, 0]] : [[.99, .1], [0, 1], [.6, .6]];
  const rho = () => axes().map(([g, s]) => g / (1 + s * s));
  function data(seed) { const r = PE.rng(seed), gs = PE.gauss(r), A = axes(); X = []; Y = [];
    for (let k = 0; k < n; k++) { const x = [], y = []; for (let i = 0; i < d; i++) { const [g, s] = A[i], u = gs(), u2 = g * u + Math.sqrt(1 - g * g) * gs(); x.push(u + s * gs()); y.push(u2 + s * gs()); } X.push(x); Y.push(y); } }
  const eye = () => [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
  const mul = (A, B) => A.map(r => B[0].map((_, j) => r.reduce((s, v, k) => s + v * B[k][j], 0)));
  const mv = (A, v) => A.map(r => r.reduce((s, a, k) => s + a * v[k], 0));
  const T = A => A[0].map((_, j) => A.map(r => r[j]));
  function init(seed) { const r = PE.rng(seed + 11); const orth = () => { const a = r() * 6.28, c = Math.cos(a), s = Math.sin(a); return [[c, -s, 0], [s, c, 0], [0, 0, 1]]; };
    W = []; for (let a = 0; a < S.L; a++) W.push(orth().map(row => row.map(v => v * Math.pow(eps, 1 / S.L) + (r() - .5) * .01)));
    V = orth().map(row => row.map(v => v * Math.pow(eps, 1 / S.L))); Wt = W.map(m => m.map(r => r.slice())); S.step = 0; }
  const prod = Ws => Ws.reduce((P, m) => mul(m, P), eye());
  function stepOnce() {
    const Wb = prod(W), Wtb = S.ema ? prod(Wt) : Wb, Vm = S.pred ? V : eye();
    const GV = [[0, 0, 0], [0, 0, 0], [0, 0, 0]], GW = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
    for (let k = 0; k < n; k++) { const h = mv(Wb, X[k]), p = mv(Vm, h), t = mv(Wtb, Y[k]), e = p.map((v, i) => v - t[i]);
      const Vte = mv(T(Vm), e);
      for (let i = 0; i < d; i++) for (let j = 0; j < d; j++) { GV[i][j] += e[i] * h[j] / n; GW[i][j] += (Vte[i] * X[k][j] - (S.sg ? 0 : e[i] * Y[k][j])) / n; } }
    for (let a = 0; a < S.L; a++) { const left = prod(W.slice(a + 1)), right = prod(W.slice(0, a)); const G = mul(mul(T(left), GW), T(right));
      W[a] = W[a].map((row, i) => row.map((v, j) => v - eta * G[i][j])); }
    if (S.pred) V = V.map((row, i) => row.map((v, j) => v - eta * GV[i][j]));
    if (S.ema) Wt = Wt.map((m, a) => m.map((row, i) => row.map((v, j) => .99 * v + .01 * W[a][i][j])));
    S.step++;
  }
  const s3 = PE3.scene(el, { aspect: w => w < 560 ? .9 : .56, fov: 36, dist: 6, yaw: .7, pitch: .4 });
  function scene() {
    const Wb = prod(W), pts = X.map(x => mv(Wb, x)), wn = [0, 1, 2].map(i => Math.hypot(...Wb.map(r => r[i])));
    const items = [{ type: 'axes', len: 1.3, color: '--muted', labels: ['slow', 'noisy', S.bg ? 'fixed' : 'mixed'] },
      { type: 'points', pts: pts.map(p => [p[0], p[2], p[1]]), color: '--blue', size: 4.5, alpha: .85, link: 'cloud' },
      { type: 'line', pts: [[0, 0, 0], [wn[0] * 1.2, 0, 0]], color: '--yellow', width: 3, link: 'slow' },
      { type: 'line', pts: [[0, 0, 0], [0, 0, wn[1] * 1.2]], color: '--red', width: 3, link: 'noisy' }];
    s3.items = items; s3.redraw();
    const rr = rho(), pred = rr.map(v => Math.pow(v, S.L));
    out.textContent = `Step ${S.step}. Encoder weight per axis: slow ${wn[0].toFixed(2)} (theory ${pred[0].toFixed(2)}), noisy ${wn[1].toFixed(2)} (theory ${pred[1].toFixed(2)}), ${S.bg ? 'fixed background' : 'mixed'} ${wn[2].toFixed(2)} (theory ${pred[2].toFixed(2)}).` +
      (S.step && wn.every(v => v < .05) ? ' Collapsed: every input maps to the same point.' : '') + (!S.sg || !S.pred ? ' (Theory values assume stop-gradient and a predictor.)' : '');
  }
  function reset() { data(3); init(3); scene(); }
  // Trains up to STEPS in total, so a second press (or a cue replay) is a no-op rather than 400 more steps.
  function run() { stopRun(); if (PE.instant()) { while (S.step < STEPS) stepOnce(); scene(); return; }
    const tick = () => { for (let i = 0; i < 12 && S.step < STEPS; i++) stepOnce(); scene(); if (S.step < STEPS) id = requestAnimationFrame(tick); }; let id = requestAnimationFrame(tick); stopRun = () => cancelAnimationFrame(id); }
  fig.querySelectorAll('[data-key]').forEach(b => b.addEventListener('click', () => { const k = b.dataset.key;
    if (k === 'bg') { S.bg = 1; b.setAttribute('aria-pressed', 'true'); }  // sets a state (reset clears it) so a cue replay is idempotent else { S[k] = +b.dataset.val; fig.querySelectorAll(`[data-key="${k}"]`).forEach(x => x.setAttribute('aria-pressed', String(x === b))); }
    reset(); }));
  document.getElementById('btn-collapse-run').addEventListener('click', run);
  document.getElementById('btn-collapse-reset').addEventListener('click', () => { stopRun(); Object.assign(S, { sg: 1, pred: 1, ema: 0, L: 2, bg: 0 });
    fig.querySelectorAll('[data-key]').forEach(x => { const k = x.dataset.key; x.setAttribute('aria-pressed', String(k === 'bg' ? false : +x.dataset.val === S[k])); }); reset(); });
  reset();
})();
</script>
