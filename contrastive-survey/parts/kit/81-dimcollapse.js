/* ch-beyond toy: dimensional collapse in a single linear layer, after Jing et al. (2110.09348) Sec. 4 and Fig. 3.
   Data x ~ N(0, I) in 3-d; each view adds Gaussian noise with variance 0.1 on axes 1 and 2 and k on axis 3.
   z = W x (3 x 3, not normalized). Loss: Jing Eq. 2 (InfoNCE with squared distances; negatives = the other
   images' first views) for 'healthy' (k = 0.1) and 'dimensional' (k = 2, larger than the data variance 1);
   'complete' trains the pull term alone (chapter 1's failure) with k = 0.1. Plain gradient descent, 300 steps.
   Bars: singular values (= eigenvalues) of the embedding covariance, Jing Eq. 1. Numbers: notes/sim-beyond.txt. */
KIT.DC = { N: 64, LR: .1, STEPS: 300, SEED: 3, KB: .1, SNAP: 5, SHOW: 160, PAIRS: 10, FLOOR: 1e-6,
  MODES: { healthy: { k: .1, loss: 'nce' }, dimensional: { k: 2, loss: 'nce' }, complete: { k: .1, loss: 'pull' } } };
KIT.dcData = k3 => {
  const { N, SEED, KB } = KIT.DC, g = PE.gauss(PE.rng(SEED)), sd = [Math.sqrt(KB), Math.sqrt(KB), Math.sqrt(k3)];
  const X = [], V1 = [], V2 = [];
  for (let n = 0; n < N; n++) { const x = [g(), g(), g()]; X.push(x); V1.push(x.map((v, d) => v + sd[d] * g())); V2.push(x.map((v, d) => v + sd[d] * g())); }
  const W = [[1, 0, 0], [0, 1, 0], [0, 0, 1]].map(r => r.map(v => 1.5 * v + .15 * g()));
  return { X, V1, V2, W };
};
KIT.dcEmb = (W, v) => [0, 1, 2].map(r => W[r][0] * v[0] + W[r][1] * v[1] + W[r][2] * v[2]);
// One plain gradient step; returns the loss before the step.
KIT.dcStep = (st, loss) => {
  const { N, LR } = KIT.DC, { V1, V2, W } = st, Z1 = V1.map(v => KIT.dcEmb(W, v)), Z2 = V2.map(v => KIT.dcEmb(W, v));
  const d1 = Z1.map(() => [0, 0, 0]), d2 = Z2.map(() => [0, 0, 0]), sq = (a, b) => (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2;
  let L = 0;
  for (let i = 0; i < N; i++) {
    if (loss === 'pull') { L += sq(Z1[i], Z2[i]) / 2; for (let c = 0; c < 3; c++) { const e = Z1[i][c] - Z2[i][c]; d1[i][c] += e; d2[i][c] -= e; } continue; }
    const lp = -sq(Z1[i], Z2[i]) / 2, ln = [];
    for (let j = 0; j < N; j++) if (j !== i) ln.push([j, -sq(Z1[i], Z1[j]) / 2]);
    const m = Math.max(lp, ...ln.map(a => a[1])); let Zs = Math.exp(lp - m); ln.forEach(a => { Zs += Math.exp(a[1] - m); });
    L += -lp + m + Math.log(Zs);
    const gp = Math.exp(lp - m) / Zs - 1;
    for (let c = 0; c < 3; c++) { const e = Z1[i][c] - Z2[i][c]; d1[i][c] -= gp * e; d2[i][c] += gp * e; }
    ln.forEach(([j, l]) => { const p = Math.exp(l - m) / Zs; for (let c = 0; c < 3; c++) { const e = Z1[i][c] - Z1[j][c]; d1[i][c] -= p * e; d1[j][c] += p * e; } });
  }
  const GW = [0, 1, 2].map(r => [0, 1, 2].map(d => { let s = 0; for (let n = 0; n < N; n++) s += d1[n][r] * V1[n][d] + d2[n][r] * V2[n][d]; return s / N; }));
  st.W = W.map((w, r) => w.map((v, d) => v - LR * GW[r][d]));
  return L / N;
};
// Singular values (= eigenvalues, C is symmetric PSD) of the 3 x 3 embedding covariance of the clean inputs (Jing Eq. 1), largest first (Jacobi sweeps).
KIT.dcSpectrum = (W, X) => {
  const Z = X.map(x => KIT.dcEmb(W, x)), n = Z.length, mu = [0, 1, 2].map(c => Z.reduce((s, z) => s + z[c], 0) / n);
  let A = [0, 1, 2].map(a => [0, 1, 2].map(b => Z.reduce((s, z) => s + (z[a] - mu[a]) * (z[b] - mu[b]), 0) / n));
  for (let it = 0; it < 60; it++) {
    let p = 0, q = 1; for (const [i, j] of [[0, 2], [1, 2]]) if (Math.abs(A[i][j]) > Math.abs(A[p][q])) { p = i; q = j; }
    if (Math.abs(A[p][q]) < 1e-18) break;
    const th = .5 * Math.atan2(2 * A[p][q], A[q][q] - A[p][p]), c = Math.cos(th), s = Math.sin(th), B = A.map(r => r.slice());
    for (let k = 0; k < 3; k++) { B[p][k] = c * A[p][k] - s * A[q][k]; B[q][k] = s * A[p][k] + c * A[q][k]; }
    const E = B.map(r => r.slice()); for (let k = 0; k < 3; k++) { E[k][p] = c * B[k][p] - s * B[k][q]; E[k][q] = s * B[k][p] + c * B[k][q]; } A = E;
  }
  return [A[0][0], A[1][1], A[2][2]].sort((a, b) => b - a);
};
// A fixed held-out set (seed 17) on which the spectrum is measured, as Jing measure on the validation set.
KIT.dcHeld = () => { const g = PE.gauss(PE.rng(17)); return Array.from({ length: KIT.DC.SHOW }, () => [g(), g(), g()]); };
KIT.dcRuns = {};
KIT.dcRun = mode => {
  if (KIT.dcRuns[mode]) return KIT.dcRuns[mode];
  const { k, loss } = KIT.DC.MODES[mode], st = KIT.dcData(k), snaps = [st.W.map(r => r.slice())];
  for (let s = 1; s <= KIT.DC.STEPS; s++) { KIT.dcStep(st, loss); if (s % KIT.DC.SNAP === 0) snaps.push(st.W.map(r => r.slice())); }
  return (KIT.dcRuns[mode] = { snaps, k, spec: KIT.dcSpectrum(st.W, KIT.dcHeld()) });
};

// Drawing: a pe3 cloud of embedded inputs plus a 2D bar chart of the spectrum on a log axis.
KIT.dimcollapse = (canvas, bars, opts = {}) => {
  const { PAIRS, FLOOR } = KIT.DC, base = KIT.dcData(.1), W0 = base.W.map(r => r.slice());
  const show = KIT.dcHeld(), g = PE.gauss(PE.rng(23)), noise = Array.from({ length: PAIRS }, () => [g(), g(), g(), g(), g(), g()]);
  const state = { mode: null, W: W0, k: .1 };
  const SC = .28, P = p => [p[0] * SC, p[2] * SC, p[1] * SC];  // embedding axis 3 drawn vertically
  const glow = opts.fig ? PE.linkPulses(opts.fig, () => cb.redraw()) : () => 0;
  const s3 = PE3.scene(canvas, { aspect: w => KIT.narrow(w) ? .95 : .6, fov: 36, dist: 6.4, yaw: .75, pitch: .2 });
  const scene = () => {
    const Wm = state.W, sk = Math.sqrt(state.k), sb = Math.sqrt(KIT.DC.KB), pairs = [], yel = [], blu = show.map(x => P(KIT.dcEmb(Wm, x)));
    noise.forEach((e, i) => { const x = show[i], v1 = [x[0] + sb * e[0], x[1] + sb * e[1], x[2] + sk * e[2]], v2 = [x[0] + sb * e[3], x[1] + sb * e[4], x[2] + sk * e[5]];
      const a = P(KIT.dcEmb(Wm, v1)), b = P(KIT.dcEmb(Wm, v2)); pairs.push({ type: 'line', pts: [a, b], color: '--green', width: 1.5, alpha: .8, link: 'aug' }); yel.push(b); blu.push(a); });
    s3.items = [
      { type: 'axes', len: 1.55, color: '--muted', labels: ['axis 1', 'axis 3 (augmented)', 'axis 2'] },
      { type: 'points', pts: blu, color: '--blue', size: 3.6, alpha: .85, link: 'cov' },
      ...pairs, { type: 'points', pts: yel, color: '--yellow', size: 3.2, link: 'aug' }];
    s3.redraw();
  };
  const drawBars = (ctx, Wd, H) => {
    if (!Wd) return;
    const G = PE.css('--green'), M = PE.css('--muted'), FG = PE.css('--fg'), RU = PE.css('--rule');
    ctx.clearRect(0, 0, Wd, H);
    const spec = KIT.dcSpectrum(state.W, show), small = KIT.narrow(Wd), gs = glow('sv'), fs = small ? 10 : 12;
    const b = { x: 46, y: 26, w: Wd - 60, h: H - 54 }, lo = Math.log10(FLOOR), hi = 1.5;
    PE.label(ctx, 'singular values of the embedding covariance (log scale)', 12, 16, gs ? FG : M, fs + gs * 2);
    [-6, -4, -2, 0].forEach(e => { const y = b.y + b.h - (e - lo) / (hi - lo) * b.h;
      ctx.fillStyle = RU; ctx.fillRect(b.x, y, b.w, 1); PE.label(ctx, '1e' + e, b.x - 6, y + 4, M, 10, 'right', 400); });
    const bw = b.w / 3;
    spec.forEach((v, i) => {
      const lv = Math.log10(Math.max(v, FLOOR)), h = (lv - lo) / (hi - lo) * b.h, dead = v < spec[0] / 20 || v < .01, x = b.x + i * bw + bw * .22;
      ctx.fillStyle = dead ? M : G; ctx.globalAlpha = .85 + .15 * gs; ctx.fillRect(x, b.y + b.h - h, bw * .56, h); ctx.globalAlpha = 1;
      PE.label(ctx, v < 1e-3 ? v.toExponential(1) : v.toFixed(v < 1 ? 3 : 2), x + bw * .28, b.y + b.h - h - 6, dead ? M : FG, Math.max(10, fs - 1), 'center', 600);
      if (dead) PE.label(ctx, 'collapsed', x + bw * .28, b.y + b.h - h - 19, M, 10, 'center', 400);
      PE.label(ctx, '\u03c3' + (i + 1), x + bw * .28, b.y + b.h + 16, M, fs, 'center', 400);
    });
  };
  const cb = PE.canvas(bars, w => KIT.narrow(w) ? .62 : .26, drawBars);
  let cancel = () => {};
  const redraw = () => { scene(); cb.redraw(); };
  const set = mode => {
    cancel(); state.mode = mode;
    if (!mode) { state.W = W0; state.k = .1; redraw(); return; }
    const run = KIT.dcRun(mode), n = run.snaps.length - 1; state.k = run.k;
    const at = u => { const s = u * n, i = Math.min(n - 1, Math.floor(s)), f = s - i, A = run.snaps[i], B = run.snaps[i + 1];
      state.W = A.map((r, a) => r.map((v, d) => v + (B[a][d] - v) * f)); redraw(); };
    if (PE.instant()) { at(1); return; }
    cancel = PE.animate(2600, at);
  };
  redraw();
  return { set, state, redraw, resize: () => { cb.resize(); s3.redraw(); }, destroy: () => { cancel(); cb.destroy(); s3.destroy(); } };
};
KIT.dimcollapseText = mode => {
  const f = v => v < 1e-3 ? v.toExponential(1) : v.toFixed(v < 1 ? 3 : 2);
  if (!mode) { const s = KIT.dcSpectrum(KIT.dcData(.1).W, KIT.dcHeld());
    return 'Before training: a random linear layer. Singular values ' + s.map(f).join(', ') + '. The cloud uses all three axes.'; }
  const r = KIT.dcRun(mode), s = r.spec;
  if (mode === 'healthy') return 'Healthy (InfoNCE, augmentation variance 0.1 on every axis), after 300 steps: singular values ' + s.map(f).join(', ') + '. All three directions are used.';
  if (mode === 'dimensional') return 'Dimensional collapse (InfoNCE, augmentation variance 2 on axis 3, data variance 1), after 300 steps: singular values ' + s.map(f).join(', ') + '. The smallest is ' + Math.round(s[1] / s[2]) + ' times below the next, yet not zero: the cloud is a flat pancake, not a point.';
  return 'Complete collapse (pull term only, no negatives), after 300 steps: singular values ' + s.map(f).join(', ') + '. Every direction shrinks toward a single point.';
};
