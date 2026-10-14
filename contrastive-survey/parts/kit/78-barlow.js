/* ch-beyond toy: Barlow Twins on a linear encoder. 256 inputs in 16-d: 8 "content" dimensions with small, unequal
   view noise and 8 "nuisance" dimensions where the view noise is larger than the signal. A shared 8 x 16 matrix W
   (rows start almost identical, so the 8 features start redundant) maps both views to 8 features; each feature is
   standardized over the batch; C = Z_A^T Z_B / N (Barlow Eq. 2); loss = sum (1 - C_ii)^2 + lambda sum C_ij^2 (Eq. 1).
   Exact gradient (checked against finite differences in build/sim/beyond.js), Adam, 600 steps.
   The paper's lambda is set for 8192 features (8191 off-diagonal entries per row); the toy has 7 per row, so it
   multiplies lambda by 8191 / 7 (our scaling). Numbers the reader sees: notes/sim-beyond.txt. */
KIT.BT = { N: 256, DI: 16, D: 8, LR: .06, STEPS: 600, SEED: 5, SNAP: 10, SCALE: 8191 / 7 };
KIT.btData = () => {
  const { N, DI, D, SEED } = KIT.BT, g = PE.gauss(PE.rng(SEED));
  const an = Array.from({ length: DI }, (_, d) => d < 8 ? .1 + .06 * d : 1.2), sv = Array.from({ length: DI }, (_, d) => d < 8 ? 1 : .5);
  const XA = new Float64Array(N * DI), XB = new Float64Array(N * DI);
  for (let n = 0; n < N; n++) { const s = sv.map(v => v * g());
    for (let d = 0; d < DI; d++) XA[n * DI + d] = s[d] + an[d] * g();
    for (let d = 0; d < DI; d++) XB[n * DI + d] = s[d] + an[d] * g(); }
  const u = Array.from({ length: DI }, () => g()), W = new Float64Array(D * DI);
  for (let i = 0; i < D; i++) for (let d = 0; d < DI; d++) W[i * DI + d] = u[d] + .35 * g();
  return { XA, XB, W, m: new Float64Array(D * DI), v: new Float64Array(D * DI), t: 0 };
};
// Standardized features H (N x D) and per-feature std for one view.
KIT.btStd = (X, W) => {
  const { N, DI, D } = KIT.BT, H = new Float64Array(N * D), sd = new Float64Array(D);
  for (let n = 0; n < N; n++) for (let i = 0; i < D; i++) { let s = 0; for (let d = 0; d < DI; d++) s += W[i * DI + d] * X[n * DI + d]; H[n * D + i] = s; }
  for (let i = 0; i < D; i++) {
    let mu = 0; for (let n = 0; n < N; n++) mu += H[n * D + i]; mu /= N;
    let va = 0; for (let n = 0; n < N; n++) { H[n * D + i] -= mu; va += H[n * D + i] ** 2; }
    sd[i] = Math.sqrt(va / N + 1e-8); for (let n = 0; n < N; n++) H[n * D + i] /= sd[i];
  }
  return { H, sd };
};
// Loss, C (D x D, row-major) and, unless noGrad, dL/dW for toy weight lam.
KIT.btLoss = (st, lam, noGrad) => {
  const { N, DI, D } = KIT.BT, A = KIT.btStd(st.XA, st.W), Bv = KIT.btStd(st.XB, st.W);
  const C = new Float64Array(D * D), G = new Float64Array(D * D); let L = 0;
  for (let i = 0; i < D; i++) for (let j = 0; j < D; j++) {
    let s = 0; for (let n = 0; n < N; n++) s += A.H[n * D + i] * Bv.H[n * D + j]; const c = s / N; C[i * D + j] = c;
    if (i === j) { L += (1 - c) ** 2; G[i * D + j] = -2 * (1 - c); } else { L += lam * c * c; G[i * D + j] = 2 * lam * c; }
  }
  if (noGrad) return { L, C };
  const back = (S, other, transpose) => {
    const dH = new Float64Array(N * D);
    for (let n = 0; n < N; n++) for (let i = 0; i < D; i++) { let s = 0;
      for (let j = 0; j < D; j++) s += (transpose ? G[j * D + i] : G[i * D + j]) * other[n * D + j]; dH[n * D + i] = s / N; }
    for (let i = 0; i < D; i++) { let md = 0, mh = 0;
      for (let n = 0; n < N; n++) { md += dH[n * D + i]; mh += dH[n * D + i] * S.H[n * D + i]; } md /= N; mh /= N;
      for (let n = 0; n < N; n++) dH[n * D + i] = (dH[n * D + i] - md - S.H[n * D + i] * mh) / S.sd[i]; }
    return dH;
  };
  const dA = back(A, Bv.H, false), dB = back(Bv, A.H, true), GW = new Float64Array(D * DI);
  for (let i = 0; i < D; i++) for (let d = 0; d < DI; d++) { let s = 0;
    for (let n = 0; n < N; n++) s += dA[n * D + i] * st.XA[n * DI + d] + dB[n * D + i] * st.XB[n * DI + d]; GW[i * DI + d] = s; }
  return { L, C, GW };
};
KIT.btStep = (st, lam) => {
  const r = KIT.btLoss(st, lam), b1 = .9, b2 = .999, k = ++st.t;
  for (let x = 0; x < st.W.length; x++) { const g = r.GW[x];
    st.m[x] = b1 * st.m[x] + (1 - b1) * g; st.v[x] = b2 * st.v[x] + (1 - b2) * g * g;
    st.W[x] -= KIT.BT.LR * (st.m[x] / (1 - b1 ** k)) / (Math.sqrt(st.v[x] / (1 - b2 ** k)) + 1e-8); }
  return r;
};
KIT.btSummary = C => { const D = KIT.BT.D; let dg = 0, od = 0;
  for (let i = 0; i < D; i++) for (let j = 0; j < D; j++) if (i === j) dg += C[i * D + j] / D; else od += C[i * D + j] ** 2 / (D * D - D);
  return { diag: dg, off: Math.sqrt(od) }; };
// Full run for a paper-scale lambda: snapshots of C every SNAP steps plus the final loss. Cached: deterministic.
KIT.btRuns = {};
KIT.btRun = lamPaper => {
  if (KIT.btRuns[lamPaper]) return KIT.btRuns[lamPaper];
  const st = KIT.btData(), lam = lamPaper * KIT.BT.SCALE, snaps = [KIT.btLoss(st, lam, true).C];
  for (let s = 1; s <= KIT.BT.STEPS; s++) { KIT.btStep(st, lam); if (s % KIT.BT.SNAP === 0) snaps.push(KIT.btLoss(st, lam, true).C); }
  const fin = KIT.btLoss(st, lam, true);
  return (KIT.btRuns[lamPaper] = { snaps, L: fin.L, ...KIT.btSummary(fin.C) });
};

// Drawing: the 8 x 8 cross-correlation matrix. Rows = view A features (blue), columns = view B features (yellow).
// Diagonal cells fill green as C_ii -> 1; off-diagonal cells fill red with |C_ij|.
KIT.barlow = (canvas, opts = {}) => {
  const D = KIT.BT.D, init = KIT.btLoss(KIT.btData(), 0, true).C;
  const state = { lam: null, C: init, step: 0 };
  const glow = opts.fig ? PE.linkPulses(opts.fig, () => cv.redraw()) : () => 0;
  let cancel = () => {};
  const draw = (ctx, W, H) => {
    if (!W) return;
    const B = PE.css('--blue'), Y = PE.css('--yellow'), G = PE.css('--green'), R = PE.css('--red'), M = PE.css('--muted'), FG = PE.css('--fg'), BG = PE.css('--bg'), RU = PE.css('--rule');
    ctx.clearRect(0, 0, W, H);
    const small = KIT.narrow(W), fs = small ? 10 : 12;
    const side = small ? Math.min(W - 60, H * .62) : Math.min(H - 70, W * .52), cs = side / D, x0 = small ? 40 : 64, y0 = small ? 46 : 52;
    const gd = glow('diag'), go = glow('offdiag'), ga = glow('za'), gb = glow('zb'), gc = glow('cell'), gl = glow('lambda');
    PE.label(ctx, 'view B feature j', x0 + side / 2, y0 - 22, gb ? FG : Y, fs + 1 + 2 * gb, 'center');
    ctx.save(); ctx.translate(x0 - 26, y0 + side / 2); ctx.rotate(-Math.PI / 2); PE.label(ctx, 'view A feature i', 0, 0, ga ? FG : B, fs + 1 + 2 * ga, 'center'); ctx.restore();
    for (let i = 0; i < D; i++) {
      PE.label(ctx, String(i + 1), x0 - 8, y0 + (i + .5) * cs + 4, B, Math.max(10, fs - 1), 'right', 400);
      PE.label(ctx, String(i + 1), x0 + (i + .5) * cs, y0 - 6, Y, Math.max(10, fs - 1), 'center', 400);
      for (let j = 0; j < D; j++) {
        const c = state.C[i * D + j], dia = i === j, u = PE.clamp(Math.abs(c), 0, 1), g = dia ? gd : go;
        ctx.fillStyle = PE.mix(BG, dia ? G : R, Math.min(1, u * .95 + .35 * g));
        ctx.fillRect(x0 + j * cs + 1, y0 + i * cs + 1, cs - 2, cs - 2);
        if (cs > 34) PE.label(ctx, Math.abs(c) < .005 ? '0.00' : c.toFixed(2), x0 + (j + .5) * cs, y0 + (i + .5) * cs + 4, u > .55 ? BG : FG, 11, 'center', 400);
      }
    }
    if (gc) PE.ring(ctx, x0 + 1.5 * cs, y0 + .5 * cs, cs * .62, FG, 2.5, gc);
    ctx.save(); ctx.strokeStyle = RU; ctx.strokeRect(x0 + .5, y0 + .5, side, side); ctx.restore();
    // Side panel: the two summary gauges and lambda.
    const s = KIT.btSummary(state.C), px = small ? 16 : x0 + side + 40, py = small ? y0 + side + 30 : y0 + 10, pw = small ? W - 32 : W - px - 20;
    const gauge = (y, label, v, col, g, target) => {
      PE.label(ctx, label, px, y, g ? FG : col, fs + 1 + g * 2);
      ctx.fillStyle = RU; ctx.fillRect(px, y + 8, pw, 10);
      ctx.fillStyle = col; ctx.fillRect(px, y + 8, pw * PE.clamp(v, 0, 1), 10);
      PE.label(ctx, v.toFixed(3) + '  (goal ' + target + ')', px, y + 32, FG, fs, 'left', 400);
    };
    gauge(py, 'mean diagonal C_ii', s.diag, G, gd, '1');
    gauge(py + (small ? 58 : 64), 'off-diagonal size (RMS)', s.off, R, go, '0');
    const ly = py + (small ? 114 : 136);
    const lt = state.lam === null ? 'not trained yet' : state.lam === 0 ? 'lambda = 0 (no redundancy term)' : 'lambda = ' + state.lam.toFixed(3) + ' (paper scale)';
    PE.label(ctx, lt, px, ly, gl ? FG : M, fs + 1 + 2 * gl);
    if (state.lam) PE.label(ctx, 'toy uses ' + (state.lam * KIT.BT.SCALE).toFixed(2) + ' = lambda x 8191 / 7', px, ly + 17, M, fs, 'left', 400);
    if (state.step) PE.label(ctx, 'step ' + state.step + ' of ' + KIT.BT.STEPS, px, ly + 34, M, fs, 'left', 400);
  };
  const cv = PE.canvas(canvas, w => KIT.narrow(w) ? 1.6 : .56, draw);
  const set = lam => {
    cancel(); state.lam = lam; state.step = 0;
    if (lam === null) { state.C = init; cv.redraw(); return; }
    const run = KIT.btRun(lam), n = run.snaps.length - 1;
    const at = u => { const s = u * n, i = Math.min(n - 1, Math.floor(s)), f = s - i, A = run.snaps[i], Bc = run.snaps[i + 1];
      state.C = A.map((v, x) => v + (Bc[x] - v) * f); state.step = Math.round(u * KIT.BT.STEPS); cv.redraw(); };
    if (PE.instant()) { at(1); return; }
    cancel = PE.animate(2800, at);
  };
  return { set, state, redraw: cv.redraw, resize: cv.resize, destroy: () => { cancel(); cv.destroy(); } };
};
KIT.barlowText = lam => {
  if (lam === null) { const s = KIT.btSummary(KIT.btLoss(KIT.btData(), 0, true).C);
    return 'Before training: the 8 features are near-copies of one mix of the inputs. Mean diagonal ' + s.diag.toFixed(2) + ', off-diagonal size ' + s.off.toFixed(2) + '.'; }
  const r = KIT.btRun(lam), f = x => x.toFixed(3);
  return lam === 0
    ? 'lambda = 0, after 600 steps: mean diagonal ' + f(r.diag) + ', off-diagonal size ' + f(r.off) + '. Each feature is invariant, but most of them chase the same input directions: redundant.'
    : 'lambda = ' + lam.toFixed(3) + ' (toy ' + (lam * KIT.BT.SCALE).toFixed(2) + '), after 600 steps: mean diagonal ' + f(r.diag) + ', off-diagonal size ' + f(r.off) + '. C is close to the identity: invariant and decorrelated.';
};
