/* viz-gptq: GPTQ Algorithm 1 (Frantar et al., 2210.17323v2) run exactly on a toy 8 x 16 weight block W with 64 toy
   calibration inputs X (16 x 64; channels mix four shared signals, so H is far from diagonal). H = 2 X X^T + lambda I with
   lambda = 1% of mean diag(H); U = upper Cholesky factor of H^-1 (H^-1 = U^T U); err = (w - q(w)) / U_jj; W[:, j:] -= err U[j, j:].
   Grid: per-row asymmetric min-max, fixed before the loop (our reading). Blocks of B = 4 columns instead of 128; the
   lazy batch update does not change the result (Step 2), and the toy gives identical codes for B = 1, 4 and 16. */
KIT.GPTQ = { R: 8, C: 16, M: 64, B: 4, steps: [0, 1, 2, 4, 8, 16] };
KIT.gptqData = (() => {
  let cache = null;
  return () => {
    if (cache) return cache;
    const { R, C, M } = KIT.GPTQ, r = PE.rng(21), g = PE.gauss(r);
    const L = 4, mixw = Array.from({ length: C }, () => Array.from({ length: L }, () => g())), sc = Array.from({ length: C }, () => .5 + 1.5 * r());
    const X = Array.from({ length: C }, () => new Array(M).fill(0)), Z = Array.from({ length: L }, () => Array.from({ length: M }, () => g()));
    for (let i = 0; i < C; i++) for (let m = 0; m < M; m++) { let v = .35 * g(); for (let l = 0; l < L; l++) v += mixw[i][l] * Z[l][m]; X[i][m] = sc[i] * v / 2; }
    const W = Array.from({ length: R }, () => Array.from({ length: C }, () => g() * .5));
    cache = { X, W }; return cache;
  };
})();
KIT.gptqGrid = (row, b) => { const lo = Math.min(0, ...row), hi = Math.max(0, ...row), n = (1 << b) - 1, S = (hi - lo) / n || 1; return { S, Z: Math.round(-lo / S), n }; };
KIT.gptqQ = (v, G) => G.S * (Math.max(0, Math.min(G.n, Math.round(v / G.S) + G.Z)) - G.Z);
KIT.gptqU = H => {
  const n = H.length, A = H.map(r => r.slice()), I = H.map((r, i) => r.map((_, j) => +(i === j)));
  for (let c = 0; c < n; c++) {
    let p = c; for (let i = c + 1; i < n; i++) if (Math.abs(A[i][c]) > Math.abs(A[p][c])) p = i;
    [A[c], A[p]] = [A[p], A[c]]; [I[c], I[p]] = [I[p], I[c]];
    const d = A[c][c]; for (let j = 0; j < n; j++) { A[c][j] /= d; I[c][j] /= d; }
    for (let i = 0; i < n; i++) if (i !== c) { const f = A[i][c]; if (f) for (let j = 0; j < n; j++) { A[i][j] -= f * A[c][j]; I[i][j] -= f * I[c][j]; } }
  }
  const Lo = I.map(r => r.map(() => 0));
  for (let i = 0; i < n; i++) for (let j = 0; j <= i; j++) {
    let s = I[i][j]; for (let k = 0; k < j; k++) s -= Lo[i][k] * Lo[j][k];
    Lo[i][j] = i === j ? Math.sqrt(s) : s / Lo[j][j];
  }
  return { U: Lo[0].map((_, i) => Lo.map(r => r[i])), Hinv: I };
};
KIT.gptqRun = (() => {
  const cache = {};
  return (bits, B) => {
    B = B || KIT.GPTQ.B; const key = bits + ':' + B; if (cache[key]) return cache[key];
    const { R, C, M } = KIT.GPTQ, { X, W: W0 } = KIT.gptqData();
    const H = Array.from({ length: C }, (_, i) => Array.from({ length: C }, (_, j) => { let s = 0; for (let m = 0; m < M; m++) s += X[i][m] * X[j][m]; return 2 * s; }));
    const lam = .01 * H.reduce((s, r, i) => s + r[i], 0) / C;
    H.forEach((r, i) => { r[i] += lam; });
    const { U } = KIT.gptqU(H), grids = W0.map(row => KIT.gptqGrid(row, bits));
    const W = W0.map(r => r.slice()), Q = W0.map(r => r.map(() => 0)), snaps = [W.map(r => r.slice())];
    for (let i = 0; i < C; i += B) {
      const E = W.map(() => new Array(B).fill(0));
      for (let j = i; j < i + B; j++) {
        for (let r = 0; r < R; r++) { Q[r][j] = KIT.gptqQ(W[r][j], grids[r]); E[r][j - i] = (W[r][j] - Q[r][j]) / U[j][j]; }
        for (let r = 0; r < R; r++) for (let k = j; k < i + B; k++) W[r][k] -= E[r][j - i] * U[j][k];
        W.forEach((row, r) => { row[j] = Q[r][j]; });
        // Snapshot with the pending lazy updates applied, so every step shows where the remaining weights end up.
        const S = W.map(r => r.slice());
        for (let r = 0; r < R; r++) for (let jj = i; jj <= j; jj++) for (let k = i + B; k < C; k++) S[r][k] -= E[r][jj - i] * U[jj][k];
        snaps.push(S);
      }
      for (let r = 0; r < R; r++) for (let jj = 0; jj < B; jj++) for (let k = i + B; k < C; k++) W[r][k] -= E[r][jj] * U[i + jj][k];
    }
    const rtn = W0.map((row, r) => row.map(v => KIT.gptqQ(v, grids[r])));
    const outErr = Wq => { let e = 0; for (let r = 0; r < R; r++) for (let m = 0; m < M; m++) { let d = 0; for (let c = 0; c < C; c++) d += (W0[r][c] - Wq[r][c]) * X[c][m]; e += d * d; } return e; };
    const rtnSnaps = snaps.map((_, j) => W0.map((row, r) => row.map((v, k) => k < j ? rtn[r][k] : v)));
    const res = { bits, W0, Q, snaps, rtnSnaps, U, lam, grids, rtn, errG: snaps.map(outErr), errR: rtnSnaps.map(outErr) };
    cache[key] = res; return res;
  };
})();
KIT.gptq = (el, opts = {}) => {
  const { R, C } = KIT.GPTQ;
  const S = { bits: 4, method: 'gptq', cols: 0, u: 1, from: null, fromCols: 0 };
  const glow = opts.fig ? PE.linkPulses(opts.fig, () => cv.redraw()) : () => 0;
  const run = () => KIT.gptqRun(S.bits);
  const mat = (method, cols) => (method === 'gptq' ? run().snaps : run().rtnSnaps)[cols];
  function draw(ctx, w, h) {
    ctx.clearRect(0, 0, w, h);
    const narrow = w < 560, cs = n => PE.css(n), pad = opts.bottomPad || 0;
    const P = { fg: cs('--fg'), mut: cs('--muted'), blue: cs('--blue'), yellow: cs('--yellow'), red: cs('--red'), rule: cs('--rule') };
    const g = run(), cur = mat(S.method, S.cols), M = S.from ? S.from.map((row, r) => row.map((v, k) => PE.lerp(v, cur[r][k], S.u))) : cur;
    const vmax = Math.max(...g.W0.flat().map(Math.abs)) * 1.05;
    // geometry: matrix (left or top) and error chart (right or bottom)
    const mw = narrow ? w - 24 : w * .6, cell = Math.min(mw / C, narrow ? 40 : 46), mx0 = narrow ? (w - cell * C) / 2 : 40, my0 = narrow ? 64 : 70;
    const ch = cell * 1.05, top = my0 + 26;
    // row of U above the matrix: how much of column j's error each later column absorbs
    const j = S.cols - 1, gu = glow('u'), ge = glow('err');
    PE.label(ctx, narrow ? 'row j of U' : 'row j of U, the Cholesky factor of H\u207b\u00b9', mx0, my0 - 30, gu > 0 ? P.fg : P.mut, 11, 'left', 500);
    if (S.method === 'gptq') {
      // before the first column, show row 0 faintly: the row the first rounding error will use
      const jr = Math.max(0, j), faint = j < 0 ? .45 : 1, urow = g.U[jr], um = Math.max(...urow.slice(jr).map(Math.abs));
      for (let k = jr; k < C; k++) {
        const hh = (cell * .8) * Math.abs(urow[k]) / um, x = mx0 + k * cell + cell * .2;
        ctx.globalAlpha = faint * (.55 + .45 * gu) + .4 * gu * (1 - faint); ctx.fillStyle = k === j ? P.fg : P.red; ctx.fillRect(x, my0 - 4 - hh * .55, cell * .6, Math.max(1, hh * .55)); ctx.globalAlpha = 1;
      }
      if (jr < C - 1) PE.arrow(ctx, mx0 + (jr + .5) * cell, my0 + 6, mx0 + Math.min(C - .3, jr + 4.5) * cell, my0 + 6, P.red, 2 + 2 * ge, 8, faint * .9 + .1 * ge);
      if (!narrow || jr < C - 4) PE.label(ctx, j < 0 ? 'first column\'s error will go right' : 'error pushed right', mx0 + (jr + .7) * cell, my0 + 20, P.red, 11, 'left', 600);
    }
    // the matrix
    const gw = glow('w'), gq = glow('q');
    for (let r = 0; r < R; r++) for (let k = 0; k < C; k++) {
      const x = mx0 + k * cell, y = top + r * ch, cy = y + ch / 2, q = k < S.cols, half = ch * .44;
      ctx.strokeStyle = P.rule; ctx.lineWidth = 1; ctx.strokeRect(x + .5, y + .5, cell - 1, ch - 1);
      const v0 = g.W0[r][k] / vmax * half, v = M[r][k] / vmax * half;
      ctx.strokeStyle = P.mut; ctx.globalAlpha = .7; ctx.strokeRect(x + cell * .24, Math.min(cy, cy - v0), cell * .52, Math.max(1, Math.abs(v0))); ctx.globalAlpha = 1;
      const lit = q ? gq : gw;
      ctx.fillStyle = q ? P.yellow : P.blue; ctx.globalAlpha = .85 + .15 * lit;
      ctx.fillRect(x + cell * (.28 - .06 * lit), Math.min(cy, cy - v), cell * (.44 + .12 * lit), Math.max(1.5, Math.abs(v))); ctx.globalAlpha = 1;
    }
    if (j >= 0) { ctx.strokeStyle = P.fg; ctx.lineWidth = 2; ctx.strokeRect(mx0 + j * cell + 1, top + 1, cell - 2, R * ch - 2); }
    const by = top + R * ch;
    PE.label(ctx, `${S.cols} of ${C} columns on the ${S.bits}-bit grid`, mx0, by + 18, P.fg, 12, 'left', 600);
    PE.label(ctx, narrow ? 'outline: original weight' : 'outline: original weight; yellow: quantized; blue: not yet', mx0, by + 34, P.mut, 11, 'left', 400);
    // error chart: output error with the first j columns quantized, both methods
    const gl = glow('loss');
    const b = narrow ? { x: 46, y: by + 64, w: w - 64, h: h - by - 110 - pad } : { x: mx0 + C * cell + 70, y: top, w: w - (mx0 + C * cell + 70) - 20, h: R * ch - 4 - pad * 0 };
    if (b.h > 40 && b.w > 60) {
      const ymax = Math.max(...g.errR) * 1.08;
      const A = PE.axes(ctx, b, [0, C], [0, ymax], { xt: [0, 4, 8, 12, 16], yt: [0, ymax / 2, ymax].map(v => +v.toPrecision(2)), yf: v => KIT.fmt(v, 1),
        xl: 'columns quantized', yl: narrow ? 'output error' : 'output error ||WX - W\u2019X||\u00b2', ylo: 36 });
      const ptsR = g.errR.map((e, k) => [A.X(k), A.Y(e)]), ptsG = g.errG.map((e, k) => [A.X(k), A.Y(e)]);
      PE.polyline(ctx, ptsR, P.mut, 1.6 + 2 * gl, [5, 4], S.method === 'rtn' ? 1 : .8);
      PE.polyline(ctx, ptsG, P.red, 2 + 2 * gl, null, S.method === 'gptq' ? 1 : .6);
      const act = S.method === 'gptq' ? g.errG : g.errR;
      PE.dot(ctx, A.X(S.cols), A.Y(act[S.cols]), 5, S.method === 'gptq' ? P.red : P.mut);
      PE.label(ctx, 'round to nearest ' + KIT.fmt(g.errR[C], 1), A.X(C) - 4, A.Y(g.errR[C]) - 8, P.mut, 11, 'right', 600);
      PE.label(ctx, 'GPTQ ' + KIT.fmt(g.errG[C], 2), A.X(C) - 4, A.Y(g.errG[C]) - 8, P.red, 11, 'right', 600);
    }
  }
  const cv = PE.canvas(el, opts.aspect || (w => w < 560 ? 1.45 : .5), draw);
  let stop = () => {};
  function set(next) {
    const prev = mat(S.method, S.cols).map(r => r.slice()), changed = (next.bits && next.bits !== S.bits) || (next.method && next.method !== S.method) || (next.cols != null && next.cols !== S.cols);
    if (next.bits) S.bits = next.bits; if (next.method) S.method = next.method; if (next.cols != null) S.cols = next.cols;
    if (!changed) { cv.redraw(); return; }
    stop(); S.from = prev; S.u = 0;
    stop = PE.animate(700, u => { S.u = u; cv.redraw(); }, () => { S.from = null; S.u = 1; cv.redraw(); });
  }
  const info = () => { const g = run(); return { bits: S.bits, method: S.method, cols: S.cols, C, eG: g.errG[S.cols], eR: g.errR[S.cols], fG: g.errG[C], fR: g.errR[C] }; };
  return { set, info, state: S, redraw: cv.redraw, resize: cv.resize, destroy: () => { stop(); cv.destroy(); } };
};
KIT.gptqText = i => {
  const f = v => KIT.fmt(v, v < 1 ? 3 : 2), done = i.cols === i.C;
  const lead = i.method === 'gptq' ? `<b class="c-red">GPTQ</b>, ${i.bits}-bit` : `<b>Round to nearest</b>, ${i.bits}-bit`;
  if (i.cols === 0) return `${lead}: nothing quantized yet. Both methods use the same per-row grid; GPTQ will end at an output error of ${f(i.fG)}, round to nearest at ${f(i.fR)}.`;
  const now = `${lead}, ${i.cols} of ${i.C} columns quantized: output error so far ${f(i.method === 'gptq' ? i.eG : i.eR)} (the other method at this point: ${f(i.method === 'gptq' ? i.eR : i.eG)}).`;
  return done ? now + ` Final: GPTQ ${f(i.fG)} against ${f(i.fR)}, ${(i.fR / i.fG).toFixed(1)}${KIT.times} lower on the same grid.` : now;
};
