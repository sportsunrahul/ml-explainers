/* CoreS4: one small time-invariant SSM computed three ways (LSSL Fig. 1): continuous, recurrent, convolutional.
   Shared by viz-s4-views and atlas stage s4. A is HiPPO-LegS with S4's sign (S4 Eq. 2: A = -A_Thm2), B_n = sqrt(2n+1)
   (S4D Eq. 4), C drawn once with std 1 (S4D's init, seed fixed), bilinear discretization with step Delta (S4 Eq. 3).
   The continuous view solves h' = A h + B x(t) exactly for the token sequence held constant over each step of length Delta;
   the bilinear rule approximates it, so that view agrees only up to the discretization error. */
const CoreS4 = (() => {
  const N = 4, L = 64, SUB = 12;
  const eye = n => Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => +(i === j)));
  const mm = (X, Y) => X.map(r => Y[0].map((_, j) => r.reduce((s, v, k) => s + v * Y[k][j], 0)));
  const mv = (X, v) => X.map(r => r.reduce((s, a, k) => s + a * v[k], 0));
  const dot = (a, b) => a.reduce((s, v, i) => s + v * b[i], 0);
  function inv(M) {
    const n = M.length, a = M.map((r, i) => [...r, ...eye(n)[i]]);
    for (let c = 0; c < n; c++) {
      let p = c; for (let r = c + 1; r < n; r++) if (Math.abs(a[r][c]) > Math.abs(a[p][c])) p = r;
      [a[c], a[p]] = [a[p], a[c]]; const d = a[c][c]; for (let j = 0; j < 2 * n; j++) a[c][j] /= d;
      for (let r = 0; r < n; r++) if (r !== c) { const f = a[r][c]; for (let j = 0; j < 2 * n; j++) a[r][j] -= f * a[c][j]; }
    }
    return a.map(r => r.slice(n));
  }
  function expm(M) {
    let nrm = Math.max(...M.map(r => r.reduce((s, v) => s + Math.abs(v), 0))), s = 0; while (nrm > .5) { nrm /= 2; s++; }
    const X = M.map(r => r.map(v => v / 2 ** s)); let T = eye(M.length), R = eye(M.length);
    for (let k = 1; k < 18; k++) { T = mm(T, X).map(r => r.map(v => v / k)); R = R.map((r, i) => r.map((v, j) => v + T[i][j])); }
    for (let i = 0; i < s; i++) R = mm(R, R);
    return R;
  }
  const A = CoreHippo.A(N).map(r => [...r].map(v => -v)), B = [...CoreHippo.B(N)];
  const C = (() => { const g = PE.gauss(PE.rng(1)); return Array.from({ length: N }, g); })();
  // Toy input tokens: an impulse, a block, then a wave.
  const X = Array.from({ length: L }, (_, k) => k === 4 ? 1 : k >= 20 && k < 28 ? .8 : k >= 40 ? .6 * Math.sin(2 * Math.PI * (k - 40) / 8) : 0);
  const cache = {};
  function solve(dt) {
    const key = dt.toFixed(5); if (cache[key]) return cache[key];
    const I = eye(N), Mi = inv(I.map((r, i) => r.map((v, j) => v - dt / 2 * A[i][j])));
    const Ab = mm(Mi, I.map((r, i) => r.map((v, j) => v + dt / 2 * A[i][j]))), Bb = mv(Mi, B).map(v => v * dt);
    // Recurrent view: h_k = Ab h_{k-1} + Bb x_k, y_k = C h_k, h_{-1} = 0.
    const H = [], yr = []; let h = new Array(N).fill(0);
    for (let k = 0; k < L; k++) { h = mv(Ab, h).map((v, i) => v + Bb[i] * X[k]); H.push(h); yr.push(dot(C, h)); }
    // Convolutional view: K = (C Bb, C Ab Bb, ..., C Ab^{L-1} Bb), y_k = sum_j K_j x_{k-j}.
    const K = []; let v = Bb.slice(); for (let j = 0; j < L; j++) { K.push(dot(C, v)); v = mv(Ab, v); }
    const yc = X.map((_, k) => { let s = 0; for (let j = 0; j <= k; j++) s += K[j] * X[k - j]; return s; });
    // Continuous view: exact solution over SUB sub-steps per token (token k is held over ((k-1) dt, k dt]).
    const E = expm(A.map(r => r.map(a => a * dt / SUB))), F = mv(inv(A), mv(E.map((r, i) => r.map((a, j) => a - (i === j))), B));
    const ycont = [[-1, 0]], yat = []; let hc = new Array(N).fill(0);
    for (let k = 0; k < L; k++) { for (let s = 1; s <= SUB; s++) { hc = mv(E, hc).map((a, i) => a + F[i] * X[k]); ycont.push([k - 1 + s / SUB, dot(C, hc)]); } yat.push(dot(C, hc)); }
    const gapRC = Math.max(...yr.map((y, k) => Math.abs(y - yc[k]))), gapCont = Math.max(...yr.map((y, k) => Math.abs(y - yat[k])));
    const Kmax = Math.max(...K.map(Math.abs));
    let width = L; for (let j = L - 1; j >= 0; j--) if (Math.abs(K[j]) > .05 * Kmax) { width = j + 1; break; }
    return (cache[key] = { Ab, Bb, H, yr, K, yc, ycont, gapRC, gapCont, Kmax, width });
  }
  const DTMIN = .02, DTMAX = .5;
  const fromSlider = s => DTMIN * Math.pow(DTMAX / DTMIN, s / 1000), toSlider = dt => Math.round(1000 * Math.log(dt / DTMIN) / Math.log(DTMAX / DTMIN));
  const sci = v => { const e = Math.floor(Math.log10(v)); return `${(v / 10 ** e).toFixed(1)}e${e}`; };

  // st = { view: 'all'|'cont'|'rec'|'conv', dt, k (output position for the convolution), p (progress of the view's animation) }
  function draw(ctx, W, H, st, glow = () => 0) {
    const Cc = n => PE.css(n), narrow = W < 520, R = solve(st.dt), p = st.p ?? 1, v = st.view;
    ctx.clearRect(0, 0, W, H);
    const top = narrow ? { x: 30, y: 62, w: W - 42, h: H * .46 } : { x: 40, y: 58, w: W - 56, h: H * .53 };
    const bot = narrow ? { x: 30, y: top.y + top.h + 46, w: W - 42, h: H - top.h - 120 } : { x: 40, y: top.y + top.h + 44, w: W - 56, h: H - top.h - 116 };
    const ymax = Math.max(1.1, ...R.yr.map(Math.abs)) * 1.12;
    const ax = PE.axes(ctx, top, [-1, L], [-ymax, ymax], { xt: [0, 16, 32, 48, 63], yt: [0] });
    const Xp = ax.X, Y = ax.Y, B1 = Cc('--blue'), G = Cc('--green'), Yl = Cc('--yellow');
    PE.label(ctx, 'token position k', top.x + top.w / 2, top.y + top.h + 28, Cc('--muted'), 11, 'center');
    // Input tokens as blue stems.
    const gi = glow('input');
    for (let k = 0; k < L; k++) if (Math.abs(X[k]) > 1e-9) { ctx.save(); ctx.strokeStyle = B1; ctx.globalAlpha = .55 + .45 * gi; ctx.lineWidth = narrow ? 2 : 3 + 2 * gi; ctx.beginPath(); ctx.moveTo(Xp(k), Y(0)); ctx.lineTo(Xp(k), Y(X[k])); ctx.stroke(); ctx.restore(); }
    // Convolution: the flipped kernel laid over the inputs that feed output k.
    const kk = Math.round(st.k ?? 44), conv = v === 'conv';
    if (conv || glow('kernel') > 0) {
      const a = conv ? p : glow('kernel'), sc = .85 * ymax / R.Kmax;
      ctx.save(); ctx.globalAlpha = .55 * a; ctx.fillStyle = Yl;
      for (let j = 0; j <= kk; j++) { const x = Xp(kk - j), hgt = R.K[j] * sc; ctx.fillRect(x - 2.5, Math.min(Y(0), Y(hgt)), 5, Math.abs(Y(hgt) - Y(0))); }
      ctx.restore();
      if (conv) PE.label(ctx, `kernel, flipped, ending at k = ${kk}`, Xp(Math.max(0, kk - 12)), top.y + 12, Yl, 11, 'center');
    }
    // Continuous output: the exact solution, drawn up to the progress point in the continuous view.
    const go = glow('output'), gc = glow('cont');
    const upTo = v === 'cont' ? -1 + (L + 1) * p : L;
    const pts = R.ycont.filter(q => q[0] <= upTo).map(([x, y]) => [Xp(x), Y(y)]);
    PE.polyline(ctx, pts, G, (v === 'cont' ? 2.6 : 1.6) + 2 * gc + go, null, v === 'all' || v === 'cont' ? .95 : .4);
    // Recurrent outputs (dots) up to the cursor; convolutional outputs (rings).
    const cur = v === 'rec' ? Math.min(L - 1, Math.floor(p * (L - 1) + 1e-9)) : L - 1;
    for (let k = 0; k <= cur; k++) PE.dot(ctx, Xp(k), Y(R.yr[k]), (narrow ? 2.2 : 3) + 1.5 * go, G, v === 'rec' || v === 'all' ? 1 : .45);
    ctx.save(); ctx.strokeStyle = G; ctx.lineWidth = 1.3 + go; ctx.globalAlpha = conv || v === 'all' ? 1 : .4;
    for (let k = 0; k < L; k++) { if (conv && k > kk) continue; ctx.beginPath(); ctx.arc(Xp(k), Y(R.yc[k]), (narrow ? 4 : 5.5) + 1.5 * go, 0, 7); ctx.stroke(); }
    ctx.restore();
    if (conv) { PE.dot(ctx, Xp(kk), Y(R.yc[kk]), 7, G); PE.label(ctx, `y at k = ${kk}: ${R.yc[kk].toFixed(3)}`, Xp(kk) + 10, Y(R.yc[kk]) - 10, G, 11, kk > 48 ? 'right' : 'left'); }
    // Recurrent view: a cursor and the state register (N yellow cells) it carries.
    const gs = glow('state');
    if (v === 'rec' || gs > 0) {
      const k = cur, hv = R.H[k], cs = narrow ? 12 : 16, rx = PE.clamp(Xp(k) - N * cs / 2, top.x, top.x + top.w - N * cs), ry = top.y - 32;
      ctx.save(); ctx.strokeStyle = Yl; ctx.globalAlpha = .6; ctx.setLineDash([3, 3]); ctx.beginPath(); ctx.moveTo(Xp(k), ry + cs + 2); ctx.lineTo(Xp(k), top.y + top.h); ctx.stroke(); ctx.restore();
      const hm = Math.max(...R.H.flat().map(Math.abs));
      for (let n = 0; n < N; n++) { ctx.fillStyle = PE.mix(Cc('--panel2'), Yl, .15 + .85 * Math.min(1, Math.abs(hv[n]) / hm)); ctx.fillRect(rx + n * cs + 1, ry, cs - 2, cs - 2); }
      if (gs > 0) { ctx.save(); ctx.strokeStyle = Yl; ctx.lineWidth = 2.5 * gs; ctx.strokeRect(rx - 2, ry - 2, N * cs + 2, cs + 2); ctx.restore(); }
      PE.label(ctx, `state h at k = ${k}`, rx + N * cs + 6, ry + cs - 4, Yl, 11, rx + N * cs + 120 > W ? 'left' : 'left');
    }
    // The step Delta: how much continuous time one token covers.
    const gd = glow('delta'), dx0 = Xp(56), dx1 = Xp(57), dy = top.y + top.h - 10;
    ctx.save(); ctx.strokeStyle = Cc('--fg'); ctx.globalAlpha = .55 + .45 * gd; ctx.lineWidth = 1 + 1.5 * gd; ctx.beginPath(); ctx.moveTo(dx0, dy - 4); ctx.lineTo(dx0, dy); ctx.lineTo(dx1, dy); ctx.lineTo(dx1, dy - 4); ctx.stroke(); ctx.restore();
    PE.label(ctx, `\u0394 = ${st.dt.toFixed(st.dt < .1 ? 3 : 2)}`, (dx0 + dx1) / 2, dy - 8, Cc('--fg'), 10 + 2 * gd, 'center');
    // Legend.
    const lg = narrow ? [['tokens x', B1], ['outputs y', G]] : [['input tokens x', B1], ['continuous', G], ['recurrent (dots)', G], ['convolution (rings)', G]];
    let lx = top.x; lg.forEach(([t, c]) => { PE.label(ctx, t, lx, 16, c, 11); lx += t.length * 6.2 + 14; });
    // Kernel panel.
    const gk = glow('kernel'), kax = PE.axes(ctx, bot, [-1, L], [-R.Kmax * .35, R.Kmax * 1.15], { xt: [0, 16, 32, 48, 63], yt: [0] });
    for (let j = 0; j < L; j++) { ctx.save(); ctx.strokeStyle = Yl; ctx.globalAlpha = conv ? .5 + .5 * p : .85; ctx.lineWidth = (narrow ? 2 : 3) + 2 * gk; ctx.beginPath(); ctx.moveTo(kax.X(j), kax.Y(0)); ctx.lineTo(kax.X(j), kax.Y(R.K[j])); ctx.stroke(); ctx.restore(); }
    PE.label(ctx, narrow ? 'kernel K, lag j' : 'kernel K = (CB\u0304, CA\u0304B\u0304, CA\u0304\u00b2B\u0304, ...): weight of the input j steps back', bot.x + 4, bot.y - 8, Yl, 11);
    PE.label(ctx, `under 5% of its peak after ${R.width} steps`, bot.x + bot.w - 4, bot.y + 14, Cc('--muted'), 11, 'right');
  }
  function readout(st) {
    const R = solve(st.dt), nl = N * L, n2l = N * N * L, P = 64 * 16384;
    const views = { all: 'All three views of one SSM.', cont: 'Continuous view: the exact solution of the differential equation.', rec: 'Recurrent view: one state update per token.', conv: `Convolutional view: output ${Math.round(st.k ?? 44)} is the flipped kernel times the inputs before it.` }[st.view];
    return `${views} \u0394 = ${st.dt.toFixed(st.dt < .1 ? 3 : 2)}: recurrent and convolutional outputs differ by at most ${sci(Math.max(R.gapRC, 1e-17))}; the continuous solution differs from them by ${R.gapCont.toFixed(4)}, the error of the bilinear step. ` +
      `Building this kernel (L = ${L}, N = ${N}) by repeated multiplication costs N\u00b2L = ${n2l.toLocaleString('en-US')} multiply-adds, against N L = ${nl.toLocaleString('en-US')} for the Cauchy loop in S4's released code; at Path-X size (N = 64, L = 16,384) that is ${(64 * P).toLocaleString('en-US')} against ${P.toLocaleString('en-US')}.`;
  }
  return { N, L, A, B, C, X, solve, draw, readout, fromSlider, toSlider, DTMIN, DTMAX };
})();
