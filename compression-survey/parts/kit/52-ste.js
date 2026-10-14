/* The straight-through estimator on a one-weight toy. Quantizer: LSQ Eq. 1-2 (Esser 1902.08153v3), 3-bit signed,
   Q_N = 4, Q_P = 3. Model y = w_hat * x with x = 1, target 1.1, loss (y - 1.1)^2. Gradients: the true one (zero
   almost everywhere), the clipped STE (Esser Eq. 5; Krishnamoorthi Eq. 14-15; Wu Eq. 13), and the STE plus LSQ's
   step-size gradient (Esser Eq. 3) scaled by g = 1 / sqrt(N Q_P) with N = 1 weight. Simulated in notes/sim-ste.txt. */
KIT.STE = { QN: 4, QP: 3, y: 1.1, w0: .1, s0: .25, lr: .1, lrs: .1 };
KIT.steQ = (v, s) => Math.round(PE.clamp(v / s, -KIT.STE.QN, KIT.STE.QP)) * s;
KIT.steGrad = (w, s, mode) => {
  const P = KIT.STE, vh = KIT.steQ(w, s), dL = 2 * (vh - P.y), r = w / s, inside = r > -P.QN && r < P.QP;
  const dvdw = mode === 'true' ? 0 : inside ? 1 : 0;
  const dvds = r <= -P.QN ? -P.QN : r >= P.QP ? P.QP : -r + Math.round(r);
  const gs = mode === 'lsq' ? dL * dvds / Math.sqrt(1 * P.QP) : 0;
  return { vh, loss: (vh - P.y) ** 2, dL, dvdw, gw: dL * dvdw, gs, inside, r };
};
KIT.steStep = (st, mode) => { const g = KIT.steGrad(st.w, st.s, mode); return { w: st.w - KIT.STE.lr * g.gw, s: st.s - KIT.STE.lrs * g.gs }; };
KIT.ste = (el, opts = {}) => {
  const P = KIT.STE, S = { w: P.w0, s: P.s0, mode: 'ste', k: 0, end: null, trail: [P.w0], losses: [KIT.steGrad(P.w0, P.s0, 'ste').loss] };
  const glow = opts.fig ? PE.linkPulses(opts.fig, () => cv.redraw()) : () => 0;
  const xr = [-.6, 1.55];
  function draw(ctx, w, h) {
    ctx.clearRect(0, 0, w, h);
    const narrow = w < 560, cs = n => PE.css(n), pad = opts.bottomPad || 0;
    const C = { fg: cs('--fg'), mut: cs('--muted'), blue: cs('--blue'), yellow: cs('--yellow'), red: cs('--red'), green: cs('--green'), brown: cs('--brown'), rule: cs('--rule') };
    const H = h - pad, L = narrow ? 40 : 54, R = narrow ? 12 : 22, top = 26, gap = 40;
    const h1 = (H - top - gap - 40) * .66, h2 = (H - top - gap - 40) * .34;
    const b1 = { x: L, y: top, w: w - L - R, h: h1 }, b2 = { x: L, y: top + h1 + gap, w: w - L - R, h: h2 };
    const g = { grid: glow('grid'), clip: glow('clip'), round: glow('round'), shadow: glow('shadow'), ste: glow('ste'), tru: glow('true'), step: glow('step') };
    const lo = -P.QN * S.s, hi = P.QP * S.s;
    // top: the fake quantizer w -> w_hat
    const A = PE.axes(ctx, b1, xr, [-.6, 1.3], { xt: [-.5, 0, .5, 1, 1.5], yt: [-.5, 0, .5, 1], xl: '', yl: 'w_hat (used in the forward pass)', ylo: narrow ? 30 : 40 });
    const shade = (x0, x1, b, A2, al) => { ctx.save(); ctx.fillStyle = C.brown; ctx.globalAlpha = al; const a = A2.X(PE.clamp(x0, xr[0], xr[1])), c = A2.X(PE.clamp(x1, xr[0], xr[1])); if (c > a) ctx.fillRect(a, b.y, c - a, b.h); ctx.restore(); };
    shade(xr[0], lo, b1, A, .1 + .25 * g.clip); shade(hi, xr[1], b1, A, .1 + .25 * g.clip);
    PE.label(ctx, 'clipped', A.X(Math.min(xr[1], hi)) + 6, b1.y + 14, C.brown, 10.5, 'left', 600);
    for (let k = -P.QN; k <= P.QP; k++) { const yv = k * S.s; if (yv < -.6 || yv > 1.3) continue; PE.polyline(ctx, [[b1.x, A.Y(yv)], [b1.x + b1.w, A.Y(yv)]], C.yellow, 1 + 1.5 * g.grid, [2, 4], .3 + .5 * g.grid); }
    PE.polyline(ctx, [[A.X(-.6), A.Y(-.6)], [A.X(1.3), A.Y(1.3)]], C.blue, 1, [5, 5], .5);
    PE.label(ctx, 'no rounding', A.X(-.3) + 6, A.Y(-.3) + 14, C.blue, 10.5, 'left', 400);
    const pts = []; for (let i = 0; i <= 600; i++) { const x = xr[0] + (xr[1] - xr[0]) * i / 600, v = KIT.steQ(x, S.s); if (pts.length && Math.abs(pts[pts.length - 1][2] - v) > 1e-9) pts.push([A.X(x), A.Y(pts[pts.length - 1][2]), v]); pts.push([A.X(x), A.Y(v), v]); }
    PE.polyline(ctx, pts.map(p => [p[0], p[1]]), C.yellow, 2.4 + 2 * g.round, null, 1);
    if (g.step > 0) { const xa = A.X(xr[1]) - 16; PE.arrow(ctx, xa, A.Y(0), xa, A.Y(S.s), C.yellow, 2, 6, g.step); PE.label(ctx, 'step s', xa - 4, A.Y(S.s / 2) + 4, C.yellow, 11, 'right', 600); }
    PE.polyline(ctx, [[b1.x, A.Y(P.y)], [b1.x + b1.w, A.Y(P.y)]], C.red, 1.5, [6, 4], .9);
    PE.label(ctx, 'target 1.1', A.X(.15), A.Y(P.y) - 5, C.red, 11, 'left', 600);
    S.trail.forEach(t => PE.dot(ctx, A.X(t), b1.y + b1.h, 2.5, C.blue, .35));
    const vh = KIT.steQ(S.w, S.s), xw = A.X(PE.clamp(S.w, xr[0], xr[1]));
    PE.polyline(ctx, [[xw, b1.y + b1.h], [xw, A.Y(vh)]], C.blue, 1, [2, 3], .8);
    PE.polyline(ctx, [[xw, A.Y(vh)], [b1.x, A.Y(vh)]], C.yellow, 1, [2, 3], .8);
    PE.dot(ctx, xw, b1.y + b1.h, 6 + 2 * g.shadow, C.blue); PE.dot(ctx, xw, A.Y(vh), 6, C.yellow);
    const rt = xw > b1.x + b1.w * .55; PE.label(ctx, 'shadow weight w (FP32)', xw + (rt ? -10 : 10), b1.y + b1.h - 8, C.blue, 11, rt ? 'right' : 'left', 600);
    // loss history
    const lw = narrow ? 92 : 130, lh = 34, lx = b1.x + 8, ly = b1.y + 6;
    ctx.save(); ctx.fillStyle = cs('--bg'); ctx.globalAlpha = .85; ctx.fillRect(lx - 4, ly - 2, lw + 8, lh + 18); ctx.restore();
    PE.label(ctx, `loss ${S.losses[S.losses.length - 1].toFixed(4)}`, lx, ly + 10, C.red, 11, 'left', 600);
    const n = S.losses.length, lp = S.losses.map((v, i) => [lx + (n > 1 ? i / (n - 1) : 0) * lw, ly + 14 + lh - v / 1.21 * lh]);
    if (n > 1) PE.polyline(ctx, lp, C.red, 1.5, null, .9);
    // bottom: d w_hat / d w
    const B = PE.axes(ctx, b2, xr, [0, 1.25], { xt: [-.5, 0, .5, 1, 1.5], yt: [0, 1], xl: 'shadow weight w', yl: 'd w_hat / d w', ylo: narrow ? 30 : 40 });
    shade(xr[0], lo, b2, B, .1 + .25 * g.clip); shade(hi, xr[1], b2, B, .1 + .25 * g.clip);
    const trueOn = S.mode === 'true', aT = trueOn ? 1 : .4, aS = trueOn ? .35 : 1;
    PE.polyline(ctx, [[b2.x, B.Y(0)], [b2.x + b2.w, B.Y(0)]], C.mut, 2 + g.tru, null, aT);
    for (let k = -P.QN; k < P.QP; k++) { const xs = (k + .5) * S.s; if (xs < xr[0] || xs > xr[1]) continue; PE.arrow(ctx, B.X(xs), B.Y(0), B.X(xs), b2.y + 2, C.mut, 1.2 + g.tru, 6, aT * .8); }
    PE.label(ctx, 'true: 0, except spikes at each jump', b2.x + b2.w - 4, B.Y(0) - 6, C.mut, 10.5, 'right', 600);
    const ste = [[A.X(xr[0]), B.Y(0)], [B.X(lo), B.Y(0)], [B.X(lo), B.Y(1)], [B.X(Math.min(hi, xr[1])), B.Y(1)], [B.X(Math.min(hi, xr[1])), B.Y(0)], [B.X(xr[1]), B.Y(0)]];
    PE.polyline(ctx, ste, C.red, 2.4 + 2 * g.ste, null, aS);
    PE.label(ctx, 'STE: 1 inside the range, 0 outside', B.X(Math.max(lo, xr[0])) + 6, B.Y(1) - 6, C.red, 10.5, 'left', 600);
    const gr = KIT.steGrad(S.w, S.s, S.mode);
    PE.dot(ctx, xw, B.Y(gr.dvdw), 6, S.mode === 'true' ? C.mut : C.red);
  }
  const cv = PE.canvas(el, opts.aspect || (w => w < 560 ? 1.3 : .6), draw);
  let stop = () => {};
  function step(n) {
    stop(); let st = S.end ? { w: S.end.w, s: S.end.s } : { w: S.w, s: S.s };
    for (let i = 0; i < n; i++) { st = KIT.steStep(st, S.mode); S.k++; S.trail.push(st.w); S.losses.push(KIT.steGrad(st.w, st.s, S.mode).loss); }
    S.end = st; stop = KIT.tween(S, { w: st.w, s: st.s }, 450 + 90 * n, cv.redraw);
  }
  function set(next) {
    stop();
    if (next.mode && next.mode !== S.mode) { S.mode = next.mode; next.reset = true; }
    if (next.reset) { if (S.end) { S.w = S.end.w; S.s = S.end.s; } S.end = { w: P.w0, s: P.s0 }; S.k = 0; S.trail = [P.w0]; S.losses = [KIT.steGrad(P.w0, P.s0, S.mode).loss];
      stop = KIT.tween(S, { w: P.w0, s: P.s0 }, 500, cv.redraw); }
    if (next.step) step(next.step);
    cv.redraw();
  }
  // A tween may still be running: report the end state.
  const info = () => { const e = S.end || S; return { mode: S.mode, k: S.k, w: e.w, s: e.s, g: KIT.steGrad(e.w, e.s, S.mode) }; };
  return { set, info, state: S, redraw: cv.redraw, resize: cv.resize, destroy: () => { stop(); cv.destroy(); } };
};
KIT.steText = i => {
  const name = { true: 'True gradient', ste: 'Straight-through estimator', lsq: 'STE plus a learned step size (LSQ)' }[i.mode], g = i.g;
  let why;
  if (i.mode === 'true') why = 'The rounding is flat almost everywhere, so d w_hat / d w = 0 and the weight never moves.';
  else if (!g.inside) why = `w / s = ${g.r.toFixed(2)} is past the top of the range (Q_P = ${KIT.STE.QP}), so the clipped STE passes no gradient: w is stuck${i.mode === 'lsq' ? ', but the step size still learns' : ''}.`;
  else if (Math.abs(g.gw) < 1e-3) why = 'The rounded value has reached the target, so the gradient is about zero.';
  else why = `Inside the range the STE passes the gradient through as if rounding were the identity: dL/dw = ${g.gw.toFixed(2)}.`;
  return `<b>${name}</b>, step ${i.k}: w = ${i.w.toFixed(3)}, w_hat = <span class="c-yellow">${g.vh.toFixed(3)}</span>, step size s = ${i.s.toFixed(3)}, loss <b class="c-red">${g.loss.toFixed(4)}</b>. ${why}`;
};
