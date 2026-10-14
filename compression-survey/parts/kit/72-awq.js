/* viz-awq: AWQ's scale search (Lin et al., 2306.00978v6, Eq. 4-5) run exactly on a toy layer: W is 16 outputs x 32 input
   channels, X is 32 channels x 64 calibration tokens, and two input channels (7 and 24, counting from 1) carry about
   10x larger activations. s_X = mean |X| per input channel; s = s_X^alpha; W diag(s) is quantized to INT3 with one
   symmetric max-abs scale per group of 16 input channels (group size shrunk from 128 for the toy); the loss is the
   Frobenius norm || Q(W diag(s)) (diag(s)^-1 X) - W X ||. Delta = max|w| / (2^(N-1) - 1) is our convention: the paper
   prints 2^(N-1), which puts the largest weight one level outside the signed INT-N range. Grid: alpha = i / 19 for
   i = 0..19 (20 points on [0, 1]; the exact spacing is our reading). */
KIT.AWQ = { CO: 16, CI: 32, T: 64, G: 16, bits: 3, salient: [6, 23] };
KIT.awqData = (() => {
  let cache = null;
  return () => {
    if (cache) return cache;
    const A = KIT.AWQ, r = PE.rng(7), g = PE.gauss(r);
    const mag = Array.from({ length: A.CI }, (_, i) => A.salient.includes(i) ? 8 + 2 * r() : .5 + .7 * r());
    const X = Array.from({ length: A.CI }, (_, i) => Array.from({ length: A.T }, () => g() * mag[i]));
    const W = Array.from({ length: A.CO }, () => Array.from({ length: A.CI }, () => g() * .4));
    cache = { X, W, sx: X.map(row => row.reduce((s, v) => s + Math.abs(v), 0) / A.T) }; return cache;
  };
})();
KIT.awqQuant = (W, bits, G) => {
  const top = (1 << (bits - 1)) - 1;
  return W.map(row => { const out = row.slice(); for (let g0 = 0; g0 < row.length; g0 += G) {
    const D = Math.max(...row.slice(g0, g0 + G).map(Math.abs)) / top;
    for (let i = g0; i < g0 + G; i++) out[i] = D > 0 ? D * Math.round(row[i] / D) : 0; } return out; });
};
KIT.awqEval = alpha => {
  const A = KIT.AWQ, { X, W, sx } = KIT.awqData();
  const s = sx.map(v => Math.pow(v, alpha));
  const Qs = KIT.awqQuant(W.map(row => row.map((v, i) => v * s[i])), A.bits, A.G);
  const E = Qs.map((row, o) => row.map((v, i) => v / s[i] - W[o][i]));
  let L = 0; for (let o = 0; o < A.CO; o++) for (let t = 0; t < A.T; t++) { let d = 0; for (let i = 0; i < A.CI; i++) d += E[o][i] * X[i][t]; L += d * d; }
  // channel i's own term ||E[:, i]|| ||X[i, :]|| (the Frobenius norm of a rank-1 product); the total also has cross terms
  const ch = Array.from({ length: A.CI }, (_, i) => Math.sqrt(E.reduce((a, row) => a + row[i] * row[i], 0)) * Math.sqrt(X[i].reduce((a, v) => a + v * v, 0)));
  return { alpha, s, L: Math.sqrt(L), ch };
};
KIT.awqGrid = (() => { let c = null; return () => c || (c = Array.from({ length: 20 }, (_, i) => KIT.awqEval(i / 19))); })();
KIT.awqBest = () => KIT.awqGrid().reduce((a, c) => c.L < a.L ? c : a);
KIT.awq = (el, opts = {}) => {
  const A = KIT.AWQ, D = KIT.awqData();
  const S = { alpha: 0, goal: 0, shown: 0, searched: false };       // shown: how many grid points are drawn (0..20)
  const glow = opts.fig ? PE.linkPulses(opts.fig, () => cv.redraw()) : () => 0;
  function draw(ctx, w, h) {
    ctx.clearRect(0, 0, w, h);
    const narrow = w < 560, cs = n => PE.css(n), pad = opts.bottomPad || 0;
    const P = { fg: cs('--fg'), mut: cs('--muted'), blue: cs('--blue'), green: cs('--green'), red: cs('--red'), rule: cs('--rule') };
    const ev = KIT.awqEval(S.alpha), grid = KIT.awqGrid(), L0 = grid[0].L, Lmax = Math.max(...grid.map(e => e.L)) * 1.1;
    const lw = narrow ? w - 56 : w * .55 - 56, lx = 44, hA = narrow ? (h - pad) * .2 : (h - pad) * .36, hB = hA;
    const yA = 34, yB = yA + hA + 50;
    const bw = lw / A.CI, gs = glow('sx'), gc = glow('ch');
    // panel A: mean |X| per input channel; salient channels in green with their scale
    const smax = Math.max(...D.sx);
    PE.label(ctx, narrow ? 'mean |X| per input channel' : 'activation size per input channel, s_X = mean |X|', lx, yA - 12, P.fg, 12, 'left', 600);
    D.sx.forEach((v, i) => { const hh = hA * .8 * v / smax, sal = A.salient.includes(i);
      ctx.globalAlpha = .85 + .15 * gs; ctx.fillStyle = sal ? P.green : P.blue; ctx.fillRect(lx + i * bw + bw * .15, yA + hA - hh, bw * .7, hh); ctx.globalAlpha = 1;
      if (sal) PE.label(ctx, '\u00d7' + ev.s[i].toFixed(ev.s[i] < 10 ? 1 : 0), lx + (i + .5) * bw, yA + hA - hh - 5, P.green, narrow ? 10 : 11, 'center', 600); });
    ctx.strokeStyle = P.rule; ctx.beginPath(); ctx.moveTo(lx, yA + hA + .5); ctx.lineTo(lx + lw, yA + hA + .5); ctx.stroke();
    // panel B: each channel's own output-error term at the current alpha (fixed axis: alpha = 0 maximum)
    const cmax = Math.max(...grid[0].ch, ...grid[19].ch);
    PE.label(ctx, narrow ? 'each channel\'s error term' : 'each input channel\'s output-error term ||E_j|| ||X_j||', lx, yB - 12, P.fg, 12, 'left', 600);
    ev.ch.forEach((v, i) => { const hh = hB * v / cmax;
      ctx.globalAlpha = .8 + .2 * gc; ctx.fillStyle = P.red; ctx.fillRect(lx + i * bw + bw * .15, yB + hB - hh, bw * .7, hh); ctx.globalAlpha = 1; });
    ctx.strokeStyle = P.rule; ctx.beginPath(); ctx.moveTo(lx, yB + hB + .5); ctx.lineTo(lx + lw, yB + hB + .5); ctx.stroke();
    PE.label(ctx, 'input channel 1 to 32', lx + lw / 2, yB + hB + 16, P.mut, 11, 'center', 400);
    // panel C: loss against alpha
    const gl = glow('loss'), ga = glow('alpha');
    const b = narrow ? { x: 52, y: yB + hB + 58, w: w - 70, h: h - pad - (yB + hB + 58) - 44 } : { x: w * .55 + 40, y: yA, w: w * .45 - 60, h: h - pad - yA - 44 };
    const ax = PE.axes(ctx, b, [0, 1], [0, Lmax], { xt: [0, .25, .5, .75, 1], yt: [0, +(Lmax / 2).toPrecision(2), +Lmax.toPrecision(2)], yf: v => KIT.fmt(v, 0),
      xl: 'alpha (s = s_X^alpha)', yl: 'output error L', ylo: 36 });
    const pts = grid.slice(0, S.shown);
    if (pts.length > 1) PE.polyline(ctx, pts.map(e => [ax.X(e.alpha), ax.Y(e.L)]), P.red, 1.6 + 2 * gl, null, .8);
    pts.forEach(e => PE.dot(ctx, ax.X(e.alpha), ax.Y(e.L), 3.2 + gl, P.red, .9));
    if (S.searched && S.shown >= 20) { const bst = KIT.awqBest(); PE.ring(ctx, ax.X(bst.alpha), ax.Y(bst.L), 9, P.green, 2);
      PE.label(ctx, 'best of 20: alpha ' + bst.alpha.toFixed(2), ax.X(bst.alpha), ax.Y(bst.L) + 24, P.green, 11, 'center', 600); }
    const cx = ax.X(S.alpha), cy = ax.Y(ev.L);
    ctx.strokeStyle = P.mut; ctx.setLineDash([3, 3]); ctx.beginPath(); ctx.moveTo(cx, b.y); ctx.lineTo(cx, b.y + b.h); ctx.stroke(); ctx.setLineDash([]);
    PE.dot(ctx, cx, cy, 6 + 2 * ga, P.fg); PE.ring(ctx, cx, cy, 10 + 5 * ga, P.fg, 1.5, .7);
    PE.label(ctx, 'no scaling: ' + KIT.fmt(L0, 1), ax.X(0) + 16, ax.Y(L0) + 4, P.mut, 11, 'left', 500);
  }
  const cv = PE.canvas(el, opts.aspect || (w => w < 560 ? 1.55 : .5), draw);
  let stop = () => {};
  function set(next) {
    stop();
    if (next.search) {
      if (S.searched) { cv.redraw(); return; }
      S.searched = true; const a0 = S.alpha, best = KIT.awqBest().alpha; S.goal = best;
      stop = PE.animate(900, u => { S.shown = Math.round(20 * Math.min(1, u * 1.5)); S.alpha = PE.lerp(a0, best, Math.max(0, (u - .5) * 2)); cv.redraw(); },
        () => { S.shown = 20; S.alpha = best; cv.redraw(); });
      return;
    }
    if (next.reset) { S.searched = false; S.shown = 0; }
    if (next.alpha != null) {
      S.goal = next.alpha;
      if (next.instant) { S.alpha = next.alpha; cv.redraw(); return; }
      const a0 = S.alpha; stop = PE.animate(500, u => { S.alpha = PE.lerp(a0, next.alpha, u); cv.redraw(); });
    } else cv.redraw();
  }
  const info = () => { const e = KIT.awqEval(S.goal); return { alpha: S.goal, L: e.L, L0: KIT.awqGrid()[0].L, s: A.salient.map(i => e.s[i]), searched: S.searched, best: KIT.awqBest() }; };
  return { set, info, state: S, redraw: cv.redraw, resize: cv.resize, destroy: () => { stop(); cv.destroy(); } };
};
KIT.awqText = i => {
  if (i.alpha < 1e-6 && !i.searched) return `<b>alpha = 0</b>: s = 1 for every channel, so this is plain round to nearest on the INT3 grid. Output error L = ${i.L.toFixed(1)}.`;
  const head = `<b>alpha = ${i.alpha.toFixed(2)}</b>: the two large-activation channels are multiplied by ${i.s.map(v => v.toFixed(1)).join(' and ')} before rounding, and their inputs divided by the same. ` +
    `Output error L = ${i.L.toFixed(1)}, against ${i.L0.toFixed(1)} with no scaling (` + (i.L <= i.L0 ? `${(i.L0 / i.L).toFixed(2)}${KIT.times} lower).` : `${(i.L / i.L0).toFixed(2)}${KIT.times} higher: the scaled channels now set the group maximum).`);
  return i.searched ? head + ` The grid search over 20 values of alpha picked ${i.best.alpha.toFixed(2)}.` : head;
};
