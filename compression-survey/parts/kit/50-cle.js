/* Cross-layer equalization (Nagel 2021 Eq. 21-22; DFQ Eq. 11) on a toy block: 4 inputs -> 6 ReLU channels -> 3 outputs.
   Toy weights; the rescaling, the per-tensor symmetric quantizer, the outputs and the empirical bias correction
   (Nagel Eq. 27, applied to the output bias only) are computed exactly. Simulated in notes/sim-cle.txt. */
KIT.CLE_R1 = [6.0, 0.05, 2.4, 0.25, 4.0, 0.1];   // range of each layer-1 output channel: a 120x spread
KIT.CLE_R2 = [0.3, 2.5, 0.5, 1.5, 0.2, 2.0];     // range of each layer-2 input channel
KIT.cleToy = (seed = 11) => {
  const r = PE.rng(seed), g = PE.gauss(r), R1 = KIT.CLE_R1, R2 = KIT.CLE_R2, I = 4, O = 3, N = 256;
  const norm = (v, R) => { const m = Math.max(...v.map(Math.abs)); return v.map(x => x / m * R); };
  const W1 = R1.map(R => norm(Array.from({ length: I }, g), R));
  const cols = R2.map(R => norm(Array.from({ length: O }, g), R));
  const W2 = Array.from({ length: O }, (_, o) => cols.map(c => c[o]));
  const b1 = R1.map(R => .3 * R * g()), b2 = Array.from({ length: O }, () => .2 * g());
  const batch = () => Array.from({ length: N }, () => Array.from({ length: I }, () => .5 + g()));
  const calib = batch(), test = batch();
  return { W1, b1, W2, b2, calib, test, probe: test[0], I, O, N };
};
KIT.cleFwd = (W1, b1, W2, b2, x) => {
  const h = W1.map((row, i) => Math.max(0, row.reduce((s, w, k) => s + w * x[k], 0) + b1[i]));
  return W2.map((row, o) => row.reduce((s, w, i) => s + w * h[i], 0) + b2[o]);
};
// Per-tensor symmetric quantizer: one step for the whole tensor, levels -q..q with q = 2^(b-1) - 1.
KIT.cleQuant = (W, bits) => {
  const q = Math.pow(2, bits - 1) - 1, st = Math.max(...W.flat().map(Math.abs)) / q;
  return { st, q, Q: W.map(row => row.map(w => PE.clamp(Math.round(w / st), -q, q) * st)) };
};
// t = 0: original weights; t = 1: s_i = sqrt(r1_i / r2_i), i.e. Eq. 22's (1 / r2_i) sqrt(r1_i r2_i). In between: s_i^t.
KIT.cleEval = (T, t, bits, bc) => {
  const R1 = KIT.CLE_R1, R2 = KIT.CLE_R2, O = T.O;
  const sStar = R1.map((a, i) => Math.sqrt(a * R2[i]) / R2[i]), s = sStar.map(v => Math.pow(v, t));
  const V1 = T.W1.map((row, i) => row.map(w => w / s[i])), c1 = T.b1.map((b, i) => b / s[i]);
  const V2 = T.W2.map(row => row.map((w, i) => w * s[i]));
  const q1 = KIT.cleQuant(V1, bits), q2 = KIT.cleQuant(V2, bits);
  const r1 = V1.map(row => Math.max(...row.map(Math.abs))), r2 = V2[0].map((_, i) => Math.max(...V2.map(row => Math.abs(row[i]))));
  const used = (r, st, q) => r.map(v => 2 * Math.min(q, Math.round(v / st)) + 1);
  let b2q = T.b2.slice(), shift = Array(O).fill(0);
  for (const x of T.calib) { const y = KIT.cleFwd(T.W1, T.b1, T.W2, T.b2, x), yq = KIT.cleFwd(q1.Q, c1, q2.Q, T.b2, x); y.forEach((v, o) => { shift[o] += (yq[o] - v) / T.N; }); }
  if (bc) b2q = T.b2.map((b, o) => b - shift[o]);
  let se = 0, sv = 0; const mean = Array(O).fill(0), my = Array(O).fill(0), ys = T.test.map(x => KIT.cleFwd(T.W1, T.b1, T.W2, T.b2, x));
  ys.forEach(y => y.forEach((v, o) => { my[o] += v / T.N; }));
  T.test.forEach((x, n) => { const yq = KIT.cleFwd(q1.Q, c1, q2.Q, b2q, x); ys[n].forEach((v, o) => { se += (yq[o] - v) ** 2; mean[o] += (yq[o] - v) / T.N; sv += (v - my[o]) ** 2; }); });
  const yProbe0 = KIT.cleFwd(T.W1, T.b1, T.W2, T.b2, T.probe), yProbe = KIT.cleFwd(V1, c1, V2, T.b2, T.probe);
  const sd = Math.sqrt(sv / (T.N * O));
  return { s, sStar, V1, V2, r1, r2, q1, q2, lv1: used(r1, q1.st, q1.q), lv2: used(r2, q2.st, q2.q), L: 2 * q1.q + 1,
    err: Math.sqrt(se / (T.N * O)) / sd, meanErr: Math.sqrt(mean.reduce((a, v) => a + v * v, 0) / O) / sd,
    yProbe0, yProbe, maxDiff: Math.max(...yProbe.map((v, o) => Math.abs(v - yProbe0[o]))) };
};
KIT.cle = (el, opts = {}) => {
  const T = KIT.cleToy(), S = { t: 0, bits: 8, bc: 0, bcT: 0, fig: opts.fig };
  const glow = opts.fig ? PE.linkPulses(opts.fig, () => cv.redraw()) : () => 0;
  const base = { 8: KIT.cleEval(T, 0, 8, false), 4: KIT.cleEval(T, 0, 4, false) };
  const fmt = v => v.toFixed(3);
  function panel(ctx, P, E, layer, C, narrow) {
    const r = layer === 1 ? E.r1 : E.r2, V = layer === 1 ? E.V1 : E.V2[0].map((_, i) => E.V2.map(row => row[i]));
    const qz = layer === 1 ? E.q1 : E.q2, Qv = layer === 1 ? qz.Q : qz.Q[0].map((_, i) => qz.Q.map(row => row[i]));
    const lv = layer === 1 ? E.lv1 : E.lv2, labW = narrow ? 58 : 74, R = Math.max(...r);
    const ax0 = P.x + 6, ax1 = P.x + P.w - labW, cx = (ax0 + ax1) / 2, half = (ax1 - ax0) / 2, X = v => cx + v / R * half;
    const rowH = P.h / 6, gl = glow(layer === 1 ? 'w1' : 'w2'), gr = glow('ranges');
    PE.label(ctx, layer === 1 ? 'layer 1: weights out of each channel' : 'layer 2: weights into each channel', P.x + 4, P.y - 8, gl > 0 ? C.fg : C.mut, narrow ? 11 : 12, 'left', 600);
    if (gl > 0) { ctx.save(); ctx.globalAlpha = gl; ctx.strokeStyle = layer === 1 ? C.blue : C.blue; ctx.lineWidth = 2; ctx.strokeRect(P.x - 2, P.y - 2, P.w + 4, P.h + 4); ctx.restore(); }
    ctx.save(); ctx.strokeStyle = C.rule; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(cx, P.y); ctx.lineTo(cx, P.y + P.h); ctx.stroke(); ctx.restore();
    const px = qz.st / R * half;  // grid spacing in pixels
    for (let i = 0; i < 6; i++) {
      const y = P.y + (i + .5) * rowH, bh = Math.min(14, rowH * .42);
      ctx.save(); ctx.globalAlpha = .16 + .3 * gr; ctx.fillStyle = C.blue; ctx.fillRect(X(-r[i]), y - bh / 2, Math.max(1, X(r[i]) - X(-r[i])), bh);
      ctx.globalAlpha = .7; ctx.strokeStyle = C.blue; ctx.lineWidth = 1 + 1.5 * gr; ctx.strokeRect(X(-r[i]), y - bh / 2, Math.max(1, X(r[i]) - X(-r[i])), bh); ctx.restore();
      if (px >= 3) { ctx.save(); ctx.strokeStyle = C.yellow; ctx.globalAlpha = .45; ctx.lineWidth = 1; ctx.beginPath();
        for (let k = -qz.q; k <= qz.q; k++) { const xx = cx + k * px; ctx.moveTo(xx, y - bh / 2 - 3); ctx.lineTo(xx, y + bh / 2 + 3); } ctx.stroke(); ctx.restore(); }
      V[i].forEach((v, k) => {
        const xq = X(Qv[i][k]), xv = X(v);
        if (Math.abs(xq - xv) > .8) PE.polyline(ctx, [[xv, y], [xq, y]], C.red, 2.2, null, .9);
        ctx.save(); ctx.strokeStyle = C.yellow; ctx.lineWidth = 2; ctx.beginPath(); ctx.moveTo(xq, y - bh / 2 - 2); ctx.lineTo(xq, y + bh / 2 + 2); ctx.stroke(); ctx.restore();
        PE.dot(ctx, xv, y, 3.2, C.blue);
      });
      const few = lv[i] <= (S.bits === 8 ? 9 : 3);
      PE.label(ctx, narrow ? `${lv[i]}/${E.L}` : `${lv[i]} of ${E.L}`, ax1 + 8, y + 4, few ? C.red : C.yellow, narrow ? 10.5 : 11.5, 'left', 600);
    }
    PE.label(ctx, 'levels', ax1 + 8, P.y - 8, C.mut, 10.5, 'left', 400);
    // the per-tensor ruler: one step for every channel of this tensor
    const ry = P.y + P.h + 8; ctx.save(); ctx.strokeStyle = C.yellow; ctx.globalAlpha = .9; ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.moveTo(X(-R), ry); ctx.lineTo(X(R), ry); ctx.stroke();
    const every = px >= 3 ? 1 : Math.pow(2, Math.ceil(Math.log2(3 / px)));
    ctx.lineWidth = 1; ctx.beginPath(); for (let k = -qz.q; k <= qz.q; k += 1) if (k % every === 0) { const xx = cx + k * px; ctx.moveTo(xx, ry - 3); ctx.lineTo(xx, ry + 3); } ctx.stroke(); ctx.restore();
    PE.label(ctx, `one ${S.bits}-bit grid for the tensor: step ${qz.st.toPrecision(2)}${every > 1 ? `, tick = ${every} levels` : ''}`, cx, ry + 16, C.yellow, 10.5, 'center', 400);
  }
  function draw(ctx, w, h) {
    ctx.clearRect(0, 0, w, h);
    const narrow = w < 560, cs = n => PE.css(n);
    const C = { fg: cs('--fg'), mut: cs('--muted'), blue: cs('--blue'), yellow: cs('--yellow'), red: cs('--red'), green: cs('--green'), rule: cs('--rule') };
    const E = KIT.cleEval(T, S.t, S.bits, false), Eb = KIT.cleEval(T, S.t, S.bits, true);
    const pad = opts.bottomPad || 0, outH = narrow ? 96 : 74, H = h - pad;
    let P1, P2, sx;
    if (!narrow) { const midW = 96, pw = (w - 20 - midW) / 2, ph = H - outH - 26 - 30;
      P1 = { x: 10, y: 26, w: pw, h: ph }; P2 = { x: 10 + pw + midW, y: 26, w: pw, h: ph }; sx = 10 + pw + midW / 2; }
    else { const sW = 50, ph = (H - outH - 2 * 24 - 2 * 26) / 2;
      P1 = { x: sW, y: 24, w: w - sW - 4, h: ph }; P2 = { x: sW, y: 24 + ph + 26 + 24, w: w - sW - 4, h: ph }; sx = sW / 2; }
    panel(ctx, P1, E, 1, C, narrow); panel(ctx, P2, E, 2, C, narrow);
    // the scale of each channel: layer 1 divides by it, layer 2 multiplies by it
    const gs = glow('scale');
    PE.label(ctx, narrow ? 's_i' : 'scale s_i', sx, P1.y - 8, gs > 0 ? C.green : C.mut, 11, 'center', 600);
    for (let i = 0; i < 6; i++) {
      const y = P1.y + (i + .5) * P1.h / 6, v = E.s[i];
      PE.label(ctx, v >= 10 ? v.toFixed(1) : v.toFixed(2), sx, y + 4, Math.abs(v - 1) > .01 ? C.green : C.fg, 11.5 + 1.5 * gs, 'center', 600);
      
    }
    // output strip: the FP32 output does not move; the INT8 error does
    const oy = H - outH + 8, go = glow('out'), gsh = glow('shift');
    ctx.save(); ctx.strokeStyle = C.rule; ctx.beginPath(); ctx.moveTo(8, oy - 6); ctx.lineTo(w - 8, oy - 6); ctx.stroke(); ctx.restore();
    const ow = narrow ? w - 16 : w * .48;
    PE.label(ctx, 'FP32 output, one test input', 10, oy + 10, go > 0 ? C.fg : C.mut, 11, 'left', 600);
    PE.label(ctx, 'y = (' + E.yProbe0.map(fmt).join(', ') + ')  original', 10, oy + 28, C.blue, 11.5 + go, 'left', 600);
    PE.label(ctx, 'y = (' + E.yProbe.map(fmt).join(', ') + ')  rescaled', 10, oy + 45, C.blue, 11.5 + go, 'left', 600);
    const mx = narrow ? 10 : ow + 24, my = narrow ? oy + 58 : oy, mw = (narrow ? w - 20 : w - mx - 12);
    const err = PE.lerp(E.err, Eb.err, S.bc), mean = PE.lerp(E.meanErr, Eb.meanErr, S.bc), ref = base[S.bits].err;
    const bar = (y, v, lab, al, g) => {
      const bw = mw - (narrow ? 118 : 150), x0 = mx + (narrow ? 118 : 150);
      PE.label(ctx, lab, mx, y + 9, C.mut, 10.5, 'left', 400);
      ctx.save(); ctx.fillStyle = C.rule; ctx.fillRect(x0, y, bw, 10); ctx.globalAlpha = al; ctx.fillStyle = C.red; ctx.fillRect(x0, y, Math.max(1.5, bw * Math.min(1, v / ref)), 10);
      if (g > 0) { ctx.globalAlpha = g; ctx.strokeStyle = C.red; ctx.lineWidth = 2; ctx.strokeRect(x0 - 2, y - 2, bw + 4, 14); } ctx.restore();
      PE.label(ctx, (v * 100).toFixed(2) + '%', x0 + Math.min(bw - 40, bw * Math.min(1, v / ref) + 6), y + 9, C.fg, 10.5, 'left', 600);
    };
    if (!narrow) PE.label(ctx, `per-tensor ${S.bits}-bit weights, 256 test inputs`, mx, oy + 10, C.mut, 11, 'left', 600);
    bar(narrow ? my : oy + 20, err, 'output error (RMS)', .95, 0);
    bar(narrow ? my + 16 : oy + 38, mean, 'its mean shift', .55, gsh);
  }
  const cv = PE.canvas(el, opts.aspect || (w => w < 560 ? 1.55 : .54), draw);
  let stopT = () => {}, stopB = () => {};
  function set(next) {
    if (next.bits && next.bits !== S.bits) S.bits = next.bits;
    if (next.t != null && next.t !== S.t) { stopT(); stopT = next.instant ? (S.t = next.t, () => {}) : KIT.tween(S, { t: next.t }, 800, cv.redraw); }
    if (next.bc != null && next.bc !== S.bcT) { S.bcT = next.bc; stopB(); stopB = KIT.tween(S, { bc: next.bc }, 600, cv.redraw); }
    cv.redraw();
  }
  function info(t = S.t) {
    const E = KIT.cleEval(T, t, S.bits, !!S.bcT), E0 = KIT.cleEval(T, 0, S.bits, !!S.bcT);
    return { t, bits: S.bits, bc: !!S.bcT, E, E0, lvMin: Math.min(...E.lv1), lvMin0: Math.min(...E0.lv1) };
  }
  return { set, info, state: S, redraw: cv.redraw, resize: cv.resize, destroy: () => { stopT(); stopB(); cv.destroy(); } };
};
KIT.cleText = i => {
  const pct = v => (v * 100).toFixed(2) + '%', y = i.E.yProbe.map(v => v.toFixed(3)).join(', ');
  const where = i.t <= 0 ? 'Original weights (every s_i = 1).' : i.t >= 1 ? 'Equalized: every s_i at its Eq. 22 value.' : `Part way: each s_i raised to the power ${i.t.toFixed(2)}.`;
  return `${where} Per-tensor ${i.bits}-bit weights${i.bc ? ' with bias correction' : ''}: output error <b class="c-red">${pct(i.E.err)}</b> of the output's spread` +
    (i.t > 0 ? ` (${pct(i.E0.err)} before equalizing)` : '') + `; mean shift ${pct(i.E.meanErr)}. ` +
    `The smallest channel uses ${i.lvMin} of ${i.E.L} levels${i.t > 0 ? ` (${i.lvMin0} before)` : ''}. ` +
    `FP32 output for the test input: (${y}), largest change from the original ${i.E.maxDiff === 0 ? 0 : i.E.maxDiff.toExponential(0)}.`;
};
