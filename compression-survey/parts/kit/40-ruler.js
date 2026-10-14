/* viz-ruler: a toy activation sample on a number line, quantized exactly for a chosen range, bit-width and grid.
   Asymmetric grid: unsigned codes 0..2^b-1 with a zero-point Z, range nudged so 0.0 is a grid point (Jacob Eq. 1,
   Krishnamoorthi Eq. 1-2). Symmetric grid: signed codes in [-(2^(b-1)-1), 2^(b-1)-1], Z = 0 (Wu Eq. 6-7).
   S is the step size. Errors are exact sums over the toy sample, divided by N: a value whose code hit the clamp
   is clipping error, measured to the end grid point; every other value is rounding error. */
KIT.RULER_X = [-2.5, 7.5];
KIT.RULER_PROBE = 1.37;
KIT.rulerSample = (() => {
  let cache = null;
  return () => {
    if (cache) return cache;
    const r = PE.rng(41), g = PE.gauss(r), N = 4000, xs = new Float64Array(N);
    for (let i = 0; i < N; i++) xs[i] = r() < .8 ? .5 + .55 * g() : .3 - .9 * Math.log(1 - r());
    let mn = Infinity, mx = -Infinity;
    for (const x of xs) { if (x < mn) mn = x; if (x > mx) mx = x; }
    cache = { xs, min: mn, max: mx, N };
    return cache;
  };
})();
KIT.RULER_PRESETS = { tight: () => [-.5, 1.6], minmax: () => { const s = KIT.rulerSample(); return [s.min, s.max]; }, wide: () => KIT.RULER_X.slice() };
KIT.rulerGrid = (lo, hi, bits, sym) => {
  if (sym) { const qmax = (1 << (bits - 1)) - 1, m = Math.max(-lo, hi, 1e-6); return { S: m / qmax, Z: 0, qmin: -qmax, qmax, sym, bits, levels: 2 * qmax + 1 }; }
  const qmax = (1 << bits) - 1, S = Math.max(hi - lo, 1e-6) / qmax, Z = PE.clamp(Math.round(-lo / S), 0, qmax);
  return { S, Z, qmin: 0, qmax, sym, bits, levels: qmax + 1 };
};
KIT.rulerCode = (G, r) => PE.clamp(Math.round(r / G.S) + G.Z, G.qmin, G.qmax);
KIT.rulerStats = (xs, G) => {
  let er = 0, ec = 0, nc = 0;
  const counts = new Map();
  for (const x of xs) {
    const raw = Math.round(x / G.S) + G.Z, q = Math.min(G.qmax, Math.max(G.qmin, raw)), d = x - G.S * (q - G.Z);
    if (raw !== q) { ec += d * d; nc++; } else er += d * d;
    counts.set(q, (counts.get(q) || 0) + 1);
  }
  const N = xs.length;
  return { round: er / N, clip: ec / N, mse: (er + ec) / N, nClip: nc, used: counts.size, counts, N };
};
KIT.rulerNum = v => v === 0 ? '0' : Math.abs(v) < 1e-3 ? v.toExponential(1) : (+v.toPrecision(2)).toString();
// Two horizontal log-scale error bars; shared with viz-calib.
KIT.rulerBars = (ctx, x0, y0, w, rows, glow) => {
  const L = v => v <= 0 ? 0 : PE.clamp((Math.log10(v) + 7) / 7, 0, 1);
  const red = PE.css('--red'), mut = PE.css('--muted'), fg = PE.css('--fg');
  const lw = Math.min(150, w * .34), bx = x0 + lw, bwid = Math.max(40, w - lw - 70);
  rows.forEach((row, i) => {
    const y = y0 + i * 22, g = glow ? glow(row.link || '') : 0;
    PE.label(ctx, row.name, x0, y + 11, g > 0 ? fg : mut, 11, 'left', 500);
    ctx.save(); ctx.globalAlpha = .25; ctx.fillStyle = mut; ctx.fillRect(bx, y + 3, bwid, 10); ctx.restore();
    const len = bwid * L(row.v);
    ctx.save(); ctx.globalAlpha = row.alpha == null ? .9 : row.alpha; ctx.fillStyle = red; ctx.fillRect(bx, y + 3 - 2 * g, Math.max(len, row.v > 0 ? 2 : 0), 10 + 4 * g); ctx.restore();
    PE.label(ctx, KIT.rulerNum(row.v), bx + Math.max(len, 0) + 6, y + 12, fg, 11, 'left', 600);
  });
  PE.label(ctx, 'log scale: 1e-7 to 1', bx + bwid, y0 + rows.length * 22 + 8, mut, 10, 'right', 400);
};

KIT.ruler = (el, opts = {}) => {
  const smp = KIT.rulerSample(), xs = smp.xs, XR = KIT.RULER_X;
  const S = { lo: smp.min, hi: smp.max, bits: 8, sym: false, u: 1, active: 'hi', drag: null };
  let G = KIT.rulerGrid(S.lo, S.hi, S.bits, S.sym), st = KIT.rulerStats(xs, G), Gp = G, stp = st;
  const NB = 120, bw = (XR[1] - XR[0]) / NB, hist = new Float64Array(NB);
  for (const x of xs) { const i = Math.floor((x - XR[0]) / bw); if (i >= 0 && i < NB) hist[i]++; }
  const dmax = Math.max(...hist) / (smp.N * bw);
  const glow = opts.fig ? PE.linkPulses(opts.fig, () => cv.redraw()) : () => 0;
  let L = null;
  const recompute = () => { G = KIT.rulerGrid(S.lo, S.hi, S.bits, S.sym); st = KIT.rulerStats(xs, G); };

  function layout(w, h) {
    const narrow = w < 560, pad = opts.bottomPad || 0;
    const x0 = narrow ? 12 : 24, x1 = w - (narrow ? 12 : 24);
    const top = narrow ? 34 : 30, usable = h - top - pad - (narrow ? 96 : 92);
    const hTop = usable * .56, hBot = usable * .44 - 30;
    const yA = top + hTop, yQ = yA + 34;
    return { narrow, x0, x1, top, yA, yQ, hTop, hBot, yBars: yQ + hBot + (narrow ? 34 : 30), X: v => x0 + (v - XR[0]) / (XR[1] - XR[0]) * (x1 - x0), V: px => XR[0] + (px - x0) / (x1 - x0) * (XR[1] - XR[0]) };
  }
  function drawGrid(ctx, Lg, Gr, sv, alpha, gl) {
    const yel = PE.css('--yellow'), sc = Lg.hBot / (dmax * 1.08);
    ctx.save(); ctx.globalAlpha = alpha; ctx.strokeStyle = yel; ctx.fillStyle = yel;
    const pxStep = Gr.S / (XR[1] - XR[0]) * (Lg.x1 - Lg.x0);
    for (let q = Gr.qmin; q <= Gr.qmax; q++) {
      const v = Gr.S * (q - Gr.Z); if (v < XR[0] || v > XR[1]) continue;
      const x = Lg.X(v);
      ctx.lineWidth = pxStep < 3 ? .6 : 1.2 + 1.5 * gl; ctx.globalAlpha = alpha * (pxStep < 3 ? .55 : .95);
      ctx.beginPath(); ctx.moveTo(x, Lg.yA + 2); ctx.lineTo(x, Lg.yA + (q === Gr.Z ? 16 : 10)); ctx.stroke();
    }
    // quantized values as a density: each code's count spread over one step, hanging below the ruler
    ctx.globalAlpha = alpha * .85;
    ctx.beginPath(); ctx.rect(Lg.x0, Lg.yQ - 2, Lg.x1 - Lg.x0, Lg.hBot + 8); ctx.clip();
    const wpx = Math.max(1, pxStep * .92);
    sv.counts.forEach((c, q) => {
      const v = Gr.S * (q - Gr.Z), hgt = Math.min(Lg.hBot, c / (sv.N * Gr.S) * sc);
      ctx.fillRect(Lg.X(v) - wpx / 2, Lg.yQ, wpx, hgt);
      if (c / (sv.N * Gr.S) * sc > Lg.hBot + 1) { ctx.globalAlpha = alpha; ctx.fillRect(Lg.X(v) - 4, Lg.yQ + Lg.hBot + 3, 8, 2); ctx.globalAlpha = alpha * .85; }
    });
    ctx.restore();
  }
  function draw(ctx, w, h) {
    ctx.clearRect(0, 0, w, h);
    const Lg = L = layout(w, h), C = PE.css, blue = C('--blue'), yel = C('--yellow'), red = C('--red'), mut = C('--muted'), fg = C('--fg'), rule = C('--rule');
    const fs = Lg.narrow ? 10 : 11, sc = Lg.hTop / (dmax * 1.08);
    const gLo = G.S * (G.qmin - G.Z), gHi = G.S * (G.qmax - G.Z);
    const gLoP = Gp.S * (Gp.qmin - Gp.Z), gHiP = Gp.S * (Gp.qmax - Gp.Z);
    const eLo = PE.lerp(gLoP, gLo, S.u), eHi = PE.lerp(gHiP, gHi, S.u);
    // clipped zones
    const gc = glow('clip');
    ctx.save(); ctx.fillStyle = red; ctx.globalAlpha = .1 + .15 * gc;
    if (eLo > XR[0]) ctx.fillRect(Lg.x0, Lg.top - 8, Lg.X(eLo) - Lg.x0, Lg.yQ + Lg.hBot - Lg.top + 8);
    if (eHi < XR[1]) ctx.fillRect(Lg.X(eHi), Lg.top - 8, Lg.x1 - Lg.X(eHi), Lg.yQ + Lg.hBot - Lg.top + 8);
    ctx.restore();
    if (eHi < XR[1] - .6) PE.label(ctx, 'clipped', Lg.X(eHi) + 12, Lg.top + 16, red, fs, 'left', 600);
    // codes no value uses (symmetric grid on one-sided data)
    if (S.sym && -G.S * G.qmax < smp.min - .3) {
      const a = Math.max(XR[0], -G.S * G.qmax), b = smp.min;
      ctx.save(); ctx.globalAlpha = .18 * S.u; ctx.fillStyle = mut; ctx.fillRect(Lg.X(a), Lg.yA + 2, Lg.X(b) - Lg.X(a), 14); ctx.restore();
      if (Lg.X(b) - Lg.X(a) > 70) PE.label(ctx, 'codes no value uses', (Lg.X(a) + Lg.X(b)) / 2, Lg.yA - 8, mut, fs, 'center', 500);
    }
    // float histogram
    const gr = glow('real');
    ctx.save(); ctx.fillStyle = blue; ctx.globalAlpha = .75 + .25 * gr;
    for (let i = 0; i < NB; i++) { const d = hist[i] / (smp.N * bw), x = Lg.X(XR[0] + i * bw); ctx.fillRect(x + .5, Lg.yA - d * sc, Math.max(1, Lg.X(XR[0] + bw) - Lg.x0 - 1), d * sc); }
    ctx.restore();
    PE.label(ctx, Lg.narrow ? 'float values' : 'float values (toy sample of ' + smp.N + ')', Lg.x0, Lg.top - 12, blue, fs, 'left', 600);
    // axis
    PE.polyline(ctx, [[Lg.x0, Lg.yA + .5], [Lg.x1, Lg.yA + .5]], rule, 1);
    for (let v = Math.ceil(XR[0]); v <= XR[1]; v++) { if (Lg.narrow && v % 2) continue; PE.label(ctx, String(v), Lg.X(v), Lg.yA + 28, mut, fs, 'center', 400); }
    // grids: previous fades out, current fades in
    const gs = glow('scale');
    if (S.u < 1) drawGrid(ctx, Lg, Gp, stp, 1 - S.u, 0);
    drawGrid(ctx, Lg, G, st, S.u, gs);
    PE.label(ctx, `after quantization: ${G.bits}-bit ${G.sym ? 'symmetric' : 'asymmetric'} grid`, Lg.x0, Lg.yQ + Lg.hBot + 16, yel, fs, 'left', 600);
    // zero-point marker
    const gz = glow('zero'), xz = Lg.X(0);
    ctx.save(); ctx.fillStyle = yel; ctx.beginPath(); ctx.moveTo(xz, Lg.yA + 17); ctx.lineTo(xz - 5 - 2 * gz, Lg.yA + 25 + 3 * gz); ctx.lineTo(xz + 5 + 2 * gz, Lg.yA + 25 + 3 * gz); ctx.closePath(); ctx.fill(); ctx.restore();
    PE.label(ctx, (Lg.narrow ? '\u25b2 0.0 = code ' : '\u25b2 real 0.0 = code ') + G.Z, Lg.x1, Lg.top - 12, gz > 0 ? fg : yel, fs, 'right', 600);
    // range handles
    [['lo', S.lo], ['hi', S.hi]].forEach(([k, v]) => {
      const x = Lg.X(v), act = S.active === k && document.activeElement === el;
      PE.polyline(ctx, [[x, Lg.top - 2], [x, Lg.yA]], fg, act ? 2 : 1.3, [3, 3], .8);
      PE.dot(ctx, x, Lg.top - 2, act ? 7 : 6, fg); PE.dot(ctx, x, Lg.top - 2, 3, C('--bg'));
    });
    if (S.sym) { const xm = Lg.X(Math.max(XR[0], -G.S * G.qmax)); PE.polyline(ctx, [[xm, Lg.top + 6], [xm, Lg.yA]], yel, 1.2, [2, 4], .8 * S.u); }
    // probe value
    const gp = glow('code'), pr = KIT.RULER_PROBE, pq = KIT.rulerCode(G, pr), pv = G.S * (pq - G.Z);
    PE.dot(ctx, Lg.X(pr), Lg.yA - 3, 4 + 2 * gp, blue); PE.ring(ctx, Lg.X(pv), Lg.yA + 7, 5 + 3 * gp, yel, 1.6);
    // error bars
    KIT.rulerBars(ctx, Lg.x0, Lg.yBars, Lg.x1 - Lg.x0, [
      { name: 'rounding error (MSE)', v: PE.lerp(stp.round, st.round, S.u), link: 'round' },
      { name: 'clipping error (MSE)', v: PE.lerp(stp.clip, st.clip, S.u), link: 'clip', alpha: .6 }], glow);
  }
  const cv = PE.canvas(el, opts.aspect || (w => w < 560 ? 1.05 : .56), draw);
  let stop = () => {};
  const tgt = { lo: S.lo, hi: S.hi };   // targets survive an interrupted tween
  function set(next) {
    if (next.lo != null) tgt.lo = next.lo;
    if (next.hi != null) tgt.hi = next.hi;
    const to = { lo: tgt.lo, hi: tgt.hi, u: 1 };
    const gridChange = (next.bits && next.bits !== S.bits) || (next.sym != null && next.sym !== S.sym);
    stop();
    if (gridChange) { Gp = G; stp = st; S.u = 0; }
    if (next.bits) S.bits = next.bits;
    if (next.sym != null) S.sym = next.sym;
    if (!gridChange) { Gp = G; stp = st; }
    stop = KIT.tween(S, to, 650, () => { recompute(); cv.redraw(); }, () => { Gp = G; stp = st; cv.redraw(); });
  }
  // dragging the range ends, and arrow keys when the canvas has focus
  el.tabIndex = 0;
  const pick = e => { const r = el.getBoundingClientRect(); return { x: e.clientX - r.left, y: e.clientY - r.top }; };
  const move = v => {
    if (S.active === 'lo') S.lo = PE.clamp(v, XR[0], Math.min(0, S.hi - .2));
    else S.hi = PE.clamp(v, Math.max(0, S.lo + .2), XR[1]);
    tgt.lo = S.lo; tgt.hi = S.hi; Gp = G; stp = st; S.u = 1; recompute(); cv.redraw();
  };
  const onDown = e => {
    if (!L) return;
    const p = pick(e), dl = Math.abs(p.x - L.X(S.lo)), dh = Math.abs(p.x - L.X(S.hi));
    if (Math.min(dl, dh) > 28) return;
    stop(); S.active = dl < dh ? 'lo' : 'hi'; S.drag = true; el.setPointerCapture(e.pointerId); e.preventDefault();
  };
  const onMove = e => { if (S.drag) move(L.V(pick(e).x)); };
  const onUp = () => { if (!S.drag) return; S.drag = null; opts.onInput && opts.onInput(); };
  const onKey = e => {
    if (e.key === 'ArrowUp' || e.key === 'ArrowDown') { S.active = S.active === 'lo' ? 'hi' : 'lo'; cv.redraw(); e.preventDefault(); return; }
    if (e.key !== 'ArrowLeft' && e.key !== 'ArrowRight') return;
    e.preventDefault(); stop();
    move((S.active === 'lo' ? S.lo : S.hi) + (e.key === 'ArrowRight' ? 1 : -1) * (e.shiftKey ? .5 : .1));
    opts.onInput && opts.onInput();
  };
  const onFocus = () => cv.redraw();
  el.addEventListener('pointerdown', onDown); el.addEventListener('pointermove', onMove);
  el.addEventListener('pointerup', onUp); el.addEventListener('pointercancel', onUp); el.addEventListener('keydown', onKey);
  el.addEventListener('focus', onFocus); el.addEventListener('blur', onFocus);
  function info() {
    const pq = KIT.rulerCode(G, KIT.RULER_PROBE);
    return { G, st, lo: S.lo, hi: S.hi, probe: { r: KIT.RULER_PROBE, q: pq, rhat: G.S * (pq - G.Z), bits: KIT.bits(pq, G.bits, G.sym) } };
  }
  return { set, info, state: S, redraw: cv.redraw, resize: cv.resize,
    destroy: () => { stop(); cv.destroy(); el.removeEventListener('pointerdown', onDown); el.removeEventListener('pointermove', onMove);
      el.removeEventListener('pointerup', onUp); el.removeEventListener('pointercancel', onUp); el.removeEventListener('keydown', onKey);
      el.removeEventListener('focus', onFocus); el.removeEventListener('blur', onFocus); } };
};
KIT.rulerText = i => {
  const G = i.G, s = i.st, f = v => (+v.toPrecision(3)).toString();
  return `<b>${G.bits}-bit ${G.sym ? 'symmetric' : 'asymmetric'}</b> on [${i.lo.toFixed(2)}, ${i.hi.toFixed(2)}]: ` +
    `S = ${f(G.S)}, Z = ${G.Z}. The data use ${s.used} of ${G.levels} codes. ` +
    `<b class="c-red">Rounding error ${KIT.rulerNum(s.round)}</b>, <b class="c-red">clipping error ${KIT.rulerNum(s.clip)}</b> ` +
    `(${s.nClip} of ${s.N} values clipped). ` +
    `r = ${i.probe.r} &rarr; q = ${i.probe.q} = <span class="bits">${i.probe.bits}</span> &rarr; ${f(i.probe.rhat)}.`;
};
