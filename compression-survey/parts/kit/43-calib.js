/* viz-calib: four calibrators pick a clip value T for a toy long-tailed ReLU activation sample; the grid is
   symmetric signed 8-bit with S = T/127 (ReLU outputs use codes 0..127), as for a symmetric INT8 activation.
   max: largest value. percentile: the 99.99th percentile of |x| over all values (Wu Sec. 3.4).
   entropy: our reading of the KL method Wu Sec. 3.4 describes in words: a 2048-bin histogram of the nonzero values
   on [0, max]; for each candidate end bin i >= 128, P = the first i bins with all mass above added to bin i, Q = the
   first i bins merged into 128 groups and spread back evenly over each group's nonempty bins; T = the edge with the
   smallest KL(P || Q). The exact zeros that ReLU makes are left out (zero is exact on the grid; our choice).
   MSE: the T that minimizes the quantization MSE on the sample, Nagel Eq. 16, by a grid search (100 coarse values, then 100 finer ones around the best). */
KIT.CALIB_N = 100000;
KIT.calibSample = (() => {
  let cache = null;
  return () => {
    if (cache) return cache;
    const r = PE.rng(17), g = PE.gauss(r), N = KIT.CALIB_N, xs = new Float64Array(N);
    for (let i = 0; i < N; i++) {
      const v = r() < .97 ? .1 + .3 * g() : .2 + .5 * Math.abs(g()) * Math.abs(g());
      xs[i] = v > 0 ? v : 0;
    }
    const sorted = Float64Array.from(xs).sort();
    cache = { xs, sorted, N, max: sorted[N - 1] };
    return cache;
  };
})();
KIT.calibErr = (xs, T) => {
  const s = T / 127; let er = 0, ec = 0, nc = 0;
  const n = xs.length, inv = 1 / s;
  for (let i = 0; i < n; i++) { const x = xs[i], raw = Math.round(x * inv); if (raw > 127) { const d = x - T; ec += d * d; nc++; } else { const d = x - s * raw; er += d * d; } }
  return { round: er / n, clip: ec / n, mse: (er + ec) / n, nClip: nc };
};
KIT.calibKL = (smp, NB = 2048, NQ = 128) => {
  const bw = smp.max / NB, h = new Float64Array(NB);
  for (const x of smp.xs) if (x > 0) h[Math.min(NB - 1, Math.floor(x / bw))]++;
  let best = Infinity, bi = NB; const kls = [], suf = new Float64Array(NB + 1);
  for (let j = NB - 1; j >= 0; j--) suf[j] = suf[j + 1] + h[j];
  for (let i = NQ; i <= NB; i++) {
    const tail = suf[i];
    const P = h.slice(0, i); P[i - 1] += tail;
    const Q = new Float64Array(i);
    for (let gq = 0; gq < NQ; gq++) {
      const a = Math.floor(gq * i / NQ), b = Math.floor((gq + 1) * i / NQ);
      let sum = 0, nz = 0; for (let j = a; j < b; j++) { sum += h[j]; if (h[j] > 0) nz++; }
      for (let j = a; j < b; j++) Q[j] = h[j] > 0 ? sum / nz : 0;
    }
    let sp = 0, sq = 0; for (let j = 0; j < i; j++) { sp += P[j]; sq += Q[j]; }
    let kl = 0;
    for (let j = 0; j < i; j++) if (P[j] > 0) { const p = P[j] / sp, q = Q[j] / sq; kl += q > 0 ? p * Math.log(p / q) : 1e3; }
    kls.push(kl); if (kl <= best) { best = kl; bi = i; }
  }
  return { T: bi * bw, kl: best, kls, bw };
};
KIT.calibMSE = smp => {
  // coarse grid of 100 values, then 100 finer values around the best; exact MSE on the whole sample
  let best = Infinity, bT = smp.max;
  for (let k = 1; k <= 100; k++) { const T = smp.max * k / 100, e = KIT.calibErr(smp.xs, T).mse; if (e < best) { best = e; bT = T; } }
  const lo = bT - smp.max / 100;
  for (let k = 0; k <= 100; k++) { const T = lo + smp.max / 50 * k / 100, e = KIT.calibErr(smp.xs, T).mse; if (e < best) { best = e; bT = T; } }
  return { T: bT };
};
KIT.calibClips = (() => {
  let cache = null;
  return () => {
    if (cache) return cache;
    const smp = KIT.calibSample(), N = smp.N;
    const pct = smp.sorted[Math.min(N - 1, Math.ceil(.9999 * N) - 1)];
    cache = { max: smp.max, p9999: pct, entropy: KIT.calibKL(smp).T, mse: KIT.calibMSE(smp).T };
    return cache;
  };
})();
KIT.CALIB_NAMES = { max: 'max', p9999: 'percentile 99.99', entropy: 'entropy (KL)', mse: 'MSE' };
KIT.calib = (el, opts = {}) => {
  const smp = KIT.calibSample(), clips = KIT.calibClips(), errs = {};
  for (const k in clips) errs[k] = KIT.calibErr(smp.xs, clips[k]);
  const S = { method: 'max', T: clips.max, u: 1, prev: 'max' };
  const NB = 160, XM = smp.max * 1.04, bw = XM / NB, h = new Float64Array(NB);
  for (const x of smp.xs) h[Math.min(NB - 1, Math.floor(x / bw))]++;
  const glow = opts.fig ? PE.linkPulses(opts.fig, () => cv.redraw()) : () => 0;
  function draw(ctx, w, h0) {
    ctx.clearRect(0, 0, w, h0);
    const C = PE.css, narrow = w < 560, pad = opts.bottomPad || 0, fs = narrow ? 10 : 11;
    const b = { x: narrow ? 40 : 56, y: 28, w: w - (narrow ? 52 : 76), h: h0 - 28 - pad - (narrow ? 120 : 108) };
    const A = PE.axes(ctx, b, [0, XM], [.5, 1e5], { logy: true, xt: (narrow ? [0, 1, 2, 3, 4] : [0, .5, 1, 1.5, 2, 2.5, 3, 3.5, 4]).filter(v => v < XM), yt: [1, 10, 100, 1000, 10000],
      yf: v => v >= 1000 ? v / 1000 + 'k' : String(v), xl: 'activation value after ReLU', yl: 'count per bin (log)', ylo: narrow ? 30 : 42 });
    const T = S.T, gh = glow('hist');
    ctx.save(); ctx.globalAlpha = .8 + .2 * gh;
    for (let i = 0; i < NB; i++) { if (!h[i]) continue; const x0 = A.X(i * bw), x1 = A.X((i + 1) * bw), y = A.Y(h[i]);
      ctx.fillStyle = (i + .5) * bw > T ? C('--red') : C('--blue'); ctx.fillRect(x0 + .3, y, Math.max(1, x1 - x0 - .6), b.y + b.h - y); }
    ctx.restore();
    // clipped zone and the 8-bit grid up to T
    const gc = glow('clip');
    ctx.save(); ctx.fillStyle = C('--red'); ctx.globalAlpha = .1 + .2 * gc; ctx.fillRect(A.X(T), b.y, b.x + b.w - A.X(T), b.h); ctx.restore();
    const s = T / 127, gg = glow('grid');
    ctx.save(); ctx.strokeStyle = C('--yellow'); ctx.globalAlpha = .55 + .4 * gg; ctx.lineWidth = .7 + gg;
    for (let q = 0; q <= 127; q++) { const x = A.X(q * s); ctx.beginPath(); ctx.moveTo(x, b.y + b.h + 1); ctx.lineTo(x, b.y + b.h + (q % 16 ? 5 : 9)); ctx.stroke(); }
    ctx.restore();
    // the four clip values
    const order = ['p9999', 'mse', 'entropy', 'max'].sort((p, q) => clips[p] - clips[q]);
    order.forEach((k, j) => {
      const x = A.X(clips[k]), sel = k === S.method, gk = glow(k === 'entropy' ? 'kl' : k === 'mse' ? 'mse' : '');
      PE.polyline(ctx, [[x, b.y + 4], [x, b.y + b.h]], sel ? C('--fg') : C('--muted'), sel ? 2 + gk : 1 + 2 * gk, [4, 3], sel ? 1 : .8);
      PE.label(ctx, KIT.CALIB_NAMES[k], x + (j % 2 ? -4 : 4), b.y + 10 + (j % 2) * 14 + (narrow ? (j > 1 ? 26 : 0) : 0), sel ? C('--fg') : C('--muted'), fs, j % 2 ? 'right' : 'left', sel ? 700 : 500);
    });
    const e = errs[S.method], ep = errs[S.prev];
    KIT.rulerBars(ctx, b.x - (narrow ? 30 : 40), b.y + b.h + 44, b.w + (narrow ? 30 : 40), [
      { name: 'rounding error (MSE)', v: Math.pow(10, PE.lerp(Math.log10(ep.round || 1e-9), Math.log10(e.round || 1e-9), S.u)), link: 'round' },
      { name: 'clipping error (MSE)', v: S.u >= 1 ? e.clip : Math.pow(10, PE.lerp(Math.log10(ep.clip || 1e-9), Math.log10(e.clip || 1e-9), S.u)), link: 'clip', alpha: .6 }], glow);
  }
  const cv = PE.canvas(el, opts.aspect || (w => w < 560 ? 1.05 : .52), draw);
  let stop = () => {};
  function set(next) {
    if (!next.method || !clips[next.method]) return;
    S.prev = S.method; S.method = next.method; S.u = 0; stop();
    stop = KIT.tween(S, { T: clips[next.method], u: 1 }, 700, cv.redraw);
  }
  return { set, info: () => ({ method: S.method, T: clips[S.method], e: errs[S.method], N: smp.N, clips, errs }), state: S, redraw: cv.redraw, resize: cv.resize, destroy: () => { stop(); cv.destroy(); } };
};
KIT.calibText = i => {
  const f = v => KIT.rulerNum(v), e = i.e;
  return `<b>${KIT.CALIB_NAMES[i.method]}</b>: clip at T = ${i.T.toFixed(3)}, step S = T/127 = ${(i.T / 127).toFixed(4)}. ` +
    `${e.nClip} of ${i.N.toLocaleString('en-US')} values clipped. Total MSE ${f(e.mse)} = <b class="c-red">rounding ${f(e.round)}</b> + <b class="c-red">clipping ${f(e.clip)}</b>.`;
};
