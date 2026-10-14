/* viz-intmm: one output value of a 3x3 conv computed with integers only, as in Jacob Sec. 2.2-2.4.
   Toy weights, inputs, bias and ranges; the arithmetic is exact: uint8 codes, an int32 accumulator, the
   zero-point cross terms of Eq. 7-8, an int32 bias with S_bias = S1 S2 (Eq. 11), and requantization by
   M = S1 S2 / S3 = 2^-n M0 (Eq. 5-6): a rounding fixed-point multiply by M0 (as gemmlowp does it, high 32 bits
   of the doubled product) then a rounding right shift by n, ties away from zero (Jacob App. B). */
KIT.INTMM = {
  w: [-.62, .05, .71, -.88, .12, .79, -.47, -.03, .54],
  x: [-.71, .18, .86, -.64, .25, .93, -.58, .31, .77].map(v => v * .55),
  bias: .15, wr: [-1, .8], xr: [-1, 1], yr: [-2, 4],
};
KIT.intmmAffine = (a, b) => { const S = (b - a) / 255; return { S, Z: Math.min(255, Math.max(0, Math.round(-a / S))) }; };
KIT.intmmRun = (P = KIT.INTMM) => {
  const A1 = KIT.intmmAffine(...P.wr), A2 = KIT.intmmAffine(...P.xr), A3 = KIT.intmmAffine(...P.yr);
  const code = (r, A) => Math.min(255, Math.max(0, Math.round(r / A.S) + A.Z));
  const q1 = P.w.map(r => code(r, A1)), q2 = P.x.map(r => code(r, A2)), N = q1.length;
  let core = 0, sum1 = 0, sum2 = 0, yf = P.bias;
  for (let j = 0; j < N; j++) { core += q1[j] * q2[j]; sum1 += q1[j]; sum2 += q2[j]; yf += P.w[j] * P.x[j]; }
  const t1 = -A1.Z * sum2, t2 = -A2.Z * sum1, t3 = N * A1.Z * A2.Z;
  const Sb = A1.S * A2.S, qb = Math.round(P.bias / Sb);
  const acc = core + t1 + t2 + t3 + qb;
  const M = A1.S * A2.S / A3.S;
  let n = 0, M0 = M; while (M0 < .5) { M0 *= 2; n++; }
  const M0i = Math.round(M0 * 2 ** 31);
  const ab = BigInt(acc) * BigInt(M0i), nudge = ab >= 0n ? (1n << 30n) : (1n - (1n << 30n));
  const hi = Number((ab + nudge) / (1n << 31n));
  const mask = (1 << n) - 1, rem = hi & mask, thr = (mask >> 1) + (hi < 0 ? 1 : 0);
  const sh = (hi >> n) + (rem > thr ? 1 : 0);
  const q3raw = A3.Z + sh, q3 = Math.min(255, Math.max(0, q3raw)), yhat = A3.S * (q3 - A3.Z);
  return { P, A1, A2, A3, q1, q2, N, core, sum1, sum2, t1, t2, t3, Sb, qb, acc, M, n, M0, M0i, hi, sh, q3raw, q3, yhat, yf, err: yhat - yf };
};
KIT.INTMM_STEPS = ['real numbers', 'uint8 codes', 'int32 sum', 'zero-point terms', 'requantize'];
KIT.intmmNum = v => (v < 0 ? '\u2212' : '') + Math.abs(v).toLocaleString('en-US');
KIT.intmm = (el, opts = {}) => {
  const R = KIT.intmmRun(), S = { step: 1, a: [1, 0, 0, 0, 0] };
  const glow = opts.fig ? PE.linkPulses(opts.fig, () => cv.redraw()) : () => 0;
  const nm = KIT.intmmNum, f4 = v => v.toFixed(4);
  function rows() {
    const A1 = R.A1, A2 = R.A2;
    return [
      { s: 2, t: `int32 += uint8 * uint8:  \u03a3 q1\u00b7q2 = ${nm(R.core)}`, c: '--yellow', link: 'core' },
      { s: 3, t: `\u2212 Z1\u00b7\u03a3q2 = \u2212${A1.Z} \u00d7 ${nm(R.sum2)} = ${nm(R.t1)}`, c: '--red', link: 'cross' },
      { s: 3, t: `\u2212 Z2\u00b7\u03a3q1 = \u2212${A2.Z} \u00d7 ${nm(R.sum1)} = ${nm(R.t2)}`, c: '--red', link: 'cross' },
      { s: 3, t: `+ N\u00b7Z1\u00b7Z2 = ${R.N} \u00d7 ${A1.Z} \u00d7 ${A2.Z} = ${nm(R.t3)}`, c: '--red', link: 'const' },
      { s: 3, t: `+ bias in int32, scale S1\u00b7S2: ${nm(R.qb)}`, c: '--muted', link: 'bias' },
      { s: 3, t: `= accumulator ${nm(R.acc)}`, c: '--fg', link: 'acc', bold: true },
      { s: 4, t: `M = S1\u00b7S2/S3 = ${R.M.toExponential(4)} = 2^\u2212${R.n} \u00d7 ${R.M0.toFixed(5)}`, c: '--green', link: 'mult' },
      { s: 4, t: `\u00d7 M0 (int32 ${nm(R.M0i)}, then >> 31): ${nm(R.hi)}`, c: '--green', link: 'mult' },
      { s: 4, t: `rounding shift >> ${R.n}: ${nm(R.sh)};  + Z3 = ${R.A3.Z}: ${R.q3raw}`, c: '--green', link: 'shift' },
      { s: 4, t: `uint8 output q3 = ${R.q3}`, c: '--yellow', link: 'zout', bold: true, bits: KIT.bits(R.q3, 8) },
    ];
  }
  function cellGrid(ctx, x0, y0, cs, vals, codes, A, name, sub, gl) {
    const C = PE.css, a2 = S.a[1];
    PE.label(ctx, name, x0, y0 - 8, C('--fg'), 12, 'left', 600);
    for (let k = 0; k < 9; k++) {
      const x = x0 + (k % 3) * cs, y = y0 + Math.floor(k / 3) * cs;
      ctx.save(); ctx.strokeStyle = gl > 0 ? C('--fg') : C('--rule'); ctx.lineWidth = 1 + gl; ctx.fillStyle = C('--panel');
      ctx.beginPath(); ctx.roundRect(x + 1, y + 1, cs - 2, cs - 2, 5); ctx.fill(); ctx.stroke(); ctx.restore();
      const fz = cs < 46 ? 10.5 : 12;
      if (a2 < 1) { ctx.save(); ctx.globalAlpha = 1 - a2; PE.label(ctx, vals[k].toFixed(2), x + cs / 2, y + cs / 2 + 4, C('--blue'), fz, 'center', 500); ctx.restore(); }
      if (a2 > 0) { ctx.save(); ctx.globalAlpha = a2; PE.label(ctx, String(codes[k]), x + cs / 2, y + cs / 2 + 4, C('--yellow'), fz + 1, 'center', 600); ctx.restore(); }
    }
    const t = a2 > .5 ? `S = ${A.S.toFixed(5)}, Z = ${A.Z}` : sub;
    PE.label(ctx, t, x0, y0 + 3 * cs + 15, C('--muted'), 10.5, 'left', 400);
  }
  function draw(ctx, w, h) {
    ctx.clearRect(0, 0, w, h);
    const C = PE.css, narrow = w < 560, pad = opts.bottomPad || 0;
    const cs = narrow ? Math.min(44, (w - 40) / 6.6) : Math.min(52, w / 17);
    const gx = narrow ? 10 : 18, gy = 34, gc = glow('codes');
    cellGrid(ctx, gx, gy, cs, R.P.w, R.q1, R.A1, 'weights r1', 'layer range [\u22121, 0.8]', gc);
    cellGrid(ctx, gx + 3 * cs + (narrow ? 22 : 28), gy, cs, R.P.x, R.q2, R.A2, 'inputs r2', 'pixels in [\u22121, 1]', gc);
    const lx = narrow ? 10 : gx + 6 * cs + 70, ly0 = narrow ? gy + 3 * cs + 46 : gy + 2;
    const lh = narrow ? 19 : 21, fs = narrow ? 11 : 12.5;
    // float reference, always shown
    PE.label(ctx, `float reference: \u03a3 w\u00b7x + b = ${f4(R.yf)}`, lx, ly0 + 4, C('--blue'), fs, 'left', 600);
    rows().forEach((r, i) => {
      const al = S.a[r.s]; if (al <= 0) return;
      const y = ly0 + 4 + (i + 1) * lh + (r.s >= 3 ? 6 : 0) + (r.s >= 4 ? 6 : 0), g = glow(r.link);
      ctx.save(); ctx.globalAlpha = al;
      if (g > 0) { ctx.fillStyle = C('--yellow'); ctx.globalAlpha = al * .18 * g; ctx.fillRect(lx - 4, y - lh + 5, (narrow ? w - 16 : w - lx - 8), lh); ctx.globalAlpha = al; }
      PE.label(ctx, r.t, lx + (1 - al) * 18, y, C(r.c), fs, 'left', r.bold ? 700 : 500);
      if (r.bits) { ctx.font = `700 ${fs}px system-ui, sans-serif`; const off = ctx.measureText(r.t).width;
        ctx.font = `600 ${fs}px ui-monospace, Menlo, monospace`; ctx.fillStyle = C('--yellow'); ctx.fillText('= ' + r.bits, lx + (1 - al) * 18 + off + 8, y); }
      ctx.restore();
    });
    const al = S.a[4];
    if (al > 0) {
      const y = ly0 + 4 + 11 * lh + 18;
      ctx.save(); ctx.globalAlpha = al;
      PE.label(ctx, `S3\u00b7(q3 \u2212 Z3) = ${f4(R.yhat)}  vs float ${f4(R.yf)}`, lx, y, C('--fg'), fs, 'left', 600);
      PE.label(ctx, `error ${R.err >= 0 ? '+' : '\u2212'}${Math.abs(R.err).toFixed(4)}, half an output step S3/2 = ${(R.A3.S / 2).toFixed(4)}`, lx, y + lh, C('--red'), fs, 'left', 600);
      ctx.restore();
    }
    void pad;
  }
  const cv = PE.canvas(el, opts.aspect || (w => w < 560 ? 1.32 : .37), draw);
  let stop = () => {};
  function set(next) {
    if (!next.step) return;
    S.step = next.step;
    const to = {}; S.a.forEach((v, i) => { to['a' + i] = i < next.step ? 1 : 0; });
    const box = {}; S.a.forEach((v, i) => { box['a' + i] = v; });
    stop(); stop = KIT.tween(box, to, 600, () => { S.a = S.a.map((_, i) => box['a' + i]); cv.redraw(); });
  }
  return { set, info: () => ({ R, step: S.step }), state: S, redraw: cv.redraw, resize: cv.resize, destroy: () => { stop(); cv.destroy(); } };
};
KIT.intmmText = i => {
  const R = i.R, n = KIT.intmmNum;
  return [
    `Step 1. Nine real weights and nine real inputs (a 3x3 patch of pixels scaled to [&minus;1, 1]). In floating point the output is ${R.yf.toFixed(4)}.`,
    `Step 2. Each number becomes a uint8 code q = round(r/S) + Z. Weights: S1 = ${R.A1.S.toFixed(5)}, Z1 = ${R.A1.Z}. Inputs: S2 = ${R.A2.S.toFixed(5)}, Z2 = ${R.A2.Z}. The first weight, ${R.P.w[0]}, is code ${R.q1[0]} = <span class="bits">${KIT.bits(R.q1[0], 8)}</span>.`,
    `Step 3. Multiply codes pairwise and add in a 32-bit integer: ${n(R.core)}. This is the only part that grows with the layer size.`,
    `Step 4. Subtract the zero-point terms and add the bias (stored as int32 with scale S1&middot;S2): accumulator ${n(R.acc)} = &Sigma;(q1 &minus; Z1)(q2 &minus; Z2) + bias.`,
    `Step 5. Scale by M = 2<sup>&minus;${R.n}</sup> &times; ${R.M0.toFixed(5)} with an integer multiply and a shift, add Z3 = ${R.A3.Z}: output code ${R.q3} = <span class="bits">${KIT.bits(R.q3, 8)}</span>, which means ${R.yhat.toFixed(4)}. Float gives ${R.yf.toFixed(4)}; the error is ${R.err.toFixed(4)}, less than half an output step (${(R.A3.S / 2).toFixed(4)}).`,
  ][i.step - 1];
};
