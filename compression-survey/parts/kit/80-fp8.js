/* FP8 formats. The code tables are enumerated from Micikevicius 2209.05433 Table 1 (E4M3: bias 7, no inf, NaN only
   S.1111.111; E5M2: bias 15, inf and NaN at exponent field 31). Rounding is round-to-nearest-even with saturation,
   as TensorRT's castToFp8. The tensors and the max scaling are ours; notes/sim-fp8.txt has the numbers. */
KIT.FP8_FMT = { e4m3: { E: 4, M: 3, bias: 7, name: 'E4M3' }, e5m2: { E: 5, M: 2, bias: 15, name: 'E5M2' } };
KIT.fp8Decode = (code, kind) => {
  const F = KIT.FP8_FMT[kind], s = code >> 7, e = (code >> F.M) & ((1 << F.E) - 1), m = code & ((1 << F.M) - 1);
  const top = (1 << F.E) - 1;
  let v, cls;
  if (kind === 'e4m3' && e === top && m === (1 << F.M) - 1) { v = NaN; cls = 'nan'; }
  else if (kind === 'e5m2' && e === top) { v = m === 0 ? Infinity : NaN; cls = m === 0 ? 'inf' : 'nan'; }
  else if (e === 0) { v = (m / (1 << F.M)) * Math.pow(2, 1 - F.bias); cls = m === 0 ? 'zero' : 'sub'; }
  else { v = (1 + m / (1 << F.M)) * Math.pow(2, e - F.bias); cls = 'normal'; }
  return { v: s && !isNaN(v) ? -v : v, cls, s, e, m };
};
KIT._fp8Grids = {};
// Non-negative finite values with their codes, ascending (zero included).
KIT.fp8Grid = kind => {
  if (KIT._fp8Grids[kind]) return KIT._fp8Grids[kind];
  const pos = [];
  for (let c = 0; c < 128; c++) { const d = KIT.fp8Decode(c, kind); if (isFinite(d.v)) pos.push({ v: d.v, c, cls: d.cls }); }
  pos.sort((a, b) => a.v - b.v);
  return (KIT._fp8Grids[kind] = pos);
};
KIT.fp8Max = kind => { const g = KIT.fp8Grid(kind); return g[g.length - 1].v; };
KIT.fp8Round = (x, kind) => {
  const pos = KIT.fp8Grid(kind), a = Math.abs(x);
  let lo = 0, hi = pos.length - 1;
  if (a >= pos[hi].v) return x < 0 ? -pos[hi].v : pos[hi].v;
  while (hi - lo > 1) { const mid = (lo + hi) >> 1; if (pos[mid].v <= a) lo = mid; else hi = mid; }
  const dl = a - pos[lo].v, dh = pos[hi].v - a;
  const p = dl < dh ? pos[lo] : dh < dl ? pos[hi] : ((pos[lo].c & 1) ? pos[hi] : pos[lo]);
  return x < 0 ? -p.v : p.v;
};
KIT.rne = x => { const f = Math.floor(x), d = x - f; return d > .5 ? f + 1 : d < .5 ? f : (f % 2 === 0 ? f : f + 1); };
KIT.int8Round = x => Math.max(-128, Math.min(127, KIT.rne(x)));
KIT.FP8_N = 16384;
KIT.FP8_DISTS = {
  uniform: { name: 'Uniform', long: 'uniform' },
  normal: { name: 'Normal', long: 'normal (Gaussian)' },
  student: { name: 'Student-t', long: 'Student-t with 3 degrees of freedom (heavy tails)' },
};
KIT._fp8Cache = {};
KIT.fp8Sample = dist => {
  const r = PE.rng(7), g = PE.gauss(r), n = KIT.FP8_N, nu = 3, xs = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    let x;
    if (dist === 'uniform') x = 2 * r() - 1;
    else if (dist === 'normal') x = g();
    else { let c = 0; for (let k = 0; k < nu; k++) { const z = g(); c += z * z; } x = g() / Math.sqrt(c / nu); }
    xs[i] = x;
  }
  let ss = 0; for (let i = 0; i < n; i++) ss += xs[i] * xs[i];
  const rms = Math.sqrt(ss / n); for (let i = 0; i < n; i++) xs[i] /= rms;
  return xs;
};
// Per-tensor max scaling: the format's largest value lands on the tensor's largest |x|.
KIT.fp8Stats = dist => {
  if (KIT._fp8Cache[dist]) return KIT._fp8Cache[dist];
  const xs = KIT.fp8Sample(dist), n = xs.length;
  let M = 0; for (let i = 0; i < n; i++) M = Math.max(M, Math.abs(xs[i]));
  const scale = { int8: M / 127, e4m3: M / KIT.fp8Max('e4m3'), e5m2: M / KIT.fp8Max('e5m2') }, mse = {};
  for (const k of ['int8', 'e4m3', 'e5m2']) {
    const s = scale[k]; let e = 0;
    for (let i = 0; i < n; i++) { const q = k === 'int8' ? KIT.int8Round(xs[i] / s) * s : KIT.fp8Round(xs[i] / s, k) * s; e += (q - xs[i]) * (q - xs[i]); }
    mse[k] = e / n;
  }
  const bits = {}; for (const k in mse) bits[k] = -Math.log2(Math.sqrt(mse[k]));
  const best = Object.keys(mse).reduce((a, b) => mse[a] <= mse[b] ? a : b);
  return (KIT._fp8Cache[dist] = { dist, n, M, scale, mse, bits, best, xs });
};
KIT.FP8_AX = [Math.pow(2, -18), Math.pow(2, 6)];
KIT.FP8_BINS = 72;
KIT.fp8Hist = dist => {
  const st = KIT.fp8Stats(dist), [a, b] = KIT.FP8_AX.map(Math.log2), nb = KIT.FP8_BINS, h = new Array(nb).fill(0);
  for (const x of st.xs) { const ax = Math.abs(x); if (ax <= 0) continue; const i = Math.floor((Math.log2(ax) - a) / (b - a) * nb); if (i >= 0 && i < nb) h[i]++; }
  return h;
};
KIT.fp8 = (el, opts = {}) => {
  const S = { dist: 'normal', logM: 0, hist: [] };
  const tgt = d => ({ logM: Math.log2(KIT.fp8Stats(d).M), hist: KIT.fp8Hist(d) });
  Object.assign(S, tgt(S.dist));
  const glow = opts.fig ? PE.linkPulses(opts.fig, () => cv.redraw()) : () => 0;
  const NAMES = { int8: 'INT8', e4m3: 'E4M3', e5m2: 'E5M2' };
  function codeMap(ctx, kind, x, y, w, h, C) {
    const F = KIT.FP8_FMT[kind], cols = 1 << F.E, half = 1 << F.M, rows = 2 * half;
    const cw = Math.min(w / cols, 26), ch = Math.max(4, Math.min(h / rows, Math.max(5, cw * .55))), mw = cw * cols, mh = ch * rows;
    const gs = glow('sign'), ge = glow('exp'), gm = glow('mant'), gsub = glow('sub');
    PE.label(ctx, F.name + ': all 256 codes', x, y - 6, C('--fg'), 12, 'left', 600);
    for (let s = 0; s < 2; s++) for (let m = 0; m < half; m++) for (let e = 0; e < cols; e++) {
      const code = (s << 7) | (e << F.M) | m, d = KIT.fp8Decode(code, kind), cx = x + e * cw, cy = y + (s * half + m) * ch;
      ctx.save();
      if (d.cls === 'nan') { ctx.fillStyle = C('--red'); ctx.globalAlpha = .85; ctx.fillRect(cx + .5, cy + .5, cw - 1, ch - 1); }
      else if (d.cls === 'inf') { ctx.strokeStyle = C('--red'); ctx.lineWidth = 1.5; ctx.strokeRect(cx + 1.2, cy + 1.2, cw - 2.4, ch - 2.4); }
      else if (d.cls === 'zero') { ctx.fillStyle = C('--muted'); ctx.globalAlpha = .6; ctx.fillRect(cx + .5, cy + .5, cw - 1, ch - 1); }
      else { ctx.fillStyle = C('--yellow'); ctx.globalAlpha = d.cls === 'sub' ? .38 : .5 + .45 * (e / cols); ctx.fillRect(cx + .5, cy + .5, cw - 1, ch - 1); }
      ctx.restore();
    }
    if (gsub > 0) { ctx.save(); ctx.strokeStyle = C('--red'); ctx.lineWidth = 1.5 + 2 * gsub; ctx.globalAlpha = gsub; ctx.strokeRect(x, y, cw, mh); ctx.restore(); }
    if (gs > 0) { ctx.save(); ctx.strokeStyle = C('--fg'); ctx.lineWidth = 1 + 2 * gs; ctx.globalAlpha = gs;
      ctx.strokeRect(x - 2, y - 1, mw + 4, half * ch + 1); ctx.strokeRect(x - 2, y + half * ch + 1, mw + 4, half * ch); ctx.restore(); }
    PE.label(ctx, '+', x - 5, y + half * ch / 2 + 4, gs > 0 ? C('--fg') : C('--muted'), 12, 'right', 600);
    PE.label(ctx, '\u2212', x - 5, y + half * ch * 1.5 + 4, gs > 0 ? C('--fg') : C('--muted'), 12, 'right', 600);
    const lc = n => n > 0 ? C('--fg') : C('--muted');
    const two = mw < 330;
    PE.label(ctx, F.E + ' exponent bits: ' + cols + ' columns', x, y + mh + 14, lc(ge), 10.5, 'left', ge > 0 ? 600 : 400);
    PE.label(ctx, F.M + ' mantissa bits: ' + half + ' rows per sign', two ? x : x + mw, y + mh + (two ? 27 : 14), lc(gm), 10.5, two ? 'left' : 'right', gm > 0 ? 600 : 400);
    if (ge > 0) { ctx.save(); ctx.strokeStyle = C('--fg'); ctx.globalAlpha = .5 * ge; for (let e = 1; e < cols; e++) { ctx.beginPath(); ctx.moveTo(x + e * cw, y); ctx.lineTo(x + e * cw, y + mh); ctx.stroke(); } ctx.restore(); }
    if (gm > 0) { ctx.save(); ctx.strokeStyle = C('--fg'); ctx.globalAlpha = .5 * gm; for (let r = 1; r < rows; r++) { ctx.beginPath(); ctx.moveTo(x, y + r * ch); ctx.lineTo(x + mw, y + r * ch); ctx.stroke(); } ctx.restore(); }
    return mh + (two ? 31 : 18);
  }
  function draw(ctx, w, h) {
    ctx.clearRect(0, 0, w, h);
    const narrow = KIT.narrow(w), C = n => PE.css(n), H = h - (opts.bottomPad || 0);
    const st = KIT.fp8Stats(S.dist), M = Math.pow(2, S.logM);
    const L = narrow ? 44 : 66, R = narrow ? 10 : 170, top = narrow ? 30 : 34;
    const hh = Math.max(56, H * (narrow ? .17 : .22)), rh = narrow ? 30 : PE.clamp(H * .06, 20, 30);
    const b = { x: L, y: top, w: w - L - R, h: hh };
    const [la, lb] = KIT.FP8_AX.map(Math.log2), X = v => b.x + (Math.log2(v) - la) / (lb - la) * b.w;
    // decade grid and labels under the tick rows
    const rowsY = b.y + hh + 10, axY = rowsY + 3 * rh;
    ctx.save(); ctx.strokeStyle = C('--rule'); ctx.lineWidth = 1; ctx.font = '10.5px system-ui, sans-serif'; ctx.fillStyle = C('--muted'); ctx.textAlign = 'center';
    const dec = narrow ? [1e-4, 1e-2, 1, 10] : [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10];
    dec.forEach(v => { const x = X(v); ctx.beginPath(); ctx.moveTo(x, b.y); ctx.lineTo(x, axY); ctx.stroke(); ctx.fillText(v >= 1 ? String(v) : v >= 1e-2 ? String(v) : '1e' + Math.round(Math.log10(v)), x, axY + 13); });
    ctx.fillText('|x|, tensor scaled to RMS 1 (log scale)', b.x + b.w / 2, axY + 27); ctx.restore();
    // histogram of |x|
    const mx = Math.max(...S.hist), bw = b.w / KIT.FP8_BINS;
    ctx.save(); ctx.fillStyle = C('--blue'); ctx.globalAlpha = .75;
    S.hist.forEach((c, i) => { if (c <= 0) return; const bh = Math.max(2, c / mx * (hh - 8)); ctx.fillRect(b.x + i * bw + .5, b.y + hh - bh, bw - 1, bh); });
    ctx.restore();
    PE.label(ctx, KIT.FP8_DISTS[S.dist].name + ' tensor, ' + KIT.FP8_N + ' values', b.x + 2, b.y - 8, C('--blue'), 11.5, 'left', 600);
    // the tensor max: every format's largest value is scaled onto it
    const gc = glow('clip'), xM = X(M);
    PE.polyline(ctx, [[xM, b.y + 4], [xM, axY]], gc > 0 ? C('--red') : C('--fg'), 1.2 + 2.5 * gc, [4, 3], .9);
    PE.label(ctx, 'max |x| = ' + M.toFixed(1), xM + (xM > b.x + b.w - 90 ? -4 : 4), b.y + 10, gc > 0 ? C('--red') : C('--fg'), 11, xM > b.x + b.w - 90 ? 'right' : 'left', 600);
    // tick rows
    const gsc = glow('scale'), gsub = glow('sub'), gex = glow('exp'), gman = glow('mant');
    const rowsDef = [['int8', null], ['e4m3', 'e4m3'], ['e5m2', 'e5m2']];
    const maxMse = Math.max(...Object.values(st.mse));
    rowsDef.forEach(([k, kind], r) => {
      const y = rowsY + r * rh + rh / 2, s = M / (kind ? KIT.fp8Max(kind) : 127);
      PE.label(ctx, NAMES[k], b.x - (narrow ? 4 : 8), y + (narrow ? -1 : 4), C('--yellow'), narrow ? 11 : 12, 'right', 600);
      const band = Math.max(gsc, kind === 'e4m3' ? glow('cast') : 0);
      if (band > 0) { ctx.save(); ctx.fillStyle = C('--yellow'); ctx.globalAlpha = .16 * band; ctx.fillRect(b.x, y - rh * .5, b.w, rh); ctx.restore(); }
      ctx.save(); ctx.strokeStyle = C('--rule'); ctx.beginPath(); ctx.moveTo(b.x, y + rh * .32); ctx.lineTo(b.x + b.w, y + rh * .32); ctx.stroke(); ctx.restore();
      const vals = kind ? KIT.fp8Grid(kind).filter(p => p.v > 0) : Array.from({ length: 127 }, (_, i) => ({ v: i + 1, cls: 'int' }));
      let off = 0;
      ctx.save(); ctx.lineWidth = 1 + .8 * gsc;
      for (const p of vals) {
        const v = p.v * s; if (v < KIT.FP8_AX[0]) { off++; continue; }
        const x = X(v), sub = p.cls === 'sub';
        ctx.strokeStyle = sub && gsub > 0 ? C('--red') : C('--yellow'); ctx.globalAlpha = sub ? .45 + .55 * gsub : .85;
        ctx.beginPath(); ctx.moveTo(x, y - rh * .32); ctx.lineTo(x, y + rh * .32); ctx.stroke();
      }
      ctx.restore();
      // exponent: binade edges of this format; mantissa: the marks inside one binade
      if (kind === 'e4m3' && (gex > 0 || gman > 0)) {
        ctx.save();
        for (let e = 1; e < 16; e++) { const v = Math.pow(2, e - 7) * s; if (v < KIT.FP8_AX[0]) continue;
          if (gex > 0) { ctx.strokeStyle = C('--fg'); ctx.globalAlpha = gex; ctx.lineWidth = 1.5; ctx.beginPath(); ctx.moveTo(X(v), y - rh * .5); ctx.lineTo(X(v), y + rh * .5); ctx.stroke(); } }
        if (gman > 0) { const lo = X(Math.pow(2, 0) * s), hi = X(Math.pow(2, 1) * s); ctx.fillStyle = C('--fg'); ctx.globalAlpha = .18 * gman; ctx.fillRect(lo, y - rh * .5, hi - lo, rh); }
        ctx.restore();
      }
      if (off) { const t = '\u2190 ' + off + ' more'; ctx.save(); ctx.font = '10px system-ui, sans-serif'; const tw = ctx.measureText(t).width; ctx.fillStyle = C('--bg'); ctx.fillRect(b.x, y - 7, tw + 6, 13); ctx.restore();
        PE.label(ctx, t, b.x + 3, y + 4, C('--muted'), 10, 'left', 600); }
      // error column
      const best = st.best === k, txt = 'MSE ' + st.mse[k].toExponential(1);
      if (narrow) PE.label(ctx, st.mse[k].toExponential(1), b.x - 4, y + 11, best ? C('--green') : C('--red'), 10, 'right', 600);
      else {
        const bx = b.x + b.w + 16, bwid = (R - 90) * st.mse[k] / maxMse;
        ctx.save(); ctx.fillStyle = C('--red'); ctx.globalAlpha = .8; ctx.fillRect(bx, y - 5, Math.max(2, bwid), 10); ctx.restore();
        PE.label(ctx, txt, bx + Math.max(2, bwid) + 6, y + 4, best ? C('--green') : C('--fg'), 11, 'left', best ? 700 : 400);
      }
    });
    if (!narrow) PE.label(ctx, 'error, our computation', b.x + b.w + 16, rowsY - 4, C('--muted'), 10.5, 'left', 400);
    // code maps, then their legend
    const my = axY + 52, avail = H - my - 46;
    if (avail < (narrow ? 200 : 90)) return;   // short stages (the atlas on a phone) keep only the number line
    let ly;
    if (narrow) { const h1 = codeMap(ctx, 'e4m3', L, my, w - L - 12, avail - 125, C); ly = my + h1 + 22 + codeMap(ctx, 'e5m2', L, my + h1 + 22, w - L - 12, avail - h1 - 53, C) + 14; }
    else { const half = (w - L - 40) / 2; ly = my + Math.max(codeMap(ctx, 'e4m3', L, my, half - 20, avail, C), codeMap(ctx, 'e5m2', L + half + 30, my, half - 10, avail, C)) + 14; }
    let lx = L; const items = [['normal', 'n'], ['subnormal', 's'], ['zero', 'z'], ['NaN', 'nan'], ['infinity', 'inf']];
    ctx.save(); ctx.font = '10.5px system-ui, sans-serif';
    for (const [t, k] of items) {
      if (k === 'inf') { ctx.strokeStyle = C('--red'); ctx.lineWidth = 1.5; ctx.strokeRect(lx + .75, ly - 9.25, 9.5, 9.5); }
      else { ctx.fillStyle = k === 'nan' ? C('--red') : k === 'z' ? C('--muted') : C('--yellow'); ctx.globalAlpha = k === 's' ? .38 : .85; ctx.fillRect(lx, ly - 10, 11, 11); ctx.globalAlpha = 1; }
      ctx.fillStyle = C('--muted'); ctx.fillText(t, lx + 15, ly); lx += 15 + ctx.measureText(t).width + (narrow ? 9 : 16);
    }
    ctx.restore();
  }
  const cv = PE.canvas(el, opts.aspect || (w => w < 560 ? 1.6 : Math.min(.8, Math.max(.54, 330 / w + .25))), draw);
  let stop = () => {};
  function set(next) {
    if (next.dist && next.dist !== S.dist) {
      S.dist = next.dist; stop();
      const from = { logM: S.logM, hist: S.hist.slice() }, to = tgt(S.dist);
      stop = PE.animate(700, u => { S.logM = PE.lerp(from.logM, to.logM, u); S.hist = to.hist.map((v, i) => PE.lerp(from.hist[i] || 0, v, u)); cv.redraw(); });
    }
    cv.redraw();
  }
  const info = () => KIT.fp8Stats(S.dist);
  return { set, info, state: S, redraw: cv.redraw, resize: cv.resize, destroy: () => { stop(); cv.destroy(); } };
};
KIT.fp8Text = i => {
  const f = k => i.mse[k].toExponential(2), n = { int8: 'INT8', e4m3: 'E4M3', e5m2: 'E5M2' };
  return `<b>${KIT.FP8_DISTS[i.dist].long}</b>: ${KIT.FP8_N} values scaled to RMS 1, largest |x| = ${i.M.toFixed(2)}. ` +
    `Mean squared error: INT8 ${f('int8')}, E4M3 ${f('e4m3')}, E5M2 ${f('e5m2')}. ` +
    `Lowest: <b class="c-green">${n[i.best]}</b>, about ${i.bits[i.best].toFixed(1)} bits of accuracy (our computation).`;
};
