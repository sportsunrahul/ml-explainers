/* Deep Compression (Han, Mao, Dally 1510.00149v5) run for real on 256 toy weights.
   Prune: Han 2015 (1506.02626v3) Sec. 4 rule, threshold = quality parameter q x the layer's standard deviation.
   Store survivors as relative indices (gap to the previous survivor) in gapBits bits; a longer gap needs a filler zero.
   Share: 1-D k-means (Eq. 2) with linear initialization over [min, max] of the survivors (Sec. 3.2), k = 8 (3-bit codes).
   Fine-tune: centroid gradient = sum of its members' gradients (Eq. 3); the toy gradients are random, like Fig. 3.
   Huffman: real Huffman codes for the cluster indices and for the gaps. Simulated in notes/sim-dc.txt. */
KIT.DC = { n: 256, sigma: .1, q: 1, k: 8, gapBits: 5, seed: 23, gradSeed: 5, gradScale: .02, lr: .1 };
KIT.dcHuffman = freq => { // freq: {symbol: count}; returns {symbol: code string}
  const keys = Object.keys(freq).filter(k => freq[k] > 0);
  if (keys.length === 1) return { [keys[0]]: '0' };
  let nodes = keys.map(k => ({ f: freq[k], s: [k] })); const code = {}; keys.forEach(k => (code[k] = ''));
  while (nodes.length > 1) {
    nodes.sort((a, b) => a.f - b.f || a.s[0].localeCompare(b.s[0]));
    const a = nodes.shift(), b = nodes.shift();
    a.s.forEach(k => (code[k] = '0' + code[k])); b.s.forEach(k => (code[k] = '1' + code[k]));
    nodes.push({ f: a.f + b.f, s: a.s.concat(b.s) });
  }
  return code;
};
KIT.dcKmeans = (vals, k) => {
  const lo = Math.min(...vals), hi = Math.max(...vals), init = Array.from({ length: k }, (_, j) => lo + (hi - lo) * j / (k - 1));
  let c = init.slice(), a = vals.map(() => -1), it = 0;
  for (; it < 100; it++) {
    const na = vals.map(v => { let b = 0; c.forEach((cj, j) => { if (Math.abs(v - cj) < Math.abs(v - c[b])) b = j; }); return b; });
    const same = na.every((x, i) => x === a[i]); a = na;
    if (same) break;
    c = c.map((cj, j) => { const m = vals.filter((_, i) => a[i] === j); return m.length ? m.reduce((s, x) => s + x, 0) / m.length : cj; });
  }
  return { init, c, a, iters: it };
};
KIT.dcRun = () => {
  if (KIT.DC.run) return KIT.DC.run;
  const P = KIT.DC, g = PE.gauss(PE.rng(P.seed)), W = Array.from({ length: P.n }, () => +(P.sigma * g()).toFixed(3));
  const mean = W.reduce((s, x) => s + x, 0) / P.n, std = Math.sqrt(W.reduce((s, x) => s + (x - mean) ** 2, 0) / P.n), t = P.q * std;
  const keptIdx = W.map((_, i) => i).filter(i => Math.abs(W[i]) > t), kept = keptIdx.map(i => W[i]);
  const gaps = keptIdx.map((i, j) => i - (j ? keptIdx[j - 1] : -1)), span = 1 << P.gapBits;
  const fillers = gaps.reduce((s, d) => s + Math.floor((d - 1) / span), 0);
  const km = KIT.dcKmeans(kept, P.k), cb = Math.log2(P.k);
  const gg = PE.gauss(PE.rng(P.gradSeed)), grad = kept.map(() => +(P.gradScale * gg()).toFixed(4));
  const cgrad = km.c.map((_, j) => grad.reduce((s, x, i) => s + (km.a[i] === j ? x : 0), 0));
  const tuned = km.c.map((cj, j) => cj - P.lr * cgrad[j]);
  const cfreq = {}, gfreq = {}; km.a.forEach(j => (cfreq[j] = (cfreq[j] || 0) + 1)); gaps.forEach(d => (gfreq[d] = (gfreq[d] || 0) + 1));
  const hc = KIT.dcHuffman(cfreq), hg = KIT.dcHuffman(gfreq), E = kept.length + fillers;
  const huffW = km.a.reduce((s, j) => s + hc[j].length, 0), huffI = gaps.reduce((s, d) => s + hg[d].length, 0);
  const bits = {
    dense: { val: 32 * P.n, idx: 0, book: 0 },
    prune: { val: 32 * E, idx: P.gapBits * E, book: 0 },
    share: { val: cb * E, idx: P.gapBits * E, book: 32 * P.k },
    huff: { val: huffW, idx: huffI, book: 32 * P.k },
  };
  for (const k in bits) { const b = bits[k]; b.tot = b.val + b.idx + b.book; b.per = b.tot / P.n; b.ratio = 32 * P.n / b.tot; }
  return (P.run = { W, std, t, keptIdx, kept, gaps, fillers, km, grad, cgrad, tuned, cfreq, hc, hg, bits, huffW, huffI, cb });
};
KIT.DC_STAGES = ['dense', 'prune', 'share', 'huff'];
KIT.dcText = (st, tuned) => {
  const R = KIT.dcRun(), P = KIT.DC, b = R.bits[st], f1 = x => x.toFixed(1), f2 = x => x.toFixed(2);
  const tail = ` <b>${f2(b.per)} bits per weight</b>, ${f1(b.ratio)}${KIT.times} smaller than FP32.`;
  if (st === 'dense') return `<b>Dense</b>: ${P.n} weights stored as FP32, <b>32 bits per weight</b>.`;
  if (st === 'prune') return `<b>Prune</b>: threshold t = q${KIT.times}std = ${f2(R.t)} keeps ${R.kept.length} of ${P.n} weights. ` +
    `Each survivor stores its FP32 value plus a ${P.gapBits}-bit gap to the previous survivor` + (R.fillers ? ` (${R.fillers} filler zeros).` : '; no gap is too long, so no filler zeros.') + tail;
  if (st === 'share') return `<b>Share</b>: k-means from linear initialization converged in ${R.km.iters} iterations to ${P.k} shared values, so each survivor stores a ${R.cb}-bit code` +
    ` plus its gap; the codebook adds ${P.k} ${KIT.times} 32 bits.` + (R.km.c.some((_, j) => !R.cfreq[j]) ? ' One linear-init centroid starts inside the pruned gap and gets no members.' : '') + (tuned ? ` One fine-tuning step moved the centroids by the summed gradients of their members (largest move ${f2(Math.max(...R.cgrad.map(x => Math.abs(P.lr * x))))}).` : '') + tail;
  return `<b>Huffman</b>: frequent codes get short bit strings. Codes average ${f2(R.huffW / R.kept.length)} bits (from ${R.cb}), gaps ${f2(R.huffI / R.kept.length)} bits (from ${P.gapBits}); code tables are not counted.` + tail;
};
KIT.dc = (el, opts = {}) => {
  const P = KIT.DC, R = KIT.dcRun();
  const S = { stage: 'dense', tuned: false, u: { p: 0, s: 0, h: 0, t: 0 } };
  const glow = opts.fig ? PE.linkPulses(opts.fig, () => cv.redraw()) : () => 0;
  const jr = PE.rng(4), jit = R.W.map(() => jr());
  const keptPos = new Map(R.keptIdx.map((i, j) => [i, j]));
  const big = R.km.c.map((_, j) => R.km.a.filter(a => a === j).length).reduce((b, x, j, arr) => x > arr[b] ? j : b, 0);
  let stop = () => {};
  function draw(ctx, w, h) {
    ctx.clearRect(0, 0, w, h);
    const narrow = w < 560, cs = k => PE.css(k), pad = opts.bottomPad || 0, H = h - pad, u = S.u;
    const C = { fg: cs('--fg'), mut: cs('--muted'), green: cs('--green'), brown: cs('--brown'), blue: cs('--blue'), yellow: cs('--yellow'), red: cs('--red'), rule: cs('--rule') };
    const gC = glow('centroid'), gG = glow('grad'), gM = glow('member');
    const L = narrow ? 12 : 24, Rr = narrow ? 12 : 24;
    // band 1: every weight on a value axis
    const b1 = { x: L, y: 30, w: w - L - Rr, h: narrow ? H * .4 : H * .5 };
    const lim = Math.max(...R.W.map(Math.abs)) * 1.05, X = v => b1.x + (v + lim) / (2 * lim) * b1.w;
    PE.label(ctx, `${P.n} weights of one layer, placed by value`, b1.x, 16, C.fg, narrow ? 11 : 12, 'left', 600);
    const axY = b1.y + b1.h;
    PE.polyline(ctx, [[b1.x, axY], [b1.x + b1.w, axY]], C.mut, 1);
    [-.3, -.2, -.1, 0, .1, .2, .3].filter(v => Math.abs(v) < lim).forEach(v => { PE.polyline(ctx, [[X(v), axY], [X(v), axY + 4]], C.mut, 1); PE.label(ctx, v.toFixed(1), X(v), axY + 15, C.mut, 10, 'center', 400); });
    // threshold band
    if (u.p > 0) {
      ctx.save(); ctx.globalAlpha = .12 * u.p; ctx.fillStyle = C.brown; ctx.fillRect(X(-R.t), b1.y, X(R.t) - X(-R.t), b1.h); ctx.restore();
      [-R.t, R.t].forEach(v => PE.polyline(ctx, [[X(v), b1.y], [X(v), axY]], C.fg, 1, [4, 3], .7 * u.p));
      PE.label(ctx, '\u00b1t', X(R.t) + 3, b1.y + 10, C.fg, 10.5, 'left', 600);
      if (!narrow) PE.label(ctx, 'pruned', X(0), b1.y + 10, C.brown, 10.5, 'center', 600);
    }
    // centroids
    const cpos = j => PE.lerp(PE.lerp(R.km.init[j], R.km.c[j], Math.min(1, u.s * 1.6)), R.tuned[j], u.t);
    if (u.s > 0 || gC > 0 || gM > 0) {
      R.km.c.forEach((_, j) => {
        const x = X(cpos(j)), gl = Math.max(gC, j === big ? gM : 0);
        PE.polyline(ctx, [[x, b1.y + 14], [x, axY]], C.yellow, 1.2 + 2 * gl, null, (.5 + .5 * gl) * Math.max(u.s, gl));
        if (u.s < 1) PE.polyline(ctx, [[X(R.km.init[j]), axY - 6], [X(R.km.init[j]), axY]], C.yellow, 1, null, .6 * (1 - u.s));
      });
      if (u.s > 0 && (!narrow || u.h === 0)) PE.label(ctx, 'shared values (centroids)', X(cpos(P.k - 1)), b1.y + 10 + (u.p > 0 ? 12 : 0), C.yellow, 10.5, 'right', 600);
    }
    // dots
    const dotR = narrow ? 2.2 : 2.8;
    R.W.forEach((v, i) => {
      const j = keptPos.get(i), y0 = b1.y + 18 + jit[i] * (b1.h - 26);
      if (j === undefined) { // pruned: fade to brown and sink
        const y = y0, a = 1 - .75 * u.p;
        PE.dot(ctx, X(v), y, dotR, u.p > .5 ? C.brown : C.blue, a); return;
      }
      const cj = R.km.a[j], x = X(PE.lerp(v, cpos(cj), u.s)), col = u.s > .5 ? C.yellow : C.blue;
      const gl = cj === big ? gM : 0;
      PE.dot(ctx, x, y0, dotR + 1.5 * gl, col, .9);
      const ga = Math.max(u.t, gG) * Math.min(1, u.s);
      if (ga > 0) { const dx = -R.grad[j] * b1.w * .45; PE.arrow(ctx, x, y0, x + dx, y0, C.red, 1.2, 4, ga * .8); }
    });
    if (u.t > 0 || gG > 0) { const a = Math.max(u.t, gG) * u.s;
      R.km.c.forEach((_, j) => { const x = X(cpos(j)), dx = -R.cgrad[j] * b1.w * .45; if (Math.abs(dx) > 1) PE.arrow(ctx, x, b1.y + 26, x + dx, b1.y + 26, C.red, 2.4, 7, a); });
      PE.label(ctx, narrow ? 'red: minus each gradient; thick: sum per centroid' : 'red arrows: minus each weight gradient; thick arrows: their sum for each centroid', b1.x, b1.y + b1.h + 30, C.red, 10.5, 'left', 600); }
    // band 2: bits per weight, one bar per stage
    const by = axY + (narrow ? 54 : 50), stages = KIT.DC_STAGES, reached = stages.indexOf(S.stage);
    const names = { dense: 'dense FP32', prune: 'pruned', share: 'shared', huff: 'Huffman' };
    const cw = narrow ? w - L - Rr : (w - L - Rr) * .62, lw = narrow ? 64 : 82, bx = L + lw, bl = cw - lw - (narrow ? 70 : 80);
    const rowH = Math.min(26, (H - by - 18) / 5);
    PE.label(ctx, 'bits per original weight', L, by, C.fg, narrow ? 11 : 12, 'left', 600);
    stages.forEach((st, k) => {
      const b = R.bits[st], y = by + 10 + k * rowH, on = k <= reached, a = on ? 1 : .18, sc = bl / 32 / P.n;
      const valCol = st === 'dense' || st === 'prune' ? C.blue : C.yellow;
      PE.label(ctx, names[st], L, y + rowH * .5 + 3, on ? C.fg : C.mut, narrow ? 10 : 11, 'left', 400);
      let x = bx; ctx.save(); ctx.globalAlpha = a;
      [[b.val, valCol], [b.idx, C.mut], [b.book, C.green]].forEach(([v, col]) => { ctx.fillStyle = col; ctx.fillRect(x, y + 3, v * sc, rowH - 8); x += v * sc; });
      ctx.restore();
      if (on) PE.label(ctx, `${b.per.toFixed(2)}  (${b.ratio.toFixed(1)}${KIT.times})`, x + 6, y + rowH * .5 + 3, C.fg, narrow ? 10 : 11, 'left', 600);
    });
    const ky = by + 14 + 4 * rowH + 4;
    [['value', C.blue], ['code', C.yellow], ['index', C.mut], ['codebook', C.green]].forEach(([t, col], k) => {
      const x = bx + k * (narrow ? 56 : 70); ctx.fillStyle = col; ctx.fillRect(x, ky, 9, 9); PE.label(ctx, t, x + 12, ky + 9, C.mut, 10, 'left', 400); });
    // Huffman code table (right of the bars, or hidden on narrow screens until reached)
    if (u.h > 0) {
      const tx = narrow ? L : L + cw + 16, ty = narrow ? ky + 30 : by, colW = narrow ? (w - 2 * L) / 2 : w - tx - Rr;
      ctx.save(); ctx.globalAlpha = u.h;
      PE.label(ctx, 'code  members  Huffman bits', tx, ty, C.fg, narrow ? 10.5 : 11.5, 'left', 600);
      const order = R.km.c.map((_, j) => j).sort((a, b) => (R.cfreq[b] || 0) - (R.cfreq[a] || 0));
      const lh = narrow ? 13 : Math.min(15, (H - ty - 8) / (P.k + 1));
      order.forEach((j, r) => {
        const x = narrow ? tx + (r % 2) * colW : tx, y = narrow ? ty + 14 + Math.floor(r / 2) * lh : ty + 14 + r * lh;
        ctx.font = `${narrow ? 10 : 11}px ui-monospace, Menlo, Consolas, monospace`; ctx.fillStyle = C.yellow; ctx.textAlign = 'left';
        ctx.fillText(KIT.bits(j, R.cb), x, y);
        ctx.fillStyle = C.mut; ctx.fillText(String(R.cfreq[j] || 0).padStart(4, ' '), x + 26, y);
        ctx.fillStyle = C.fg; ctx.fillText(R.hc[j] || '-', x + 66, y);
      });
      ctx.restore();
    }
  }
  const cv = PE.canvas(el, opts.aspect || (w => w < 560 ? 1.55 : .56), draw);
  const target = () => { const k = KIT.DC_STAGES.indexOf(S.stage); return { p: k >= 1 ? 1 : 0, s: k >= 2 ? 1 : 0, h: k >= 3 ? 1 : 0, t: S.tuned && k >= 2 ? 1 : 0 }; };
  function set(next) {
    if (next.stage) S.stage = next.stage;
    if (next.tuned != null) S.tuned = next.tuned;
    if (S.tuned && KIT.DC_STAGES.indexOf(S.stage) < 2) S.stage = 'share';
    stop(); stop = KIT.tween(S.u, target(), 850, cv.redraw);
  }
  return { set, state: S, text: () => KIT.dcText(S.stage, S.tuned), redraw: cv.redraw, resize: cv.resize, destroy: () => { stop(); cv.destroy(); } };
};
