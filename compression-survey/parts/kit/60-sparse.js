/* Sparsity patterns on one toy layer: 8 output channels x 16 inputs (inputs = the GEMM reduction dimension K).
   Exact magnitude pruning for three patterns: unstructured (global |w| threshold), 2:4 (keep the two largest of each
   group of four inputs, Mishra 2104.08378 Sec. 3.1), whole channels (drop rows with the smallest L1 norm, Li 1608.08710
   Sec. 3.1). Readout: what a dense GPU kernel can skip and the share of total |w| that survives. */
KIT.SPARSE = { R: 8, C: 16, rowScale: [1, .7, 1.1, .55, .9, .75, 1.05, .6] };
KIT.sparseW = () => {
  if (KIT.SPARSE.W) return KIT.SPARSE.W;
  const P = KIT.SPARSE, g = PE.gauss(PE.rng(11)), W = [];
  for (let r = 0; r < P.R; r++) for (let c = 0; c < P.C; c++) W.push(+(P.rowScale[r] * .45 * g()).toFixed(2) || .01);
  return (P.W = W);
};
KIT.sparseMask = (pattern, s) => {
  const P = KIT.SPARSE, W = KIT.sparseW(), n = W.length, m = new Array(n).fill(1);
  let rowsOut = [];
  if (pattern === '24') {
    for (let r = 0; r < P.R; r++) for (let g0 = 0; g0 < P.C; g0 += 4) {
      const idx = [0, 1, 2, 3].map(j => r * P.C + g0 + j).sort((a, b) => Math.abs(W[a]) - Math.abs(W[b]));
      m[idx[0]] = 0; m[idx[1]] = 0;
    }
  } else if (pattern === 'chan') {
    const k = Math.round(s * P.R), l1 = [];
    for (let r = 0; r < P.R; r++) { let a = 0; for (let c = 0; c < P.C; c++) a += Math.abs(W[r * P.C + c]); l1.push([a, r]); }
    rowsOut = l1.sort((a, b) => a[0] - b[0]).slice(0, k).map(x => x[1]).sort((a, b) => a - b);
    rowsOut.forEach(r => { for (let c = 0; c < P.C; c++) m[r * P.C + c] = 0; });
  } else {
    const k = Math.round(s * n), idx = W.map((_, i) => i).sort((a, b) => Math.abs(W[a]) - Math.abs(W[b]));
    for (let i = 0; i < k; i++) m[idx[i]] = 0;
  }
  return { m, rowsOut };
};
KIT.sparseInfo = (pattern, s) => {
  const P = KIT.SPARSE, W = KIT.sparseW(), n = W.length, { m, rowsOut } = KIT.sparseMask(pattern, s);
  let tot = 0, keptMag = 0, kept = 0, thr = 0, minKept = Infinity, maxOut = 0;
  W.forEach((w, i) => { const a = Math.abs(w); tot += a; if (m[i]) { kept++; keptMag += a; minKept = Math.min(minKept, a); } else maxOut = Math.max(maxOut, a); });
  if (pattern === 'unstructured') thr = kept < n ? (minKept + maxOut) / 2 : 0;
  const mults = pattern === 'unstructured' ? n : pattern === '24' ? n / 2 : kept;
  return { pattern, s: 1 - kept / n, kept, n, mag: keptMag / tot, thr, mults, rowsOut, m, bigLost: maxOut > minKept,
    rowsLeft: P.R - rowsOut.length, R: P.R, C: P.C };
};
KIT.SPARSE_NAMES = { unstructured: 'Unstructured', '24': '2:4', chan: 'Whole channels' };
KIT.sparseText = i => {
  const pct = x => (100 * x).toFixed(1) + '%', head = `<b>${KIT.SPARSE_NAMES[i.pattern]}, ${Math.round(100 * i.s)}% zeros</b>: `;
  const mag = `keeps ${i.kept} of ${i.n} weights and <b class="c-green">${pct(i.mag)}</b> of their total magnitude. `;
  if (i.pattern === 'unstructured') return head + mag + (i.kept === i.n ? 'Nothing is pruned yet. ' :
    `A dense GPU kernel skips <b class="c-brown">nothing</b>: it still does all ${i.n} multiplies, ${i.n - i.kept} of them by zero. `) +
    'Only a sparse kernel could skip them, and on a GPU that pays off in narrow cases (next figure).';
  if (i.pattern === '24') return head + mag + `Sparse tensor cores skip <b class="c-green">half the multiplies</b>: ${i.mults} of ${i.n}, ` +
    `up to <span data-claim="mishra2021-math-speedup">2</span>${KIT.times} math throughput. Stored as ${i.kept} FP16 values with a 2-bit index each: ` +
    `${i.kept * 18} bits instead of ${i.n * 16}, ${(100 * (1 - i.kept * 18 / (i.n * 16))).toFixed(1)}% less.`;
  return head + mag + (i.rowsOut.length ? `What is left is a <b class="c-green">smaller dense layer</b>, ${i.rowsLeft} ${KIT.times} ${i.C}: an ordinary dense kernel does ${i.mults} of ${i.n} multiplies, ` +
    `and the next layer drops the matching ${i.rowsOut.length} input channel${i.rowsOut.length > 1 ? 's' : ''}.` : 'No channel is removed yet.');
};
KIT.sparse = (el, opts = {}) => {
  const P = KIT.SPARSE, W = KIT.sparseW(), n = W.length, wmax = Math.max(...W.map(Math.abs));
  const S = { pattern: 'unstructured', s: .5, keep: KIT.sparseMask('unstructured', .5).m.slice(), panel: 1, info: KIT.sparseInfo('unstructured', .5) };
  const glow = opts.fig ? PE.linkPulses(opts.fig, () => cv.redraw()) : () => 0;
  let stop = () => {}, stopP = () => {};
  function draw(ctx, w, h) {
    ctx.clearRect(0, 0, w, h);
    const narrow = w < 560, cs = k => PE.css(k), pad = opts.bottomPad || 0, H = h - pad;
    const C = { fg: cs('--fg'), mut: cs('--muted'), green: cs('--green'), brown: cs('--brown'), blue: cs('--blue'), yellow: cs('--yellow'), red: cs('--red'), rule: cs('--rule'), bg: cs('--bg') };
    const I = S.info, gK = glow('kept'), gP = glow('pruned'), gT = glow('thr'), gM = glow('mag');
    // matrix block
    const lab = narrow ? 34 : 46, top = 40;
    const mw = narrow ? w - lab - 10 : w * .56 - lab;
    const mhMax = narrow ? H * .42 : H - top - 30;
    const cell = Math.max(8, Math.min(mw / P.C, (mhMax) / P.R)), mx = lab, my = top;
    const tall = !narrow && H - top > cell * P.R * 1.8 + 120, short = narrow && H / w < 1.3;
    PE.label(ctx, `one layer: ${P.R} output channels ${KIT.times} ${P.C} inputs`, mx, 16, C.fg, narrow ? 11 : 12, 'left', 600);
    PE.label(ctx, 'inputs (the dimension a GEMM sums over)', mx, my + P.R * cell + 16, C.mut, narrow ? 10 : 11, 'left', 400);
    for (let r = 0; r < P.R; r++) {
      const out = I.rowsOut.includes(r);
      PE.label(ctx, (narrow ? '' : 'ch ') + (r + 1), mx - 6, my + r * cell + cell / 2 + 4, out ? C.brown : C.mut, narrow ? 10 : 11, 'right', out ? 600 : 400);
      for (let c = 0; c < P.C; c++) {
        const i = r * P.C + c, k = S.keep[i], a = Math.abs(W[i]) / wmax, x = mx + c * cell, y = my + r * cell, gap = cell > 14 ? 1.5 : 1;
        ctx.save();
        ctx.globalAlpha = (.18 + .82 * a) * k; ctx.fillStyle = C.green; ctx.fillRect(x + gap, y + gap, cell - 2 * gap, cell - 2 * gap);
        ctx.globalAlpha = (1 - k) * (.34 + .3 * gP); ctx.fillStyle = C.brown; ctx.fillRect(x + gap, y + gap, cell - 2 * gap, cell - 2 * gap);
        if (k > .5 && gK > 0) { ctx.globalAlpha = gK; ctx.strokeStyle = C.green; ctx.lineWidth = 2; ctx.strokeRect(x + gap, y + gap, cell - 2 * gap, cell - 2 * gap); }
        ctx.restore();
        if (cell >= 30) PE.label(ctx, (W[i] < 0 ? '-' : '') + Math.abs(W[i]).toFixed(2).replace(/^0/, ''), x + cell / 2, y + cell / 2 + 4, k > .5 ? (a > .55 ? C.bg : C.fg) : C.brown, 10, 'center', k > .5 ? 600 : 400);
      }
      if (out) PE.polyline(ctx, [[mx - 2, my + r * cell + cell / 2], [mx + P.C * cell + 2, my + r * cell + cell / 2]], C.brown, 1.5, null, .8);
    }
    // groups of four along K
    for (let g0 = 0; g0 <= P.C; g0 += 4) {
      const x = mx + g0 * cell, on = S.pattern === '24';
      PE.polyline(ctx, [[x, my - (on ? 6 : 2)], [x, my + P.R * cell + (on ? 4 : 0)]], on ? C.fg : C.rule, on ? 1.5 : 1, null, on ? .9 : 1);
      if (on && g0 < P.C && cell * 4 > 50) PE.label(ctx, 'group of 4', x + 2 * cell, my - 9, C.mut, 10, 'center', 400);
    }
    // right (or lower) panels: |w| histogram, then what the GPU runs
    const px = narrow ? 12 : w * .6 + 8, pw = narrow ? w - 24 : w * .4 - 20;
    const hy = narrow ? my + P.R * cell + 56 : top + (tall ? 16 : 0), hh = short ? 0 : narrow ? H * .2 : (H - top) * (tall ? .5 : .38);
    const nb = 12, bins = Array.from({ length: nb }, () => [0, 0]);
    W.forEach((v, i) => { const b = Math.min(nb - 1, Math.floor(Math.abs(v) / wmax * nb)); bins[b][S.keep[i] > .5 ? 0 : 1]++; });
    const bmax = Math.max(...bins.map(b => b[0] + b[1])), bw = pw / nb;
    if (hh > 0) {
    PE.label(ctx, 'how many weights have each size |w|', px, hy - 8, C.fg, narrow ? 11 : 12, 'left', 600);
    bins.forEach((b, j) => {
      const x = px + j * bw, hk = b[0] / bmax * hh, hp = b[1] / bmax * hh, base = hy + hh;
      ctx.save(); ctx.globalAlpha = .55 + .45 * gM; ctx.fillStyle = C.brown; ctx.fillRect(x + 1, base - hp, bw - 2, hp);
      ctx.fillStyle = C.green; ctx.fillRect(x + 1, base - hp - hk, bw - 2, hk); ctx.restore();
    });
    PE.polyline(ctx, [[px, hy + hh], [px + pw, hy + hh]], C.mut, 1);
    PE.label(ctx, '0', px, hy + hh + 13, C.mut, 10, 'left', 400);
    PE.label(ctx, 'largest |w|', px + pw, hy + hh + 13, C.mut, 10, 'right', 400);
    if (S.pattern === 'unstructured' && I.thr > 0) {
      const tx = px + I.thr / wmax * pw;
      PE.polyline(ctx, [[tx, hy - 2], [tx, hy + hh + 2]], C.fg, 1.5 + 2.5 * gT, [4, 3], 1);
      PE.label(ctx, 'threshold t', tx + 4, hy + 10, C.fg, 10.5, 'left', 600);
    } else if (I.bigLost) PE.label(ctx, 'some large weights are removed too', px + pw, hy + 10, C.brown, 10.5, 'right', 600);
    }
    // what the GPU runs
    const gy = tall ? my + P.R * cell + 58 : short ? hy : hy + hh + (narrow ? 34 : 40), gh = H - gy - 10;
    const gpx = tall ? mx : px, gpw = tall ? P.C * cell : pw;
    ctx.save(); ctx.globalAlpha = S.panel;
    PE.label(ctx, 'what the GPU multiplies', gpx, gy, C.fg, narrow ? 11 : 12, 'left', 600);
    const rows = S.pattern === 'chan' ? I.rowsLeft : P.R, cols = S.pattern === '24' ? P.C / 2 : P.C;
    const mc = Math.max(3, Math.min((gpw - 10) / P.C * (tall ? .75 : 1), (gh - 52) / P.R));
    const gx0 = gpx, gy0 = gy + 10;
    for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) {
      let zero = false;
      if (S.pattern === 'unstructured') zero = !I.m[r * P.C + c];
      ctx.fillStyle = zero ? C.brown : C.green; ctx.globalAlpha = S.panel * (zero ? .45 : .8);
      ctx.fillRect(gx0 + c * mc + .5, gy0 + r * mc + .5, mc - 1, mc - 1);
    }
    ctx.globalAlpha = S.panel;
    if (S.pattern === '24') { // 2-bit index metadata next to the compressed values
      for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) { ctx.fillStyle = C.mut; ctx.globalAlpha = S.panel * .6; ctx.fillRect(gx0 + cols * mc + 6 + c * mc / 4, gy0 + r * mc + .5, mc / 4 - .5, mc - 1); }
      ctx.globalAlpha = S.panel;
      PE.label(ctx, '2-bit index', gx0 + cols * mc + 8 + cols * mc / 4, gy0 + mc * rows / 2 + 4, C.mut, 10, 'left', 400);
    }
    const ty = gy0 + rows * mc + 16, txt = S.pattern === 'unstructured' ? `${I.mults} of ${I.n} multiplies, zeros included` :
      S.pattern === '24' ? `${I.mults} of ${I.n}: half, on sparse tensor cores` : `${I.mults} of ${I.n}: a smaller dense layer`;
    PE.label(ctx, txt, gx0, ty, S.pattern === 'unstructured' && I.kept < I.n ? C.brown : C.green, narrow ? 10.5 : 11.5, 'left', 600);
    // multiplies bar
    const by = ty + 8, bl = Math.min(gpw, 240), useful = (S.pattern === 'unstructured' ? I.kept : I.mults) / I.n, done = I.mults / I.n;
    ctx.fillStyle = C.rule; ctx.fillRect(gx0, by, bl, 7);
    ctx.fillStyle = C.brown; ctx.globalAlpha = S.panel * .7; ctx.fillRect(gx0, by, bl * done, 7);
    ctx.fillStyle = C.green; ctx.globalAlpha = S.panel; ctx.fillRect(gx0, by, bl * useful, 7);
    ctx.restore();
  }
  const cv = PE.canvas(el, opts.aspect || (w => w < 560 ? 1.45 : .44), draw);
  function set(next) {
    const pattern = next.pattern || S.pattern, s = pattern === '24' ? .5 : (next.s != null ? next.s : S.s);
    const changed = pattern !== S.pattern;
    S.pattern = pattern; S.s = s; S.info = KIT.sparseInfo(pattern, s);
    stop(); stop = KIT.tween(S.keep, Object.assign({}, S.info.m), 600, cv.redraw);
    if (changed) { stopP(); S.panel = 0; stopP = KIT.tween(S, { panel: 1 }, 500, cv.redraw); }
    cv.redraw();
  }
  return { set, state: S, info: () => S.info, redraw: cv.redraw, resize: cv.resize, destroy: () => { stop(); stopP(); cv.destroy(); } };
};
