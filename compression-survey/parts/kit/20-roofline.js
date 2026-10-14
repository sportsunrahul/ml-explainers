/* Roofline: FLOPs and bytes per layer are exact for the stated shape (each tensor read or written once);
   the GPU limits are a toy with a 1:8:16 FP32:FP16:INT8 math ratio and one memory bandwidth. */
KIT.ROOF_GPU = { bw: 500e9, peak: { fp32: 12.5e12, fp16: 100e12, int8: 200e12 } };
KIT.ROOF_PREC = { fp32: { w: 4, a: 4, math: 'fp32', name: 'FP32' }, fp16: { w: 2, a: 2, math: 'fp16', name: 'FP16' },
  int8: { w: 1, a: 1, math: 'int8', name: 'INT8' }, w4: { w: .5, a: 2, math: 'fp16', name: '4-bit weights, FP16 math' } };
KIT.ROOF_LAYERS = [
  { key: 'c3', name: '3x3 conv, 256 ch, 40x40', short: '3x3 conv', kind: 'conv', cin: 256, cout: 256, k: 3, hw: 40 },
  { key: 'hi', name: '3x3 conv, 64 ch, 160x160', short: 'early conv', kind: 'conv', cin: 64, cout: 64, k: 3, hw: 160 },
  { key: 'c1', name: '1x1 conv, 256 -> 64, 40x40', short: '1x1 conv', kind: 'conv', cin: 256, cout: 64, k: 1, hw: 40 },
  { key: 'dw', name: '3x3 depthwise, 256 ch, 40x40', short: 'depthwise', kind: 'dw', c: 256, k: 3, hw: 40 },
  { key: 'hd', name: '3x3 head conv, 256 ch, 5x5', short: 'head on 5x5', kind: 'conv', cin: 256, cout: 256, k: 3, hw: 5 },
  { key: 'fc', name: 'linear 4096 -> 4096, 1 token', short: 'LLM, 1 token', kind: 'fc', din: 4096, dout: 4096, tokens: 1 },
];
KIT.roofCost = (L, prec) => {
  const P = KIT.ROOF_PREC[prec];
  let flops, params, acts;
  if (L.kind === 'conv') { const px = L.hw * L.hw; flops = 2 * px * L.cout * L.cin * L.k * L.k; params = L.cin * L.cout * L.k * L.k; acts = px * (L.cin + L.cout); }
  else if (L.kind === 'dw') { const px = L.hw * L.hw; flops = 2 * px * L.c * L.k * L.k; params = L.c * L.k * L.k; acts = 2 * px * L.c; }
  else { flops = 2 * L.tokens * L.din * L.dout; params = L.din * L.dout; acts = L.tokens * (L.din + L.dout); }
  const bytes = params * P.w + acts * P.a, ai = flops / bytes, G = KIT.ROOF_GPU, peak = G.peak[P.math];
  const tMath = flops / peak, tMem = bytes / G.bw, time = Math.max(tMath, tMem);
  return { flops, bytes, params, acts, ai, tMath, tMem, time, bound: tMath >= tMem ? 'math' : 'memory', perf: flops / time };
};
KIT.roofline = (el, opts = {}) => {
  const S = { prec: 'fp16', sel: 'c3', ai: {}, perf: {}, roof: 1, fig: opts.fig };
  const keys = KIT.ROOF_LAYERS.map(L => L.key);
  const target = prec => { const o = {}; KIT.ROOF_LAYERS.forEach(L => { const c = KIT.roofCost(L, prec); o['a_' + L.key] = Math.log10(c.ai); o['p_' + L.key] = Math.log10(c.perf); }); o.roof = Math.log10(KIT.ROOF_GPU.peak[KIT.ROOF_PREC[prec].math]); return o; };
  Object.assign(S, target('fp16'));
  const glow = opts.fig ? PE.linkPulses(opts.fig, () => cv.redraw()) : () => 0;
  function draw(ctx, w, h) {
    ctx.clearRect(0, 0, w, h);
    const narrow = w < 560, C = n => PE.css(n);
    const b = { x: narrow ? 46 : 64, y: narrow ? 30 : 22, w: w - (narrow ? 58 : 84), h: h - (narrow ? 78 : 66) - (opts.bottomPad || 0) };
    const xr = [.25, 4000], yr = [1e10, 4e14];
    const A = PE.axes(ctx, b, xr, yr, { logx: true, logy: true, xt: [1, 10, 100, 1000], yt: [1e11, 1e12, 1e13, 1e14],
      yf: v => (v / 1e12) + 'T', xl: 'arithmetic intensity: operations per byte moved (log)', yl: 'ops per second (log)', ylo: narrow ? 36 : 48 });
    const bw = KIT.ROOF_GPU.bw, mut = C('--muted');
    // faint roofs for the other precisions, bold roof for the current one
    const curMath = KIT.ROOF_PREC[S.prec].math;
    for (const p of ['fp32', 'fp16', 'int8']) {
      const cur = p === curMath, peak = cur ? Math.pow(10, S.roof) : KIT.ROOF_GPU.peak[p], ridge = peak / bw;
      const pts = [[A.X(xr[0]), A.Y(Math.max(yr[0], xr[0] * bw))], [A.X(ridge), A.Y(peak)], [A.X(xr[1]), A.Y(peak)]];
      const gl = glow(cur ? 'roof' : '');
      PE.polyline(ctx, pts, cur ? C('--fg') : mut, cur ? 2.2 + 2 * gl : 1, cur ? null : [4, 4], cur ? 1 : .55);
      PE.label(ctx, KIT.ROOF_PREC[p].name + ' math limit', A.X(xr[1]) - 4, A.Y(peak) - 5, cur ? C('--fg') : mut, 11, 'right', cur ? 600 : 400);
    }
    const gm = glow('mem');
    PE.label(ctx, 'memory limit', A.X(1.1), A.Y(1.1 * bw) - 8, gm > 0 ? C('--red') : mut, 11, 'left', 400);
    if (gm > 0) PE.polyline(ctx, [[A.X(xr[0]), A.Y(Math.max(yr[0], xr[0] * bw))], [A.X(Math.pow(10, S.roof) / bw), A.Y(Math.pow(10, S.roof))]], C('--red'), 2 + 3 * gm, null, gm);
    // layer points; an unselected label that would collide with the selected dot is dropped
    const selL = KIT.ROOF_LAYERS.find(L => L.key === S.sel), sx = A.X(Math.pow(10, S['a_' + selL.key])), sy = A.Y(Math.pow(10, S['p_' + selL.key]));
    KIT.ROOF_LAYERS.forEach(L => {
      const x = A.X(Math.pow(10, S['a_' + L.key])), y = A.Y(Math.pow(10, S['p_' + L.key])), sel = L.key === S.sel;
      const color = S.prec === 'fp32' || S.prec === 'fp16' ? C('--blue') : C('--yellow');
      PE.dot(ctx, x, y, sel ? 7 : 4.5, color, sel ? 1 : .55);
      if (sel) PE.ring(ctx, x, y, 11 + 6 * glow('point'), color, 1.5, .8);
      if (sel || (!narrow && (Math.abs(x - sx) > 60 || Math.abs(y - sy) > 24))) PE.label(ctx, L.short, x + (sel ? 12 : 8), y + (sel ? -10 : 14), sel ? C('--fg') : mut, sel ? 12 : 10, 'left', sel ? 600 : 400);
    });
  }
  const cv = PE.canvas(el, opts.aspect || (w => w < 560 ? .95 : .56), draw);
  let stop = () => {};
  function set(next) {
    if (next.sel) S.sel = next.sel;
    if (next.prec && next.prec !== S.prec) { S.prec = next.prec; stop(); stop = KIT.tween(S, target(S.prec), 700, cv.redraw); }
    cv.redraw();
  }
  function info() {
    const L = KIT.ROOF_LAYERS.find(l => l.key === S.sel), c = KIT.roofCost(L, S.prec), base = KIT.roofCost(L, 'fp32');
    return { L, c, speedup: base.time / c.time, prec: KIT.ROOF_PREC[S.prec].name };
  }
  return { set, info, state: S, redraw: cv.redraw, resize: cv.resize, destroy: () => { stop(); cv.destroy(); } };
};
KIT.roofText = i => `<b>${i.L.name}</b>, ${i.prec}: ${KIT.fmt(i.c.flops)} operations, ${KIT.bytes(i.c.bytes)} moved, ` +
  `${i.c.ai >= 10 ? Math.round(i.c.ai) : i.c.ai.toFixed(1)} operations per byte. ` +
  `<b class="${i.c.bound === 'memory' ? 'c-red' : 'c-green'}">${i.c.bound === 'memory' ? 'Memory-bound' : 'Math-bound'}</b>: ` +
  `${(i.c.time * 1e6).toFixed(i.c.time < 1e-5 ? 2 : 1)} \u00b5s on the toy GPU, ${i.speedup.toFixed(1)}${KIT.times} faster than FP32.`;
