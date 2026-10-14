/* CoreBill: what attention's cache and an SSM's state cost at context length L, from two published configs.
   Shared by the long page (viz-bill) and atlas stage 1. Sizes assume 16-bit values for both models. */
const CoreBill = (() => {
  const LLAMA = { layers: 32, kvHeads: 8, headDim: 128, params: 8e9 };       // Llama 3 8B (Llama 3 paper, Table 3)
  const MAMBA = { layers: 64, dModel: 2560, expand: 2, dState: 16, dConv: 4 }; // Mamba 2.8B (Mamba paper; layers, width, conv from the official README)
  const BYTES = 2;
  const kvPerToken = 2 * LLAMA.layers * LLAMA.kvHeads * LLAMA.headDim * BYTES;
  // Per layer the recurrent cache is the SSM state (E*D x N) plus the last d_conv - 1 inputs the causal convolution needs.
  const ssmBytes = MAMBA.layers * MAMBA.expand * MAMBA.dModel * (MAMBA.dState + MAMBA.dConv - 1) * BYTES;
  const weightBytes = LLAMA.params * BYTES;
  const kvBytes = L => kvPerToken * L;
  const pairs = L => L * (L + 1) / 2;
  const LMIN = 128, LMAX = 1048576;
  function fmtBytes(b) {
    const u = ['B', 'KiB', 'MiB', 'GiB', 'TiB']; let i = 0;
    while (b >= 1024 && i < u.length - 1) { b /= 1024; i++; }
    return (b >= 100 ? b.toFixed(0) : b >= 10 ? b.toFixed(1) : b.toFixed(2)).replace(/\.0+$/, '') + ' ' + u[i];
  }
  function fmtTok(L) { return L >= 1048576 ? (L / 1048576).toFixed(L % 1048576 ? 1 : 0) + 'M' : L >= 1024 ? Math.round(L / 1024) + 'K' : String(L); }
  function fmtNum(x) {
    if (x < 1e6) return Math.round(x).toLocaleString('en-US');
    const [d, w] = x >= 1e12 ? [1e12, 'trillion'] : x >= 1e9 ? [1e9, 'billion'] : [1e6, 'million'];
    return `${(x / d).toFixed(1)} ${w}`;
  }
  // Slider position s in [0, 1000] <-> L on a log scale.
  const fromSlider = s => Math.round(LMIN * Math.pow(LMAX / LMIN, s / 1000));
  const toSlider = L => Math.round(1000 * Math.log(L / LMIN) / Math.log(LMAX / LMIN));
  const crossWeights = weightBytes / kvPerToken, crossState = ssmBytes / kvPerToken;

  // draw(ctx, W, H, L, glow): log-log chart of bytes held vs context length; glow(word) in [0, 1] for linked terms.
  function draw(ctx, W, H, L, glow = () => 0) {
    const C = n => PE.css(n), narrow = W < 520;
    ctx.clearRect(0, 0, W, H);
    const b = narrow ? { x: 46, y: 40, w: W - 60, h: H - 86 } : { x: 66, y: 26, w: W - 236, h: H - 70 };
    const yr = [Math.pow(2, 20), Math.pow(2, 38)];
    const ax = PE.axes(ctx, b, [LMIN, LMAX], yr, { logx: true, logy: true,
      xt: [128, 1024, 8192, 65536, 524288], xf: fmtTok, yt: [Math.pow(2, 20), Math.pow(2, 25), Math.pow(2, 30), Math.pow(2, 35)],
      yf: v => fmtBytes(v).replace(' ', ''), xl: 'context length L (tokens, log scale)', yl: narrow ? '' : 'memory held (log scale)', ylo: 54 });
    const X = ax.X, Y = ax.Y, M = C('--muted');
    // Weights reference.
    PE.polyline(ctx, [[b.x, Y(weightBytes)], [b.x + b.w, Y(weightBytes)]], M, 1.4, [5, 4], .9);
    PE.label(ctx, 'Llama 3 8B weights', b.x + 6, Y(weightBytes) - 6, M, 11);
    // KV cache line (slope 1 on log-log), state line (flat).
    const kvPts = [], n = 60;
    for (let i = 0; i <= n; i++) { const l = LMIN * Math.pow(LMAX / LMIN, i / n); kvPts.push([X(l), Y(Math.min(Math.max(kvBytes(l), yr[0]), yr[1]))]); }
    PE.polyline(ctx, kvPts, C('--brown'), 3 + 3 * glow('cache'));
    PE.polyline(ctx, [[b.x, Y(ssmBytes)], [b.x + b.w, Y(ssmBytes)]], C('--yellow'), 3 + 3 * glow('state'));
    const lx = narrow ? b.x + b.w - 4 : b.x + b.w + 8, la = narrow ? 'right' : 'left';
    if (narrow) PE.label(ctx, 'KV cache, Llama 3 8B', X(1024) - 6, Y(kvBytes(1024)) - 12, C('--brown'), 12);
    else PE.label(ctx, 'KV cache, Llama 3 8B', lx, Math.max(b.y + 10, Y(kvBytes(LMAX)) + 4), C('--brown'), 12, la);
    PE.label(ctx, 'state, Mamba 2.8B', lx, Y(ssmBytes) + (narrow ? -8 : 4), C('--yellow'), 12, la);
    // Current L.
    const x = X(L);
    ctx.save(); ctx.strokeStyle = C('--fg'); ctx.globalAlpha = .35; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(x, b.y); ctx.lineTo(x, b.y + b.h); ctx.stroke(); ctx.restore();
    PE.dot(ctx, x, Y(Math.min(kvBytes(L), yr[1])), 6, C('--brown'));
    PE.dot(ctx, x, Y(ssmBytes), 6, C('--yellow'));
    // Red gap between the two at L.
    if (kvBytes(L) > ssmBytes * 1.3) PE.arrow(ctx, x + 10, Y(ssmBytes) - 4, x + 10, Y(Math.min(kvBytes(L), yr[1])) + 8, C('--red'), 2, 8, .9);
    PE.label(ctx, `L = ${fmtTok(L)}`, Math.min(x + 6, b.x + b.w - 60), b.y + 12, C('--fg'), 12);
  }
  function readout(L) {
    const kv = kvBytes(L), r = kv / ssmBytes;
    return `At L = ${L.toLocaleString('en-US')} tokens the Llama 3 8B cache holds ${fmtBytes(kv)} per sequence; the Mamba 2.8B state holds ${fmtBytes(ssmBytes)} at any length` +
      (r >= 1.05 ? ` (${r >= 10 ? Math.round(r).toLocaleString('en-US') : r.toFixed(1)} times less).` : r <= .95 ? ` (the cache is still smaller).` : ' (about the same).') +
      ` Reading a prompt this long, attention scores ${fmtNum(pairs(L))} token pairs per head per layer; the SSM makes ${L.toLocaleString('en-US')} state updates per channel.`;
  }
  return { LLAMA, MAMBA, BYTES, kvPerToken, ssmBytes, weightBytes, kvBytes, pairs, fmtBytes, fmtTok, fmtNum, fromSlider, toSlider, crossWeights, crossState, draw, readout, LMIN, LMAX };
})();
