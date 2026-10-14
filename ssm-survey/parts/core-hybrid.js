/* CoreHybrid: a stack of 32 sequence-mixing layers (Jamba's count), each Mamba or attention, and the KV cache that the
   attention layers hold at context length L. Per attention layer the cache is 4 KiB per token at 16 bits: back-solved from
   Jamba Table 1 (our reading), and equal to 2 x 8 KV heads x 128 channels x 2 bytes in the Nemotron-H and NVIDIA hybrid
   configs. Published models are drawn as markers. Shared by the long page (viz-hybrid) and atlas stage 10. */
const CoreHybrid = (() => {
  const LAYERS = 32, KV_LAYER = 4096, GiB = Math.pow(2, 30), T256 = 262144;
  // Attention layers spread evenly; for 4 of 32 this gives 0-indexed 4, 12, 20, 28, Jamba's place in each 8-layer block.
  function target(n) {
    const t = new Array(LAYERS).fill(0);
    for (let i = 0; i < n; i++) t[Math.floor((i + .5) * LAYERS / n)] = 1;
    return t;
  }
  // bytes are at 256K tokens and 16 bits; sourced = the paper's own KV-cache table, else computed from the paper's config.
  const POINTS = [
    { name: 'Jamba', share: 4 / 32, bytes: 4 * GiB, sourced: true, note: '4GB' },
    { name: 'Jamba-1.5-Large', share: 9 / 72, bytes: 9 * GiB, sourced: true, note: '9GB' },
    { name: 'NVIDIA 8B hybrid, Nemotron-H 8B', share: 4 / 28, bytes: 16384 * T256, sourced: false },
    { name: 'Nemotron-H 56B', share: 10 / 64, bytes: 40960 * T256, sourced: false },
    { name: 'Mistral, Mixtral', share: 1, bytes: 32 * GiB, sourced: true, note: '32GB' },
    { name: 'Llama-2 7B (full multi-head)', share: 1, bytes: 128 * GiB, sourced: true, note: '128GB' },
  ];
  const nAttn = S => S.attn.reduce((s, a) => s + a, 0);
  const kvBytes = (n, L) => n * KV_LAYER * L;
  function fmtBytes(b) {
    const u = ['B', 'KiB', 'MiB', 'GiB', 'TiB']; let i = 0;
    while (b >= 1024 && i < u.length - 1) { b /= 1024; i++; }
    return (b >= 100 ? b.toFixed(0) : b >= 10 ? b.toFixed(1) : b.toFixed(2)).replace(/\.?0+$/, '') + ' ' + u[i];
  }
  const fmtTok = L => L >= 1024 ? Math.round(L / 1024) + 'K' : String(L);

  function readout(S) {
    const n = Math.round(nAttn(S)), m = LAYERS - n, all = kvBytes(LAYERS, S.L), kv = kvBytes(n, S.L);
    const ratio = n ? (n === LAYERS ? 'the all-attention size' : `1/${(LAYERS / n) % 1 ? (LAYERS / n).toFixed(1) : LAYERS / n} of the ${fmtBytes(all)} an all-attention stack would hold`) : `nothing, against ${fmtBytes(all)} for an all-attention stack`;
    const r = n && m ? ` (1:${(m / n) % 1 ? (m / n).toFixed(1) : m / n})` : '';
    return `${n} of ${LAYERS} sequence-mixing layers are attention, ${Math.round(1000 * n / LAYERS) / 10}%${r}. KV cache at ${fmtTok(S.L)} tokens: ${fmtBytes(kv)}, ${ratio}. ` +
      (m ? `The ${m} Mamba layers each keep a state of fixed size that does not grow with the context (Jamba does not publish its size).` : 'No Mamba layers are left.');
  }

  // draw(ctx, W, H, S, glow): S = { attn: [32 values in 0..1], L }; glow(word) in [0, 1].
  function draw(ctx, W, H, S, glow = () => 0) {
    const C = n => PE.css(n), narrow = W < 520;
    const Y = C('--yellow'), Br = C('--brown'), M = C('--muted'), F = C('--fg'), BG = C('--bg'), Bl = C('--blue');
    ctx.clearRect(0, 0, W, H);
    const ga = glow('attn'), gc = glow('cache'), gs = glow('state'), s = S.L / T256;
    // The stack, input at the left. Below each attention layer hangs its cache, as long as the context is.
    const px = narrow ? 14 : 20, top = narrow ? 52 : 44, cw = (W - 2 * px) / LAYERS, chh = narrow ? 20 : 26, barMax = narrow ? 46 : 70;
    PE.label(ctx, narrow ? '32 sequence-mixing layers' : '32 sequence-mixing layers, input at the left (MLP and MoE layers between them not drawn)', px, top - 10, M, 11);
    for (let i = 0; i < LAYERS; i++) {
      const a = S.attn[i], x = px + i * cw, w = Math.max(2, cw - (cw > 8 ? 2 : 1));
      ctx.fillStyle = PE.mix(Y, Br, a); ctx.fillRect(x, top, w, chh);
      if (a < .5 && cw > 7) {   // a tiny state register on each Mamba layer
        ctx.fillStyle = BG; ctx.globalAlpha = .55 * (1 - 2 * a);
        const g = Math.min(4, (w - 4) / 2.6);
        for (let r = 0; r < 2; r++) for (let c = 0; c < 2; c++) ctx.fillRect(x + w / 2 - g - .5 + c * (g + 1), top + chh / 2 - g - .5 + r * (g + 1), g, g);
        ctx.globalAlpha = 1;
      }
      const hb = a * barMax * s;
      if (hb > .3) { ctx.fillStyle = Br; ctx.globalAlpha = .75 * a; ctx.fillRect(x + w * .2, top + chh + 3, w * .6, hb); ctx.globalAlpha = 1; }
      const pulse = a > .5 ? ga : gs;
      if (pulse > .02) { ctx.save(); ctx.strokeStyle = a > .5 ? Br : Y; ctx.globalAlpha = pulse; ctx.lineWidth = 2.5; ctx.strokeRect(x - 1.5, top - 1.5, w + 3, chh + 3); ctx.restore(); }
      if (gc > .02 && hb > .3) { ctx.save(); ctx.strokeStyle = Br; ctx.globalAlpha = gc; ctx.lineWidth = 2; ctx.strokeRect(x + w * .2 - 1.5, top + chh + 1.5, w * .6 + 3, hb + 3); ctx.restore(); }
    }
    PE.label(ctx, `brown bars: each attention layer's cache at ${fmtTok(S.L)} tokens`, px, top + chh + barMax + 18, Br, 11);
    // Chart: KV cache against the attention share, log scale.
    const b = narrow ? { x: 50, y: top + chh + barMax + 52, w: W - 66, h: H - (top + chh + barMax + 52) - 50 }
      : { x: 76, y: top + chh + barMax + 50, w: W - 110, h: H - (top + chh + barMax + 50) - 52 };
    const yr = [Math.pow(2, 29) * s, Math.pow(2, 38) * s];
    const yt = [30, 32, 34, 36, 38].map(e => Math.pow(2, e) * s);
    const ax = PE.axes(ctx, b, [0, 1], yr, { logy: true, xt: [0, .25, .5, .75, 1], xf: v => Math.round(v * 100) + '%', yt, yf: v => fmtBytes(v).replace(' ', ''),
      xl: narrow ? 'attention share of sequence mixers' : 'attention layers as a share of sequence-mixing layers (attention + Mamba)', yl: narrow ? '' : `KV cache at ${fmtTok(S.L)} tokens, 16-bit (log)`, ylo: 58 });
    const X = ax.X, YY = ax.Y;
    const line = []; for (let n = 1; n <= LAYERS; n++) line.push([X(n / LAYERS), YY(kvBytes(n, S.L))]);
    PE.polyline(ctx, line, Br, 2.4 + 3 * gc, null, .85);
    PE.label(ctx, narrow ? '32-layer stack, 4 KiB/token per attention layer' : 'this stack: 4 KiB per token per attention layer (our reading)', X(.42), YY(kvBytes(.42 * LAYERS, S.L)) - 10, Br, 11, narrow ? 'center' : 'right');
    // Published models. Filled: the paper's own KV-cache table; hollow: computed from the paper's config (our reading).
    const cl = POINTS.filter(p => p.share < .5).sort((p, q) => q.bytes - p.bytes);
    const lx = X(narrow ? .3 : .26), ly0 = YY(4 * GiB * s) + 18;
    cl.forEach((p, i) => {
      const x = X(p.share), y = YY(p.bytes * s), ty = ly0 + i * 15;
      ctx.save(); ctx.strokeStyle = M; ctx.globalAlpha = .6; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(x, y); ctx.lineTo(lx - 4, ty - 4); ctx.stroke(); ctx.restore();
      PE.label(ctx, `${p.name}: ${fmtBytes(p.bytes * s)}`, lx, ty, F, narrow ? 10 : 11);
    });
    POINTS.forEach(p => {
      const x = X(p.share), y = YY(p.bytes * s);
      if (p.sourced) PE.dot(ctx, x, y, 5.5, F);
      else { ctx.save(); ctx.strokeStyle = F; ctx.lineWidth = 1.8; ctx.beginPath(); ctx.arc(x, y, 5, 0, 7); ctx.stroke(); ctx.restore(); }
    });
    POINTS.filter(p => p.share === 1).forEach(p => PE.label(ctx, `${p.name}: ${fmtBytes(p.bytes * s)}`, X(1) - 10, YY(p.bytes * s) + 4, F, narrow ? 10 : 11, 'right'));
    // The current stack.
    const n = nAttn(S);
    if (n >= .5) {
      const x = X(n / LAYERS), y = YY(kvBytes(n, S.L));
      ctx.save(); ctx.strokeStyle = F; ctx.globalAlpha = .3; ctx.beginPath(); ctx.moveTo(x, b.y); ctx.lineTo(x, b.y + b.h); ctx.stroke(); ctx.restore();
      PE.dot(ctx, x, y, 8, Br); PE.dot(ctx, x, y, 3.5, BG);
    } else {
      PE.dot(ctx, X(0), b.y + b.h, 8, Y);
      PE.label(ctx, 'no attention: no KV cache', X(0) + 12, b.y + b.h - 8, Y, 11);
    }
    if (!narrow) PE.label(ctx, 'filled: from the paper\'s KV table    hollow: computed from its config', b.x + b.w, b.y - 8, M, 10, 'right');
    else PE.label(ctx, 'filled: paper table; hollow: computed', b.x + b.w, b.y - 8, M, 10, 'right');
  }
  return { LAYERS, KV_LAYER, POINTS, target, nAttn, kvBytes, fmtBytes, fmtTok, readout, draw };
})();
