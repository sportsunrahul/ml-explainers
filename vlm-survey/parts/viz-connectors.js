<script>
(() => {
  const fig = document.getElementById('viz-connectors'), el = document.getElementById('connectors-canvas');
  const out = document.getElementById('connectors-readout');
  // Token counts: Flamingo 64 and BLIP-2 32 are stated by those papers; 256 and 576 are (224/14)^2 and (336/14)^2.
  const M = {
    flamingo: { name: 'Flamingo', conn: 'Perceiver Resampler', grid: 10, gridLab: 'frozen NFNet-F6 feature grid', tok: 64, lat: [8, 8], entry: 'xattn', frozenLM: true },
    qformer: { name: 'BLIP-2', conn: 'Q-Former + linear', grid: 16, gridLab: 'frozen ViT-g/14 at 224: 16 x 16', tok: 32, lat: [4, 8], entry: 'prefix', frozenLM: true },
    llava: { name: 'LLaVA', conn: 'linear W', grid: 16, gridLab: 'frozen CLIP ViT-L/14 at 224: 16 x 16', tok: 256, entry: 'inline', frozenLM: false },
    llava15: { name: 'LLaVA-1.5', conn: '2-layer MLP', grid: 24, gridLab: 'frozen CLIP ViT-L/14 at 336: 24 x 24', tok: 576, entry: 'inline', frozenLM: false },
  };
  const order = ['flamingo', 'qformer', 'llava', 'llava15'], TEXT = 40, MAXSEQ = 576 + TEXT;
  const S = { m: 'llava', from: 'llava', u: 1, compare: 0 };
  const glow = PE.linkPulses(fig, () => cv.redraw());
  let stop = () => {};
  const inSeq = k => M[k].entry === 'xattn' ? 0 : M[k].tok;

  function rrect(ctx, x, y, w, h, r, fill, stroke, lw = 1.5) {
    ctx.beginPath(); ctx.moveTo(x + r, y); ctx.arcTo(x + w, y, x + w, y + h, r); ctx.arcTo(x + w, y + h, x, y + h, r);
    ctx.arcTo(x, y + h, x, y, r); ctx.arcTo(x, y, x + w, y, r); ctx.closePath();
    if (fill) { ctx.fillStyle = fill; ctx.fill(); } if (stroke) { ctx.strokeStyle = stroke; ctx.lineWidth = lw; ctx.stroke(); }
  }
  const layout = (W, H) => {
    if (W < 560) { const gs = Math.min(W * .36, 150);
      return { narrow: true, gx: 10, gy: 70, gs, cx: gs + 40, cy: 70 + gs / 2 - 34, cw: W - gs - 52, ch: 68,
        lx: 10, ly: gs + 116, lw: W - 20, lh: H - gs - 130 }; }
    const gs = Math.min(W * .21, 160), cw = Math.min(150, W * .2);
    return { narrow: false, gx: 16, gy: 70, gs, cx: 16 + gs + 42, cy: 70 + gs / 2 - 36, cw, ch: 72,
      lx: 16 + gs + 42 + cw + 42, ly: 50, lw: W - (16 + gs + 42 + cw + 42) - 14, lh: H - 64 };
  };

  const cv = PE.canvas(el, w => w < 560 ? 1.36 : .56, (ctx, W, H) => {
    const B = PE.css('--blue'), Y = PE.css('--yellow'), G = PE.css('--green'), Br = PE.css('--brown'), F = PE.css('--fg'),
      Mu = PE.css('--muted'), Rule = PE.css('--rule'), BG = PE.css('--bg');
    ctx.fillStyle = BG; ctx.fillRect(0, 0, W, H);
    if (S.compare > 0) return drawCompare(ctx, W, H, { B, Y, G, F, Mu, Rule });
    const L = layout(W, H), a = M[S.from], b = M[S.m], u = S.u;
    const e1 = PE.clamp(u / .45, 0, 1), e2 = PE.clamp((u - .3) / .7, 0, 1);
    const fs = L.narrow ? 11 : 12;
    // Header: which model, how many tokens per image.
    const tokNow = Math.round(PE.lerp(a.tok, b.tok, e1));
    ctx.save(); ctx.font = `600 ${L.narrow ? 15 : 17}px system-ui, sans-serif`; const nw = ctx.measureText(b.name).width; ctx.restore();
    PE.label(ctx, b.name, L.gx, 24, F, L.narrow ? 15 : 17);
    if (L.narrow) PE.label(ctx, tokNow + ' visual tokens per image', L.gx, 42, B, 12.5); else PE.label(ctx, tokNow + ' visual tokens per image', L.gx + nw + 10, 24, B, 15);
    PE.label(ctx, b.entry === 'xattn' ? 'read by cross-attention, none added to the sequence' : b.entry === 'prefix' ? 'prepended to the text as a soft prompt' : 'placed in the sequence, one token per patch',
      L.gx, L.narrow ? 58 : 42, Mu, L.narrow ? 10.5 : 11.5);

    // 1. Encoder output grid (blue). The grid size tweens between models.
    const gp = glow('grid'), n = Math.round(PE.lerp(a.grid, b.grid, e1)), c = L.gs / n;
    ctx.fillStyle = PE.mix(BG.length === 7 ? BG : '#16181d', B.length === 7 ? B : '#58c4dd', .28 + .3 * gp); ctx.fillRect(L.gx, L.gy, L.gs, L.gs);
    ctx.strokeStyle = B; ctx.lineWidth = n > 20 ? .6 : 1; ctx.beginPath();
    for (let i = 0; i <= n; i++) { ctx.moveTo(L.gx + i * c, L.gy); ctx.lineTo(L.gx + i * c, L.gy + L.gs); ctx.moveTo(L.gx, L.gy + i * c); ctx.lineTo(L.gx + L.gs, L.gy + i * c); }
    ctx.stroke();
    if (gp > 0) { ctx.strokeStyle = F; ctx.lineWidth = 3 * gp; ctx.strokeRect(L.gx - 2, L.gy - 2, L.gs + 4, L.gs + 4); }
    const lines = b.gridLab.split(': ');
    PE.label(ctx, lines[0], L.gx, L.gy + L.gs + 16, Br, L.narrow ? 10 : 11);
    if (lines[1]) PE.label(ctx, lines[1] + ' = ' + (b.grid * b.grid) + ' patches', L.gx, L.gy + L.gs + 30, B, L.narrow ? 10 : 11);
    else PE.label(ctx, '(any size)', L.gx, L.gy + L.gs + 30, B, L.narrow ? 10 : 11);

    // 2. Connector (green). Resampler and Q-Former output a fixed set of latents; linear and MLP map patch by patch.
    const pp = glow('proj');
    rrect(ctx, L.cx, L.cy, L.cw, L.ch, 8, PE.mix(BG.length === 7 ? BG : '#16181d', G.length === 7 ? G : '#83c167', .14 + .3 * pp), G, 2 + 2 * pp);
    PE.label(ctx, b.conn, L.cx + L.cw / 2, L.cy + 20, G, fs, 'center');
    PE.label(ctx, b.lat ? 'learned queries' : 'same map for each patch', L.cx + L.cw / 2, L.cy + 36, Mu, L.narrow ? 9.5 : 10.5, 'center');
    if (b.lat) { const [r, k] = b.lat, s = Math.min(4, (L.cw - 24) / (k * 1.6)); for (let i = 0; i < r; i++) for (let j = 0; j < k; j++)
      ctx.fillStyle = B, ctx.fillRect(L.cx + L.cw / 2 - k * s * .8 + j * s * 1.6, L.cy + 44 + i * s * 1.2 * (r > 4 ? .75 : 1), s, s * (r > 4 ? .75 : 1)); }
    else { for (let j = 0; j < 8; j++) PE.arrow(ctx, L.cx + 14 + j * (L.cw - 28) / 7, L.cy + 46, L.cx + 14 + j * (L.cw - 28) / 7, L.cy + 62, B, 1.2, 5); }
    PE.arrow(ctx, L.gx + L.gs + 6, L.gy + L.gs / 2, L.cx - 4, L.cy + L.ch / 2, Mu, 1.5, 7);

    // 3. The language model: a stack of layers plus its input sequence along the bottom.
    const layerN = 6, stackTop = L.ly + (L.narrow ? 40 : 30), seqY = L.ly + L.lh - (L.narrow ? 46 : 58), lh = (seqY - 26 - stackTop) / (layerN * 1.5);
    PE.label(ctx, b.frozenLM ? 'language model (frozen)' : 'language model', L.lx, L.ly + (L.narrow ? 32 : 11), b.frozenLM ? Br : Y, fs);
    const xa = b.entry === 'xattn' ? 1 : 0;
    for (let i = 0; i < layerN; i++) {
      const y = stackTop + i * lh * 1.5;
      rrect(ctx, L.lx + 22, y, L.lw - 44, lh, 4, PE.mix(BG.length === 7 ? BG : '#16181d', (b.frozenLM ? Br : Y), .22), b.frozenLM ? Br : Y, 1.2);
      if (xa) { // Flamingo: a new gated cross-attention layer sits between frozen layers; the latents are its keys and values.
        const yy = y + lh * 1.08, ww = L.lw - 44; ctx.globalAlpha = e2; rrect(ctx, L.lx + 22, yy, ww, lh * .34, 3, G, null); ctx.globalAlpha = 1;
      }
    }
    if (xa) {
      PE.label(ctx, 'green: gated cross-attention layers', L.lx + L.lw - 22, stackTop - 6, G, L.narrow ? 9.5 : 10.5, 'right');
      for (let i = 0; i < layerN; i++) { const y = stackTop + i * lh * 1.5 + lh * 1.25;
        PE.arrow(ctx, L.lx + 4, y, L.lx + 20, y, G, 1.3, 5, e2); }
      PE.polyline(ctx, [[L.cx + L.cw / 2, L.cy + L.ch + 2], [L.cx + L.cw / 2, L.narrow ? L.ly + 14 : L.cy + L.ch + 20], [L.lx + 4, L.narrow ? L.ly + 14 : L.cy + L.ch + 20], [L.lx + 4, stackTop + (layerN - 1) * lh * 1.5 + lh * 1.25]], G, 1.5, null, e2);
    }
    // Input sequence strip: blue = visual tokens in the sequence, yellow = text tokens. Width is proportional to token count.
    const sw = L.lw - 44, sx = L.lx + 22, tp = glow('tokens');
    const vis = PE.lerp(inSeq(S.from), inSeq(S.m), e1), vw = sw * vis / MAXSEQ, tw = sw * TEXT / MAXSEQ;
    ctx.fillStyle = B; ctx.fillRect(sx, seqY, vw, 18);
    if (tp > 0) { ctx.strokeStyle = F; ctx.lineWidth = 3 * tp; ctx.strokeRect(sx - 2, seqY - 2, vw + 4, 22); }
    ctx.fillStyle = Y; ctx.fillRect(sx + vw + 2, seqY, tw, 18);
    if (xa) { ctx.fillStyle = F; ctx.globalAlpha = .8; ctx.fillRect(sx + vw + 2 + tw * .3, seqY - 3, 2, 24); ctx.globalAlpha = 1;
      PE.label(ctx, '<image> marker', sx + vw + 2 + tw * .3 + 6, seqY - 6, Mu, 10); }
    PE.arrow(ctx, sx + sw / 2, seqY - 4, sx + sw / 2, stackTop + (layerN - 1) * lh * 1.5 + lh + 4, Mu, 1.3, 6);
    PE.label(ctx, 'input: ' + Math.round(vis) + ' visual + ' + TEXT + ' text tokens', sx, seqY + 34, Mu, L.narrow ? 9.5 : 10.5);
    PE.label(ctx, 'bar length = tokens the LLM must read', sx, seqY + (L.narrow ? 47 : 48), Mu, L.narrow ? 9.5 : 10.5);
    if (!xa) PE.polyline(ctx, [[L.cx + L.cw / 2, L.cy + L.ch + 2], [L.cx + L.cw / 2, L.narrow ? L.ly + 14 : seqY + 9], [sx - 6, L.narrow ? L.ly + 14 : seqY + 9]], B, 1.5, [4, 3], e2);
    if (!xa && L.narrow) PE.polyline(ctx, [[sx - 6, L.ly + 14], [sx - 12, L.ly + 14], [sx - 12, seqY + 9], [sx - 2, seqY + 9]], B, 1.5, [4, 3], e2);

    // Tokens in flight: dots from the grid through the connector to where they enter the LLM.
    if (u < 1) for (let k = 0; k < 14; k++) {
      const t = PE.clamp(e2 * 1.4 - k * .03, 0, 1); if (t <= 0 || t >= 1) continue;
      const p0 = [L.gx + L.gs * ((k * 37) % 13 + .5) / 13, L.gy + L.gs * ((k * 17) % 11 + .5) / 11], p1 = [L.cx + L.cw / 2, L.cy + L.ch / 2],
        p2 = xa ? [L.lx + 20, stackTop + (k % layerN) * lh * 1.5 + lh * 1.25] : [sx + vw * ((k % 7) + .5) / 7, seqY + 9];
      const q = t < .5 ? [PE.lerp(p0[0], p1[0], t * 2), PE.lerp(p0[1], p1[1], t * 2)] : [PE.lerp(p1[0], p2[0], t * 2 - 1), PE.lerp(p1[1], p2[1], t * 2 - 1)];
      PE.dot(ctx, q[0], q[1], 3.2, t < .5 ? B : (xa ? G : B), .9);
    }
  });

  function drawCompare(ctx, W, H, C) {
    const narrow = W < 560, u = S.compare;
    // Narrow screens: name and connector on one line above each bar, so nothing is clipped at the left edge.
    const x0 = narrow ? 12 : 150, bw = W - x0 - (narrow ? 44 : 90), rowH = Math.min(narrow ? 70 : 62, (H - 110) / 4), top = narrow ? 18 : 4;
    PE.label(ctx, narrow ? 'Visual tokens per image' : 'Visual tokens per image, and where they enter the language model', 12, 36, C.F, narrow ? 12 : 14);
    order.forEach((k, i) => {
      const m = M[k], y = 58 + i * rowH, w = bw * m.tok / 576 * u;
      if (narrow) {
        PE.label(ctx, m.name, x0, y + 10, C.F, 11.5);
        ctx.save(); ctx.font = '600 11.5px system-ui, sans-serif'; const nw = ctx.measureText(m.name).width; ctx.restore();
        PE.label(ctx, m.conn, x0 + nw + 8, y + 10, C.G, 9.5);
      } else {
        PE.label(ctx, m.name, x0 - 10, y + 16, C.F, 13, 'right');
        PE.label(ctx, m.conn, x0 - 10, y + 31, C.G, 10.5, 'right');
      }
      ctx.fillStyle = C.Rule; ctx.fillRect(x0, y + top, bw, 18);
      if (m.entry === 'xattn') { ctx.strokeStyle = C.G; ctx.lineWidth = 2; ctx.setLineDash([4, 3]); ctx.strokeRect(x0, y + top, w, 18); ctx.setLineDash([]); }
      else { ctx.fillStyle = C.B; ctx.fillRect(x0, y + top, w, 18); }
      PE.label(ctx, String(Math.round(m.tok * u)), x0 + w + 6, y + top + 14, C.B, 12);
      PE.label(ctx, m.entry === 'xattn' ? 'via cross-attention: adds 0 tokens to the sequence' : m.entry === 'prefix' ? 'prepended soft prompt' : 'inline in the sequence', x0, y + top + 34, C.Mu, narrow ? 9.5 : 10.5);
    });
    if (narrow) {
      PE.label(ctx, 'dashed = not in the sequence;', 12, H - 28, C.Mu, 9.5);
      PE.label(ctx, 'solid = sequence length the LLM pays for', 12, H - 14, C.Mu, 9.5);
    } else PE.label(ctx, 'dashed = not in the sequence; solid = sequence length the LLM pays for', 12, H - 14, C.Mu, 10.5);
  }

  function readout() {
    if (S.compare > 0) { out.textContent = 'Side by side: Flamingo 64 latents (read by cross-attention, so its sequence does not grow), BLIP-2 32 queries prepended, LLaVA 256 and LLaVA-1.5 576 patch tokens placed in the sequence. 256 and 576 are computed from the patch grids, not stated by the papers.'; return; }
    const m = M[S.m], t = {
      flamingo: 'Flamingo: a Perceiver Resampler turns the frozen encoder\'s grid into 64 latents, whatever the image size. They are never put in the text sequence; new cross-attention layers inside the frozen LM read them.',
      qformer: 'BLIP-2: 32 learned queries ask the frozen encoder what matters; a linear layer maps their 32 outputs into the frozen LLM\'s input, ahead of the text.',
      llava: 'LLaVA: one linear map W turns each of the 16 x 16 = 256 patch features into a word-sized vector. All 256 go into the sequence next to the text.',
      llava15: 'LLaVA-1.5: a two-layer MLP instead of W, and a 336 px encoder: 24 x 24 = 576 tokens, 2.25 times as many as LLaVA for a 1.5 times wider image.' }[S.m];
    out.textContent = t;
  }
  function show(k) {
    stop(); S.from = S.compare > 0 ? k : S.m; S.m = k; S.compare = 0; S.u = 0;
    order.forEach(o => document.getElementById('btn-connectors-' + o).setAttribute('aria-pressed', String(o === k)));
    document.getElementById('btn-connectors-compare').setAttribute('aria-pressed', 'false');
    readout(); stop = PE.animate(1300, v => { S.u = v; cv.redraw(); });
  }
  order.forEach(k => document.getElementById('btn-connectors-' + k).addEventListener('click', () => show(k)));
  document.getElementById('btn-connectors-compare').addEventListener('click', () => {
    stop(); order.forEach(o => document.getElementById('btn-connectors-' + o).setAttribute('aria-pressed', 'false'));
    document.getElementById('btn-connectors-compare').setAttribute('aria-pressed', 'true');
    S.compare = .001; readout(); stop = PE.animate(1000, v => { S.compare = Math.max(.001, v); cv.redraw(); });
  });
  document.getElementById('btn-connectors-reset').addEventListener('click', () => {
    stop(); S.m = S.from = 'llava'; S.u = 1; S.compare = 0;
    order.forEach(o => document.getElementById('btn-connectors-' + o).setAttribute('aria-pressed', String(o === 'llava')));
    document.getElementById('btn-connectors-compare').setAttribute('aria-pressed', 'false'); cv.redraw(); readout();
  });
  readout();
})();
</script>
