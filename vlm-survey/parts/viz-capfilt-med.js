<script>
(() => {
  const fig = document.getElementById('viz-capfilt-med'), el = document.getElementById('capfilt-med-canvas');
  const out = document.getElementById('capfilt-med-readout');
  const K = ['itc', 'itm', 'lm'];
  const COL = {
    itc: { title: 'text encoder', t2: ['text', 'encoder'], hs: 'vs image', loss: 'ITC loss', head: 'compare with image', sa: 'bi-directional SA', ca: null, tok: '[CLS] a dog ...' },
    itm: { title: 'image-grounded encoder', t2: ['image-grounded', 'encoder'], hs: 'match?', loss: 'ITM loss', head: 'linear head: match?', sa: 'bi-directional SA', ca: 'cross-attention', tok: '[Encode] a dog ...' },
    lm: { title: 'image-grounded decoder', t2: ['image-grounded', 'decoder'], hs: 'next word', loss: 'LM loss', head: 'predict next word', sa: 'causal SA', ca: 'cross-attention', tok: '[Decode] a dog ...' },
  };
  const S = { a: { itc: 1, itm: 1, lm: 1 }, sel: 'all' };
  let stop = () => {};
  const glow = PE.linkPulses(fig, () => cv.redraw());

  const layout = W => {
    const sm = W < 560, vitH = sm ? 40 : 0, x0 = sm ? 4 : 132, top = sm ? 88 : 34, gap = sm ? 6 : 14;
    const cw = (W - x0 - 4 - 2 * gap) / 3, bh = sm ? 30 : 34, step = bh + (sm ? 14 : 16);
    return { sm, x0, top, gap, cw, bh, step, vitH, colX: i => x0 + i * (cw + gap), H: top + 22 + 5 * step + (sm ? 54 : 40) };
  };
  const cv = PE.canvas(el, w => layout(w).H / w, (ctx, W, H) => {
    const B = PE.css('--blue'), Y = PE.css('--yellow'), G = PE.css('--green'), Rd = PE.css('--red'), F = PE.css('--fg'),
      M = PE.css('--muted'), Rule = PE.css('--rule'), BG = PE.css('--bg');
    const bg7 = BG.length === 7 ? BG : '#16181d';
    ctx.fillStyle = BG; ctx.fillRect(0, 0, W, H);
    const L = layout(W), fs = L.sm ? 10 : 11.5;
    // Rows from the top: loss, head, feed-forward, cross-attention, self-attention, input tokens.
    const rowY = r => L.top + 22 + r * L.step;
    const box = (x, y, w, h, fill, stroke, text, tc, dash, lw = 1.5) => {
      ctx.save(); ctx.fillStyle = fill; ctx.fillRect(x, y, w, h); ctx.strokeStyle = stroke; ctx.lineWidth = lw; if (dash) ctx.setLineDash(dash);
      ctx.strokeRect(x + .5, y + .5, w - 1, h - 1); ctx.restore(); if (text) PE.label(ctx, text, x + w / 2, y + h / 2 + fs * .36, tc, fs, 'center'); };
    // The ViT: one forward pass per pair, shared by all three objectives. Trainable in BLIP (blue frame, not brown).
    let vit;
    if (L.sm) { vit = { x: 4, y: 30, w: W - 8, h: L.vitH }; box(vit.x, vit.y, vit.w, vit.h, PE.mix(bg7, B, .18), B, 'ViT image encoder (trained), one pass per pair', B); }
    else { vit = { x: 4, y: rowY(1), w: 104, h: 3 * L.step + L.bh }; box(vit.x, vit.y, vit.w, vit.h, PE.mix(bg7, B, .18), B, '', B);
      PE.label(ctx, 'ViT image', vit.x + vit.w / 2, vit.y + vit.h / 2 - 8, B, 12, 'center'); PE.label(ctx, 'encoder', vit.x + vit.w / 2, vit.y + vit.h / 2 + 8, B, 12, 'center');
      PE.label(ctx, '(trained)', vit.x + vit.w / 2, vit.y + vit.h / 2 + 24, M, 10.5, 'center'); PE.label(ctx, 'one pass per pair', vit.x + vit.w / 2, vit.y + vit.h + 16, M, 10.5, 'center');
      PE.arrow(ctx, vit.x + vit.w + 2, vit.y + vit.h / 2, L.x0 - 3, vit.y + vit.h / 2, B, 2, 8); }
    const encSA = PE.mix(bg7, Y, .22), ffn = PE.mix(bg7, Y, .12), caF = PE.mix(bg7, G, .2);
    K.forEach((k, i) => {
      const c = COL[k], x = L.colX(i), w = L.cw, a = S.a[k], g = glow(k);
      ctx.save(); ctx.globalAlpha = a;
      if (L.sm) c.t2.forEach((t, j) => PE.label(ctx, t, x + w / 2, L.top - 6 + 12 * j, F, 9.5, 'center'));
      else PE.label(ctx, c.title, x + w / 2, L.top + 6, F, 11.5, 'center');
      box(x, rowY(0), w, L.bh, PE.mix(bg7, Rd, .12 + .25 * g), Rd, c.loss, Rd, null, 1.5 + 2 * g);
      box(x, rowY(1), w, L.bh, PE.mix(bg7, Y, .06), Y, L.sm ? c.hs : c.head, F);
      box(x, rowY(2), w, L.bh, ffn, Y, 'feed-forward', F);
      if (c.ca) box(x, rowY(3), w, L.bh, caF, G, c.ca, F); else box(x, rowY(3), w, L.bh, bg7, Rule, 'no cross-attn', M, [4, 4]);
      if (k === 'lm') box(x, rowY(4), w, L.bh, bg7, Y, c.sa, F, [5, 3], 2); else box(x, rowY(4), w, L.bh, encSA, Y, c.sa, F);
      PE.label(ctx, c.tok, x + w / 2, rowY(4) + L.bh + 18, Y, L.sm ? 9.5 : 11, 'center');
      for (let r = 4; r > 0; r--) PE.arrow(ctx, x + w / 2, rowY(r) - 1, x + w / 2, rowY(r - 1) + L.bh + 1, M, 1.2, 6);
      if (g > 0) { ctx.strokeStyle = F; ctx.globalAlpha = g * a; ctx.lineWidth = 2.5; ctx.strokeRect(x - 3, L.top - 8, w + 6, rowY(4) + L.bh + 26 - L.top); }
      // Vision enters through cross-attention (ITM, LM), or as the image vector for the contrastive score (ITC).
      ctx.globalAlpha = a; const tgt = c.ca ? rowY(3) + L.bh / 2 : rowY(1) + L.bh / 2;
      ctx.fillStyle = B; ctx.fillRect(x, tgt - L.bh / 2, 5, L.bh);
      ctx.restore();
    });
    // Same weights: SA shared by the two encoder passes; cross-attention and feed-forward shared by encoder and decoder.
    const link = (r, i0, i1, txt) => { const y = rowY(r) + L.bh / 2;
      ctx.save(); ctx.strokeStyle = F; ctx.lineWidth = 1.5; ctx.setLineDash([2, 3]);
      for (let i = i0; i < i1; i++) { ctx.beginPath(); ctx.moveTo(L.colX(i) + L.cw, y); ctx.lineTo(L.colX(i + 1), y); ctx.stroke(); }
      ctx.restore(); if (txt) for (let i = i0; i < i1; i++) PE.label(ctx, '=', L.colX(i) + L.cw + L.gap / 2, y - 5, F, 12, 'center'); };
    link(2, 0, 2, true); link(3, 1, 2, true); link(4, 0, 1, true);
    const ly = rowY(4) + L.bh + 38;
    const leg = ['"=" and matching fill: same weights. Dashed causal SA: the decoder\'s own weights.', 'Blue edge: where the ViT\'s image features enter.'];
    if (L.sm) ['"=" and matching fill: same weights.', 'Dashed causal SA: the decoder\'s own weights.', 'Blue edge: where image features enter.'].forEach((t, j) => PE.label(ctx, t, 4, ly - 6 + 13 * j, M, 9.5));
    else leg.forEach((t, j) => PE.label(ctx, t, 4, ly + 14 * j - 4, M, 11));
  });

  const TXT = {
    all: 'One image goes through the ViT once; its caption goes through the text transformer three times, once per objective. Layers joined by "=" are the same weights; only the decoder\'s causal self-attention is separate.',
    itc: 'ITC pass: the text runs through the encoder alone, with no cross-attention; its [CLS] vector is scored against the image\'s vector. This is the chapter 1 matching game, with a momentum encoder supplying soft targets.',
    itm: 'ITM pass: the same encoder weights, now with cross-attention reading the image in every block. A linear head on the [Encode] token says match or no match; the negatives are sampled, favouring the wrong captions the contrastive score found most similar.',
    lm: 'LM pass: causal self-attention with its own weights, but the same cross-attention and feed-forward layers. The model writes the caption one word at a time; this pass becomes the captioner.',
  };
  function select(sel) {
    stop(); S.sel = sel; const a0 = { ...S.a }, tg = {}; K.forEach(k => tg[k] = sel === 'all' || sel === k ? 1 : .18);
    fig.querySelectorAll('.controls button[aria-pressed]').forEach(b => b.setAttribute('aria-pressed', b.id === 'btn-capfilt-med-' + sel));
    out.textContent = TXT[sel];
    stop = PE.animate(600, u => { K.forEach(k => S.a[k] = PE.lerp(a0[k], tg[k], u)); cv.redraw(); });
  }
  ['all', ...K].forEach(k => document.getElementById('btn-capfilt-med-' + k).addEventListener('click', () => select(k)));
  document.getElementById('btn-capfilt-med-reset').addEventListener('click', () => { stop(); K.forEach(k => S.a[k] = 1); select('all'); });
  out.textContent = TXT.all;
})();
</script>
