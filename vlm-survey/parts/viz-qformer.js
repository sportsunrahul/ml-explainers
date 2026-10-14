<script>
(() => {
  const fig = document.getElementById('viz-qformer'), el = document.getElementById('qformer-canvas');
  const out = document.getElementById('qformer-readout');
  const NQ = 8, TXT = ['a', 'cat', 'wearing', 'sunglasses'], NT = TXT.length + 1, N = NQ + NT;
  // Rows = the token that attends, columns = the token it may attend to (BLIP-2 Fig. 2, right). Checked in build/sim/qformer-masks.cjs.
  const allow = (m, i, j) => {
    const qi = i < NQ, qj = j < NQ;
    if (qi && qj) return 1;
    if (m === 'itm') return 1;
    if (qi) return 0;
    if (qj) return m === 'itg' ? 1 : 0;
    return m === 'itc' ? 1 : (j <= i ? 1 : 0);
  };
  const maskOf = m => Array.from({ length: N }, (_, i) => Array.from({ length: N }, (_, j) => allow(m, i, j)));
  const NAME = { itc: 'ITC: uni-modal mask', itm: 'ITM: bi-directional mask', itg: 'ITG: multimodal causal mask' };
  const S = { view: 'itm', mask: maskOf('itm'), s2: 0, enc: 0, first: '[CLS]' };
  let stop = () => {};
  const glow = PE.linkPulses(fig, () => cv.redraw());
  // Parameter counts: BLIP-2 Table 2 (ViT-g FlanT5-XXL row, Flamingo80B row) and Sec. 3.1 (Q-Former total).
  const BARS = [
    ['BLIP-2: trained (Table 2)', 108e6, 'g'], ['Q-Former, all of it (Sec. 3.1)', 188e6, 'g2'], ['BLIP-2: total', 12.1e9, 'b'],
    ['Flamingo80B: trained', 10.2e9, 'g'], ['Flamingo80B: total', 80e9, 'b']];
  const fmt = v => v >= 1e9 ? +(v / 1e9).toFixed(1) + 'B' : Math.round(v / 1e6) + 'M';

  const layout = W => {
    const sm = W < 640, lab = sm ? 70 : 84, cell = sm ? Math.min(21, (W - lab - 12) / N) : Math.min(22, (W * .56 - lab - 20) / N);
    const gx = lab, gy = 122, gw = cell * N, left = { x: 0, y: 0, w: sm ? W : gx + gw + 10, h: gy + gw + 30 };
    const right = sm ? { x: 0, y: left.h + 10, w: W, h: 230 } : { x: left.w + 22, y: 30, w: W - left.w - 26, h: 230 };
    return { sm, lab, cell, gx, gy, gw, left, right, H: sm ? right.y + right.h + 6 : Math.max(left.h, right.y + right.h) + 6 };
  };
  const cv = PE.canvas(el, w => layout(w).H / w, (ctx, W, H) => {
    const B = PE.css('--blue'), Y = PE.css('--yellow'), G = PE.css('--green'), Rd = PE.css('--red'), Br = PE.css('--brown'),
      F = PE.css('--fg'), M = PE.css('--muted'), Rule = PE.css('--rule'), BG = PE.css('--bg');
    const bg7 = BG.length === 7 ? BG : '#16181d';
    ctx.fillStyle = BG; ctx.fillRect(0, 0, W, H);
    const L = layout(W), gq = glow('queries'), gm = glow('mask'), go = glow('open'), gp = glow('prompt'), gt = glow('text');
    const tok = j => j < NQ ? 'q' + (j + 1) : j === NQ ? S.first : TXT[j - NQ - 1];

    // ---- Left panel A: the stage-1 self-attention mask over [queries | text]. ----
    ctx.save(); ctx.globalAlpha = 1 - S.s2;
    const view = S.view in NAME ? S.view : S.last || 'itm';
    PE.label(ctx, NAME[view], 4, 18, F, L.sm ? 12 : 13);
    PE.label(ctx, `${NQ} of the 32 queries drawn`, 4, 36, M, 11);
    PE.label(ctx, 'row: token that looks; column: token it may look at', 4, 52, M, L.sm ? 10 : 11);
    const { gx, gy, cell } = L;
    for (let j = 0; j < N; j++) {
      const isQ = j < NQ, c = isQ ? G : Y, bump = isQ ? gq + gp : gt;
      ctx.save(); ctx.translate(gx + j * cell + cell * (L.sm ? .7 : .62), gy - 4); ctx.rotate(L.sm ? -Math.PI / 2 : -Math.PI / 3);
      PE.label(ctx, tok(j), 0, 0, c, (L.sm ? 9 : 10) + 1.5 * bump); ctx.restore();
      PE.label(ctx, tok(j), gx - 5, gy + j * cell + cell * .68, c, (L.sm ? 9.5 : 10.5) + 1.5 * bump, 'right');
    }
    for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) {
      const a = S.mask[i][j], x = gx + j * cell, y = gy + i * cell, c = j < NQ ? G : Y;
      ctx.globalAlpha = 1 - S.s2; ctx.fillStyle = PE.mix(bg7, Br, .2); ctx.fillRect(x + 1, y + 1, cell - 2, cell - 2);
      ctx.globalAlpha = (1 - S.s2) * .8 * a; ctx.fillStyle = c; ctx.fillRect(x + 1, y + 1, cell - 2, cell - 2); ctx.globalAlpha = 1 - S.s2;
      if (a < .5 && gm > 0) { ctx.strokeStyle = Br; ctx.lineWidth = 2 * gm; ctx.strokeRect(x + 2, y + 2, cell - 4, cell - 4); }
      if (a >= .5 && go > 0) { ctx.strokeStyle = F; ctx.globalAlpha = go * (1 - S.s2); ctx.lineWidth = 1.5; ctx.strokeRect(x + 2, y + 2, cell - 4, cell - 4); }
      ctx.globalAlpha = 1 - S.s2;
    }
    ctx.strokeStyle = F; ctx.lineWidth = 1.5; ctx.beginPath();
    ctx.moveTo(gx + NQ * cell, gy - 2); ctx.lineTo(gx + NQ * cell, gy + N * cell + 2); ctx.moveTo(gx - 2, gy + NQ * cell); ctx.lineTo(gx + N * cell + 2, gy + NQ * cell); ctx.stroke();
    let open = 0; S.mask.forEach(r => r.forEach(v => open += v >= .5));
    PE.label(ctx, `open ${open} of ${N * N} cells; dark = masked`, gx, gy + N * cell + 18, M, 11);
    ctx.restore();

    // ---- Left panel B (stage 2): frozen ViT -> Q-Former -> FC -> soft prompt -> frozen LLM. ----
    if (S.s2 > 0) {
      ctx.save(); ctx.globalAlpha = S.s2;
      const x0 = 4, w = L.left.w - 12, bh = 30, fs = L.sm ? 10.5 : 11.5;
      const box = (x, y, bw, fill, stroke, text, tc, lw = 1.5) => { ctx.fillStyle = fill; ctx.fillRect(x, y, bw, bh); ctx.strokeStyle = stroke; ctx.lineWidth = lw; ctx.strokeRect(x + .5, y + .5, bw - 1, bh - 1);
        PE.label(ctx, text, x + bw / 2, y + bh / 2 + fs * .36, tc, fs, 'center'); };
      PE.label(ctx, S.enc < .5 ? 'Stage 2, decoder-only LLM (OPT)' : 'Stage 2, encoder-decoder LLM (FlanT5)', 4, 18, F, L.sm ? 11.5 : 13);
      box(x0, 34, w, PE.mix(bg7, B, .2), Br, 'frozen image encoder', B, 2.5);
      PE.arrow(ctx, x0 + w / 2, 34 + bh, x0 + w / 2, 76, M, 1.4, 6);
      box(x0, 78, w, PE.mix(bg7, G, .22), G, 'Q-Former (trained)', F);
      PE.label(ctx, `${NQ} of 32 output queries, 768 wide each`, x0 + w / 2 + 6, 128, M, 10.5, 'center');
      PE.arrow(ctx, x0 + w / 2, 78 + bh, x0 + w / 2, 140, M, 1.4, 6);
      box(x0 + w * .3, 142, w * .4, PE.mix(bg7, G, .22), G, 'FC layer (trained)', F);
      PE.arrow(ctx, x0 + w / 2, 142 + bh, x0 + w / 2, 196, M, 1.4, 6);
      // Token strip: the projected queries become a soft prompt in front of the text.
      const ty = 200, ts = Math.min(22, w * .38 / NQ), words = ['a', 'cat', 'wearing', 'sunglasses'];
      const split = PE.lerp(0, 1, S.enc), sufGap = 14 * split;
      for (let k = 0; k < NQ; k++) { ctx.fillStyle = G; ctx.globalAlpha = S.s2 * (.75 + .25 * gp); ctx.fillRect(x0 + k * ts + 1, ty, ts - 3, ts - 3);
        if (gp > 0) { ctx.strokeStyle = F; ctx.lineWidth = 2 * gp; ctx.strokeRect(x0 + k * ts, ty - 1, ts - 1, ts - 1); } }
      ctx.globalAlpha = S.s2;
      let tx = x0 + NQ * ts + 6; const wpos = [];
      ctx.font = `600 ${fs}px system-ui, sans-serif`;
      words.forEach((wd, k) => { if (k === 2) tx += sufGap; const ww = ctx.measureText(wd).width + 8;
        ctx.strokeStyle = Y; ctx.lineWidth = 1 + gt; ctx.strokeRect(tx, ty, ww, ts - 3); PE.label(ctx, wd, tx + 4, ty + ts * .62, Y, fs); wpos.push([tx, ww]); tx += ww + 3; });
      PE.label(ctx, 'soft visual prompt', x0, ty + ts + 12, G, 10.5);
      // Which tokens carry the loss: all text (decoder-only) or only the suffix (encoder-decoder, prefix LM).
      ctx.strokeStyle = Rd; ctx.lineWidth = 3;
      wpos.forEach(([x, ww], k) => { const onLoss = k >= 2 ? 1 : 1 - S.enc; if (onLoss > .02) { ctx.globalAlpha = S.s2 * onLoss; ctx.beginPath(); ctx.moveTo(x, ty + ts + 1); ctx.lineTo(x + ww, ty + ts + 1); ctx.stroke(); } });
      ctx.globalAlpha = S.s2;
      const lossX = wpos[2][0], lossLab = S.enc < .5 ? 'loss on every text token' : 'loss on the suffix only';
      PE.label(ctx, lossLab, Math.min(lossX, x0 + w - 150), ty + ts + 26, Rd, 10.5);
      const ly = ty + ts + 40;
      if (S.enc < .5) { box(x0, ly, w, PE.mix(bg7, Y, .16), Br, 'frozen LLM decoder: predicts the text', F, 2.5); }
      else { const pre = wpos[1][0] + wpos[1][1] - x0;
        box(x0, ly, pre, PE.mix(bg7, Y, .16), Br, 'frozen encoder', F, 2.5);
        box(x0 + pre + 8, ly, w - pre - 8, PE.mix(bg7, Y, .16), Br, 'frozen decoder', F, 2.5);
        PE.label(ctx, 'prompt + prefix', x0 + pre / 2, ly + bh + 14, M, 10, 'center'); PE.label(ctx, 'generates suffix', x0 + pre + 8 + (w - pre - 8) / 2, ly + bh + 14, M, 10, 'center'); }
      PE.label(ctx, 'brown frame = frozen; green = trained', x0, ly + bh + 30, M, 10.5);
      ctx.restore();
    }

    // ---- Right panel: trainable vs frozen parameters, log scale. ----
    const R = L.right, ax = R.x + 6, aw = R.w - 60, lo = Math.log10(5e7), hi = Math.log10(1.5e11);
    const X = v => ax + (Math.log10(v) - lo) / (hi - lo) * aw;
    PE.label(ctx, 'parameters (log scale)', R.x + 6, R.y + 12, F, 12.5);
    [1e8, 1e9, 1e10, 1e11].forEach(v => { ctx.strokeStyle = Rule; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(X(v), R.y + 22); ctx.lineTo(X(v), R.y + 26 + 5 * 36); ctx.stroke();
      PE.label(ctx, fmt(v), X(v), R.y + 42 + 5 * 36, M, 10.5, 'center'); });
    BARS.forEach(([name, v, k], i) => {
      const y = R.y + 36 + i * 36, c = k === 'b' ? Br : G;
      PE.label(ctx, name, ax, y - 3, k === 'b' ? M : F, 11);
      ctx.fillStyle = c; ctx.globalAlpha = k === 'g2' ? .55 : 1; ctx.fillRect(ax, y, X(v) - ax, 12); ctx.globalAlpha = 1;
      PE.label(ctx, fmt(v), X(v) + 5, y + 11, c, 11);
      if (i === 2) { ctx.strokeStyle = Rule; ctx.beginPath(); ctx.moveTo(R.x, y + 22); ctx.lineTo(R.x + R.w, y + 22); ctx.stroke(); }
    });
  });

  const TXTS = {
    itm: 'ITM, bi-directional: every query and every word see each other (all 169 cells open). Each output query goes through a two-class linear classifier and the 32 logits are averaged into one match score.',
    itc: 'ITC, uni-modal: queries see only queries and words see only words (89 of 169 cells open). The image side and the text side are encoded apart, then the best of the 32 query-vs-[CLS] similarities is the match score.',
    itg: 'ITG, multimodal causal: queries still see only queries; each word sees all queries and the words before it (119 of 169 cells open). Words can get the image only through the queries.',
    dec: 'Stage 2, decoder-only LLM: one fully connected layer maps the query outputs to the LLM\'s embedding size, and they go in front of the text as a soft prompt. The frozen LLM predicts the text; gradients reach only the Q-Former and that FC layer.',
    encdec: 'Stage 2, encoder-decoder LLM: the soft prompt plus the first part of the text (the prefix) go into the frozen encoder; the frozen decoder must generate the rest (the suffix).',
  };
  function setMask(m) {
    stop(); S.view = m; S.last = m; const a0 = S.mask.map(r => r.slice()), tg = maskOf(m), s20 = S.s2;
    const first = m === 'itg' ? '[DEC]' : '[CLS]'; out.textContent = TXTS[m];
    stop = PE.animate(800, u => { S.mask = a0.map((r, i) => r.map((v, j) => PE.lerp(v, tg[i][j], u))); S.s2 = PE.lerp(s20, 0, u); if (u > .5) S.first = first; cv.redraw(); });
  }
  function setStage2(k) {
    stop(); S.view = k; const s20 = S.s2, e0 = S.enc, e1 = k === 'encdec' ? 1 : 0; out.textContent = TXTS[k];
    stop = PE.animate(800, u => { S.s2 = PE.lerp(s20, 1, u); S.enc = s20 < .5 ? e1 : PE.lerp(e0, e1, u); cv.redraw(); });
  }
  const press = id => fig.querySelectorAll('.controls button[aria-pressed]').forEach(b => b.setAttribute('aria-pressed', b.id === id));
  ['itc', 'itm', 'itg'].forEach(m => document.getElementById('btn-qformer-' + m).addEventListener('click', () => { press('btn-qformer-' + m); setMask(m); }));
  ['dec', 'encdec'].forEach(k => document.getElementById('btn-qformer-' + k).addEventListener('click', () => { press('btn-qformer-' + k); setStage2(k); }));
  document.getElementById('btn-qformer-reset').addEventListener('click', () => {
    stop(); S.view = 'itm'; S.last = 'itm'; S.mask = maskOf('itm'); S.s2 = 0; S.enc = 0; S.first = '[CLS]'; press('btn-qformer-itm'); out.textContent = TXTS.itm; cv.redraw(); });
  press('btn-qformer-itm'); out.textContent = TXTS.itm;
})();
</script>
