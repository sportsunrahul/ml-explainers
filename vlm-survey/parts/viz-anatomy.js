<script>
(() => {
  const fig = document.getElementById('viz-anatomy'), el = document.getElementById('anatomy-canvas');
  const out = document.getElementById('anatomy-readout'), info = document.getElementById('anatomy-info');
  const PARTS = ['resolution', 'encoder', 'merger', 'mrope', 'video', 'deepstack', 'llm'];
  const NAMES = { resolution: 'native resolution', encoder: 'SigLIP-2 encoder', merger: '2x2 MLP merger', mrope: 'interleaved M-RoPE positions',
    video: 'text timestamps for video', deepstack: 'DeepStack', llm: 'Qwen3 language model' };
  const S = { sel: null, hl: Object.fromEntries(PARTS.map(p => [p, 0])), ds: 0, hits: [] };
  let stopSel = () => {}, stopDs = () => {};
  const glow = PE.linkPulses(fig, () => cv.redraw());
  const ENC_N = 7, TAPS = [1, 3, 5], LLM_COLS = ['L1', 'L2', 'L3', 'L4', '...', 'LN'];
  const TOKS = [['Describe', 't'], ['<0.0 seconds>', 's'], ['vis', 'v'], ['vis', 'v'], ['<2.0 seconds>', 's'], ['vis', 'v'], ['vis', 'v'], ['the clip', 't']];

  function rrect(ctx, x, y, w, h, r) { if (w <= 0 || h <= 0) { ctx.beginPath(); return; } r = Math.min(r, w / 2, h / 2); ctx.beginPath(); ctx.moveTo(x + r, y); ctx.arcTo(x + w, y, x + w, y + h, r);
    ctx.arcTo(x + w, y + h, x, y + h, r); ctx.arcTo(x, y + h, x, y, r); ctx.arcTo(x, y, x + w, y, r); ctx.closePath(); }
  const layout = (W, H) => {
    if (W >= 560) return { st: false,
      img: { x: W * .01, y: H * .33, w: W * .13, h: H * .22 }, vid: { x: W * .015, y: H * .64, w: W * .12, h: H * .24 },
      enc: { x: W * .17, y: H * .3, w: W * .19, h: H * .6 }, mer: { x: W * .39, y: H * .52, w: W * .07, h: H * .15 },
      tok: { x: W * .49, y: H * .27, w: W * .145, h: H * .6 }, llm: { x: W * .665, y: H * .27, w: W * .31, h: H * .6 }, apex: H * .14 };
    const top = 34, eh = W * .5;
    return { st: true,
      img: { x: W * .02, y: top + 4, w: W * .28, h: eh * .42 }, vid: { x: W * .02, y: top + eh * .55, w: W * .28, h: eh * .42 },
      enc: { x: W * .36, y: top, w: W * .40, h: eh }, mer: { x: W * .81, y: top + eh * .3, w: W * .17, h: eh * .3 },
      tok: { x: W * .02, y: top + eh + 70, w: W * .37, h: H - (top + eh + 70) - 40 }, llm: { x: W * .44, y: top + eh + 70, w: W * .54, h: H - (top + eh + 70) - 40 } };
  };
  const encCol = (L, i) => { const cw = L.enc.w / ENC_N, t = L.st ? 22 : 0; return { x: L.enc.x + i * cw + cw * .15, y: L.enc.y + t, w: cw * .7, h: L.enc.h - t - 8 }; };
  const llmCol = (L, i) => { const cw = L.llm.w / LLM_COLS.length; return { x: L.llm.x + i * cw + cw * .12, y: L.llm.y, w: cw * .76, h: L.llm.h }; };
  const tokRow = (L, i) => { const rh = L.tok.h / TOKS.length; return { x: L.tok.x, y: L.tok.y + i * rh + rh * .12, w: L.tok.w, h: rh * .76 }; };
  const bez = (p0, p1, p2, p3, t) => { const u = 1 - t; return [0, 1].map(k => u * u * u * p0[k] + 3 * u * u * t * p1[k] + 3 * u * t * t * p2[k] + t * t * t * p3[k]); };
  function arcPts(L, j) {
    const e = encCol(L, TAPS[j]), l = llmCol(L, j);
    if (!L.st) { const p0 = [e.x + e.w / 2, e.y], p3 = [l.x + l.w / 2, l.y - 2], ay = L.apex + j * 10;
      return [p0, [p0[0], ay], [p3[0], ay], p3]; }
    const p0 = [e.x + e.w / 2, e.y + e.h], p3 = [l.x + l.w / 2, l.y - 2], my = (p0[1] + p3[1]) / 2;
    return [p0, [p0[0], my + 10], [p3[0], my - 10], p3];
  }

  const cv = PE.canvas(el, w => w < 560 ? 1.55 : .56, (ctx, W, H) => {
    const B = PE.css('--blue'), Y = PE.css('--yellow'), G = PE.css('--green'), F = PE.css('--fg'), M = PE.css('--muted'),
      Rule = PE.css('--rule'), BG = PE.css('--bg'), Br = PE.css('--brown');
    ctx.fillStyle = BG; ctx.fillRect(0, 0, W, H);
    if (W < 200) return;
    const L = layout(W, H), any = S.sel ? 1 : 0, fs = L.st ? 10 : 11;
    const dim = p => 1 - .45 * any * (1 - S.hl[p]);
    const outline = (r, p, color) => { const h = S.hl[p]; if (h <= .01) return; ctx.save(); ctx.globalAlpha = h; ctx.strokeStyle = F; ctx.lineWidth = 2.5;
      ctx.setLineDash([6, 4]); rrect(ctx, r.x - 5, r.y - 5, r.w + 10, r.h + 10, 9); ctx.stroke(); ctx.restore(); };
    const gl = glow('level'), gm = glow('merge'), gh = glow('hidden');
    S.hits = [];
    PE.label(ctx, 'Qwen3-VL, redrawn from Sec. 2 (not to scale)', 6, 14, M, 10.5);

    // Residual streams: each token's row runs through every LLM layer.
    ctx.save(); ctx.globalAlpha = dim('llm');
    TOKS.forEach((t, i) => { const r = tokRow(L, i), c = t[1] === 'v' ? B : Y; PE.polyline(ctx, [[L.tok.x + L.tok.w, r.y + r.h / 2], [L.llm.x + L.llm.w, r.y + r.h / 2]], c, 1.2, null, .35); });
    LLM_COLS.forEach((n, i) => { const r = llmCol(L, i), hot = i < 3 ? gh : 0;
      ctx.fillStyle = PE.mix(BG, Y, n === '...' ? .04 : .13); rrect(ctx, r.x, r.y, r.w, r.h, 6); ctx.fill();
      if (n !== '...') { ctx.strokeStyle = hot > 0 ? F : Y; ctx.lineWidth = 1.4 + 2 * hot; ctx.stroke(); }
      PE.label(ctx, n, r.x + r.w / 2, r.y + r.h + 13, Y, fs, 'center'); });
    ctx.restore();
    PE.label(ctx, 'language model layers', L.llm.x + L.llm.w / 2, L.llm.y + L.llm.h + 28, Y, fs, 'center');
    S.hits.push(['llm', L.llm]); outline(L.llm, 'llm');
    if (S.hl.llm > .01) { ctx.save(); ctx.globalAlpha = S.hl.llm;
      const r = { x: L.llm.x + 8, y: L.llm.y + L.llm.h * .36, w: L.llm.w - 16, h: 44 }; ctx.fillStyle = PE.css('--panel'); rrect(ctx, r.x, r.y, r.w, r.h, 6); ctx.fill();
      PE.label(ctx, L.st ? '4 dense + 2 MoE sizes' : '4 dense + 2 mixture-of-experts sizes', r.x + r.w / 2, r.y + 18, F, fs, 'center');
      PE.label(ctx, L.st ? 'context: 256K tokens' : 'native context: 256K tokens', r.x + r.w / 2, r.y + 34, F, fs, 'center'); ctx.restore(); }

    // The one sequence: text, timestamps (both text) and visual tokens.
    ctx.save(); ctx.globalAlpha = Math.min(dim('mrope'), 1);
    TOKS.forEach((t, i) => { const r = tokRow(L, i), c = t[1] === 'v' ? B : Y, hotTs = t[1] === 's' ? S.hl.video : 0;
      ctx.fillStyle = PE.mix(BG, c, .2 + .25 * hotTs); rrect(ctx, r.x, r.y, r.w, r.h, 5); ctx.fill();
      ctx.strokeStyle = hotTs > .01 ? F : c; ctx.lineWidth = 1.2 + 1.8 * hotTs; ctx.stroke();
      PE.label(ctx, t[0], r.x + r.w / 2, r.y + r.h / 2 + 4, c, t[1] === 's' ? fs - .5 : fs, 'center'); });
    ctx.restore();
    PE.label(ctx, 'one sequence', L.tok.x + L.tok.w / 2, L.tok.y + L.tok.h + 13, M, fs, 'center');
    S.hits.push(['mrope', L.tok]); outline(L.tok, 'mrope');

    // Inputs.
    const im = L.img, iw = Math.min(im.w, im.h * 1.5), ih = iw / 1.5, ix = im.x + (im.w - iw) / 2, iy = im.y + (im.h - ih) / 2;
    ctx.save(); ctx.globalAlpha = dim('resolution'); ctx.fillStyle = PE.mix(BG, B, .25); ctx.fillRect(ix, iy, iw, ih); ctx.strokeStyle = B; ctx.lineWidth = 1.5; ctx.strokeRect(ix, iy, iw, ih);
    const gA = .25 + .75 * S.hl.resolution, nx = 9, ny = 6; ctx.globalAlpha = dim('resolution') * gA; ctx.strokeStyle = B; ctx.lineWidth = 1;
    for (let k = 1; k < nx; k++) { ctx.beginPath(); ctx.moveTo(ix + k * iw / nx, iy); ctx.lineTo(ix + k * iw / nx, iy + ih); ctx.stroke(); }
    for (let k = 1; k < ny; k++) { ctx.beginPath(); ctx.moveTo(ix, iy + k * ih / ny); ctx.lineTo(ix + iw, iy + k * ih / ny); ctx.stroke(); }
    ctx.restore(); PE.label(ctx, 'image, own size', im.x + im.w / 2, iy - 6, B, fs, 'center');
    S.hits.push(['resolution', im]); outline(im, 'resolution');
    const vd = L.vid; ctx.save(); ctx.globalAlpha = dim('video');
    for (let k = 0; k < 3; k++) { const fw = vd.w * .62, fh = vd.h * .42, fx = vd.x + k * vd.w * .17, fy = vd.y + vd.h * .12 + k * vd.h * .16;
      ctx.fillStyle = PE.mix(BG, B, .18 + .06 * k); ctx.fillRect(fx, fy, fw, fh); ctx.strokeStyle = B; ctx.lineWidth = 1.2; ctx.strokeRect(fx, fy, fw, fh); }
    ctx.restore(); PE.label(ctx, 'video frames', vd.x + vd.w / 2, vd.y + vd.h + 4, B, fs, 'center');
    S.hits.push(['video', vd]); outline(vd, 'video');

    // Encoder: layers left to right; three levels tapped.
    ctx.save(); ctx.globalAlpha = dim('encoder');
    for (let i = 0; i < ENC_N; i++) { const r = encCol(L, i); ctx.fillStyle = PE.mix(BG, B, .16); rrect(ctx, r.x, r.y, r.w, r.h, 4); ctx.fill(); ctx.strokeStyle = B; ctx.lineWidth = 1.3; ctx.stroke(); }
    ctx.restore();
    PE.label(ctx, 'SigLIP-2 encoder', L.enc.x + L.enc.w / 2, L.st ? L.enc.y + 12 : L.enc.y + L.enc.h + 19, B, fs + .5, 'center');
    S.hits.push(['encoder', L.enc]); outline(L.enc, 'encoder');
    PE.arrow(ctx, im.x + im.w + 2, im.y + im.h / 2, L.enc.x + 3, L.st ? L.enc.y + L.enc.h * .35 : im.y + im.h / 2, B, 1.8, 7, dim('resolution'));
    PE.arrow(ctx, vd.x + vd.w + 2, vd.y + vd.h / 2, L.enc.x + 3, L.st ? L.enc.y + L.enc.h * .75 : vd.y + vd.h / 2, B, 1.8, 7, dim('video'));

    // Main merger: last encoder layer -> 2x2 MLP -> visual tokens.
    const mr = L.mer, last = encCol(L, ENC_N - 1);
    PE.arrow(ctx, last.x + last.w + 2, L.st ? mr.y + mr.h / 2 : mr.y + mr.h / 2, mr.x - 3, mr.y + mr.h / 2, B, 2, 8, dim('merger'));
    ctx.save(); ctx.globalAlpha = dim('merger'); ctx.fillStyle = PE.mix(BG, G, .2);
    ctx.beginPath(); ctx.moveTo(mr.x, mr.y); ctx.lineTo(mr.x + mr.w, mr.y + mr.h * .22); ctx.lineTo(mr.x + mr.w, mr.y + mr.h * .78); ctx.lineTo(mr.x, mr.y + mr.h); ctx.closePath(); ctx.fill();
    ctx.strokeStyle = G; ctx.lineWidth = 2; ctx.stroke(); ctx.restore();
    PE.label(ctx, '2x2 -> 1', mr.x + mr.w / 2, mr.y + mr.h / 2 + 4, G, fs - .5, 'center');
    PE.label(ctx, 'merger', mr.x + mr.w / 2, mr.y + mr.h + 13, G, fs, 'center');
    S.hits.push(['merger', mr]); outline(mr, 'merger');
    const v0 = tokRow(L, 2), v1 = tokRow(L, 6);
    if (!L.st) PE.arrow(ctx, mr.x + mr.w + 2, mr.y + mr.h / 2, L.tok.x - 3, (v0.y + v1.y + v1.h) / 2, G, 2, 8, dim('merger'));
    else { const p = [mr.x + mr.w / 2, mr.y + mr.h + 18], q = [L.tok.x + L.tok.w * .6, L.tok.y - 16];
      ctx.save(); ctx.globalAlpha = dim('merger'); ctx.strokeStyle = G; ctx.lineWidth = 2; ctx.beginPath(); ctx.moveTo(p[0], p[1]); ctx.bezierCurveTo(p[0], p[1] + 30, q[0] + 40, q[1] - 34, q[0], q[1]); ctx.stroke(); ctx.restore();
      PE.arrow(ctx, q[0] + 2, q[1] - 6, q[0], q[1] + 2, G, 2, 8, dim('merger')); }

    // DeepStack: three encoder levels -> own mergers -> added into LLM layers 1-3 at the visual tokens.
    let bb = null;
    for (let j = 0; j < 3; j++) {
      const pts = arcPts(L, j), p = PE.clamp(S.ds * 3 - j * .6, 0, 1), show = Math.max(p, gl, gm, gh) > 0, a = Math.max(p > 0 ? 1 : 0, gl, gm, gh) * dim('deepstack');
      const e = encCol(L, TAPS[j]), tapY = L.st ? e.y + e.h : e.y;
      PE.dot(ctx, e.x + e.w / 2, tapY, 3.5 + 3 * gl, B, Math.max(.35, a));
      if (!show) continue;
      const P = []; const tEnd = Math.max(p, gl, gm, gh > 0 ? 1 : 0);
      for (let k = 0; k <= 40; k++) P.push(bez(...pts, tEnd * k / 40));
      PE.polyline(ctx, P, G, 2, null, a);
      const mid = bez(...pts, .5); if (tEnd >= .5) { ctx.save(); ctx.globalAlpha = a; ctx.fillStyle = PE.mix(BG, G, .35); const s = 9 + 3 * gm; ctx.fillRect(mid[0] - s, mid[1] - s * .7, 2 * s, 1.4 * s);
        ctx.strokeStyle = G; ctx.lineWidth = 1.5 + 1.5 * gm; ctx.strokeRect(mid[0] - s, mid[1] - s * .7, 2 * s, 1.4 * s); ctx.restore(); }
      if (tEnd >= 1) { const l = llmCol(L, j);
        TOKS.forEach((t, i) => { if (t[1] !== 'v') return; const r = tokRow(L, i), cx = l.x + l.w / 2, cy = r.y + r.h / 2;
          PE.dot(ctx, cx, cy, 6.5 + 2 * gh, G, a); PE.label(ctx, '+', cx, cy + 4, BG, 11, 'center'); }); }
      P.forEach(([x, y]) => { bb = bb ? { x0: Math.min(bb.x0, x), y0: Math.min(bb.y0, y), x1: Math.max(bb.x1, x), y1: Math.max(bb.y1, y) } : { x0: x, y0: y, x1: x, y1: y }; });
    }
    if (S.ds > .01 || gl > 0) { if (L.st) { const y = L.enc.y + L.enc.h + 22; PE.label(ctx, 'DeepStack:', 8, y, G, fs); PE.label(ctx, '3 levels -> L1-L3', 8, y + 14, G, fs); }
      else PE.label(ctx, 'DeepStack: 3 encoder levels (layers not stated) -> added into LLM layers 1-3', W * .5, 32, G, fs, 'center'); }
    const dsr = bb ? { x: bb.x0, y: bb.y0, w: bb.x1 - bb.x0, h: bb.y1 - bb.y0 } : (L.st ? { x: L.enc.x, y: L.enc.y + L.enc.h, w: L.llm.x + L.llm.w * .5 - L.enc.x, h: 60 } : { x: L.enc.x, y: L.apex, w: L.llm.x + L.llm.w * .5 - L.enc.x, h: L.enc.y - L.apex });
    S.hits.unshift(['deepstack', dsr]); outline(dsr, 'deepstack');

    // Positions inset: which rotary dimensions each axis gets.
    if (S.hl.mrope > .01) {
      ctx.save(); ctx.globalAlpha = S.hl.mrope;
      const r = L.st ? { x: L.llm.x, y: L.llm.y + 20, w: L.llm.w, h: 118 } : { x: L.llm.x - 4, y: L.llm.y + 22, w: L.llm.w + 8, h: 128 };
      ctx.fillStyle = PE.css('--panel'); rrect(ctx, r.x, r.y, r.w, r.h, 8); ctx.fill(); ctx.strokeStyle = F; ctx.lineWidth = 1; ctx.stroke();
      const n = 12, cw = (r.w - 16) / n, rows = [['Qwen2-VL: blocks', 'tttthhhhwwww'], [L.st ? 'Qwen3-VL: interleaved*' : 'Qwen3-VL: interleaved (order illustrative)', 'thwthwthwthw']];
      rows.forEach(([lab, pat], k) => { const y = r.y + 22 + k * 46; PE.label(ctx, lab, r.x + 8, y, F, fs);
        for (let i = 0; i < n; i++) { const x = r.x + 8 + i * cw; ctx.fillStyle = PE.mix(BG, F, .28 - .22 * i / (n - 1)); ctx.fillRect(x + 1, y + 6, cw - 2, 18);
          PE.label(ctx, pat[i], x + cw / 2, y + 19, F, fs, 'center'); } });
      PE.label(ctx, 'fast rotation', r.x + 8, r.y + r.h - 6, M, fs - 1); PE.label(ctx, 'slow rotation', r.x + r.w - 8, r.y + r.h - 6, M, fs - 1, 'right');
      if (L.st) PE.label(ctx, '* order illustrative', r.x + r.w / 2, r.y + r.h - 20, M, fs - 1, 'center');
      ctx.restore();
    }
  });

  function showInfo(p) { info.querySelectorAll('[data-part]').forEach(d => { d.hidden = d.dataset.part !== (p || 'none'); }); }
  function readout() {
    const ds = S.ds > .5 ? 'DeepStack on: three encoder levels are added into language-model layers 1 to 3 at the visual tokens, with no extra tokens.' : 'DeepStack off: the language model sees only the last encoder layer, through the merger.';
    out.textContent = (S.sel ? `Selected: ${NAMES[S.sel]}. ` : 'Click a part of the diagram or use the part buttons. ') + ds;
  }
  function select(p) {
    stopSel(); S.sel = p; showInfo(p); const h0 = { ...S.hl };
    PARTS.forEach(q => { const b = document.getElementById('btn-anatomy-' + q); b.setAttribute('aria-pressed', q === p); });
    stopSel = PE.animate(500, u => { PARTS.forEach(q => { S.hl[q] = PE.lerp(h0[q], q === p ? 1 : 0, u); }); cv.redraw(); });
    readout();
  }
  function setDs(on) {
    stopDs(); const d0 = on ? 0 : S.ds; S.ds = d0;
    document.getElementById('btn-anatomy-before').setAttribute('aria-pressed', !on);
    document.getElementById('btn-anatomy-after').setAttribute('aria-pressed', on);
    stopDs = PE.animate(on ? 1600 : 500, u => { S.ds = PE.lerp(d0, on ? 1 : 0, u); cv.redraw(); readout(); });
  }
  PARTS.forEach(p => document.getElementById('btn-anatomy-' + p).addEventListener('click', () => select(p)));
  document.getElementById('btn-anatomy-before').addEventListener('click', () => setDs(false));
  document.getElementById('btn-anatomy-after').addEventListener('click', () => { setDs(true); });
  document.getElementById('btn-anatomy-reset').addEventListener('click', () => {
    stopSel(); stopDs(); S.sel = null; S.ds = 0; PARTS.forEach(q => { S.hl[q] = 0; document.getElementById('btn-anatomy-' + q).setAttribute('aria-pressed', false); });
    document.getElementById('btn-anatomy-before').setAttribute('aria-pressed', true); document.getElementById('btn-anatomy-after').setAttribute('aria-pressed', false);
    showInfo(null); cv.redraw(); readout(); });
  el.addEventListener('click', e => { const b = el.getBoundingClientRect(), x = e.clientX - b.left, y = e.clientY - b.top;
    const hit = S.hits.find(([, r]) => x >= r.x - 4 && x <= r.x + r.w + 4 && y >= r.y - 4 && y <= r.y + r.h + 4); if (hit) select(hit[0]); });
  el.addEventListener('keydown', e => {
    if (e.key !== 'ArrowRight' && e.key !== 'ArrowLeft') return; e.preventDefault();
    const i = PARTS.indexOf(S.sel), n = PARTS.length; select(PARTS[(i + (e.key === 'ArrowRight' ? 1 : n - 1) + (i < 0 && e.key === 'ArrowLeft' ? 1 : 0)) % n]); });
  showInfo(null); readout();
})();
</script>
