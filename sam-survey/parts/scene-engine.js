/* ch-engine: SAM's fully automatic stage on the desk scene. A toy point grid; each point gives up to three
   candidate masks (TOY.three); background points give the wall or table region and the whole image. Then the
   paper's filters in order (App. B thresholds). Predicted IoU and stability are toy stand-ins computed from
   the drawn shapes; NMS and the cover filter use the real rule on the toy rasters. */
const ENGINE_KIT = (() => {
  const rectPath = (x, y, w, h) => { const p = new Path2D(); p.rect(x, y, w, h); return p; };
  const pseudo = { wall: { id: 'x-wall', name: 'wall', mask: rectPath(0, 0, TOY.WW, .62), kids: [] }, table: { id: 'x-table', name: 'table', mask: rectPath(0, .62, TOY.WW, TOY.WH - .62), kids: [] },
    whole: { id: 'x-whole', name: 'whole image', mask: rectPath(0, 0, TOY.WW, TOY.WH), kids: [] } };
  const GX = 160, GY = 100, stats = new Map();
  function stat(n) { if (stats.has(n.id)) return stats.get(n.id);
    const m = TOY.raster(n, GX, GY); let A = 0, B = 0;
    for (let k = 0; k < m.length; k++) if (m[k]) { A++; const i = k % GX, j = (k / GX) | 0; if ((i && !m[k - 1]) || (i < GX - 1 && !m[k + 1]) || (j && !m[k - GX]) || (j < GY - 1 && !m[k + GX])) B++; }
    // toy stability: moving the logit threshold from -1 to +1 shifts the edge by about d cells, about half a
    // pixel at SAM's 1024 input (our assumption); the IoU of the shrunk and grown masks is then about 1 - 2 d B / A.
    const d = .08, s = { A, B, cover: A / (GX * GY), stab: PE.clamp(1 - 2 * d * B / Math.max(A, 1), 0, 1), box: TOY.bbox(n) };
    stats.set(n.id, s); return s; }
  function candidates(n, seed) {
    const r = PE.rng(seed), out = [];
    for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) {
      const x = (i + .5) / n * TOY.WW, y = (j + .5) / n * TOY.WH, pt = [x, y];
      let nodes = TOY.three(x, y); if (!nodes.length) nodes = [y < .62 ? pseudo.wall : pseudo.table, pseudo.whole];
      nodes.forEach(nd => { const st = stat(nd), thin = 1 - st.stab;
        // toy predicted IoU: high for compact shapes, lower for thin ones, with noise
        const piou = PE.clamp(.96 - 1.2 * thin + (r() - .5) * .16, 0, 1); out.push({ pt, node: nd, piou, stab: st.stab, cover: st.cover, box: st.box }); });
    }
    return out;
  }
  const boxIoU = (a, b) => { const iw = Math.max(0, Math.min(a[2], b[2]) - Math.max(a[0], b[0])), ih = Math.max(0, Math.min(a[3], b[3]) - Math.max(a[1], b[1])), I = iw * ih;
    const U = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - I; return U ? I / U : 0; };
  // stages: 0 all, 1 predicted IoU >= 0.88, 2 stability >= 0.95, 3 cover < 95%, 4 box NMS 0.7, 5 small-part cleanup
  function stages(c) {
    const s = [c];
    s.push(s[0].filter(k => k.piou >= .88));
    s.push(s[1].filter(k => k.stab >= .95));
    s.push(s[2].filter(k => k.cover < .95));
    const sorted = s[3].slice().sort((a, b) => b.piou - a.piou), kept = [];
    sorted.forEach(k => { if (kept.every(q => boxIoU(q.box, k.box) <= .7)) kept.push(k); });
    s.push(kept); s.push(kept.slice());
    return s;
  }
  return { candidates, stages };
})();

SCENES.engine = ctx => {
  const el = mkCanvas(ctx.stage), slider = ctx.q('[data-grid]'), gv = ctx.q('[data-grid-value]');
  const S = { step: 0, n: +(slider ? slider.value : 8), u: 1 };
  let C = [], ST = [], cancel = () => {};
  const NAMES = ['All candidates', 'Predicted IoU', 'Stability', 'Not the whole image', 'Box NMS', 'Clean-up'];
  function compute() { C = ENGINE_KIT.candidates(S.n, 101); ST = ENGINE_KIT.stages(C); }
  compute();
  const cv = PE.canvas(el, S3ASPECT(ctx.stage), draw);
  const offs = [];
  if (slider) offs.push(listen(slider, 'input', () => { S.n = +slider.value; if (gv) gv.textContent = `${S.n} x ${S.n}`; compute(); cv.redraw(); readout(); }));
  if (gv) gv.textContent = `${S.n} x ${S.n}`;
  function draw(c, W, H) {
    c.clearRect(0, 0, W, H);
    const narrow = W < 620, pad = 12, top = 34, bottom = ctx.reserve || 0;
    const imgW = narrow ? W - 2 * pad : W * .64, imgH = narrow ? (H - bottom - top) * .6 : Math.min(H - bottom - top - pad, imgW / TOY.WW);
    const top2 = narrow ? top : Math.max(top, (H - bottom - imgH) / 2), V = TOY.fit(pad, top2, imgW, imgH);
    TOY.drawWorld(c, V, { dim: .3 });
    const cur = ST[S.step], prev = S.step ? ST[S.step - 1] : cur, gone = prev.filter(k => !cur.includes(k));
    const uniq = [...new Set(cur.map(k => k.node))];
    if (S.step >= 4) uniq.forEach(nd => TOY.drawMask(c, V, nd, { alpha: S.u, lift: 0, fill: nd.id.startsWith('x-') ? .06 : .32, dash: nd.id.startsWith('x-') ? [4, 4] : null }));
    else { c.save(); c.translate(V.ox, V.oy); c.scale(V.s, V.s); c.lineWidth = 1.4 / V.s;
      c.strokeStyle = PE.css('--green'); c.globalAlpha = .55; uniq.forEach(nd => c.stroke(nd.mask));
      c.strokeStyle = PE.css('--red'); c.globalAlpha = .8 * (1 - S.u); [...new Set(gone.map(k => k.node))].filter(nd => !uniq.includes(nd)).forEach(nd => c.stroke(nd.mask)); c.restore(); }
    const pts = new Map(); C.forEach(k => pts.set(k.pt.join(), k.pt));
    pts.forEach(p => PE.dot(c, V.X(p[0]), V.Y(p[1]), S.n > 12 ? 2 : 3, PE.css('--yellow'), .85));
    const b = narrow ? { x: pad, y: top2 + imgH + 18, w: W - 2 * pad, h: H - bottom - (top2 + imgH + 18) - 4 } : { x: pad + imgW + 18, y: top2, w: W - imgW - 2 * pad - 18, h: imgH };
    funnel(c, b, narrow);
  }
  function funnel(c, b, narrow) {
    const F = PE.css('--fg'), M = PE.css('--muted'), G = PE.css('--green'); const max = ST[0].length;
    PE.label(c, 'Masks left after each step', b.x, b.y + 12, F, 11, 'left', 700);
    const rh = Math.min(narrow ? 18 : 30, (b.h - 22) / 6); if (rh < 9) return;
    ST.forEach((s, i) => { const y = b.y + 22 + i * rh, on = i <= S.step, w = (b.w - (narrow ? 150 : 40)) * s.length / max;
      c.globalAlpha = on ? 1 : .3; PE.label(c, NAMES[i], b.x, y + rh * .55, i === S.step ? F : M, narrow ? 9 : 10, 'left', i === S.step ? 700 : 400);
      const bx = narrow ? b.x + 118 : b.x, by = narrow ? y + rh * .2 : y + rh * .62; c.fillStyle = i >= 4 ? G : PE.css('--blue'); c.fillRect(bx, by, Math.max(1, w), narrow ? rh * .5 : rh * .26);
      PE.label(c, String(s.length), bx + Math.max(1, w) + 4, by + (narrow ? rh * .45 : rh * .26), M, 10, 'left', 600); c.globalAlpha = 1; });
  }
  const TEXT = [
    n => `${n} x ${n} toy grid of points; each point gives up to three masks: ${ST[0].length} candidates, most of them duplicates. SAM used a 32x32 grid.`,
    () => `Keep masks whose predicted IoU is at least 88.0 (as a fraction, 0.88): ${ST[1].length} left. Toy scores: thin shapes score lower.`,
    () => `Keep stable masks: thresholding the logits at -1 and at +1 must give masks with IoU of at least 95.0: ${ST[2].length} left. Thin and small parts fail first.`,
    () => `Drop masks that cover 95% or more of the image: ${ST[3].length} left.`,
    () => `Remove duplicates with box NMS at 0.7: ${ST[4].length} distinct masks remain.`,
    () => `Clean up: fill holes and remove stray parts under 100 pixels. Our drawn masks have none, so the count stays ${ST[5].length}.`];
  function readout() { ctx.hud(badge('toy', TEXT[S.step](S.n))); }
  readout();
  return { flat: true, step(n) { S.step = n; readout(); cancel(); cancel = PE.animate(600, u => { S.u = u; cv.redraw(); }); },
    resize: () => cv.resize(), destroy: () => { cancel(); offs.forEach(f => f()); cv.destroy(); } };
};

/* ch-engine: the three stages of the data engine as a sourced chart. */
SCENES.stages = ctx => {
  const el = mkCanvas(ctx.stage);
  const STG = [
    { name: 'Assisted-manual', images: '120k', masks: '4.3M', mpi: [20, 44], t: '34 s -> 14 s per mask', who: 'annotators click; 6 retrains' },
    { name: 'Semi-automatic', images: '180k', masks: '5.9M', mpi: [44, 72], t: '34 s per mask', who: 'a detector pre-fills; 5 retrains' },
    { name: 'Fully automatic', images: '11M', masks: '1.1B', mpi: [100, 100], t: 'no annotator; bar = SA-1B average', who: '32x32 grid, filters, NMS' }];
  let u = 1, cancel = () => {};
  const cv = PE.canvas(el, S3ASPECT(ctx.stage), draw);
  function draw(c, W, H) {
    c.clearRect(0, 0, W, H);
    const narrow = W < 620, pad = 14, top = 40, bottom = ctx.reserve || 0, F = PE.css('--fg'), M = PE.css('--muted'), B = PE.css('--blue'), G = PE.css('--green');
    const b = { x: pad + 36, y: top + 10, w: W - 2 * pad - 50, h: (H - bottom - top) * (narrow ? .4 : .62) };
    const ax = PE.axes(c, b, [0, 3], [0, 110], { yt: [0, 20, 44, 72, 100], yl: 'masks per image', ylo: 30 });
    STG.forEach((s, i) => { const x0 = ax.X(i + .1), x1 = ax.X(i + .9), y0 = ax.Y(s.mpi[0]), y1 = ax.Y(PE.lerp(s.mpi[0], s.mpi[1], u));
      c.fillStyle = i < 2 ? B : G; c.globalAlpha = .55; c.fillRect(x0, s.mpi[0] === s.mpi[1] ? ax.Y(s.mpi[1] * u) : y1, x1 - x0, ax.Y(0) - (s.mpi[0] === s.mpi[1] ? ax.Y(s.mpi[1] * u) : y1)); c.globalAlpha = 1;
      if (s.mpi[0] !== s.mpi[1]) PE.polyline(c, [[x0, y0], [x1, y1]], F, 2);
      PE.label(c, s.name, (x0 + x1) / 2, ax.Y(0) + 16, F, narrow ? 10 : 12, 'center', 700); });
    const ty = b.y + b.h + (narrow ? 30 : 34);
    STG.forEach((s, i) => { const x = narrow ? pad : pad + i * (W - 2 * pad) / 3, y = narrow ? ty + i * 34 : ty;
      PE.label(c, `${s.masks} masks on ${s.images} images`, x, y, i < 2 ? B : G, narrow ? 11 : 12, 'left', 700);
      PE.label(c, s.t, x, y + 16, M, narrow ? 10 : 11, 'left', 400);
      if (!narrow) PE.label(c, s.who, x, y + 31, M, 11, 'left', 400); });
  }
  ctx.hud(badge('sourced', 'Masks per image rose from 20 to 44 in stage 1 and to 72 in stage 2 (manual plus automatic). The third bar is a different count: SA-1B, automatic masks only, averages 100 per image. The paper gives no rise within stage 3. SAM paper, Sec. 4 and 5.'));
  function play() { cancel(); cancel = PE.animate(900, t => { u = t; cv.redraw(); }); }
  play();
  return { flat: true, replay: play, resize: () => cv.resize(), destroy: () => { cancel(); cv.destroy(); } };
};
