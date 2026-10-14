<script>
/* Chapter 13 figure modules, shared with the atlas. Data from WHEN_DATA (generated from notes/claims).
   SceneRename: GLIP Table 10 and FSOD-VLM Sec. 4.3, AP with the original name vs a plain description.
   SceneTiny:   computed; how many model-input pixels a defect keeps after each model's resize.
   SceneFppc:   D3 Supp. Table 7; Grounding DINO false alarms on images without the target vs mAP, by threshold. */
const SceneRename = (el, o = {}) => {
  const D = WHEN_DATA.rename, glow = o.glow || (() => 0), bottom = o.bottom || 0;
  const P = { Aquarium: ['penguin', 'stingray which is flat and round'], Rabbits: ['Cottontail-Rabbits', 'rabbit'], EgoHands: ['hand', 'hand of a person'],
    Mushrooms: ['Cow. Chanterelle', 'flat mushroom. yellow mushroom'], Packages: ['package', 'there is a package on the porch'],
    Pothole: ['pothole', 'there are some holes on the road'], 'pushable-pullable': ['pushable-pullable', 'GPT-4o description'] };
  let S = { step: 'names', u: 1 }, stop = () => {};
  function draw(ctx, w, h) {
    const k = DK.C(); ctx.clearRect(0, 0, w, h); if (w < 80) return;
    const hh = h - bottom, narrow = w < 620, left = narrow ? 8 : 150, b = { x: left, y: 16, w: w - left - (narrow ? 12 : 30), h: hh - 52 };
    const { X } = PE.axes(ctx, b, [0, 80], [0, 1], { xt: [0, 20, 40, 60, 80], xl: 'zero-shot AP' });
    const n = D.length, rowH = b.h / n, gl = glow('describe');
    D.forEach((r, i) => {
      const y = b.y + rowH * (i + .5) + (narrow ? 6 : 0), x0 = X(r.orig), x1 = X(PE.lerp(r.orig, r.new, S.step === 'described' ? S.u : 0));
      ctx.save(); ctx.font = `600 ${narrow ? 10.5 : 12}px system-ui, sans-serif`; ctx.fillStyle = k.fg;
      if (narrow) { ctx.textAlign = 'left'; ctx.fillText(r.set, b.x, y - rowH * .32); } else { ctx.textAlign = 'right'; ctx.fillText(r.set, b.x - 10, y + 4); }
      ctx.restore();
      if (S.step === 'described' && Math.abs(r.new - r.orig) > .2) PE.arrow(ctx, x0, y, x1, y, r.new > r.orig ? k.G : k.R, 2 + 2 * gl, 8);
      ctx.save(); ctx.beginPath(); ctx.arc(x0, y, 5, 0, 7); ctx.strokeStyle = k.Y; ctx.lineWidth = 1.6; ctx.fillStyle = k.bg; ctx.fill(); ctx.stroke(); ctx.restore();
      if (S.step === 'described') PE.dot(ctx, x1, y, 5.5, k.Y);
      ctx.save(); ctx.font = `${narrow ? 10 : 11.5}px system-ui, sans-serif`; ctx.fillStyle = k.mu;
      const lab = S.step === 'described' ? P[r.set][1] : P[r.set][0], right = b.x + b.w - (Math.max(x0, x1) + 10), left = Math.min(x0, x1) - 10 - b.x;
      const fit = (t, room) => ctx.measureText(t).width <= room ? t : t.slice(0, Math.max(3, Math.floor(t.length * room / ctx.measureText(t).width) - 1)) + '...';
      if (right >= left && right > 60) ctx.fillText(fit(lab, right), Math.max(x0, x1) + 10, y + 4);
      else if (left > 60) { ctx.textAlign = 'right'; ctx.fillText(fit(lab, left), Math.min(x0, x1) - 10, y + 4); }
      ctx.restore();
    });
  }
  const cv = PE.canvas(el, o.aspect || (w => (w < 620 ? 1.15 : .46)), draw);
  const say = () => o.onSay && o.onSay(S.step === 'names'
    ? 'Zero-shot AP with the dataset\'s own class names (hollow dots).'
    : 'The same models with the names rewritten as plain descriptions. Mushrooms jumps the most; Pothole does not move.');
  function set(step) { stop(); const same = step === S.step; S.step = step; S.u = same ? 1 : 0; stop = PE.animate(800, u => { S.u = same ? 1 : u; cv.redraw(); }); say(); }
  function reset() { stop(); S = { step: 'names', u: 1 }; cv.redraw(); say(); }
  say();
  return { set, reset, redraw: cv.redraw, destroy: () => { stop(); cv.destroy && cv.destroy(); } };
};

const SceneTiny = (el, o = {}) => {
  // Input sizes: OWL-ViT B/32 768 and OWLv2 L/14 1008 (papers), GLIP 800 short / 1333 long and P3 stride 8
  // (GLIP official config), YOLO-World 640 and P3 stride 8 (official code; YOLOv8 convention).
  // Resize rule per model is our reading of each preprocessing; the camera image is assumed 4:3.
  const M = { owlvit: { name: 'OWL-ViT B/32', side: 768, cell: 32, cellName: 'patch' }, owlv2: { name: 'OWLv2 L/14', side: 1008, cell: 14, cellName: 'patch' },
    glip: { name: 'GLIP-T', short: 800, long: 1333, cell: 8, cellName: 'finest stride' }, yolow: { name: 'YOLO-World', side: 640, cell: 8, cellName: 'finest stride' } };
  const bottom = o.bottom || 0;
  let S = { m: 'owlvit', L: 4096, d: 20, u: 1, from: null }, stop = () => {};
  const scale = (m, L) => { const q = M[m], Sh = .75 * L; return q.side ? q.side / L : Math.min(q.short / Sh, q.long / L); };
  const calc = () => { const s = scale(S.m, S.L), px = S.d * s; return { s, px, frac: px / M[S.m].cell }; };
  function draw(ctx, w, h) {
    const k = DK.C(); ctx.clearRect(0, 0, w, h); if (w < 80) return;
    const hh = h - bottom, narrow = w < 620, c = calc(), q = M[S.m];
    // left: the camera image with the defect drawn to scale; right: a zoom on the model input's cell grid
    const A = narrow ? { x: 12, y: 30, w: w - 24, h: (hh - 60) * .42 } : { x: 16, y: 34, w: w * .44, h: hh - 70 };
    const Z = narrow ? { x: 12, y: A.y + A.h + 40, w: w - 24, h: (hh - 60) * .5 } : { x: w * .52, y: 34, w: w * .46, h: hh - 70 };
    ctx.save(); ctx.font = `600 ${narrow ? 11 : 12.5}px system-ui, sans-serif`; ctx.fillStyle = k.fg;
    ctx.fillText(`Your image, ${S.L} px wide`, A.x, A.y - 10); ctx.fillText(`${q.name} input, zoomed on the defect`, Z.x, Z.y - 10); ctx.restore();
    const iw = Math.min(A.w, A.h / .75), ih = iw * .75;
    ctx.strokeStyle = k.rule; ctx.lineWidth = 1.2; ctx.strokeRect(A.x, A.y, iw, ih);
    const dpx = Math.max(1.5, S.d / S.L * iw);
    ctx.fillStyle = k.R; ctx.beginPath(); ctx.arc(A.x + iw * .62, A.y + ih * .45, dpx / 2, 0, 7); ctx.fill();
    DK.brackets(ctx, A.x + iw * .62 - Math.max(8, dpx), A.y + ih * .45 - Math.max(8, dpx), Math.max(16, 2 * dpx), Math.max(16, 2 * dpx), k.B, { lw: 1.6 });
    // zoom: show a 4 x 4 block of cells; one cell = Z.w / 4 screen pixels
    const side = Math.min(Z.w, Z.h), cellPx = side / 4;
    ctx.strokeStyle = k.rule; ctx.lineWidth = 1;
    for (let i = 0; i <= 4; i++) { ctx.beginPath(); ctx.moveTo(Z.x + i * cellPx, Z.y); ctx.lineTo(Z.x + i * cellPx, Z.y + side); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(Z.x, Z.y + i * cellPx); ctx.lineTo(Z.x + side, Z.y + i * cellPx); ctx.stroke(); }
    const r = Math.max(1, c.frac * cellPx / 2);
    ctx.fillStyle = k.R; ctx.beginPath(); ctx.arc(Z.x + 1.5 * cellPx + cellPx * .3, Z.y + 1.5 * cellPx + cellPx * .2, Math.min(r, side / 2), 0, 7); ctx.fill();
    ctx.save(); ctx.font = `${narrow ? 10.5 : 11.5}px system-ui, sans-serif`; ctx.fillStyle = k.mu;
    ctx.fillText(`one square = one ${q.cellName} (${q.cell} input px)`, Z.x, Z.y + side + 16); ctx.restore();
  }
  const cv = PE.canvas(el, o.aspect || (w => (w < 620 ? 1.25 : .46)), draw);
  function describe() { const c = calc(), q = M[S.m];
    return `${q.name}: a ${S.d} px defect in a ${S.L} px wide 4:3 image becomes ${c.px.toFixed(1)} px at the model input, ${c.frac < 1 ? 'about ' + (c.frac * 100).toFixed(0) + '% of' : (c.frac).toFixed(1) + ' times'} one ${q.cellName} of ${q.cell} px.`; }
  const say = () => o.onSay && o.onSay(describe());
  function set(m) { if (!M[m]) return; S.m = m; cv.redraw(); say(); }
  function sizes(L, d) { S.L = L; S.d = d; cv.redraw(); say(); }
  function reset() { S = { m: 'owlvit', L: 4096, d: 20, u: 1, from: null }; cv.redraw(); say(); }
  say();
  return { set, sizes, reset, redraw: cv.redraw, destroy: () => { stop(); cv.destroy && cv.destroy(); } };
};

const SceneFppc = (el, o = {}) => {
  const D = WHEN_DATA.d3, glow = o.glow || (() => 0), bottom = o.bottom || 0;
  let S = { i: 0, u: 1, from: 0 }, stop = () => {};
  function draw(ctx, w, h) {
    const k = DK.C(); ctx.clearRect(0, 0, w, h); if (w < 80) return;
    const hh = h - bottom, narrow = w < 620, b = { x: narrow ? 40 : 58, y: 20, w: w - (narrow ? 56 : 90), h: hh - 64 };
    const { X, Y } = PE.axes(ctx, b, [0, D.thr.length - 1], [0, 100], { yt: [0, 20, 40, 60, 80, 100], yl: 'percent', ylo: narrow ? 30 : 38 });
    ctx.save(); ctx.font = `${narrow ? 10.5 : 11.5}px system-ui, sans-serif`; ctx.fillStyle = k.mu; ctx.textAlign = 'center';
    D.thr.forEach((t, i) => ctx.fillText(t, X(i), b.y + b.h + 14)); ctx.fillText('score threshold', b.x + b.w / 2, b.y + b.h + 30); ctx.restore();
    const gF = glow('fppc'), gM = glow('map');
    const lf = D.fppc.map((v, i) => [X(i), Y(v)]), lm = D.map.map((v, i) => [X(i), Y(v)]);
    PE.polyline(ctx, lf, k.R, 2.4 + 2 * gF); lf.forEach(([x, y]) => PE.dot(ctx, x, y, 3.5, k.R));
    PE.polyline(ctx, lm, k.B, 2.4 + 2 * gM); lm.forEach(([x, y]) => PE.dot(ctx, x, y, 3.5, k.B));
    ctx.save(); ctx.font = `600 ${narrow ? 10.5 : 12}px system-ui, sans-serif`;
    // labels sit near the left end, where the two lines are far apart
    ctx.fillStyle = k.R; ctx.fillText('false alarms on images without it', X(1) + 8, Y(D.fppc[1]) - 10);
    ctx.fillStyle = k.B; ctx.fillText('mAP', X(1) + 8, Y(D.map[1]) - 10);
    ctx.restore();
    // the chosen threshold
    const xi = X(PE.lerp(S.from, S.i, S.u));
    ctx.save(); ctx.strokeStyle = k.Y; ctx.lineWidth = 1.6; ctx.setLineDash([4, 4]); ctx.beginPath(); ctx.moveTo(xi, b.y); ctx.lineTo(xi, b.y + b.h); ctx.stroke(); ctx.restore();
  }
  const cv = PE.canvas(el, o.aspect || (w => (w < 620 ? .95 : .42)), draw);
  const say = () => o.onSay && o.onSay(`Threshold ${D.thr[S.i]}: Grounding DINO still fires on ${D.fppc[S.i]}% of images that do not contain the described thing, at ${D.map[S.i]} mAP.`);
  function set(i) { i = Math.max(0, Math.min(D.thr.length - 1, i)); stop(); S.from = PE.lerp(S.from, S.i, S.u); S.i = i; S.u = 0;
    stop = PE.animate(600, u => { S.u = u; cv.redraw(); }); say(); }
  function reset() { stop(); S = { i: 0, u: 1, from: 0 }; cv.redraw(); say(); }
  say();
  return { set, reset, redraw: cv.redraw, destroy: () => { stop(); cv.destroy && cv.destroy(); } };
};
</script>
