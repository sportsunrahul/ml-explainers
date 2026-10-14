/* ch-clicks: interactive segmentation as a loop. A toy model's mask (green) is compared with the true object
   (dashed outline); error regions are red. The simulated user clicks the centre of the largest error region (the
   benchmark protocol: the point farthest from that region's boundary). Each click fixes the region it lands in,
   and a band of boundary noise shrinks. An IoU-vs-clicks curve builds; NoC@90 = clicks needed to reach 90% IoU. */
SCENES.clicks = ctx => {
  const el = mkCanvas(ctx.stage), GX = 160, GY = 100, N = GX * GY;
  const TARGETS = { mug: ['mug', 'mugset'], plant: ['plant', 'leaves'], lamp: ['lamp', 'shade'], window: ['window', 'pane2'] };
  const S = { target: 'mug', P: null, T: null, clicks: [], ious: [], seed: 3, u: 1 };
  const tBtns = ctx.qa('[data-target]'), nextBtn = ctx.q('[data-next]'), autoBtn = ctx.q('[data-auto]');
  let cancel = () => {}, geo = null;
  const glow = {}; const onLink = e => { const k = e.detail; PE.animate(1100, u => { glow[k] = Math.sin(Math.PI * u); cv.redraw(); }, () => { glow[k] = 0; cv.redraw(); }); };
  ctx.article.addEventListener('pe-link', onLink);
  const cv = PE.canvas(el, S3ASPECT(ctx.stage), draw);
  const pressed = tBtns.find(b => b.getAttribute('aria-pressed') === 'true'); if (pressed) S.target = pressed.dataset.target;
  function reset() { const [t, start] = TARGETS[S.target]; S.T = TOY.raster(TOY.byId[t], GX, GY); S.P = Uint8Array.from(TOY.raster(TOY.byId[start], GX, GY)); S.clicks = []; S.noise = .5; S.ious = []; S.r = PE.rng(S.seed); applyNoise(); S.ious.push(iou()); }
  function boundaryBand() { const band = new Uint8Array(N); for (let j = 0; j < GY; j++) for (let i = 0; i < GX; i++) { const k = j * GX + i, v = S.T[k];
    for (let dj = -2; dj <= 2 && !band[k]; dj++) for (let di = -2; di <= 2; di++) { const ii = i + di, jj = j + dj; if (ii >= 0 && jj >= 0 && ii < GX && jj < GY && S.T[jj * GX + ii] !== v) { band[k] = 1; break; } } } return band; }
  let BAND = null;
  function applyNoise() { BAND = BAND && BAND.t === S.target ? BAND : Object.assign(boundaryBand(), { t: S.target });
    for (let k = 0; k < N; k++) if (BAND[k]) S.P[k] = S.r() < S.noise ? 1 - S.T[k] : S.T[k]; }
  function iou() { let I = 0, U = 0; for (let k = 0; k < N; k++) { I += S.P[k] & S.T[k]; U += S.P[k] | S.T[k]; } return U ? I / U : 1; }
  function components() { const lab = new Int32Array(N).fill(-1), comps = []; for (let k = 0; k < N; k++) { if (lab[k] >= 0 || S.P[k] === S.T[k]) continue;
      const q = [k], c = { cells: [], fn: S.T[k] === 1 }; lab[k] = comps.length;
      while (q.length) { const x = q.pop(); c.cells.push(x); const i = x % GX, j = (x / GX) | 0;
        [[1, 0], [-1, 0], [0, 1], [0, -1]].forEach(([di, dj]) => { const ii = i + di, jj = j + dj, y = jj * GX + ii; if (ii >= 0 && jj >= 0 && ii < GX && jj < GY && lab[y] < 0 && S.P[y] !== S.T[y] && S.T[y] === S.T[k]) { lab[y] = comps.length; q.push(y); } }); }
      comps.push(c); } return { lab, comps }; }
  function centreOf(c, lab, id) { // farthest-from-boundary cell of the component (BFS distance transform)
    const dist = new Map(), q = []; c.cells.forEach(x => { const i = x % GX, j = (x / GX) | 0;
      if ([[1, 0], [-1, 0], [0, 1], [0, -1]].some(([di, dj]) => { const ii = i + di, jj = j + dj; return ii < 0 || jj < 0 || ii >= GX || jj >= GY || lab[jj * GX + ii] !== id; })) { dist.set(x, 0); q.push(x); } });
    let best = q[0], h = 0; while (h < q.length) { const x = q[h++], d = dist.get(x), i = x % GX, j = (x / GX) | 0; if (d > dist.get(best)) best = x;
      [[1, 0], [-1, 0], [0, 1], [0, -1]].forEach(([di, dj]) => { const ii = i + di, jj = j + dj, y = jj * GX + ii; if (ii >= 0 && jj >= 0 && ii < GX && jj < GY && lab[y] === id && !dist.has(y)) { dist.set(y, d + 1); q.push(y); } }); }
    return best; }
  function clickAt(k) { const { lab, comps } = components(), id = lab[k];
    S.clicks.push({ k, pos: S.T[k] === 1 });
    if (id >= 0) comps[id].cells.forEach(x => { S.P[x] = S.T[x]; });
    S.noise = Math.max(.12, S.noise * .7); applyNoise(); S.ious.push(iou()); }
  function next() { if (S.clicks.length >= 20) return; const { lab, comps } = components(); if (!comps.length) { S.clicks.push({ k: -1 }); S.ious.push(iou()); return; }
    let big = 0; comps.forEach((c, i) => { if (c.cells.length > comps[big].cells.length) big = i; }); clickAt(centreOf(comps[big], lab, big)); }
  function draw(c, W, H) {
    c.clearRect(0, 0, W, H); const narrow = W < 620, pad = 12, top = 34, bottom = ctx.reserve || 0;
    const imgW = narrow ? W - 2 * pad : W * .62, imgH = narrow ? (H - bottom) * .58 : Math.min(H - bottom - top - pad, imgW / TOY.WW);
    const V = TOY.fit(pad, top, imgW, imgH); geo = V;
    TOY.drawWorld(c, V, { dim: .2 });
    const cw = V.w / GX, ch = V.h / GY, G = PE.css('--green'), R = PE.css('--red');
    c.save(); c.globalAlpha = .5 * S.u; c.fillStyle = G; for (let k = 0; k < N; k++) if (S.P[k]) c.fillRect(V.ox + (k % GX) * cw, V.oy + ((k / GX) | 0) * ch, cw + .4, ch + .4);
    c.globalAlpha = .55 * S.u; c.fillStyle = R; for (let k = 0; k < N; k++) if (S.P[k] !== S.T[k]) c.fillRect(V.ox + (k % GX) * cw, V.oy + ((k / GX) | 0) * ch, cw + .4, ch + .4); c.restore();
    const tn = TOY.byId[TARGETS[S.target][0]]; c.save(); c.translate(V.ox, V.oy); c.scale(V.s, V.s); c.strokeStyle = PE.css('--fg'); c.lineWidth = 1.5 / V.s; c.setLineDash([4 / V.s, 3 / V.s]); c.stroke(tn.mask); c.restore();
    S.clicks.forEach(p => { if (p.k < 0) return; TOY.point(c, V.ox + (p.k % GX + .5) * cw, V.oy + (((p.k / GX) | 0) + .5) * ch, p.pos ? 'pos' : 'neg', 1, 5); });
    // IoU curve
    const b = narrow ? { x: pad + 30, y: top + imgH + 22, w: W - 2 * pad - 40, h: H - bottom - top - imgH - 52 } : { x: imgW + 3 * pad + 30, y: top + 10, w: W - imgW - 5 * pad - 30, h: imgH - 40 };
    const ax = PE.axes(c, b, [0, 20], [0, 1], { xt: [0, 5, 10, 15, 20], yt: narrow ? [0, .5, .9] : [0, .5, .9, 1], yf: v => Math.round(v * 100) + '%', xl: 'clicks', ylo: 30 });
    c.save(); c.strokeStyle = PE.css('--yellow'); c.lineWidth = 1 + 3 * (glow.threshold || 0); c.setLineDash([4, 3]); c.beginPath(); c.moveTo(b.x, ax.Y(.9)); c.lineTo(b.x + b.w, ax.Y(.9)); c.stroke(); c.restore();
    PE.polyline(c, S.ious.map((v, i) => [ax.X(i), ax.Y(v)]), G, 2.5 + 3 * (glow.curve || 0));
    S.ious.forEach((v, i) => PE.dot(c, ax.X(i), ax.Y(v), 3, G));
    const noc = S.ious.findIndex((v, i) => i > 0 && v >= .9);
    if (noc > 0) { PE.ring(c, ax.X(noc), ax.Y(S.ious[noc]), 7 + 5 * (glow.noc || 0), PE.css('--yellow'), 2); }
    PE.label(c, 'IoU with the true mask', b.x, b.y - 8, PE.css('--muted'), 11, 'left', 400);
  }
  function readout() { const n = S.clicks.length, v = S.ious[S.ious.length - 1], noc = S.ious.findIndex((x, i) => i > 0 && x >= .9);
    ctx.hud(badge('toy', `${n} click${n === 1 ? '' : 's'}: IoU ${(v * 100).toFixed(1)}%.` + (noc > 0 ? ` It first reached 90% after ${noc} clicks (NoC@90 = ${noc}).` : ' The simulated user clicks the centre of the largest red region next.'))); }
  function redraw() { cv.redraw(); readout(); }
  const offs = [
    ...tBtns.map(b => listen(b, 'click', () => { S.target = b.dataset.target; press(tBtns, x => x === b); reset(); redraw(); })),
    listen(nextBtn, 'click', () => { cancel(); next(); redraw(); }),
    listen(autoBtn, 'click', () => { cancel(); reset(); redraw();
      if (PE.instant()) { for (let k = 0; k < 12; k++) next(); redraw(); return; }
      let n = 0; const id = setInterval(() => { next(); redraw(); if (++n >= 12) clearInterval(id); }, 350); cancel = () => clearInterval(id); }),
    listen(el, 'click', e => { if (!geo) return; const r = el.getBoundingClientRect(), x = (e.clientX - r.left - geo.ox) / geo.w, y = (e.clientY - r.top - geo.oy) / geo.h;
      if (x < 0 || y < 0 || x >= 1 || y >= 1 || S.clicks.length >= 20) return; cancel(); clickAt(Math.floor(y * GY) * GX + Math.floor(x * GX)); redraw(); })];
  press(tBtns, b => b.dataset.target === S.target);
  reset(); readout();
  return { flat: true, resize: () => cv.resize(), destroy: () => { cancel(); offs.forEach(f => f()); ctx.article.removeEventListener('pe-link', onLink); cv.destroy(); } };
};
