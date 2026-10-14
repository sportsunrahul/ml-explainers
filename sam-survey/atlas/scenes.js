/* TOY: the drawn desk scene every figure reuses. World units: 1.6 x 1 (x right, y down). Each object is a
   node in a part hierarchy (logo < mug < mug and saucer); a node's mask is the union of its own shape and
   its descendants. Hidden nodes draw but are never offered as a mask. Shared by the long page and the atlas. */
const TOY = (() => {
  const WW = 1.6, WH = 1;
  const poly = pts => { const p = new Path2D(); pts.forEach(([x, y], i) => i ? p.lineTo(x, y) : p.moveTo(x, y)); p.closePath(); return p; };
  const ell = (cx, cy, rx, ry, rot = 0) => { const p = new Path2D(); p.ellipse(cx, cy, rx, ry, rot, 0, Math.PI * 2); return p; };
  const rrect = (x, y, w, h, r) => { const p = new Path2D(); p.roundRect(x, y, w, h, r); return p; };
  const band = (cx, cy, r1, r2, a0, a1, n = 24) => { const pts = [];
    for (let i = 0; i <= n; i++) { const a = a0 + (a1 - a0) * i / n; pts.push([cx + r2 * Math.cos(a), cy + r2 * Math.sin(a)]); }
    for (let i = n; i >= 0; i--) { const a = a0 + (a1 - a0) * i / n; pts.push([cx + r1 * Math.cos(a), cy + r1 * Math.sin(a)]); }
    return poly(pts); };
  // tone: 0..1 lightness between bg and fg; shapes are plain fills so the masks carry the color.
  const N = (id, name, shape, tone, kids = [], o = {}) => ({ id, name, shape, tone, kids, ...o });
  const root = N('scene', 'scene', null, 0, [
    N('window', 'window', rrect(.55, .1, .4, .32, .012), .2, [
      N('pane1', 'window pane', rrect(.57, .12, .175, .13, .006), .34), N('pane2', 'window pane', rrect(.755, .12, .175, .13, .006), .34),
      N('pane3', 'window pane', rrect(.57, .27, .175, .13, .006), .3), N('pane4', 'window pane', rrect(.755, .27, .175, .13, .006), .3)]),
    N('lamp', 'lamp', null, 0, [
      N('lampbase', 'lamp base', ell(.25, .9, .1, .026), .5),
      N('arm', 'lamp arm', poly([[.24, .9], [.33, .52], [.205, .335], [.218, .325], [.35, .52], [.262, .9]]), .45),
      N('shade', 'lamp shade', poly([[.1, .38], [.15, .25], [.235, .27], [.245, .41]]), .62)]),
    N('book', 'notebook', null, 0, [
      N('cover', 'notebook cover', poly([[.5, .93], [.56, .77], [.96, .79], [.93, .955]]), .4, [], { hidden: true }),
      N('spine', 'notebook spine', poly([[.5, .93], [.56, .77], [.585, .772], [.525, .932]]), .58)]),
    N('pencil', 'pencil', poly([[.6, .9], [.88, .815], [.905, .815], [.885, .827], [.607, .912]]), .7),
    N('mugset', 'mug and saucer', null, 0, [
      N('saucer', 'saucer', ell(1.12, .885, .19, .042), .5),
      N('mug', 'mug', null, 0, [
        N('body', 'mug body', poly([[1.015, .55], [1.225, .55], [1.21, .85], [1.03, .85]]), .66, [], { hidden: true }),
        N('handle', 'mug handle', band(1.222, .7, .045, .075, -Math.PI / 2.2, Math.PI / 2.2), .6),
        N('logo', 'logo on the mug', ell(1.12, .69, .042, .042), .3, [], { short: 'logo' })])]),
    N('plant', 'plant', null, 0, [
      N('pot', 'pot', poly([[1.37, .72], [1.54, .72], [1.515, .9], [1.395, .9]]), .48),
      N('leaves', 'leaves', null, 0, [
        N('leaf1', 'leaf', ell(1.41, .6, .03, .085, -.5), .36), N('leaf2', 'leaf', ell(1.455, .55, .03, .1, -.05), .4),
        N('leaf3', 'leaf', ell(1.5, .6, .03, .085, .5), .36), N('leaf4', 'leaf', ell(1.39, .67, .025, .06, -1.1), .33),
        N('leaf5', 'leaf', ell(1.525, .67, .025, .06, 1.1), .33)])])]);
  root.kids.find(n => n.id === 'mugset').short = 'mug + saucer';
  const all = [], byId = {};
  (function walk(n, parent, depth) { n.parent = parent; n.depth = depth; all.push(n); byId[n.id] = n; n.kids.forEach(k => walk(k, n, depth + 1)); })(root, null, 0);
  all.forEach(n => { const p = new Path2D(); (function add(m) { if (m.shape) p.addPath(m.shape); m.kids.forEach(add); })(n); n.mask = p; });
  const probe = document.createElement('canvas').getContext('2d');
  const inside = (n, x, y) => n.mask && probe.isPointInPath(n.mask, x, y);
  // The chain of maskable nodes under a point, smallest first (excludes the scene itself).
  function chain(x, y) { const hits = all.filter(n => n !== root && !n.hidden && inside(n, x, y) && n.kids.length + (n.shape ? 1 : 0) > 0);
    hits.sort((a, b) => b.depth - a.depth); return hits; }
  // SAM-style three outputs: subpart, part, whole (padded with the last when the hierarchy is shallow).
  function three(x, y) { const c = chain(x, y); return c.length >= 3 ? [c[0], c[c.length >> 1], c[c.length - 1]] : c; }
  // Fit the world into a rect; returns the transform and inverse.
  function fit(x, y, w, h) { const s = Math.min(w / WW, h / WH), ox = x + (w - WW * s) / 2, oy = y + (h - WH * s) / 2;
    return { s, ox, oy, w: WW * s, h: WH * s, X: wx => ox + wx * s, Y: wy => oy + wy * s, ix: px => (px - ox) / s, iy: py => (py - oy) / s }; }
  const col = t => PE.mix(PE.css('--bg'), PE.css('--fg'), t);
  function drawWorld(c, V, o = {}) {
    c.save(); c.translate(V.ox, V.oy); c.scale(V.s, V.s);
    c.fillStyle = col(.07); c.fillRect(0, 0, WW, .62); c.fillStyle = col(.13); c.fillRect(0, .62, WW, WH - .62);
    c.strokeStyle = col(.22); c.lineWidth = 1.2 / V.s; c.beginPath(); c.moveTo(0, .62); c.lineTo(WW, .62); c.stroke();
    all.forEach(n => { if (!n.shape) return; c.fillStyle = col(n.tone * (o.dim != null ? 1 - o.dim * .55 : 1) + (o.dim ? o.dim * .05 : 0)); c.fill(n.shape); });
    c.restore();
  }
  // A mask as a cutout: a bg-colored hole, the piece lifted up-left, a green tint and a crisp outline.
  function drawMask(c, V, n, o = {}) {
    const a = o.alpha == null ? 1 : o.alpha, lift = (o.lift == null ? 3 : o.lift) * a, color = o.color || PE.css('--green');
    c.save(); c.globalAlpha = a;
    c.save(); c.translate(V.ox, V.oy); c.scale(V.s, V.s); c.fillStyle = PE.css('--bg'); c.fill(n.mask); c.restore();
    c.save(); c.translate(-lift, -lift); c.translate(V.ox, V.oy); c.scale(V.s, V.s); c.clip(n.mask); c.scale(1 / V.s, 1 / V.s); c.translate(-V.ox, -V.oy);
    drawWorld(c, V); c.translate(V.ox, V.oy); c.scale(V.s, V.s);
    c.globalAlpha = a * (o.fill == null ? .38 : o.fill); c.fillStyle = color; c.fill(n.mask);
    c.globalAlpha = a; c.strokeStyle = color; c.lineWidth = (o.lw || 2) / V.s; if (o.dash) c.setLineDash(o.dash.map(d => d / V.s)); c.stroke(n.mask);
    c.restore(); c.restore();
  }
  // A soft probability map: average of several masks (what a single-output model learns for an ambiguous click).
  function drawAverage(c, V, nodes, o = {}) {
    const a = o.alpha == null ? 1 : o.alpha;
    c.save(); c.translate(V.ox, V.oy); c.scale(V.s, V.s); c.fillStyle = PE.css('--green');
    nodes.forEach(n => { c.globalAlpha = a * .62 / nodes.length * 1.6; c.fill(n.mask); }); c.restore();
  }
  function point(c, x, y, kind = 'pos', a = 1, r = 6) {
    c.save(); c.globalAlpha = a;
    if (kind === 'pos') { PE.dot(c, x, y, r + 2, PE.css('--bg'), a); PE.dot(c, x, y, r, PE.css('--yellow'), a); }
    else { PE.dot(c, x, y, r + 2, PE.css('--bg'), a); PE.ring(c, x, y, r - 1, PE.css('--brown'), 2.5, a); c.strokeStyle = PE.css('--brown'); c.lineWidth = 2; c.beginPath(); c.moveTo(x - r * .5, y - r * .5); c.lineTo(x + r * .5, y + r * .5); c.moveTo(x + r * .5, y - r * .5); c.lineTo(x - r * .5, y + r * .5); c.stroke(); }
    c.restore();
  }
  function boxPrompt(c, x0, y0, x1, y1, a = 1) { c.save(); c.globalAlpha = a; c.strokeStyle = PE.css('--yellow'); c.lineWidth = 2; c.setLineDash([6, 4]); c.strokeRect(x0, y0, x1 - x0, y1 - y0); c.restore(); }
  // Rasterize a node's mask on a gx x gy grid over the world (for IoU, grids, automatic generation).
  const rcache = new Map();
  function raster(n, gx = 96, gy = 60) { const k = n.id + ':' + gx + 'x' + gy; if (rcache.has(k)) return rcache.get(k);
    const m = new Uint8Array(gx * gy); for (let j = 0; j < gy; j++) for (let i = 0; i < gx; i++) m[j * gx + i] = inside(n, (i + .5) / gx * WW, (j + .5) / gy * WH) ? 1 : 0;
    rcache.set(k, m); return m; }
  function iou(a, b) { let I = 0, U = 0; for (let i = 0; i < a.length; i++) { I += a[i] & b[i]; U += a[i] | b[i]; } return U ? I / U : 0; }
  function bbox(n) { const m = raster(n, 160, 100); let x0 = 1e9, y0 = 1e9, x1 = -1, y1 = -1;
    for (let j = 0; j < 100; j++) for (let i = 0; i < 160; i++) if (m[j * 160 + i]) { x0 = Math.min(x0, i); x1 = Math.max(x1, i); y0 = Math.min(y0, j); y1 = Math.max(y1, j); }
    return [x0 / 160 * WW, y0 / 100 * WH, (x1 + 1) / 160 * WW, (y1 + 1) / 100 * WH]; }
  return { WW, WH, root, all, byId, inside, chain, three, fit, drawWorld, drawMask, drawAverage, point, boxPrompt, raster, iou, bbox, col };
})();
/* Hero / atlas overview: a click on the mug's logo; the image is encoded once; three cutouts lift off with
   their predicted scores (logo, mug, mug and saucer). Toy scores. */
SCENES.overview = ctx => {
  const el = mkCanvas(ctx.stage), click = [1.115, .7], outs = TOY.three(click[0], click[1]), scores = [.79, .93, .88];
  let u = 1, cancel = () => {};
  const sm = (a, b, t) => PE.clamp((t - a) / (b - a), 0, 1);
  const cv = PE.canvas(el, S3ASPECT(ctx.stage), draw);
  function draw(c, W, H) {
    c.clearRect(0, 0, W, H);
    const narrow = W < 620, pad = 12, bottom = ctx.reserve || 0;
    const mainW = narrow ? W - 2 * pad : W * .6, mainH = narrow ? (H - bottom) * .56 : H - bottom - 2 * pad;
    const V = TOY.fit(pad, pad, mainW, mainH);
    const enc = sm(.25, .45, u), drop = sm(.12, .22, u);
    TOY.drawWorld(c, V);
    if (enc > 0) { c.save(); c.globalAlpha = .22 * (enc < 1 ? 1 : .55); c.fillStyle = PE.css('--blue'); c.fillRect(V.ox, V.oy, V.w * enc, V.h); c.restore();
      PE.label(c, 'image encoder: runs once', V.ox + 8, V.oy + 18, PE.css('--blue'), 12, 'left', 600); }
    // cursor glide, then the yellow dot
    const cx = V.X(click[0]), cy = V.Y(click[1]), g = sm(0, .12, u);
    if (u < .22) { const x = PE.lerp(V.X(.9), cx, g), y = PE.lerp(V.Y(.25), cy, g); drawCursor(c, x, y); }
    // three panels
    const panels = outs.map((n, i) => {
      const t = sm(.45 + i * .14, .59 + i * .14, u);
      let px, py, pw, ph;
      if (narrow) { pw = (W - 2 * pad - 16) / 3; ph = (H - bottom) - mainH - 3 * pad - 18; px = pad + i * (pw + 8); py = mainH + 2 * pad + 10; }
      else { const top = 40; px = mainW + 2 * pad + 6; pw = W - px - pad; ph = (H - bottom - top - 3 * pad) / 3; py = top + i * (ph + pad); }
      return { n, t, px, py, pw, ph, i };
    });
    const shown = panels.filter(p => p.t > 0), focus = u >= 1 ? panels[1] : shown[shown.length - 1];
    if (focus) TOY.drawMask(c, V, focus.n, { alpha: focus.t });
    panels.forEach(p => { if (p.t <= 0) return;
      c.save(); c.globalAlpha = p.t; c.strokeStyle = PE.css('--rule'); c.lineWidth = 1; c.beginPath(); c.roundRect(p.px, p.py, p.pw, p.ph, 8); c.stroke();
      const [x0, y0, x1, y1] = TOY.bbox(p.n), m = .06, bw = x1 - x0 + 2 * m, bh = y1 - y0 + 2 * m, lab = 22;
      const s = Math.min((p.pw - 12) / bw, (p.ph - lab - 18) / bh), V2 = { s, ox: p.px + (p.pw - bw * s) / 2 - (x0 - m) * s, oy: p.py + 6 - (y0 - m) * s };
      V2.ox = Math.min(p.px, Math.max(p.px + p.pw - TOY.WW * s, V2.ox));
      V2.X = x => V2.ox + x * s; V2.Y = y => V2.oy + y * s; V2.w = TOY.WW * s; V2.h = TOY.WH * s;
      c.save(); c.beginPath(); c.rect(p.px + 1, p.py + 1, p.pw - 2, p.ph - lab - 8); c.clip(); TOY.drawWorld(c, V2); TOY.drawMask(c, V2, p.n, { lift: 2 }); TOY.point(c, V2.X(click[0]), V2.Y(click[1]), 'pos', 1, 4); c.restore();
      const by = p.py + p.ph - lab + 2, bwid = p.pw - 16;
      c.fillStyle = PE.css('--panel2'); c.fillRect(p.px + 8, by + 10, bwid, 5); c.fillStyle = PE.css('--green'); c.fillRect(p.px + 8, by + 10, bwid * scores[p.i] * p.t, 5);
      PE.label(c, narrow ? (p.n.short || p.n.name) : p.n.name, p.px + 8, by + 4, PE.css('--fg'), narrow ? 10 : 12, 'left', 600);
      c.font = `600 ${narrow ? 10 : 12}px system-ui, sans-serif`; const nm = narrow ? (p.n.short || p.n.name) : p.n.name, fits = c.measureText(nm + '  0.00').width < p.pw - 16;
      PE.label(c, scores[p.i].toFixed(2), p.px + p.pw - 8, fits ? by + 4 : by - 10, PE.css('--green'), narrow ? 10 : 12, 'right', 600);
      c.restore(); });
    if (drop > 0) TOY.point(c, cx, cy, 'pos', drop, 7);
    if (u >= 1) { PE.label(c, 'one click', cx, V.Y(.5), PE.css('--yellow'), 12, 'center', 600);
      if (!narrow) PE.label(c, 'three valid masks, each with a predicted score', panels[0].px, H - bottom - 4, PE.css('--muted'), 12, 'left', 400); }
  }
  function drawCursor(c, x, y) { c.save(); c.translate(x, y); c.fillStyle = PE.css('--fg'); c.strokeStyle = PE.css('--bg'); c.lineWidth = 1.5;
    c.beginPath(); c.moveTo(0, 0); c.lineTo(0, 16); c.lineTo(4.5, 12); c.lineTo(8, 19); c.lineTo(10.5, 18); c.lineTo(7, 11); c.lineTo(12.5, 11); c.closePath(); c.fill(); c.stroke(); c.restore(); }
  function readout() { ctx.hud(badge('toy', u < .25 ? 'You click once on the logo of the mug.' : u < .45 ? 'The heavy image encoder runs once per image.' : u < 1 ? 'The light decoder returns three masks for the one click.' : 'One click, three valid answers: the logo, the mug, and the mug with its saucer. The scores are toy values.')); }
  function play() { cancel(); if (PE.instant()) { u = 1; cv.redraw(); readout(); return; }
    let last = -1; cancel = PE.animate(6000, t => { u = t; cv.redraw(); const k = u < .25 ? 0 : u < .45 ? 1 : u < 1 ? 2 : 3; if (k !== last) { last = k; readout(); } }); }
  play();
  return { flat: true, replay: play, resize: () => cv.resize(), destroy: () => { cancel(); cv.destroy(); } };
};
/* ch-before: the same desk scene as four models would label it. Tabs: FCN (coarse score map, stride buttons),
   U-Net (full-resolution map through skip connections), Mask R-CNN (box, then a small per-box mask),
   Mask2Former (N queries, each a mask or "no object"; stuff and things). Toy throughout. */
SCENES.before = ctx => {
  const el = mkCanvas(ctx.stage);
  const CLASSES = [['cup', 'mugset'], ['plant', 'plant'], ['lamp', 'lamp'], ['book', 'book']];
  const S = { step: 0, stride: 32, u: 1 };
  const GX = 160, GY = 100;
  const rast = CLASSES.map(([, id]) => TOY.raster(TOY.byId[id], GX, GY));
  // class index per raster cell (-1 = background or not in the label list)
  const lab = new Int8Array(GX * GY).fill(-1); rast.forEach((m, k) => m.forEach((v, i) => { if (v) lab[i] = k; }));
  const strideCells = { 32: 16, 16: 8, 8: 4 };
  let cancel = () => {};
  const glow = {}; // equation terms (long page): mask, query, pixel
  const onLink = e => { const k = e.detail; if (S.step !== 3) { const t = ctx.q('[role="tab"][data-step="3"]'); if (t) t.click(); }
    PE.animate(1100, u => { glow[k] = Math.sin(Math.PI * u); cv.redraw(); }, () => { glow[k] = 0; cv.redraw(); }); };
  ctx.article.addEventListener('pe-link', onLink);
  const cv = PE.canvas(el, S3ASPECT(ctx.stage), draw);
  const strideBtns = ctx.qa('[data-stride]');
  const offs = strideBtns.map(b => listen(b, 'click', () => { S.stride = +b.dataset.stride; press(strideBtns, x => x === b); if (S.step !== 0) { const t = ctx.q('[role="tab"][data-step="0"]'); if (t) t.click(); } else animateIn(); }));
  const pressed = strideBtns.find(b => b.getAttribute('aria-pressed') === 'true'); if (pressed) S.stride = +pressed.dataset.stride; // survives a theme remount
  press(strideBtns, b => +b.dataset.stride === S.stride);
  function coarse(cell) { // majority vote per cell
    const nx = Math.ceil(GX / cell), ny = Math.ceil(GY / cell), out = new Int8Array(nx * ny).fill(-1);
    for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) { const cnt = new Array(CLASSES.length + 1).fill(0);
      for (let y = j * cell; y < Math.min(GY, (j + 1) * cell); y++) for (let x = i * cell; x < Math.min(GX, (i + 1) * cell); x++) cnt[lab[y * GX + x] + 1]++;
      let best = 0; cnt.forEach((v, k) => { if (v > cnt[best]) best = k; }); out[j * nx + i] = best - 1; }
    return { nx, ny, out };
  }
  function draw(c, W, H) {
    c.clearRect(0, 0, W, H);
    const narrow = W < 620, pad = 12, top = 34, bottom = ctx.reserve || 0;
    const imgW = narrow ? W - 2 * pad : W * .58, imgH = narrow ? (H - bottom) * .55 : Math.min(H - bottom - top - pad, imgW / TOY.WW);
    const top2 = narrow ? top : Math.max(top, (H - bottom - imgH) / 2);
    const V = TOY.fit(pad, top2, imgW, imgH), G = PE.css('--green'), u = S.u;
    TOY.drawWorld(c, V, { dim: .25 });
    const cellW = V.w / GX, cellH = V.h / GY;
    if (S.step === 0) {
      const { nx, ny, out } = coarse(strideCells[S.stride]), cs = strideCells[S.stride];
      c.save(); c.globalAlpha = .42 * u; c.fillStyle = G;
      for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) if (out[j * nx + i] >= 0) c.fillRect(V.ox + i * cs * cellW, V.oy + j * cs * cellH, Math.min(cs, GX - i * cs) * cellW, Math.min(cs, GY - j * cs) * cellH);
      c.globalAlpha = .5 * u; c.strokeStyle = PE.css('--rule'); c.lineWidth = 1;
      for (let i = 0; i <= nx; i++) { const x = V.ox + Math.min(GX, i * cs) * cellW; c.beginPath(); c.moveTo(x, V.oy); c.lineTo(x, V.oy + V.h); c.stroke(); }
      for (let j = 0; j <= ny; j++) { const y = V.oy + Math.min(GY, j * cs) * cellH; c.beginPath(); c.moveTo(V.ox, y); c.lineTo(V.ox + V.w, y); c.stroke(); }
      c.restore();
      classLabels(c, V, u);
    } else if (S.step === 1) {
      CLASSES.forEach(([, id]) => TOY.drawMask(c, V, TOY.byId[id], { alpha: u, lift: 0, fill: .42 }));
      classLabels(c, V, u);
    } else if (S.step === 2) {
      CLASSES.forEach(([name, id], k) => { const n = TOY.byId[id], [x0, y0, x1, y1] = TOY.bbox(n), t = PE.clamp(u * 1.6 - k * .15, 0, 1);
        const X0 = V.X(x0), Y0 = V.Y(y0), X1 = V.X(x1), Y1 = V.Y(y1);
        c.save(); c.globalAlpha = Math.min(1, t * 2); c.strokeStyle = G; c.lineWidth = 1.5; c.setLineDash([5, 3]); c.strokeRect(X0, Y0, X1 - X0, Y1 - Y0); c.restore();
        const m = 14, mt = PE.clamp(t * 2 - 1, 0, 1); // a small fixed-size mask grid per box (Mask R-CNN predicts m x m; drawn coarse here)
        c.save(); c.globalAlpha = .45 * mt; c.fillStyle = G;
        for (let j = 0; j < m; j++) for (let i = 0; i < m; i++) { const wx = x0 + (i + .5) / m * (x1 - x0), wy = y0 + (j + .5) / m * (y1 - y0); if (TOY.inside(n, wx, wy)) c.fillRect(X0 + i * (X1 - X0) / m, Y0 + j * (Y1 - Y0) / m, (X1 - X0) / m + .5, (Y1 - Y0) / m + .5); }
        c.restore();
        if (t > 0) { c.save(); c.globalAlpha = Math.min(1, t * 2); c.font = '600 11px system-ui, sans-serif'; c.fillStyle = PE.css('--panel'); const tw = c.measureText(name).width; c.fillRect(X0, Y0 - 16, tw + 8, 16); PE.label(c, name, X0 + 4, Y0 - 4, G, 11); c.restore(); } });
    } else {
      // panoptic: stuff (wall, table) then things; every pixel gets exactly one label
      c.save(); c.globalAlpha = .16 * u; c.fillStyle = G; c.fillRect(V.ox, V.oy, V.w, V.Y(.62) - V.oy); c.globalAlpha = .1 * u; c.fillRect(V.ox, V.Y(.62), V.w, V.oy + V.h - V.Y(.62));
      c.globalAlpha = .7 * u; c.strokeStyle = G; c.setLineDash([3, 4]); c.lineWidth = 1.5; c.strokeRect(V.ox + 2, V.oy + 2, V.w - 4, V.Y(.62) - V.oy - 3); c.strokeRect(V.ox + 2, V.Y(.62) + 1, V.w - 4, V.oy + V.h - V.Y(.62) - 3); c.restore();
      CLASSES.forEach(([, id], k) => TOY.drawMask(c, V, TOY.byId[id], { alpha: PE.clamp(u * 1.5 - k * .12, 0, 1), lift: 0, fill: .42, lw: 2 + 3 * (glow.mask || 0) }));
      PE.label(c, 'wall (stuff)', V.ox + 8, V.oy + 16, G, 11); PE.label(c, 'table (stuff)', V.ox + 8, V.oy + V.h - 8, G, 11);
      classLabels(c, V, u);
    }
    diagram(c, narrow ? { x: pad, y: top + imgH + 16, w: W - 2 * pad, h: H - bottom - top - imgH - 22 } : { x: pad + imgW + 18, y: top2, w: W - imgW - 2 * pad - 18, h: imgH }, narrow);
  }
  function classLabels(c, V, u) {
    const spots = { cup: [1.12, .52], plant: [1.455, .43], lamp: [.2, .2], book: [.73, .985] };
    c.save(); c.globalAlpha = u; Object.entries(spots).forEach(([k, [x, y]]) => { PE.label(c, k, V.X(x), V.Y(y), PE.css('--green'), 12, 'center', 700); }); c.restore();
    c.save(); c.globalAlpha = .9 * u; PE.label(c, 'window: not a class', V.X(.75), V.Y(.07), PE.css('--brown'), 11, 'center'); PE.label(c, 'pencil: not a class', V.X(.86), V.Y(.77), PE.css('--brown'), 11, 'center'); c.restore();
  }
  function diagram(c, b, narrow) {
    const B = PE.css('--blue'), G = PE.css('--green'), M = PE.css('--muted'), F = PE.css('--fg');
    c.save(); c.strokeStyle = PE.css('--rule'); c.lineWidth = 1; c.beginPath(); c.roundRect(b.x, b.y, b.w, b.h, 10); c.stroke(); c.restore();
    const x = b.x + 12, y = b.y + 12, w = b.w - 24, h = b.h - 24;
    if (S.step === 0 || S.step === 1) {
      const lv = 4, enc = [], dec = [];
      for (let i = 0; i < lv; i++) { const bh = h * (1 - i * .2) * .7, bw = Math.max(10, w * .07); enc.push({ x: x + i * w * .11, y: y + (h - bh) / 2 - (narrow ? 0 : 0), w: bw, h: bh }); }
      const bot = enc[lv - 1];
      for (let i = 0; i < lv; i++) { const e = enc[lv - 1 - i]; dec.push({ x: x + w * .45 + i * w * .11, y: e.y, w: e.w, h: e.h }); }
      enc.forEach((r, i) => { c.fillStyle = B; c.globalAlpha = .55; c.fillRect(r.x, r.y, r.w, r.h); c.globalAlpha = 1; if (i) PE.arrow(c, enc[i - 1].x + enc[i - 1].w + 2, y + h / 2, r.x - 2, y + h / 2, M, 1.2, 6); });
      PE.label(c, 'downsample', enc[0].x, b.y + b.h - 8, M, 11);
      if (S.step === 1) {
        dec.forEach((r, i) => { c.fillStyle = G; c.globalAlpha = .55; c.fillRect(r.x, r.y, r.w, r.h); c.globalAlpha = 1; if (i) PE.arrow(c, dec[i - 1].x + dec[i - 1].w + 2, y + h / 2, r.x - 2, y + h / 2, M, 1.2, 6); });
        for (let i = 0; i < lv - 1; i++) { const e = enc[i], d = dec[lv - 1 - i]; c.save(); c.strokeStyle = B; c.setLineDash([4, 3]); c.lineWidth = 1.3; c.beginPath(); c.moveTo(e.x + e.w, e.y + 6 + i * 3); c.lineTo(d.x, d.y + 6 + i * 3); c.stroke(); c.restore(); }
        PE.label(c, 'skip connections carry detail across', x + w * .02, y + 10, B, 11);
        PE.label(c, 'upsample', dec[0].x, b.y + b.h - 8, M, 11);
      } else {
        const out = { x: x + w * .58, y: bot.y, w: w * .3, h: bot.h };
        PE.arrow(c, bot.x + bot.w + 4, y + h / 2, out.x - 4, y + h / 2, M, 1.2, 6);
        c.fillStyle = G; c.globalAlpha = .55; c.fillRect(out.x, out.y, out.w, out.h); c.globalAlpha = 1;
        PE.label(c, 'class scores per cell', out.x, out.y - 6, G, 11);
        PE.label(c, `stride ${S.stride}: one cell = ${S.stride} x ${S.stride} pixels`, x, y + 10, F, 11);
        PE.label(c, 'upsample to image size', out.x, b.y + b.h - 8, M, 11);
      }
    } else if (S.step === 2) {
      const cols = ['image', 'backbone', 'boxes', 'features\nper box', 'class, box,\nmask m x m'];
      const vert = !narrow, bw = vert ? Math.min(w * .7, 180) : (w - (cols.length - 1) * 10) / cols.length, bh = vert ? Math.min(46, (h - 30 - 4 * 14) / 5) : Math.min(70, h * .5);
      cols.forEach((t, i) => { const bx = vert ? x + (w - bw) / 2 : x + i * (bw + 10), by = vert ? y + 26 + i * (bh + 14) : y + (h - bh) / 2; PE.box(c, bx, by, bw, bh, i < 2 ? B : G, null); t.split('\n').forEach((s, k) => PE.label(c, s, bx + bw / 2, by + bh / 2 + 4 + (k - (t.split('\n').length - 1) / 2) * 13, i < 2 ? B : G, narrow ? 9 : 11, 'center'));
        if (i) { if (vert) PE.arrow(c, bx + bw / 2, by - 13, bx + bw / 2, by - 1, M, 1.2, 5); else PE.arrow(c, bx - 9, by + bh / 2, bx - 1, by + bh / 2, M, 1.2, 5); } });
      PE.label(c, 'find each object, then mask inside its box', x, y + 10, F, 11);
    } else {
      const Nq = 8, qx = x + 6, used = [true, true, true, true, true, true, false, false], names = ['cup', 'plant', 'lamp', 'book', 'wall', 'table', 'no object', 'no object'];
      const pix = { x: x + w * .5, y: y + 44, w: w * .45, h: h - 52 };
      c.fillStyle = B; c.globalAlpha = .4 + .4 * (glow.pixel || 0); c.fillRect(pix.x, pix.y, pix.w, pix.h); c.globalAlpha = 1; if (glow.pixel) { c.save(); c.strokeStyle = B; c.lineWidth = 2 + 3 * glow.pixel; c.strokeRect(pix.x, pix.y, pix.w, pix.h); c.restore(); }
      PE.label(c, 'per-pixel embeddings', pix.x, pix.y - 6, B, 11);
      for (let i = 0; i < Nq; i++) { const qy = y + 44 + i * (h - 52) / Nq + 8, col = used[i] ? G : PE.css('--brown');
        PE.dot(c, qx + 6, qy, 5 + 3 * (used[i] ? glow.query || 0 : 0), col); PE.label(c, names[i], qx + 16, qy + 4, col, narrow ? 9 : 11);
        if (used[i]) PE.arrow(c, qx + w * .3, qy, pix.x - 4, pix.y + pix.h * (i + .5) / Nq, PE.css('--muted'), 1, 5, .6); }
      PE.label(c, 'N queries; each dot product with the pixel', x, y + 6, F, 11);
      PE.label(c, 'embeddings gives one mask', x, y + 19, F, 11);
    }
  }
  const text = [
    'FCN labels every pixel with one class from a fixed list. Its scores come out on a coarse grid; coarser strides blur edges.',
    'U-Net adds skip connections, so fine detail from early layers reaches the output. The map is sharp, but the class list is still fixed.',
    'Mask R-CNN finds each object first, then predicts a small mask inside its box. Two cups would be two instances, not one blob.',
    'Mask2Former predicts a set of masks, each with a class or "no object". One design covers semantic, instance and panoptic segmentation.'];
  function readout() { ctx.hud(badge('toy', text[S.step] + ' The window and the pencil are not in the label list, so no model here can mask them.')); }
  function animateIn() { cancel(); cancel = PE.animate(700, t => { S.u = t; cv.redraw(); }); }
  readout();
  return { flat: true, step(n) { S.step = n; readout(); animateIn(); }, resize: () => cv.resize(), destroy: () => { cancel(); offs.forEach(f => f()); ctx.article.removeEventListener('pe-link', onLink); cv.destroy(); } };
};
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
/* ch-promptable: click the desk scene; SAM-style outputs. Modes: one click (three masks, or the blurred average a
   single-output model learns), two clicks (the fourth token: one mask), box (toy: the object whose box fits best).
   Train view: pick the mask the user meant; per-mask loss (toy stand-in: 1 - IoU), only the lowest-loss mask
   gets gradient; the IoU head is pulled toward each mask's true IoU. Toy scores. */
SCENES.promptable = ctx => {
  const el = mkCanvas(ctx.stage); el.tabIndex = 0; el.setAttribute('role', 'img');
  el.setAttribute('aria-label', 'Desk scene. Click, or move the focus ring with the arrow keys and press Enter, to place a prompt.');
  const S = { mode: 'click', out: 'three', view: 'use', target: 1, p1: [1.115, .7], p2: null, box: null, drag: null, focus: [1.115, .7], u: 1 };
  const SC = [.79, .93, .88];
  const B = sel => ctx.qa(sel);
  const modeB = B('[data-pmode]'), outB = B('[data-pout]'), viewB = B('[data-pview]'), tgtB = B('[data-ptarget]');
  // State that must survive an atlas theme remount lives in the pressed buttons.
  const pr = (bs, k) => { const b = bs.find(x => x.getAttribute('aria-pressed') === 'true'); return b ? b.dataset[k] : null; };
  S.mode = pr(modeB, 'pmode') || S.mode; S.out = pr(outB, 'pout') || S.out; S.view = pr(viewB, 'pview') || S.view; S.target = +(pr(tgtB, 'ptarget') ?? S.target);
  let V = null, cancel = () => {}, glow = {};
  const cv = PE.canvas(el, S3ASPECT(ctx.stage), draw);
  function sync() { press(modeB, b => b.dataset.pmode === S.mode); press(outB, b => b.dataset.pout === S.out); press(viewB, b => b.dataset.pview === S.view); press(tgtB, b => +b.dataset.ptarget === S.target); }
  function fan() { cancel(); cancel = PE.animate(650, t => { S.u = t; cv.redraw(); }); }
  const offs = [
    ...modeB.map(b => listen(b, 'click', () => { S.mode = b.dataset.pmode; if (S.mode !== 'click') S.view = 'use'; if (S.mode === 'box' && !S.box) S.box = [.97, .5, 1.3, .9]; if (S.mode === 'two' && !S.p2) S.p2 = [1.26, .885]; sync(); readout(); fan(); })),
    ...outB.map(b => listen(b, 'click', () => { S.out = b.dataset.pout; S.mode = 'click'; if (S.out === 'one') S.view = 'use'; sync(); readout(); fan(); })),
    ...viewB.map(b => listen(b, 'click', () => { S.view = b.dataset.pview; if (S.view === 'train') { S.mode = 'click'; S.out = 'three'; } sync(); readout(); fan(); })),
    ...tgtB.map(b => listen(b, 'click', () => { S.target = +b.dataset.ptarget; S.view = 'train'; S.mode = 'click'; S.out = 'three'; sync(); readout(); fan(); })),
  ];
  const onLink = e => { const k = e.detail; if (S.view !== 'train') { const t = viewB.find(b => b.dataset.pview === 'train'); if (t) t.click(); }
    PE.animate(1100, u => { glow[k] = Math.sin(Math.PI * u); cv.redraw(); }, () => { glow[k] = 0; cv.redraw(); }); };
  ctx.article.addEventListener('pe-link', onLink);
  const world = ev => { const r = el.getBoundingClientRect(); return [V.ix(ev.clientX - r.left), V.iy(ev.clientY - r.top)]; };
  const inWorld = p => p[0] >= 0 && p[0] <= TOY.WW && p[1] >= 0 && p[1] <= TOY.WH;
  function place(p) {
    if (!inWorld(p)) return;
    if (S.mode === 'two') { if (!S.p2 && !S.p1) S.p1 = p; else S.p2 = p; }
    else { S.p1 = p; if (S.mode === 'box') S.mode = 'click'; }
    S.focus = p; sync(); readout(); fan();
  }
  offs.push(listen(el, 'pointerdown', ev => { if (!V) return; const p = world(ev); if (!inWorld(p)) return;
    if (S.mode === 'box') { S.drag = p; S.box = [p[0], p[1], p[0], p[1]]; el.setPointerCapture(ev.pointerId); cv.redraw(); } else place(p); }));
  offs.push(listen(el, 'pointermove', ev => { if (!S.drag) return; const p = world(ev); S.box = [Math.min(S.drag[0], p[0]), Math.min(S.drag[1], p[1]), Math.max(S.drag[0], p[0]), Math.max(S.drag[1], p[1])]; cv.redraw(); }));
  offs.push(listen(el, 'pointerup', () => { if (!S.drag) return; S.drag = null; const b = S.box; if (b[2] - b[0] < .03 || b[3] - b[1] < .03) S.box = [b[0] - .1, b[1] - .1, b[0] + .1, b[1] + .1]; readout(); fan(); }));
  offs.push(listen(el, 'keydown', ev => { const d = { ArrowLeft: [-.03, 0], ArrowRight: [.03, 0], ArrowUp: [0, -.03], ArrowDown: [0, .03] }[ev.key];
    if (d) { ev.preventDefault(); S.focus = [PE.clamp(S.focus[0] + d[0], 0, TOY.WW), PE.clamp(S.focus[1] + d[1], 0, TOY.WH)]; cv.redraw(); }
    else if (ev.key === 'Enter' || ev.key === ' ') { ev.preventDefault(); if (S.mode === 'box') { const f = S.focus; S.box = [f[0] - .12, f[1] - .12, f[0] + .12, f[1] + .12]; readout(); fan(); } else place(S.focus.slice()); } }));
  // Long-page Reset remounts the scene: clear the pressed state first so the remount starts fresh.
  ctx.qa('[data-cue-reset]').forEach(b => offs.push(listen(b, 'click', () => ctx.qa('[aria-pressed]').forEach(x => x.removeAttribute('aria-pressed')))));
  offs.push(listen(el, 'focus', () => cv.redraw()), listen(el, 'blur', () => cv.redraw()));
  // What the model returns in the current state.
  function result() {
    if (S.mode === 'two') { const a = TOY.chain(...S.p1), b = S.p2 ? TOY.chain(...S.p2) : a; const both = a.filter(n => b.includes(n));
      return { kind: 'single', nodes: both.length ? [both[0]] : [], why: both.length ? 'fourth token' : 'none' }; }
    if (S.mode === 'box') { const bx = S.box; let best = null, bi = 0;
      TOY.all.forEach(n => { if (n === TOY.root || n.hidden || !n.mask) return; const [x0, y0, x1, y1] = TOY.bbox(n);
        const ix = Math.max(0, Math.min(x1, bx[2]) - Math.max(x0, bx[0])), iy = Math.max(0, Math.min(y1, bx[3]) - Math.max(y0, bx[1])), I = ix * iy;
        const U = (x1 - x0) * (y1 - y0) + (bx[2] - bx[0]) * (bx[3] - bx[1]) - I, v = U > 0 ? I / U : 0; if (v > bi) { bi = v; best = n; } });
      return { kind: 'single', nodes: best && bi > .15 ? [best] : [], why: 'box' }; }
    const t = TOY.three(...S.p1); return { kind: S.out === 'one' ? 'avg' : 'three', nodes: t };
  }
  // The single-output average: equal-weight overlap of the valid masks, softened. Darker = more of the answers agree.
  function avgMask(c, V2, nodes, a = 1) { c.save(); try { c.filter = 'blur(' + Math.max(2, V2.s * .012) + 'px)'; } catch (e) {}
    c.translate(V2.ox, V2.oy); c.scale(V2.s, V2.s); c.fillStyle = PE.css('--green'); nodes.forEach(n => { c.globalAlpha = a * .3; c.fill(n.mask); }); c.restore(); }
  function train(nodes) { const tgt = nodes[Math.min(S.target, nodes.length - 1)], tr = TOY.raster(tgt);
    const ious = nodes.map(n => TOY.iou(TOY.raster(n), tr)), loss = ious.map(v => 1 - v); let k = 0; loss.forEach((v, i) => { if (v < loss[k]) k = i; });
    return { tgt, ious, loss, k }; }
  function draw(c, W, H) {
    c.clearRect(0, 0, W, H);
    const narrow = W < 620, pad = 12, top = 40, bottom = ctx.reserve || 0;
    const mainW = narrow ? W - 2 * pad : W * .6, mainH = narrow ? (H - bottom) * .55 : H - bottom - top - pad;
    V = TOY.fit(pad, narrow ? top - 6 : top, mainW, mainH - (narrow ? 0 : 0));
    const R = result(), u = S.u, T = S.view === 'train' && R.kind === 'three' && R.nodes.length ? train(R.nodes) : null;
    TOY.drawWorld(c, V);
    if (R.kind === 'avg' && R.nodes.length) avgMask(c, V, R.nodes, u);
    else if (R.nodes.length) { const show = T ? R.nodes[T.k] : R.kind === 'three' ? R.nodes[1] || R.nodes[0] : R.nodes[0]; TOY.drawMask(c, V, show, { alpha: u }); }
    if (T) { c.save(); c.translate(V.ox, V.oy); c.scale(V.s, V.s); c.strokeStyle = PE.css('--fg'); c.lineWidth = 2 / V.s; c.setLineDash([5 / V.s, 4 / V.s]); c.stroke(T.tgt.mask); c.restore();
      const [x0t, y0t] = TOY.bbox(T.tgt); PE.label(c, 'what the user meant', V.X(x0t), V.Y(y0t) - 8, PE.css('--fg'), 11, 'left', 600); }
    // prompts
    if (S.mode === 'box' && S.box) TOY.boxPrompt(c, V.X(S.box[0]), V.Y(S.box[1]), V.X(S.box[2]), V.Y(S.box[3]));
    else { if (S.p1) TOY.point(c, V.X(S.p1[0]), V.Y(S.p1[1]), 'pos', 1, 7); if (S.mode === 'two' && S.p2) TOY.point(c, V.X(S.p2[0]), V.Y(S.p2[1]), 'pos', 1, 7); }
    if (document.activeElement === el) PE.ring(c, V.X(S.focus[0]), V.Y(S.focus[1]), 11, PE.css('--yellow'), 1.5, .9);
    // panels
    const items = R.kind === 'three' ? R.nodes.map((n, i) => ({ n, label: n.name, score: SC[i] ?? .8, i }))
      : R.kind === 'avg' ? (R.nodes.length ? [{ avg: R.nodes, label: 'one output: the average', i: 0 }] : [])
      : R.nodes.map(n => ({ n, label: n.name, score: .95, i: 0, note: R.why === 'fourth token' ? 'fourth output token' : 'box prompt' }));
    if (!items.length) PE.label(c, S.mode === 'two' ? 'No object holds both clicks.' : 'No object under the prompt in this toy scene.', narrow ? pad : mainW + 2 * pad + 6, narrow ? mainH + top + 20 : top + 20, PE.css('--muted'), 12, 'left', 500);
    const nP = Math.max(3, items.length);
    items.forEach((it, idx) => {
      let px, py, pw, ph; const t = PE.clamp(u * 1.6 - idx * .25, 0, 1);
      if (narrow) { pw = (W - 2 * pad - 16) / 3; ph = (H - bottom) - mainH - top - pad; px = pad + idx * (pw + 8); py = mainH + top + 4; }
      else { px = mainW + 2 * pad + 6; pw = W - px - pad; ph = (H - bottom - top - 2 * pad) / nP - (pad * (nP - 1)) / nP; py = top + idx * (ph + pad); }
      drawPanel(c, it, { px, py, pw, ph, t, narrow, T });
    });
  }
  function drawPanel(c, it, o) {
    const { px, py, pw, ph, t, narrow, T } = o, lab = narrow ? 30 : 34;
    c.save(); c.globalAlpha = t;
    const isMin = T && it.i === T.k, edge = T ? (isMin ? PE.css('--red') : PE.css('--brown')) : PE.css('--rule');
    c.strokeStyle = edge; c.lineWidth = T ? 2 + (isMin ? 3 * (glow.min || 0) : 0) : 1 + 2 * (glow.masks || 0); c.beginPath(); c.roundRect(px, py, pw, ph, 8); c.stroke();
    const n = it.n || it.avg[it.avg.length - 1], [x0, y0, x1, y1] = TOY.bbox(n), m = .06, bw = x1 - x0 + 2 * m, bh = y1 - y0 + 2 * m;
    const s = Math.min((pw - 12) / bw, (ph - lab - 10) / bh), V2 = { s, ox: px + (pw - bw * s) / 2 - (x0 - m) * s, oy: py + 5 - (y0 - m) * s };
    V2.ox = Math.min(px, Math.max(px + pw - TOY.WW * s, V2.ox)); V2.X = x => V2.ox + x * s; V2.Y = y => V2.oy + y * s; V2.w = TOY.WW * s; V2.h = TOY.WH * s;
    c.save(); c.beginPath(); c.rect(px + 1, py + 1, pw - 2, ph - lab - 4); c.clip(); TOY.drawWorld(c, V2);
    if (it.avg) avgMask(c, V2, it.avg); else TOY.drawMask(c, V2, it.n, { lift: 2, lw: 2 + 2 * (glow.masks || 0) });
    if (S.p1 && S.mode !== 'box') TOY.point(c, V2.X(S.p1[0]), V2.Y(S.p1[1]), 'pos', 1, 4); c.restore();
    const by = py + ph - lab + 4, bwid = pw - 16, fs = narrow ? 10 : 12;
    c.font = `600 ${fs}px system-ui, sans-serif`; const name = narrow ? (n.short || it.label) : it.label;
    PE.label(c, it.avg ? (narrow ? 'average' : it.label) : name, px + 8, by + 8, PE.css('--fg'), fs, 'left', 600);
    if (it.score != null) {
      c.fillStyle = PE.css('--panel2'); c.fillRect(px + 8, by + 16, bwid, 5);
      c.fillStyle = PE.css('--green'); c.fillRect(px + 8, by + 16, bwid * it.score * t, 5);
      if (glow.iou) { c.save(); c.globalAlpha = glow.iou; c.strokeStyle = PE.css('--green'); c.lineWidth = 2; c.strokeRect(px + 6, by + 14, bwid + 4, 9); c.restore(); }
      if (T) { const xi = px + 8 + bwid * T.ious[it.i]; c.fillStyle = PE.css('--red'); c.fillRect(xi - 1.5, by + 12, 3, 13);
        PE.label(c, `loss ${T.loss[it.i].toFixed(2)}`, px + pw - 8, by + 8, isMin ? PE.css('--red') : PE.css('--brown'), fs, 'right', 600); }
      else if (!narrow || c.measureText(name + '  0.00').width < pw - 16) PE.label(c, it.score.toFixed(2), px + pw - 8, by + 8, PE.css('--green'), fs, 'right', 600);
    }
    c.restore();
  }
  function readout() {
    const R = result(); let s;
    if (S.mode === 'two') s = R.nodes.length ? `Two clicks: ambiguity is much rarer, so SAM returns one mask through its fourth output token: ${R.nodes[0].name}.` : 'Two clicks on different objects: no single object contains both.';
    else if (S.mode === 'box') s = R.nodes.length ? `A box says how big the thing is. In this toy, the box picks ${R.nodes[0].name}, the object whose outline fits it best.` : 'Drag a box around something.';
    else if (!R.nodes.length) s = 'No object under the click in this toy scene. Click on the mug, the plant, the lamp or the window.';
    else if (S.out === 'one') s = 'One output: a model with a single mask is trained toward all the valid answers at once, so it learns their average, a blur that is no valid mask.';
    else if (S.view === 'train') { const T = train(R.nodes); s = `Training: the user meant ${T.tgt.name}. Each output gets a loss; only the lowest (${R.nodes[T.k].name}, loss ${T.loss[T.k].toFixed(2)}) sends gradient. The score head is pulled toward each mask's true overlap (red ticks).`; }
    else s = `One click, ${R.nodes.length} valid masks: ${R.nodes.map(n => n.name).join(', ')}. Each comes with a predicted score; SAM ranks them by it.`;
    ctx.hud(badge('toy', s));
  }
  sync(); readout();
  return { flat: true, resize: () => cv.resize(), destroy: () => { cancel(); offs.forEach(f => f()); ctx.article.removeEventListener('pe-link', onLink); cv.destroy(); } };
};
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
/* ch-sam2: a toy video. The mug slides right along the table, passes behind a box (occluded), and comes out again.
   A click on frame 1 starts the track; each frame reads a memory bank of the 6 most recent frame memories (with
   their temporal position) plus the prompted-frame memories and object pointers; the occlusion head marks
   "object absent" behind the box. Without a correction the toy track grabs the box edge for a few frames after the
   occlusion; a correction click on a later frame becomes a prompted memory and fixes the rest. Toy throughout. */
SCENES.sam2 = ctx => {
  const el = mkCanvas(ctx.stage), T = 24, NREC = 6;
  const S = { started: 0, corrected: 0, f: 0 };
  let cancel = () => {};
  const WW = 1.6, WH = 1;
  const BOX = [.78, .3, .3, .52]; // occluder x, y, w, h (world units), standing on the table at y = .82
  const mugX = f => .22 + f * (1.36 - .22) / (T - 1);
  function mugPath(f) { const x = mugX(f), p = new Path2D();
    p.moveTo(x - .1, .55); p.lineTo(x + .1, .55); p.lineTo(x + .086, .82); p.lineTo(x - .086, .82); p.closePath();
    const h = new Path2D(); const cx = x + .098, cy = .67; for (let i = 0; i <= 20; i++) { const a = -1.3 + 2.6 * i / 20; const fn = i ? 'lineTo' : 'moveTo'; h[fn](cx + .07 * Math.cos(a), cy + .07 * Math.sin(a)); }
    for (let i = 20; i >= 0; i--) { const a = -1.3 + 2.6 * i / 20; h.lineTo(cx + .042 * Math.cos(a), cy + .042 * Math.sin(a)); } h.closePath(); p.addPath(h); return p; }
  const boxPath = (() => { const p = new Path2D(); p.rect(BOX[0], BOX[1], BOX[2], BOX[3]); return p; })();
  // Visible fraction of the mug (sampled), used for the occlusion state.
  const probe = document.createElement('canvas').getContext('2d');
  const vis = []; for (let f = 0; f < T; f++) { let n = 0, v = 0; const mp = mugPath(f);
    for (let y = .5; y < .86; y += .02) for (let x = mugX(f) - .12; x < mugX(f) + .2; x += .01) if (probe.isPointInPath(mp, x, y)) { n++; if (!probe.isPointInPath(boxPath, x, y)) v++; }
    vis.push(n ? v / n : 0); }
  const firstBack = vis.findIndex((v, i) => i > 6 && v >= .12 && vis[i - 1] < .12), CORR = firstBack + 1; // correct on the second frame after the mug reappears
  // Toy model state per frame: 'none' before the click, 'ok', 'absent' (occlusion head), 'drift' (wrong mask after the occlusion).
  function state(f) { if (!S.started) return 'none'; if (vis[f] < .12) return 'absent';
    if (!S.corrected || f < CORR) { if (firstBack > 0 && f >= firstBack && f < firstBack + 5) return 'drift'; }
    return 'ok'; }
  const fit = (x, y, w, h) => { const s = Math.min(w / WW, h / WH), ox = x + (w - WW * s) / 2, oy = y + (h - WH * s) / 2; return { s, ox, oy, w: WW * s, h: WH * s, X: v => ox + v * s, Y: v => oy + v * s }; };
  const col = t => PE.mix(PE.css('--bg'), PE.css('--fg'), t);
  function frame(c, V, f, o = {}) {
    c.save(); c.beginPath(); c.rect(V.ox, V.oy, V.w, V.h); c.clip(); c.translate(V.ox, V.oy); c.scale(V.s, V.s);
    c.fillStyle = col(.07); c.fillRect(0, 0, WW, .62); c.fillStyle = col(.13); c.fillRect(0, .62, WW, WH - .62);
    c.fillStyle = col(.2); c.fillRect(.2, .1, .34, .26); c.fillStyle = col(.32); c.fillRect(.22, .12, .14, .1); c.fillRect(.38, .12, .14, .1); c.fillRect(.22, .24, .14, .1); c.fillRect(.38, .24, .14, .1);
    c.fillStyle = col(.5); c.beginPath(); c.ellipse(mugX(f), .82, .15, .03, 0, 0, 7); c.fill();
    const mp = mugPath(f); c.fillStyle = col(.66); c.fill(mp); c.fillStyle = col(.3); c.beginPath(); c.arc(mugX(f), .69, .035, 0, 7); c.fill();
    c.fillStyle = col(.42); c.fill(boxPath); c.fillStyle = col(.5); c.fillRect(BOX[0], BOX[1], BOX[2], .05);
    const st = o.mask === false ? 'none' : state(f), G = PE.css('--green');
    if (st === 'ok' || st === 'drift') {
      c.save(); if (st === 'ok') { c.beginPath(); c.rect(0, 0, WW, WH); c.rect(BOX[0], BOX[1], BOX[2], BOX[3]); c.clip('evenodd'); }
      const m = st === 'ok' ? mp : (() => { const p = new Path2D(); p.rect(BOX[0] + BOX[2] - .06, BOX[1] + .1, .06, BOX[3] - .1); p.addPath(mp); return p; })();
      const colr = st === 'ok' ? G : PE.css('--red');
      c.globalAlpha = .4; c.fillStyle = colr; c.fill(m); c.globalAlpha = 1; c.strokeStyle = colr; c.lineWidth = 2 / V.s; c.stroke(m); c.restore(); }
    c.restore();
    return st;
  }
  const btnStart = ctx.q('[data-sam2="click"]'), btnCorr = ctx.q('[data-sam2="correct"]'), btnPlay = ctx.q('[data-sam2="play"]'), slider = ctx.q('[data-sam2-frame]');
  const offs = [];
  if (btnStart) offs.push(listen(btnStart, 'click', () => { S.started = 1; press([btnStart], () => true); goTo(0); readout(); }));
  if (btnCorr) offs.push(listen(btnCorr, 'click', () => { S.started = 1; S.corrected = 1; press([btnStart, btnCorr].filter(Boolean), () => true); play(CORR); }));
  if (btnPlay) offs.push(listen(btnPlay, 'click', () => { S.started = 1; press([btnStart].filter(Boolean), () => true); play(0); }));
  if (slider) { slider.max = T - 1; offs.push(listen(slider, 'input', () => { cancel(); S.f = +slider.value; cv.redraw(); readout(); })); }
  const glow = {};
  const onLink = e => { const k = e.detail; PE.animate(1100, u => { glow[k] = Math.sin(Math.PI * u); cv.redraw(); }, () => { glow[k] = 0; cv.redraw(); }); };
  ctx.article.addEventListener('pe-link', onLink);
  const cv = PE.canvas(el, S3ASPECT(ctx.stage), draw);
  function memories(f) { // frames held in the bank when frame f is processed
    if (!S.started || f === 0) return { rec: [], pr: [] };
    const pr = [0]; if (S.corrected && f > CORR) pr.push(CORR);
    const rec = []; for (let k = f - 1; k >= 0 && rec.length < NREC; k--) if (!pr.includes(k)) rec.push(k);
    return { rec, pr };
  }
  function draw(c, W, H) {
    c.clearRect(0, 0, W, H);
    const narrow = W < 620, pad = 12, top = 40, bottom = ctx.reserve || 0, B = PE.css('--blue'), Y = PE.css('--yellow'), G = PE.css('--green'), M = PE.css('--muted'), F = PE.css('--fg'), Br = PE.css('--brown'), R = PE.css('--red');
    const stripH = 34, avail = H - bottom - top - stripH - 16;
    const mainW = narrow ? W - 2 * pad : W * .56, mainH = narrow ? avail * .52 : Math.min(avail, mainW / WW);
    const top2 = narrow ? top : top + Math.max(0, (avail - mainH) / 2);
    const V = fit(pad, top2, mainW, mainH);
    const st = frame(c, V, S.f);
    c.save(); c.strokeStyle = B; c.globalAlpha = .35 + .65 * (glow.frame || 0); c.lineWidth = 1.5 + 2 * (glow.frame || 0); c.strokeRect(V.ox, V.oy, V.w, V.h); c.restore();
    if (!narrow) PE.label(c, `frame ${S.f + 1} of ${T}`, V.ox, V.oy - 8, F, 12, 'left', 600);
    if (S.started && S.f === 0) TOY.point(c, V.X(mugX(0)), V.Y(.69), 'pos', 1, 6);
    if (S.corrected && S.f === CORR) TOY.point(c, V.X(mugX(CORR)), V.Y(.69), 'pos', 1, 6);
    const stLab = { none: 'no prompt yet', ok: 'mask from memory', absent: 'occlusion head: object absent', drift: 'error: the track grabbed the box edge' }[st];
    PE.label(c, narrow ? `frame ${S.f + 1}: ${stLab}` : stLab, narrow ? V.ox : V.ox + V.w, V.oy - 8, st === 'absent' ? Br : st === 'drift' ? R : st === 'ok' ? G : M, narrow ? 11 : 12, narrow ? 'left' : 'right', 600);
    // memory bank panel
    const mb = narrow ? { x: pad, y: V.oy + V.h + 14, w: W - 2 * pad, h: avail - V.h - 14 } : { x: pad + mainW + 18, y: V.oy, w: W - mainW - 2 * pad - 18, h: V.h };
    c.save(); c.strokeStyle = B; c.globalAlpha = .5 + .5 * (glow.memory || 0); c.lineWidth = 1 + 2 * (glow.memory || 0); c.beginPath(); c.roundRect(mb.x, mb.y, mb.w, mb.h, 10); c.stroke(); c.restore();
    PE.label(c, 'memory bank read by this frame', mb.x + 10, mb.y + 18, B, 12, 'left', 600);
    const { rec, pr } = memories(S.f), slots = narrow ? 4 : 3, rows = narrow ? 2 : 3, cell = (mb.w - 20 - (slots - 1) * 8) / slots, ch = Math.min(cell / WW, (mb.h - 46 - rows * 16) / rows);
    const thumbs = [...pr.map(k => ({ k, p: 1 })), ...rec.map((k, i) => ({ k, p: 0, t: i + 1 }))];
    thumbs.slice(0, slots * rows).forEach((m, i) => {
      const cx = mb.x + 10 + (i % slots) * (cell + 8), cy = mb.y + 28 + Math.floor(i / slots) * (ch + 16);
      const V2 = fit(cx, cy, cell, ch); frame(c, V2, m.k);
      c.save(); c.strokeStyle = m.p ? Y : B; c.lineWidth = m.p ? 2 : 1; c.strokeRect(V2.ox, V2.oy, V2.w, V2.h); c.restore();
      const occ = state(m.k) === 'absent';
      PE.label(c, narrow ? (m.p ? `prompt f${m.k + 1}` : `t-${m.t}`) : (m.p ? `prompted: frame ${m.k + 1}` : `t-${m.t}: frame ${m.k + 1}`), V2.ox, V2.oy + V2.h + 12, m.p ? Y : B, narrow ? 9 : 10, 'left', 600);
      if (occ) { PE.dot(c, V2.ox + V2.w - 6, V2.oy + 6, 4, Br); }
      c.save(); c.fillStyle = G; c.globalAlpha = occ ? .3 : .9; for (let q = 0; q < 4; q++) c.fillRect(V2.ox + 4 + q * 5, V2.oy + 4, 3.5, 6); c.restore();
    });
    if (!thumbs.length) { PE.label(c, S.started ? 'Frame 1 holds the prompt.' : 'Click the mug on frame 1 to start.', mb.x + 10, mb.y + 40, M, 11, 'left', 400); if (S.started) PE.label(c, 'The bank is empty, so SAM 2 acts like SAM.', mb.x + 10, mb.y + 56, M, 11, 'left', 400); }
    else PE.label(c, narrow ? `${rec.length} recent + ${pr.length} prompted; ticks: pointers` : `${rec.length} recent + ${pr.length} prompted; green ticks: object pointers`, mb.x + 10, mb.y + mb.h - 8, M, narrow ? 9 : 10, 'left', 400);
    // timeline strip
    const sy = H - bottom - stripH, sx = pad, sw = W - 2 * pad, cw = sw / T;
    for (let f = 0; f < T; f++) { const s = state(f), colr = s === 'ok' ? G : s === 'absent' ? Br : s === 'drift' ? R : PE.css('--rule');
      c.fillStyle = colr; c.globalAlpha = f === S.f ? 1 : .55; c.fillRect(sx + f * cw + 1, sy + 8, cw - 2, 12); c.globalAlpha = 1;
      if ((f === 0 && S.started) || (f === CORR && S.corrected)) PE.dot(c, sx + f * cw + cw / 2, sy + 4, 3.5, Y); }
    c.save(); c.strokeStyle = F; c.lineWidth = 2; c.strokeRect(sx + S.f * cw, sy + 6, cw, 16); c.restore();
    PE.label(c, 'frames: green mask, brown absent, red error, yellow dot prompt', sx, sy + 33, M, narrow ? 9 : 10, 'left', 400);
  }
  function readout() {
    const st = state(S.f), { rec, pr } = memories(S.f);
    const t = { none: 'No prompt yet. Click the mug on frame 1, or press Play.',
      ok: S.f === 0 ? 'Frame 1: the click is the prompt. The memory bank is empty, so this frame is segmented exactly as SAM would.' : `Frame ${S.f + 1}: the frame's features attend to ${rec.length} recent memories and ${pr.length} prompted ${pr.length > 1 ? 'memories' : 'memory'}, then the SAM decoder draws the mask.`,
      absent: `Frame ${S.f + 1}: the mug is behind the box. The occlusion head says "no object", so no mask is drawn, and this frame's memory carries the occlusion embedding.`,
      drift: `Frame ${S.f + 1}: in this toy the track comes back wrong for a few frames. Press "Correct with one click": a click on frame ${CORR + 1} becomes a prompted memory and fixes every later frame.` }[st];
    ctx.hud(badge('toy', t));
  }
  function goTo(f) { cancel(); S.f = f; if (slider) slider.value = f; cv.redraw(); }
  function play(from) { cancel(); S.f = from; if (slider) slider.value = from; readout();
    if (PE.instant()) { S.f = T - 1; if (slider) slider.value = S.f; cv.redraw(); readout(); return; }
    let last = -1; cancel = PE.animate((T - from) * 220, u => { const f = Math.min(T - 1, from + Math.floor(u * (T - from))); if (f !== last) { last = f; S.f = f; if (slider) slider.value = f; cv.redraw(); readout(); } }); }
  readout();
  return { flat: true, replay: () => play(0), resize: () => cv.resize(), destroy: () => { cancel(); offs.forEach(f => f()); ctx.article.removeEventListener('pe-link', onLink); cv.destroy(); } };
};
/* ch-sam3: concept prompts on the toy desk. Each query proposes one region with a query score q (how well it
   matches the phrase IF the phrase is in the image); one presence token gives p (is the phrase in the image
   at all). Total score = p x q (SAM 3 v2, App. C.2); total > 0.5 keeps the mask. Hard-negative training moves
   p for look-alike phrases; exemplar boxes move q for similar regions. All scores are toy values. */
SCENES.sam3 = ctx => {
  const el = mkCanvas(ctx.stage), THR = .5, NQ = 10;
  // Toy query scores per concept (regions not listed get a small seeded baseline).
  const Q = {
    leaf: { leaf1: .93, leaf2: .95, leaf3: .92, leaf4: .44, leaf5: .41, pot: .2, leaves: .3 },
    pane: { pane1: .9, pane2: .92, pane3: .88, pane4: .9, window: .62 },
    mug: { mug: .94, saucer: .3, handle: .35, logo: .2, mugset: .4 },
    shade: { shade: .93, arm: .3, lamp: .35 },
    pencil: { pencil: .9, spine: .25 },
    ymug: { mug: .82, saucer: .25, handle: .3, logo: .45 },
    cat: { plant: .14, pot: .1 },
    none: {} };
  const EX = { leaf: { leaf4: .86, leaf5: .84 }, pane: { window: .12 } }; // exemplar effect on q
  const PRES = { leaf: [.97, .97], pane: [.96, .96], mug: [.97, .97], shade: [.95, .95], pencil: [.94, .95], ymug: [.78, .1], cat: [.05, .04], none: [.05, .05] }; // [no hard negatives, with hard negatives]
  const NAMES = { leaf: 'leaf', pane: 'window pane', mug: 'mug', shade: 'lamp shade', pencil: 'pencil', ymug: 'yellow mug', cat: 'cat', none: '' };
  const POOL = ['leaf1', 'leaf2', 'leaf3', 'leaf4', 'leaf5', 'pot', 'pane1', 'pane2', 'pane3', 'pane4', 'window', 'mug', 'saucer', 'handle', 'logo', 'shade', 'arm', 'lampbase', 'pencil', 'spine', 'leaves', 'mugset', 'lamp', 'plant'];
  const base = {}; { const r = PE.rng(31); POOL.forEach(id => { base[id] = .02 + r() * .1; }); }
  const S = { c: 'leaf', hn: 1, pres: 1, ex: 0, typed: '' };
  const cur = { q: {}, p: 0, u: 1 }, prev = { q: {}, p: 0 };
  const glow = {};
  const qOf = (c, id, ex) => { const e = ex && EX[c] && EX[c][id] != null ? EX[c][id] : null; return e != null ? e : (Q[c][id] != null ? Q[c][id] : base[id]); };
  function target() { const q = {}; POOL.forEach(id => { q[id] = qOf(S.c, id, S.ex); }); return { q, p: PRES[S.c][S.hn] }; }
  function lerpState(u) { const t = target(), o = {}; POOL.forEach(id => { o[id] = PE.lerp(prev.q[id] ?? t.q[id], t.q[id], u); }); cur.q = o; cur.p = PE.lerp(prev.p ?? t.p, t.p, u); cur.u = u; }
  const total = id => (S.pres ? cur.p : 1) * cur.q[id];
  let cancel = () => {};
  const cv = PE.canvas(el, S3ASPECT(ctx.stage), draw);
  const cBtns = ctx.qa('[data-concept]'), hnBtns = ctx.qa('[data-hn]'), prBtns = ctx.qa('[data-presence]'), exBtns = ctx.qa('[data-exemplar]');
  const input = ctx.q('[data-concept-input]'), go = ctx.q('[data-concept-go]');
  // State that must survive an atlas theme remount lives in the pressed controls.
  { const pc = cBtns.find(b => b.getAttribute('aria-pressed') === 'true'); if (pc) S.c = pc.dataset.concept;
    const ph = hnBtns.find(b => b.getAttribute('aria-pressed') === 'true'); if (ph) S.hn = +ph.dataset.hn;
    const pp = prBtns.find(b => b.getAttribute('aria-pressed') === 'true'); if (pp) S.pres = +pp.dataset.presence;
    const pe = exBtns.find(b => b.getAttribute('aria-pressed') === 'true'); if (pe) { S.ex = 1; S.c = pe.dataset.exemplar; } }
  function sync() { press(cBtns, b => b.dataset.concept === S.c && !S.typed); press(hnBtns, b => +b.dataset.hn === S.hn); press(prBtns, b => +b.dataset.presence === S.pres); press(exBtns, b => S.ex && b.dataset.exemplar === S.c); }
  function change(fn) { prev.q = { ...cur.q }; prev.p = cur.p; fn(); sync(); cancel(); cancel = PE.animate(650, u => { lerpState(u); cv.redraw(); }, readout); readout(); }
  const offs = [
    ...cBtns.map(b => listen(b, 'click', () => change(() => { S.c = b.dataset.concept; S.ex = 0; S.typed = ''; }))),
    ...hnBtns.map(b => listen(b, 'click', () => change(() => { S.hn = +b.dataset.hn; }))),
    ...prBtns.map(b => listen(b, 'click', () => change(() => { S.pres = +b.dataset.presence; }))),
    ...exBtns.map(b => listen(b, 'click', () => change(() => { S.c = b.dataset.exemplar; S.ex = 1; S.typed = ''; })))];
  function mapText(t) { t = t.toLowerCase().trim();
    if (/yellow/.test(t) && /(mug|cup)/.test(t)) return 'ymug'; if (/lea(f|ves)/.test(t)) return 'leaf'; if (/(pane|glass)/.test(t)) return 'pane';
    if (/(mug|cup)/.test(t)) return 'mug'; if (/shade/.test(t)) return 'shade'; if (/pencil/.test(t)) return 'pencil'; if (/cat/.test(t)) return 'cat'; return 'none'; }
  if (go) offs.push(listen(go, 'click', () => { const t = (input && input.value) || ''; if (!t.trim()) return; change(() => { S.c = mapText(t); S.ex = 0; S.typed = t.trim().slice(0, 40); }); }));
  if (input) offs.push(listen(input, 'keydown', e => { if (e.key === 'Enter') { e.preventDefault(); go && go.click(); } }));
  const onLink = e => { const k = e.detail; PE.animate(1100, u => { glow[k] = Math.sin(Math.PI * u); cv.redraw(); }, () => { glow[k] = 0; cv.redraw(); }); };
  ctx.article.addEventListener('pe-link', onLink);
  let nShow = NQ;
  function slots() { return POOL.slice().sort((a, b) => cur.q[b] - cur.q[a]).slice(0, nShow); }
  function draw(c, W, H) {
    c.clearRect(0, 0, W, H);
    const narrow = W < 620, pad = 12, top = 34, bottom = ctx.reserve || 0;
    const imgW = narrow ? W - 2 * pad : W * .56, imgH = narrow ? Math.min((H - bottom) * .5, imgW / TOY.WW) : Math.min(H - bottom - top - pad, imgW / TOY.WW);
    const top2 = narrow ? top : Math.max(top, (H - bottom - imgH) / 2);
    const V = TOY.fit(pad, top2, imgW, imgH), F = PE.css('--fg'), M = PE.css('--muted'), Y = PE.css('--yellow'), G = PE.css('--green'), B = PE.css('--blue');
    TOY.drawWorld(c, V);
    const panelH = narrow ? H - bottom - (top2 + imgH + 14) - 6 : imgH;
    nShow = Math.max(4, Math.min(NQ, Math.floor((panelH - 70) / 17)));
    const sl = slots(), kept = sl.filter(id => total(id) > THR);
    kept.forEach(id => TOY.drawMask(c, V, TOY.byId[id], { alpha: PE.clamp(cur.u * 1.4, 0, 1), lift: 2, lw: 2 + 3 * (glow.total || 0) }));
    // exemplar boxes
    if (S.ex && S.c === 'leaf') { const [x0, y0, x1, y1] = TOY.bbox(TOY.byId.leaf4); TOY.boxPrompt(c, V.X(x0) - 3, V.Y(y0) - 3, V.X(x1) + 3, V.Y(y1) + 3); PE.label(c, '+ exemplar', V.X(x0) - 6, V.Y(y1) + 15, Y, 11, 'right'); }
    if (S.ex && S.c === 'pane') { const [x0, y0, x1, y1] = TOY.bbox(TOY.byId.window); c.save(); c.strokeStyle = PE.css('--brown'); c.lineWidth = 2; c.setLineDash([6, 4]); c.strokeRect(V.X(x0) - 4, V.Y(y0) - 4, V.X(x1) - V.X(x0) + 8, V.Y(y1) - V.Y(y0) + 8); c.restore(); PE.label(c, '- exemplar (the frame)', V.X(x1) + 8, V.Y(y1), PE.css('--brown'), 11); }
    // prompt chip on the image
    const phrase = S.typed || NAMES[S.c]; c.save(); c.font = '600 13px system-ui, sans-serif'; const tw = c.measureText(phrase).width;
    c.fillStyle = PE.css('--panel'); c.beginPath(); c.roundRect(V.ox + 8, V.oy + 8, tw + 18, 24, 12); c.fill(); c.strokeStyle = Y; c.lineWidth = 1.5; c.stroke(); c.restore();
    PE.label(c, phrase, V.ox + 17, V.oy + 25, Y, 13);
    // the score panel
    const P = narrow ? { x: pad, y: top2 + imgH + 14, w: W - 2 * pad, h: H - bottom - (top2 + imgH + 14) - 6 } : { x: pad + imgW + 18, y: top2, w: W - imgW - 2 * pad - 18, h: imgH };
    const pres = cur.p, pg = glow.presence || 0;
    PE.label(c, S.pres ? 'presence token: is the phrase in the image?' : 'no presence head: each query decides alone', P.x, P.y + 10, S.pres ? B : M, 11);
    const bx = P.x, bw = P.w - 46, by = P.y + 18;
    c.fillStyle = PE.css('--panel2'); c.fillRect(bx, by, bw, 10);
    if (S.pres) { c.fillStyle = B; c.globalAlpha = .85; c.fillRect(bx, by, bw * pres, 10); c.globalAlpha = 1; if (pg) { c.save(); c.strokeStyle = B; c.lineWidth = 1 + 3 * pg; c.strokeRect(bx - 2, by - 2, bw + 4, 14); c.restore(); }
      PE.label(c, pres.toFixed(2), bx + bw + 6, by + 10, B, 12, 'left', 700); }
    const top3 = by + 26, rows = sl.length, rh = Math.min(22, (P.h - (top3 - P.y) - 18) / rows);
    const nameW = narrow ? 70 : Math.min(92, P.w * .34), barX = P.x + nameW, barW = P.w - nameW - 40;
    PE.label(c, narrow ? 'proposal' : 'query proposals', P.x, top3 + 2, M, 10); PE.label(c, narrow ? 'query score (outline), total (fill)' : 'query score (outline) and total (fill)', barX, top3 + 2, M, 10);
    const tx = barX + barW * THR;
    sl.forEach((id, i) => {
      const y = top3 + 10 + i * rh, q = cur.q[id], t = total(id), on = t > THR, n = TOY.byId[id];
      PE.label(c, (n.short || n.name).replace('window pane', 'pane').replace('logo on the mug', 'logo'), P.x, y + rh * .62, on ? G : M, narrow ? 10 : 11, 'left', on ? 700 : 500);
      c.save(); c.strokeStyle = F; c.globalAlpha = .7 + .3 * (glow.query || 0); c.lineWidth = 1 + 2 * (glow.query || 0); c.strokeRect(barX, y + 3, barW * q, rh - 7); c.restore();
      c.fillStyle = on ? G : PE.css('--brown'); c.globalAlpha = on ? .85 : .55; c.fillRect(barX, y + 3, barW * t, rh - 7); c.globalAlpha = 1;
      if (glow.total) { c.save(); c.strokeStyle = G; c.lineWidth = 3 * glow.total; c.strokeRect(barX - 1, y + 2, barW * t + 2, rh - 5); c.restore(); }
      PE.label(c, t.toFixed(2), barX + barW + 4, y + rh * .62, on ? G : M, 10);
    });
    c.save(); c.strokeStyle = F; c.globalAlpha = .55; c.setLineDash([3, 3]); c.beginPath(); c.moveTo(tx, top3 + 8); c.lineTo(tx, top3 + 10 + rows * rh); c.stroke(); c.restore();
    PE.label(c, 'keep if total > 0.5', tx + 3, top3 + 20 + rows * rh, M, 10);
  }
  function readout() {
    const t = target(), kept = POOL.filter(id => (S.pres ? t.p : 1) * t.q[id] > THR), phrase = S.typed || NAMES[S.c];
    let s;
    if (S.c === 'none') s = `"${phrase}" is not in this toy's vocabulary, so every query score stays low and the presence score is ${t.p.toFixed(2)}. Nothing is kept.`;
    else if (!kept.length) s = `"${phrase}": presence ${t.p.toFixed(2)}. The phrase is judged absent, so the gate closes on every query and nothing is kept.`;
    else s = `"${phrase}": presence ${S.pres ? t.p.toFixed(2) : 'not used'}; ${kept.length} instance${kept.length > 1 ? 's' : ''} kept.`;
    if (S.c === 'ymug') s += S.pres ? (S.hn ? ' Trained with hard negatives, the presence token knows a grey mug is not a yellow mug.' : ' Trained without hard negatives, the presence token is fooled by the look-alike: a false positive.') : ' Without a presence head, the mug query fires on its own: a false positive.';
    if (S.c === 'leaf' && !S.ex) s += ' The two small sideways leaves score below the line. Try the exemplar.';
    if (S.c === 'leaf' && S.ex) s += ' One exemplar box on a small leaf lifts the similar leaves above the line.';
    if (S.c === 'pane' && !S.ex) s += ' The whole window frame also scores above the line. Try the negative exemplar.';
    if (S.c === 'pane' && S.ex) s += ' A negative exemplar on the frame pushes it below the line.';
    ctx.hud(badge('toy', s));
  }
  sync(); lerpState(1); readout();
  return { flat: true, resize: () => cv.resize(), destroy: () => { cancel(); offs.forEach(f => f()); ctx.article.removeEventListener('pe-link', onLink); cv.destroy(); } };
};
/* ch-inspection (toy): a machined plate with a bolt and three defects of different kinds: a dent (clear edges), a
   thin scratch, and a faint stain (low contrast). Tabs: 0 one click per defect, as SAM answers it; 1 a box per
   defect from a detector; 2 SAM's box masks used as noisy labels to train a small student segmenter. The
   behaviours follow the failure modes the inspection papers report; the drawing and the outcomes are ours. */
SCENES.inspect = ctx => {
  const el = mkCanvas(ctx.stage), S = { step: 0, u: 1 };
  const WW = 1.6, WH = 1;
  const plate = (() => { const p = new Path2D(); p.roundRect(.12, .14, 1.36, .74, .06); return p; })();
  const ell = (cx, cy, rx, ry, a = 0) => { const p = new Path2D(); p.ellipse(cx, cy, rx, ry, a, 0, 7); return p; };
  const DEF = [
    { id: 'dent', name: 'dent', at: [.42, .55], shape: ell(.42, .55, .075, .055, .3) },
    { id: 'scratch', name: 'thin scratch', at: [.86, .42], pts: [[.66, .58], [.78, .47], [.9, .4], [1.05, .3]] },
    { id: 'stain', name: 'faint stain', at: [1.2, .66], shape: (() => { const p = new Path2D(); p.moveTo(1.12, .62); p.bezierCurveTo(1.15, .55, 1.27, .56, 1.3, .63); p.bezierCurveTo(1.33, .7, 1.22, .76, 1.16, .72); p.bezierCurveTo(1.11, .7, 1.1, .66, 1.12, .62); return p; })() }];
  const scratchPath = (a = 0, b = 1, w = .012) => { const P = DEF[1].pts, n = 40, top = [], bot = [];
    const at = t => { const s = t * (P.length - 1), i = Math.min(P.length - 2, Math.floor(s)), f = s - i; return [PE.lerp(P[i][0], P[i + 1][0], f), PE.lerp(P[i][1], P[i + 1][1], f)]; };
    for (let k = 0; k <= n; k++) { const t = PE.lerp(a, b, k / n), p = at(t), q = at(Math.min(1, t + .01)), dx = q[0] - p[0], dy = q[1] - p[1], L = Math.hypot(dx, dy) || 1; top.push([p[0] - dy / L * w, p[1] + dx / L * w]); bot.push([p[0] + dy / L * w, p[1] - dx / L * w]); }
    const pth = new Path2D(); [...top, ...bot.reverse()].forEach(([x, y], i) => i ? pth.lineTo(x, y) : pth.moveTo(x, y)); pth.closePath(); return pth; };
  // what each method returns per defect: [path, verdict, ok?]
  const OUT = [
    [[DEF[0].shape, 'good', true], [plate, 'wrong: the whole plate', false], [plate, 'wrong: the whole plate', false]],
    [[DEF[0].shape, 'good', true], [scratchPath(.3, .62, .016), 'partial: misses the thin ends', false], [ell(1.2, .655, .055, .035, .2), 'found, but the extent is wrong', false]],
    [[DEF[0].shape, 'good', true], [scratchPath(.04, .97, .014), 'closer', true], [DEF[2].shape, 'closer', true]]];
  let cancel = () => {};
  const cv = PE.canvas(el, S3ASPECT(ctx.stage), draw);
  function fit(x, y, w, h) { const s = Math.min(w / WW, h / WH); return { s, ox: x + (w - WW * s) / 2, oy: y + (h - WH * s) / 2 }; }
  function draw(c, W, H) {
    c.clearRect(0, 0, W, H); const narrow = W < 620, bottom = ctx.reserve || 0, top = 34, V = fit(10, top, W - 20, H - bottom - top - (narrow ? 70 : 10)), u = S.u;
    const tone = t => PE.mix(PE.css('--bg'), PE.css('--fg'), t), G = PE.css('--green');
    c.save(); c.translate(V.ox, V.oy); c.scale(V.s, V.s);
    c.fillStyle = tone(.06); c.fillRect(0, 0, WW, WH);
    c.fillStyle = tone(.3); c.fill(plate); c.save(); c.clip(plate); c.strokeStyle = tone(.34); c.lineWidth = 1 / V.s; for (let y = .15; y < .9; y += .012) { c.beginPath(); c.moveTo(.1, y); c.lineTo(1.5, y + .01); c.stroke(); } c.restore();
    c.fillStyle = tone(.45); c.fill(ell(1.32, .27, .055, .055)); c.fillStyle = tone(.25); c.fill(ell(.25, .27, .03, .03));
    c.fillStyle = tone(.16); c.fill(DEF[0].shape); c.strokeStyle = tone(.5); c.lineWidth = 1.5 / V.s; c.stroke(DEF[0].shape);
    c.strokeStyle = tone(.5); c.lineWidth = 2.2 / V.s; c.beginPath(); DEF[1].pts.forEach(([x, y], i) => i ? c.lineTo(x, y) : c.moveTo(x, y)); c.stroke();
    c.fillStyle = tone(.275); c.fill(DEF[2].shape);
    OUT[S.step].forEach(([p, v, ok], i) => { const a = PE.clamp(u * 1.6 - i * .25, 0, 1); if (!a) return; c.save(); c.globalAlpha = a * (p === plate ? .26 : .45); c.fillStyle = G; c.fill(p); c.globalAlpha = a; c.strokeStyle = G; c.lineWidth = 2 / V.s; c.stroke(p); c.restore(); });
    c.restore();
    const X = x => V.ox + x * V.s, Y = y => V.oy + y * V.s;
    DEF.forEach((d, i) => { const [p, v, ok] = OUT[S.step][i], a = PE.clamp(u * 1.6 - i * .25, 0, 1);
      if (S.step === 0) TOY.point(c, X(d.at[0]), Y(d.at[1]), 'pos', 1, 5);
      if (S.step >= 1) { const r = { dent: [.33, .48, .51, .62], scratch: [.64, .27, 1.07, .61], stain: [1.09, .54, 1.33, .78] }[d.id]; TOY.boxPrompt(c, X(r[0]), Y(r[1]), X(r[2]), Y(r[3]), S.step === 2 ? .45 : 1); }
      const lx = narrow ? 14 : X(d.at[0]), ly = narrow ? H - bottom - 58 + i * 18 : Y(d.id === 'scratch' ? .2 : .86) + (d.id === 'scratch' ? 0 : 26);
      c.save(); c.globalAlpha = a; PE.label(c, `${d.name}: ${v}`, lx, ly, ok ? G : PE.css('--red'), narrow ? 11 : 12, narrow ? 'left' : 'center', 600); c.restore(); });
  }
  const text = ['One click on each defect. SAM was trained on masks of objects, so a click on a defect without an object-like boundary can return the object around it: the plate (our illustration). The clear dent works.',
    'A box from a defect detector fixes most of the "which object" problem. Thin and low-contrast defects still come back partial or with the wrong extent.',
    'Use SAM\'s box masks as noisy training labels for a small, fast segmenter, with a loss that tolerates label noise. The student gets closer, not perfect; at run time only the student runs.'];
  function readout() { ctx.hud(badge('toy', text[S.step])); }
  function animateIn() { cancel(); cancel = PE.animate(800, t => { S.u = t; cv.redraw(); }); }
  readout();
  return { flat: true, step(n) { S.step = n; readout(); animateIn(); }, resize: () => cv.resize(), destroy: () => { cancel(); cv.destroy(); } };
};
/* Atlas: 2D scenes keep the bottom 130 px free on desktop, where the readout box sits. */
SCENES.inspection = SCENES.inspect;
for (const k of Object.keys(SCENES)) { const f = SCENES[k]; SCENES[k] = ctx => f(Object.assign(ctx, { reserve: matchMedia('(min-width: 861px)').matches ? 130 : 0 })); }
