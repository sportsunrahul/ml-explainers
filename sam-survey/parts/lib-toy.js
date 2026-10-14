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
