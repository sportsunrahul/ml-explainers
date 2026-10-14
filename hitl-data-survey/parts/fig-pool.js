/* FIG.pool: uncertainty sampling on the toy pool (ch-uncertainty; atlas "ask").
   Left: the 2D pool (shape = true class; brown = unlabeled; yellow ring = labeled by a person) over a blue heatmap of the
   chosen uncertainty score. Right: the 3-class probability triangle, and accuracy vs labels for the uncertainty run (blue)
   and a random run (brown) on the same pool, over the 20-pool median bands. */
FIG.pool = (el, o = {}) => {
  const SIM = DATA.poolsim, SEED = SIM.seed, B = SIM.b, ROUNDS = SIM.labels.length - 1;
  const pool = ToyPool.sample(700, SEED), test = ToyPool.sample(900, 1000 + SEED);
  const S = { measure: 'margin', view: 'unc', rounds: 0, hover: -1, lastU: [], lastR: [] };
  let U = null, R = null, heat = null, heatKey = '';
  const glow = o.fig ? PE.linkPulses(o.fig, () => cv.redraw()) : () => 0;
  function rebuild(k) {   // replay both campaigns from the start to round k (deterministic)
    U = ToyPool.session(pool, test, S.measure, { b: B, seed: SEED * 31 });
    R = ToyPool.session(pool, test, 'random', { b: B, seed: SEED * 31 });
    S.lastU = []; S.lastR = [];
    for (let t = 0; t < k; t++) { S.lastU = U.round(S.measure); S.lastR = R.round(); }
    S.rounds = k; heatKey = '';
  }
  rebuild(0);
  const span = 6.4;
  function heatmap(sess) {
    const key = S.measure + S.view + S.rounds; if (key === heatKey) return heat; heatKey = key;
    const G = 64, W = sess.model(), out = new Float32Array(G * G);
    for (let i = 0; i < G; i++) for (let j = 0; j < G; j++) {
      const x = -span / 2 + span * (j + .5) / G, y = span / 2 - span * (i + .5) / G;
      out[i * G + j] = ToyPool.score[S.measure](ToyPool.prob(W, ToyPool.feat(x, y)));
    }
    let mx = 0; for (const v of out) mx = Math.max(mx, v);
    heat = { G, v: out, mx: Math.max(mx, 1e-6) }; return heat;
  }
  function glyph(c, x, y, cls, r, fill, stroke) {
    c.beginPath();
    if (cls === 0) c.arc(x, y, r, 0, 7);
    else if (cls === 1) c.rect(x - r * .9, y - r * .9, r * 1.8, r * 1.8);
    else { c.moveTo(x, y - r * 1.15); c.lineTo(x + r * 1.05, y + r * .75); c.lineTo(x - r * 1.05, y + r * .75); c.closePath(); }
    if (fill) { c.fillStyle = fill; c.fill(); } if (stroke) { c.strokeStyle = stroke; c.lineWidth = 1.6; c.stroke(); }
  }
  const aspect = o.aspect || (w => w < 620 ? 1.62 : .5);
  const cv = PE.canvas(el, aspect, draw);
  let L = null;
  function draw(c, W, H) {
    const C = { fg: PE.css('--fg'), mu: PE.css('--muted'), rule: PE.css('--rule'), blue: PE.css('--blue'), yel: PE.css('--yellow'), brown: PE.css('--brown'), bg: PE.css('--bg'), red: PE.css('--red') };
    c.clearRect(0, 0, W, H);
    const narrow = W < 620, padB = o.padBottom || 0;
    const side = narrow ? Math.min(W - 16, (H - padB) * .52) : Math.min(H - padB - 16, W * .5);
    const pb = { x: 8, y: 8, s: side };
    const right = narrow ? { x: 8, y: pb.y + side + 14, w: W - 16, h: H - padB - side - 30 } : { x: pb.x + side + 22, y: 8, w: W - side - 38, h: H - padB - 16 };
    const sess = S.view === 'unc' ? U : R, P = sess.probs(), lab = new Set(sess.labeled), last = new Set(S.view === 'unc' ? S.lastU : S.lastR);
    // heatmap of the uncertainty score under the current model
    const hm = heatmap(sess);
    KIT.heat(c, pb.x, pb.y, side, side, hm.G, (i, j) => hm.v[i * hm.G + j] / hm.mx, C.blue, .62);
    c.strokeStyle = C.rule; c.strokeRect(pb.x, pb.y, side, side);
    const X = x => pb.x + (x + span / 2) / span * side, Y = y => pb.y + (span / 2 - y) / span * side;
    const next = S.rounds < ROUNDS ? new Set(sess.labeled.length ? (S.view === 'unc' ? nextPicks(U) : []) : []) : new Set();
    pool.forEach((p, i) => { if (!lab.has(i)) glyph(c, X(p.x), Y(p.y), p.c, 2.6, C.brown, null); });
    pool.forEach((p, i) => { if (lab.has(i)) { glyph(c, X(p.x), Y(p.y), p.c, 4.2, C.fg, null); KIT.ring(c, X(p.x), Y(p.y), last.has(i) ? 10 : 7.5, C.yel, { seed: i, lw: last.has(i) ? 2.4 : 1.4, alpha: last.has(i) ? 1 : .6 }); } });
    next.forEach(i => { const p = pool[i]; c.save(); c.setLineDash([3, 3]); c.strokeStyle = C.yel; c.lineWidth = 1.4; c.beginPath(); c.arc(X(p.x), Y(p.y), 9, 0, 7); c.stroke(); c.restore(); });
    if (S.hover >= 0) { const p = pool[S.hover]; c.strokeStyle = C.fg; c.lineWidth = 1.5; c.beginPath(); c.arc(X(p.x), Y(p.y), 7, 0, 7); c.stroke(); }
    PE.label(c, S.view === 'unc' ? `uncertainty run (${S.measure === 'lc' ? 'least confidence' : S.measure})` : 'random run', pb.x + 6, pb.y + 16, S.view === 'unc' ? C.blue : C.brown, 12);
    PE.label(c, `${sess.labeled.length} labels`, pb.x + 6, pb.y + 32, C.mu, 11);
    // probability triangle
    const th = narrow ? Math.min(right.h * .46, right.w * .48) : right.h * .44, tw = th * 1.15;
    const tri = { cx: right.x + (narrow ? tw / 2 + 10 : right.w / 2), top: right.y + 18, w: tw, h: th };
    const corner = [[tri.cx - tw / 2, tri.top + th], [tri.cx + tw / 2, tri.top + th], [tri.cx, tri.top]];
    const toXY = p => [corner[0][0] * p[0] + corner[1][0] * p[1] + corner[2][0] * p[2], corner[0][1] * p[0] + corner[1][1] * p[1] + corner[2][1] * p[2]];
    const TG = 72, bary = (i, j) => { const px = tri.cx - tw / 2 + tw * (j + .5) / TG, py = tri.top + th * (i + .5) / TG;   // pixel -> barycentric
      const d = (corner[1][1] - corner[2][1]) * (corner[0][0] - corner[2][0]) + (corner[2][0] - corner[1][0]) * (corner[0][1] - corner[2][1]);
      const a = ((corner[1][1] - corner[2][1]) * (px - corner[2][0]) + (corner[2][0] - corner[1][0]) * (py - corner[2][1])) / d;
      const b = ((corner[2][1] - corner[0][1]) * (px - corner[2][0]) + (corner[0][0] - corner[2][0]) * (py - corner[2][1])) / d; return [a, b, 1 - a - b]; };
    KIT.heat(c, tri.cx - tw / 2, tri.top, tw, th, TG, (i, j) => ToyPool.score[S.measure](bary(i, j).map(v => Math.max(v, 0))), C.blue, .55, (i, j) => bary(i, j).every(v => v >= -.01));
    c.globalAlpha = 1; c.strokeStyle = C.mu; c.lineWidth = 1; c.beginPath(); c.moveTo(...corner[0]); c.lineTo(...corner[1]); c.lineTo(...corner[2]); c.closePath(); c.stroke();
    [0, 1, 2].forEach(k => glyph(c, corner[k][0] + (k === 0 ? -10 : k === 1 ? 10 : 0), corner[k][1] + (k === 2 ? -8 : 8), k, 4, C.fg, null));
    ['lc', 'margin', 'entropy'].forEach(m => { const g = glow(m); if (g > 0) {   // equation term hover: draw that score's contour at half its max
      c.save(); c.globalAlpha = g; c.strokeStyle = C.yel; c.lineWidth = 2;
      for (let a = 0; a <= 60; a++) for (let b = 0; b <= 60 - a; b++) { const p = [a / 60, b / 60, 1 - a / 60 - b / 60], v = ToyPool.score[m](p) - .5; const p2 = [(a + 1) / 60, b / 60, 1 - (a + 1) / 60 - b / 60];
        if (p2[2] >= 0 && Math.sign(v) !== Math.sign(ToyPool.score[m](p2) - .5)) { const q = toXY(p); c.fillStyle = C.yel; c.fillRect(q[0] - 1, q[1] - 1, 2.4, 2.4); } }
      c.restore(); } });
    if (S.hover >= 0) { const q = toXY(P[S.hover]); PE.dot(c, q[0], q[1], 5, C.blue); c.strokeStyle = C.fg; c.beginPath(); c.arc(q[0], q[1], 6.5, 0, 7); c.stroke(); }
    next.forEach(i => { const q = toXY(P[i]); PE.dot(c, q[0], q[1], 3, C.yel); });
    PE.label(c, 'probabilities of one image', narrow ? tri.cx + tw / 2 + 10 : tri.cx - tw / 2, narrow ? tri.top + 10 : tri.top + th + 22, C.mu, 11);
    // learning curves
    const cb = narrow ? { x: right.x + 30, y: tri.top + th + 40, w: right.w - 40, h: right.y + right.h - (tri.top + th + 40) - 18 } : { x: right.x + 34, y: tri.top + th + 44, w: right.w - 44, h: right.y + right.h - (tri.top + th + 44) - 20 };
    const ax = PE.axes(c, cb, [SIM.labels[0], SIM.labels.at(-1)], [.4, 1], { xt: [3, 23, 43, 63], yt: [.4, .7, 1], yf: v => Math.round(v * 100) + '%', xl: 'labels' });
    const band = (q, col) => { c.fillStyle = col; c.globalAlpha = .16; c.beginPath(); q.q75.forEach((v, i) => { const x = ax.X(SIM.labels[i]), y = ax.Y(v); i ? c.lineTo(x, y) : c.moveTo(x, y); }); for (let i = q.q25.length - 1; i >= 0; i--) c.lineTo(ax.X(SIM.labels[i]), ax.Y(q.q25[i])); c.closePath(); c.fill(); c.globalAlpha = 1; };
    band(SIM[S.measure], C.blue); band(SIM.random, C.brown);
    const line = (curve, col) => { const pts = curve.map(p => [ax.X(p.n), ax.Y(p.acc)]); PE.polyline(c, pts, col, 2.2); pts.forEach(p => PE.dot(c, p[0], p[1], 2.6, col)); };
    line(R.curve, C.brown); line(U.curve, C.blue);
    PE.label(c, 'this pool, solid; 20 pools (middle half), shaded', cb.x, cb.y - 8, C.mu, 11);
    L = { pb, X, Y };
  }
  function nextPicks(sess) {   // what the uncertainty run would ask for next (preview only)
    const P = sess.probs(), lab = new Set(sess.labeled), cand = [];
    for (let i = 0; i < pool.length; i++) if (!lab.has(i)) cand.push([ToyPool.score[S.measure](P[i]), i]);
    return cand.sort((a, b) => b[0] - a[0] || a[1] - b[1]).slice(0, B).map(x => x[1]);
  }
  function say() {
    const u = U.curve.at(-1), r = R.curve.at(-1);
    o.say && o.say(`${U.labeled.length} labels each. Accuracy on held-out points: uncertainty run ${(100 * u.acc).toFixed(0)}%, random run ${(100 * r.acc).toFixed(0)}%. Dashed yellow rings: the ${B} images the ${S.measure === 'lc' ? 'least-confidence' : S.measure} score would send next.`);
  }
  function onMove(e) {
    if (!L) return; const r = el.getBoundingClientRect(), x = e.clientX - r.left, y = e.clientY - r.top;
    let best = -1, bd = 100; pool.forEach((p, i) => { const d = (L.X(p.x) - x) ** 2 + (L.Y(p.y) - y) ** 2; if (d < bd) { bd = d; best = i; } });
    if (best !== S.hover) { S.hover = best; cv.redraw(); }
  }
  el.addEventListener('pointermove', onMove); el.addEventListener('pointerleave', () => { S.hover = -1; cv.redraw(); });
  PE.onTheme(() => cv.redraw());
  say();
  return {
    measure(m) { S.measure = m; rebuild(S.rounds); cv.redraw(); say(); },
    view(v) { S.view = v; heatKey = ''; cv.redraw(); say(); },
    rounds(k) { rebuild(Math.min(k, ROUNDS)); cv.redraw(); say(); },
    step() { if (S.rounds >= ROUNDS) return; S.lastU = U.round(S.measure); S.lastR = R.round(); S.rounds++; heatKey = ''; cv.redraw(); say(); },
    reset() { S.measure = 'margin'; S.view = 'unc'; S.hover = -1; rebuild(0); cv.redraw(); say(); },
    redraw: () => cv.redraw(),
    destroy() { el.removeEventListener('pointermove', onMove); cv.destroy && cv.destroy(); },
  };
};
