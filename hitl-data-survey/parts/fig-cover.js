/* FIG.cover: filling one labeling batch (ch-coreset; atlas "cover"). A 2D "embedding" of a pool with a burst of
   near-duplicate frames sitting on the class boundary. Top-k uncertainty, greedy k-Center and random each pick b = 10.
   Circles of radius delta around every labeled point show the coverage. Toy data; the k-Center greedy is the paper's rule. */
FIG.cover = (el, o = {}) => {
  const r = PE.rng(23), g = PE.gauss(r), pts = [];
  const blobs = [[-2, 1.2, .55, 70], [-1.4, -1.3, .6, 70], [1.6, 1.4, .5, 60], [2.1, -.9, .6, 70], [0, 2.3, .45, 40], [-.2, -2.4, .45, 40]];
  blobs.forEach(([x, y, s, n]) => { for (let i = 0; i < n; i++) pts.push({ x: x + s * g(), y: y + s * g(), burst: false }); });
  for (let i = 0; i < 70; i++) pts.push({ x: .35 + .09 * g(), y: .1 + .09 * g(), burst: true });   // one moment, many frames
  const B = 10, start = [3, 75, 145, 210, 290, 330].filter(i => i < pts.length);
  // toy uncertainty: closeness to a boundary through the burst (the model's decision line x = .3 + .1 y)
  const unc = p => Math.exp(-Math.abs(p.x - .3 - .1 * p.y) * 2.2);
  const d = (a, b) => Math.hypot(a.x - b.x, a.y - b.y);
  function pick(strategy, k) {
    const lab = start.slice(), cand = pts.map((p, i) => i).filter(i => !lab.includes(i));
    if (strategy === 'topk') return cand.sort((a, b) => unc(pts[b]) - unc(pts[a]) || a - b).slice(0, k);
    if (strategy === 'random') { const rr = PE.rng(5), c2 = cand.slice(); for (let i = c2.length - 1; i > 0; i--) { const j = Math.floor(rr() * (i + 1)); [c2[i], c2[j]] = [c2[j], c2[i]]; } return c2.slice(0, k); }
    const out = [], md = pts.map(p => Math.min(...lab.map(j => d(p, pts[j]))));   // greedy k-Center: farthest point first
    for (let t = 0; t < k; t++) { let u = -1, best = -1; md.forEach((v, i) => { if (v > best && !lab.includes(i) && !out.includes(i)) { best = v; u = i; } }); out.push(u); pts.forEach((p, i) => { md[i] = Math.min(md[i], d(p, pts[u])); }); }
    return out;
  }
  const S = { strategy: 'topk', k: B, u: 1 };
  const delta = sel => { const lab = start.concat(sel); return Math.max(...pts.map(p => Math.min(...lab.map(j => d(p, pts[j]))))); };
  const cv = PE.canvas(el, o.aspect || (w => w < 620 ? 1.05 : .5), draw); let stop = () => {};
  function draw(c, W, H) {
    const C = { fg: PE.css('--fg'), mu: PE.css('--muted'), blue: PE.css('--blue'), yel: PE.css('--yellow'), brown: PE.css('--brown'), rule: PE.css('--rule'), red: PE.css('--red') };
    c.clearRect(0, 0, W, H); H = Math.max(140, H - (o.padBottom || 0));
    const sc = Math.min(W / 7.6, H / 6.2), X = x => W / 2 + x * sc, Y = y => H / 2 - y * sc;
    const sel = pick(S.strategy, S.k), dl = delta(sel), lab = start.concat(sel);
    // coverage discs
    c.save(); c.globalAlpha = .1 * S.u; c.fillStyle = C.yel; lab.forEach(j => { c.beginPath(); c.arc(X(pts[j].x), Y(pts[j].y), dl * sc, 0, 7); c.fill(); }); c.restore();
    if (S.strategy === 'topk') { c.save(); c.strokeStyle = C.blue; c.setLineDash([5, 5]); c.globalAlpha = .7; c.beginPath(); c.moveTo(X(.3 - .1 * 3.2), Y(-3.2)); c.lineTo(X(.3 + .1 * 3.2), Y(3.2)); c.stroke(); c.restore(); PE.label(c, "model's boundary", X(.62), Y(2.9), C.blue, 11); }
    pts.forEach(p => PE.dot(c, X(p.x), Y(p.y), p.burst ? 2.2 : 2.6, C.brown, p.burst ? .9 : .75));
    start.forEach(j => { PE.dot(c, X(pts[j].x), Y(pts[j].y), 4, C.fg); KIT.ring(c, X(pts[j].x), Y(pts[j].y), 7, C.yel, { seed: j, lw: 1.3, alpha: .55 }); });
    sel.forEach((j, t) => { PE.dot(c, X(pts[j].x), Y(pts[j].y), 4, C.fg); KIT.ring(c, X(pts[j].x), Y(pts[j].y), 9, C.yel, { seed: j + 3, u: PE.clamp(S.u * 1.4 - t * .04, 0, 1) }); });
    const inBurst = sel.filter(j => pts[j].burst).length;
    PE.label(c, 'a burst of near-duplicate frames', X(.35) + 18, Y(.1) + 30, C.mu, 11);
    PE.label(c, `covering radius delta = ${dl.toFixed(2)}`, 10, 18, C.yel, 12);
    PE.label(c, `${inBurst} of ${S.k} picks inside the burst`, 10, 34, C.mu, 12);
    o.say && o.say(`${({ topk: 'Top-k uncertainty', greedy: 'Greedy k-Center', random: 'Random' })[S.strategy]}: ${S.k} new labels, ${inBurst} of them in the burst of near-copies. Every pool point is within ${dl.toFixed(2)} of a labeled point (the yellow discs).`);
  }
  function anim() { stop(); if (PE.instant()) { S.u = 1; cv.redraw(); return; } S.u = 0; stop = PE.animate(700, u => { S.u = u; cv.redraw(); }); }
  PE.onTheme(() => cv.redraw());
  return { strategy(s) { S.strategy = s; S.k = B; anim(); }, step() { if (S.strategy !== 'greedy') { S.strategy = 'greedy'; S.k = 0; } S.k = S.k >= B ? 1 : S.k + 1; S.u = 1; cv.redraw(); return S.k; },
    reset() { S.strategy = 'topk'; S.k = B; S.u = 1; cv.redraw(); }, redraw: () => cv.redraw(), destroy() { stop(); cv.destroy && cv.destroy(); } };
};
