/* AdaRound's question on a plane: two weights (in units of the grid step), the four grid corners around them, and
   the layer-output error d^T H d of rounding to each corner, H = E[x x^T] of the layer inputs. Exact arithmetic.
   Reusable (GPTQ): KIT.adaround(el, { H, w }) takes any 2x2 positive-definite H; state.trail ([[x, y], ...]) and
   state.marks ([{ p: [x, y], color, label }]) draw extra paths and points on the same plane. */
KIT.quadForm = (H, d) => H[0][0] * d[0] * d[0] + 2 * H[0][1] * d[0] * d[1] + H[1][1] * d[1] * d[1];
KIT.roundCosts = (w, H) => {
  const f = w.map(Math.floor), near = w.map(Math.round);
  const cs = [[0, 0], [1, 0], [0, 1], [1, 1]].map(([a, b]) => { const p = [f[0] + a, f[1] + b], d = [p[0] - w[0], p[1] - w[1]];
    return { p, d, cost: KIT.quadForm(H, d), nearest: p[0] === near[0] && p[1] === near[1] }; });
  const best = cs.reduce((m, c) => c.cost < m.cost ? c : m); best.best = true;
  return cs;
};
// Points of the ellipse {v : (v - w)^T H (v - w) = level}, from the eigen-decomposition of the 2x2 H.
KIT.quadEllipse = (H, w, level, n = 72) => {
  const a = H[0][0], b = H[0][1], c = H[1][1], tr = (a + c) / 2, dt = Math.sqrt(((a - c) / 2) ** 2 + b * b);
  const l1 = tr + dt, l2 = tr - dt, th = Math.abs(b) < 1e-12 ? (a >= c ? 0 : Math.PI / 2) : Math.atan2(l1 - a, b);
  const r1 = Math.sqrt(level / l1), r2 = Math.sqrt(level / l2), ct = Math.cos(th), st = Math.sin(th);
  return Array.from({ length: n + 1 }, (_, k) => { const t = 2 * Math.PI * k / n, u = r1 * Math.cos(t), v = r2 * Math.sin(t);
    return [w[0] + u * ct - v * st, w[1] + u * st + v * ct]; });
};
KIT.ADA_H = { indep: [[1, 0], [0, 1]], corr: [[1, .5], [.5, 1]] };  // corr: the Hessian of AdaRound's Example 1 (Sec. 2)
KIT.adaround = (el, opts = {}) => {
  const S = { w: (opts.w || [.42, .35]).slice(), H: opts.H || KIT.ADA_H.corr, pick: null, u: 0, trail: [], marks: [] };
  const glow = opts.fig ? PE.linkPulses(opts.fig, () => cv.redraw()) : () => 0;
  let geo = null, drag = false;
  function draw(ctx, w, h) {
    ctx.clearRect(0, 0, w, h);
    const narrow = w < 560, cs = n => PE.css(n), pad = opts.bottomPad || 0;
    const C = { fg: cs('--fg'), mut: cs('--muted'), blue: cs('--blue'), yellow: cs('--yellow'), red: cs('--red'), green: cs('--green'), rule: cs('--rule') };
    const side = narrow ? Math.min(w - 40, h - pad - 150) : Math.min(h - pad - 50, w * .58);
    const ox = narrow ? (w - side) / 2 + 8 : 46, oy = 14, lo = -.75, hi = 1.75;
    const X = v => ox + (v - lo) / (hi - lo) * side, Y = v => oy + side - (v - lo) / (hi - lo) * side;
    geo = { X, Y, ix: px => lo + (px - ox) / side * (hi - lo), iy: py => lo + (oy + side - py) / side * (hi - lo) };
    const costs = KIT.roundCosts(S.w, S.H), gh = glow('hess'), gd = glow('delta'), gc = glow('corners'), gl = glow('loss');
    ctx.save(); ctx.strokeStyle = C.rule; ctx.strokeRect(X(lo), Y(hi), side, side); ctx.restore();
    // the integer grid (yellow): lines through the grid points
    for (const g of [0, 1]) { PE.polyline(ctx, [[X(g), Y(lo)], [X(g), Y(hi)]], C.yellow, 1, [3, 4], .45); PE.polyline(ctx, [[X(lo), Y(g)], [X(hi), Y(g)]], C.yellow, 1, [3, 4], .45); }
    PE.label(ctx, 'w\u2081 / step', X(hi) - 4, Y(lo) + 26, C.mut, 11, 'right', 400);
    ctx.save(); ctx.translate(X(lo) - 26, Y(hi) + 4); ctx.rotate(-Math.PI / 2); PE.label(ctx, 'w\u2082 / step', 0, 0, C.mut, 11, 'right', 400); ctx.restore();
    for (const g of [0, 1]) { PE.label(ctx, String(g), X(g), Y(lo) + 13, C.mut, 10.5, 'center', 400); PE.label(ctx, String(g), X(lo) - 6, Y(g) + 4, C.mut, 10.5, 'right', 400); }
    // error contours (red): the output error is the same along each ellipse; one passes through each corner
    ctx.save(); ctx.beginPath(); ctx.rect(X(lo), Y(hi), side, side); ctx.clip();
    const levels = costs.map(c => c.cost).sort((a, b) => a - b);
    [levels[0] * .25, levels[0] * .6, ...levels].forEach((L, k) => {
      const pts = KIT.quadEllipse(S.H, S.w, L).map(([a, b]) => [X(a), Y(b)]);
      PE.polyline(ctx, pts, C.red, k >= 2 ? 1.4 + 2 * gh : 1 + gh, k >= 2 ? null : [2, 3], .35 + .25 * (k >= 2) + .4 * gh);
    });
    if (S.trail.length > 1) PE.polyline(ctx, S.trail.map(([a, b]) => [X(a), Y(b)]), C.fg, 1.5, [4, 3], .8);
    ctx.restore();
    // corners
    costs.forEach(c => {
      const x = X(c.p[0]), y = Y(c.p[1]), out = [c.p[0] - .5, c.p[1] - .5];
      PE.dot(ctx, x, y, 6 + 2 * gc, C.yellow);
      if (c.best) PE.ring(ctx, x, y, 12 + 3 * gc, C.green, 2.2);
      if (c.nearest) { ctx.save(); ctx.setLineDash([3, 3]); PE.ring(ctx, x, y, c.best ? 17 : 12, C.yellow, 1.6); ctx.restore(); }
      const tx = x + Math.sign(out[0]) * 16, ty = y - Math.sign(out[1]) * 18 + 4;
      const tag = c.best && c.nearest ? 'nearest = best' : c.best ? 'best' : c.nearest ? 'nearest' : '';
      PE.label(ctx, c.cost.toFixed(3), tx, ty, c.best ? C.green : C.fg, 11.5, out[0] < 0 ? 'right' : 'left', 600);
      if (tag) PE.label(ctx, tag, tx, ty + 13, c.best ? C.green : C.yellow, 10.5, out[0] < 0 ? 'right' : 'left', 400);
    });
    // the rounding error vector (red arrow) from w to the chosen corner, and the moving rounded point
    const tgt = S.pick ? costs.find(c => S.pick === 'best' ? c.best : c.nearest) : null;
    const showD = tgt || (gd > 0 ? costs.find(c => c.nearest) : null);
    if (showD) PE.arrow(ctx, X(S.w[0]), Y(S.w[1]), X(PE.lerp(S.w[0], showD.p[0], tgt ? S.u : 1)), Y(PE.lerp(S.w[1], showD.p[1], tgt ? S.u : 1)), C.red, 2 + 2 * gd, 9, .9);
    S.marks.forEach(m => { PE.dot(ctx, X(m.p[0]), Y(m.p[1]), 5, m.color || C.fg); if (m.label) PE.label(ctx, m.label, X(m.p[0]) + 8, Y(m.p[1]) - 8, m.color || C.fg, 10.5, 'left', 600); });
    PE.dot(ctx, X(S.w[0]), Y(S.w[1]), 7, C.blue); PE.ring(ctx, X(S.w[0]), Y(S.w[1]), 11, C.blue, 1.2, .6);
    PE.label(ctx, 'w (FP32)' + (opts.drag !== false ? ', drag me' : ''), X(S.w[0]) + 12, Y(S.w[1]) + 18, C.blue, 11, 'left', 600);
    // the cost list: one bar per corner
    const lx = narrow ? 14 : ox + side + 34, ly = narrow ? oy + side + 52 : oy + 30, lw = narrow ? w - 28 : w - lx - 14;
    PE.label(ctx, 'output error of each corner, \u0394w^T H \u0394w', lx, ly - 12, gl > 0 ? C.fg : C.mut, 11, 'left', 600);
    const mx = Math.max(...costs.map(c => c.cost)), rowH = narrow ? 20 : 30;
    costs.forEach((c, k) => {
      const y = ly + k * rowH, bx = lx + 70, bw = Math.max(40, lw - 70 - 50);
      PE.label(ctx, `(${c.p[0]}, ${c.p[1]})`, lx, y + 11, c.best ? C.green : C.fg, 11.5, 'left', 600);
      ctx.save(); ctx.fillStyle = C.rule; ctx.fillRect(bx, y + 2, bw, 11); ctx.fillStyle = c.best ? C.green : C.red; ctx.globalAlpha = .85; ctx.fillRect(bx, y + 2, bw * c.cost / mx, 11);
      if (gl > 0) { ctx.globalAlpha = gl; ctx.strokeStyle = C.red; ctx.lineWidth = 2; ctx.strokeRect(bx - 2, y, bw + 4, 15); } ctx.restore();
      PE.label(ctx, c.cost.toFixed(3) + (c.nearest ? ' n' : ''), bx + bw + 6, y + 11, c.best ? C.green : C.fg, 11, 'left', 600);
    });
    if (!narrow) { const hy = ly + 4 * rowH + 18, H = S.H;
      PE.label(ctx, `toy H = [[${H[0][0]}, ${H[0][1]}], [${H[1][0]}, ${H[1][1]}]]`, lx, hy, gh > 0 ? C.red : C.mut, 11, 'left', 600);
      PE.label(ctx, 'n = round to nearest', lx, hy + 16, C.mut, 10.5, 'left', 400); }
  }
  const cv = PE.canvas(el, opts.aspect || (w => w < 560 ? 1.32 : .56), draw);
  let stop = () => {};
  function set(next) {
    if (next.H) S.H = next.H;
    if (next.w) S.w = next.w.slice();
    if ('pick' in next && next.pick !== S.pick) { stop(); S.pick = next.pick; S.u = 0; if (S.pick) stop = KIT.tween(S, { u: 1 }, 700, cv.redraw); else S.u = 0; }
    if (next.trail) S.trail = next.trail; if (next.marks) S.marks = next.marks;
    cv.redraw();
  }
  if (opts.drag !== false) {
    const pos = e => { const r = el.getBoundingClientRect(); return [e.clientX - r.left, e.clientY - r.top]; };
    el.addEventListener('pointerdown', e => { if (!geo) return; const [x, y] = pos(e); if (Math.hypot(x - geo.X(S.w[0]), y - geo.Y(S.w[1])) < 26) { drag = true; el.setPointerCapture(e.pointerId); } });
    el.addEventListener('pointermove', e => { if (!drag) return; const [x, y] = pos(e); S.w = [PE.clamp(geo.ix(x), .03, .97), PE.clamp(geo.iy(y), .03, .97)]; S.u = 1; cv.redraw(); });
    el.addEventListener('pointerup', () => { if (drag) { drag = false; opts.onChange && opts.onChange(); } });
  }
  const info = () => ({ w: S.w, H: S.H, pick: S.pick, costs: KIT.roundCosts(S.w, S.H) });
  return { set, info, state: S, redraw: cv.redraw, resize: cv.resize, destroy: () => { stop(); cv.destroy(); } };
};
KIT.adaroundText = i => {
  const n = i.costs.find(c => c.nearest), b = i.costs.find(c => c.best), p = c => `(${c.p[0]}, ${c.p[1]})`;
  const head = i.H[0][1] === 0 ? 'Independent inputs: H has no off-diagonal term.' : `Correlated inputs: off-diagonal H = ${i.H[0][1]}.`;
  const verdict = n === b ? `Rounding to nearest picks ${p(n)}, which is also the best corner (error ${n.cost.toFixed(3)}).`
    : `Rounding to nearest picks ${p(n)}: output error ${n.cost.toFixed(3)}. The best corner is <b class="c-green">${p(b)}</b>: ${b.cost.toFixed(3)}, ${Math.round(100 * (1 - b.cost / n.cost))}% lower.`;
  const shown = i.pick ? ` Showing: ${i.pick === 'best' ? 'the best corner' : 'round to nearest'}.` : '';
  return `w = (${i.w.map(v => v.toFixed(2)).join(', ')}) grid steps. ${head} ${verdict}${shown}`;
};
