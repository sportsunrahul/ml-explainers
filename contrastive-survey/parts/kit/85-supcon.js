/* ch-supcon toy: 15 images in 3 hidden classes, two views each, as 30 angles on the unit circle. 'self' trains the
   one-positive loss (NT-Xent, KIT.ntxentAngles); 'sup' trains SupCon L_out (Eq. 2): every same-class view is a positive.
   Settings and the numbers the reader sees are in notes/sim-supcon.txt (build/sim/supcon.js). */
KIT.SC = { N: 15, C: 3, TAU: .3, LR: .6, STEPS: 300, SEED: 5, JIT: .5 };
KIT.scInit = () => {
  const { N, SEED, JIT } = KIT.SC, r = PE.rng(SEED), th = [];
  for (let i = 0; i < N; i++) th.push(r() * 2 * Math.PI);
  for (let i = 0; i < N; i++) th.push(th[i] + (r() - .5) * 2 * JIT);
  return th;
};
KIT.scClass = k => (k % KIT.SC.N) % KIT.SC.C;
// Positives of view a: its twin only ('self') or every other view with the same class ('sup').
KIT.scPos = (a, mode) => {
  const { N } = KIT.SC, M = 2 * N, out = [];
  for (let k = 0; k < M; k++) if (k !== a && (mode === 'self' ? k % N === a % N : KIT.scClass(k) === KIT.scClass(a))) out.push(k);
  return out;
};
// L_out (SupCon Eq. 2) on angles: per anchor, the mean of -log softmax over its positives; the denominator is every view but itself.
KIT.supconAngles = (th, mode, tau) => {
  const M = th.length, z = th.map(t => [Math.cos(t), Math.sin(t)]), gz = z.map(() => [0, 0]);
  let L = 0;
  for (let a = 0; a < M; a++) {
    const P = KIT.scPos(a, mode), w = new Array(M).fill(0);
    P.forEach(p => { w[p] = 1 / P.length; });
    let mx = -Infinity; const s = [];
    for (let k = 0; k < M; k++) { s[k] = k === a ? -Infinity : (z[a][0] * z[k][0] + z[a][1] * z[k][1]) / tau; if (s[k] > mx) mx = s[k]; }
    let Z = 0; for (let k = 0; k < M; k++) if (k !== a) Z += Math.exp(s[k] - mx);
    L += mx + Math.log(Z); P.forEach(p => { L -= s[p] / P.length; });
    for (let k = 0; k < M; k++) {
      if (k === a) continue;
      const c = (Math.exp(s[k] - mx) / Z - w[k]) / tau;
      gz[a][0] += c * z[k][0]; gz[a][1] += c * z[k][1];
      gz[k][0] += c * z[a][0]; gz[k][1] += c * z[a][1];
    }
  }
  const g = th.map((t, k) => (-Math.sin(t) * gz[k][0] + Math.cos(t) * gz[k][1]) / M);
  return { loss: L / M, g };
};
KIT.scStep = (th, mode) => { const r = KIT.supconAngles(th, mode, KIT.SC.TAU); for (let k = 0; k < th.length; k++) th[k] -= KIT.SC.LR * r.g[k]; return r.loss; };
// Retrieval check: how many views have a same-class nearest neighbour, not counting their own twin.
KIT.scStats = th => {
  const { N } = KIT.SC, M = 2 * N;
  let hit = 0;
  for (let a = 0; a < M; a++) {
    let best = -2, bk = -1;
    for (let k = 0; k < M; k++) if (k % N !== a % N) { const c = Math.cos(th[a] - th[k]); if (c > best) { best = c; bk = k; } }
    if (KIT.scClass(bk) === KIT.scClass(a)) hit++;
  }
  return { hit, of: M };
};
KIT.scRun = mode => { const th = KIT.scInit(); for (let s = 0; s < KIT.SC.STEPS; s++) KIT.scStep(th, mode); return th; };
// Drawing. Class = shape (circle, square, triangle); view 1 filled on the circle, view 2 hollow just inside it.
// The anchor (view 1 of image 1) is blue, its positives yellow; green lines = the pull to each positive, red ticks = negatives.
KIT.scShape = (ctx, x, y, r, cls, color, filled, lw = 1.8) => {
  ctx.save(); ctx.strokeStyle = color; ctx.fillStyle = color; ctx.lineWidth = lw; ctx.beginPath();
  if (cls === 0) ctx.arc(x, y, r, 0, 7);
  else if (cls === 1) ctx.rect(x - r * .9, y - r * .9, r * 1.8, r * 1.8);
  else { ctx.moveTo(x, y - r * 1.15); ctx.lineTo(x + r * 1.05, y + r * .75); ctx.lineTo(x - r * 1.05, y + r * .75); ctx.closePath(); }
  filled ? ctx.fill() : ctx.stroke(); ctx.restore();
};
KIT.supcon = (canvas, opts = {}) => {
  const { N, C, STEPS } = KIT.SC, M = 2 * N, A = 0, NAMES = ['circles', 'squares', 'triangles'];
  const state = { mode: null, th: KIT.scInit(), step: 0, arrows: 0 };
  const glow = opts.fig ? PE.linkPulses(opts.fig, () => cv.redraw()) : () => 0;
  let cancel = () => {};
  const draw = (ctx, W, H) => {
    if (!W) return;
    const B = PE.css('--blue'), Y = PE.css('--yellow'), G = PE.css('--green'), R = PE.css('--red'), F = PE.css('--fg'), Mu = PE.css('--muted'), RU = PE.css('--rule');
    ctx.clearRect(0, 0, W, H);
    const small = KIT.narrow(W), cx = W / 2, cy = small ? H * .52 : H * .52, rad = Math.min(W * (small ? .36 : .3), H * (small ? .34 : .4));
    PE.ring(ctx, cx, cy, rad, RU, 1.5);
    const rOf = k => k < N ? rad : rad * .86, P = k => [cx + rOf(k) * Math.cos(state.th[k]), cy - rOf(k) * Math.sin(state.th[k])];
    const mode = state.mode, pos = mode ? KIT.scPos(A, mode) : [], isPos = new Set(pos), u = state.arrows;
    const ga = glow('anchor'), gp = glow('pos'), gd = glow('denom'), gv = glow('avg');
    const [ax, ay] = P(A);
    // Red ticks: every view in the denominator that is not a positive is pushed away from the anchor.
    if (mode || gd) for (let k = 0; k < M; k++) {
      if (k === A || isPos.has(k)) continue;
      const [x, y] = P(k), dx = x - ax, dy = y - ay, d = Math.hypot(dx, dy) || 1;
      if (d < rad * 1.1 || gd) PE.arrow(ctx, x, y, x + dx / d * 18, y + dy / d * 18, R, 1.8 + 1.5 * gd, 7, (.75 + .25 * gd) * Math.max(u, gd));
    }
    pos.forEach(k => { const [x, y] = P(k); ctx.save(); ctx.globalAlpha = u; ctx.strokeStyle = G; ctx.lineWidth = 2 + 3 * Math.max(gp, gv);
      ctx.beginPath(); ctx.moveTo(ax, ay); ctx.lineTo(x, y); ctx.stroke(); ctx.restore(); });
    for (let k = M - 1; k >= 0; k--) {
      const [x, y] = P(k), cls = KIT.scClass(k), col = k === A ? B : isPos.has(k) ? Y : Mu;
      KIT.scShape(ctx, x, y, k === A ? 7 : 5.5, cls, col, k < N);
    }
    if (ga) PE.ring(ctx, ax, ay, 14, B, 2 + 2 * ga, ga);
    PE.label(ctx, 'anchor', cx + (rad - 46) * Math.cos(state.th[A]), cy - (rad - 46) * Math.sin(state.th[A]) + 4, B, small ? 11 : 12, 'center');
    if (gv && pos.length) PE.label(ctx, 'each green line weighs 1/|P(i)| = 1/' + pos.length, 12, 56, G, small ? 11 : 12);
    // A class squeezed to one spot reads as one shape, so say how many views sit there.
    for (let c = 0; c < C; c++) {
      const ks = []; for (let k = 0; k < M; k++) if (KIT.scClass(k) === c) ks.push(k);
      const m = Math.atan2(ks.reduce((s, k) => s + Math.sin(state.th[k]), 0), ks.reduce((s, k) => s + Math.cos(state.th[k]), 0));
      const sp = ks.reduce((s, k) => s + Math.abs(Math.atan2(Math.sin(state.th[k] - m), Math.cos(state.th[k] - m))), 0) / ks.length;
      if (sp < .08) { const lx = cx + (rad + 22) * Math.cos(m), ly = cy - (rad + 22) * Math.sin(m) + 4;
        const t = ks.length + ' ' + NAMES[c], fz = small ? 11 : 12; ctx.font = `600 ${fz}px system-ui, sans-serif`;
        const tw = ctx.measureText(t).width, al = Math.cos(m) > .3 ? 'left' : Math.cos(m) < -.3 ? 'right' : 'center';
        const x0 = al === 'left' ? lx : al === 'right' ? lx - tw : lx - tw / 2, dx = PE.clamp(x0, 6, W - 6 - tw) - x0;
        PE.label(ctx, t, lx + dx, ly + (dx ? 18 : 0), F, fz, al); }
    }
    PE.label(ctx, mode === 'self' ? 'one positive per anchor' : mode === 'sup' ? 'all same-class views are positives' : 'before training', 12, 20, Mu, small ? 12 : 13);
    if (state.step) PE.label(ctx, 'step ' + state.step + ' of ' + STEPS, 12, 38, Mu, small ? 11 : 12, 'left', 400);
    if (small) { PE.label(ctx, 'shape = class (hidden from the one-positive loss)', 12, H - 30, Mu, 10, 'left', 400); PE.label(ctx, 'filled = view 1, hollow = view 2', 12, H - 16, Mu, 10, 'left', 400); }
    else PE.label(ctx, 'shape = class (hidden from the one-positive loss); filled = view 1, hollow = view 2', W - 12, H - 12, Mu, 11, 'right', 400);
  };
  const cv = PE.canvas(canvas, w => KIT.narrow(w) ? 1.05 : .6, draw);
  const set = mode => {
    cancel(); state.th = KIT.scInit(); state.mode = mode; state.step = 0; state.arrows = 0;
    if (!mode) { cv.redraw(); return; }
    const run = n => { while (state.step < n) { KIT.scStep(state.th, mode); state.step++; } };
    if (PE.instant()) { state.arrows = 1; run(STEPS); cv.redraw(); return; }
    // First draw the arrows for the chosen loss, then train.
    cancel = PE.animate(3000, u => { state.arrows = Math.min(1, u / .2); run(Math.round(PE.clamp((u - .25) / .75, 0, 1) * STEPS)); cv.redraw(); });
  };
  return { set, state, redraw: cv.redraw, resize: cv.resize, destroy: () => { cancel(); cv.destroy(); }, STEPS };
};
KIT.supconText = mode => {
  const f0 = KIT.scStats(KIT.scInit());
  if (!mode) return 'Before training: 30 views of 15 images at random spots. ' + f0.hit + ' of ' + f0.of + ' views have a same-class nearest neighbour (own twin not counted), about what chance gives.';
  const th = KIT.scRun(mode), s = KIT.scStats(th), P = KIT.scPos(0, mode).length;
  return mode === 'self'
    ? 'One positive per anchor (' + P + ' green line), after ' + KIT.SC.STEPS + ' steps: every view sits on its twin, but only ' + s.hit + ' of ' + s.of + ' views have a same-class nearest neighbour. The loss never saw the classes, so it spread them at random.'
    : 'All same-class views are positives (' + P + ' green lines), after ' + KIT.SC.STEPS + ' steps: ' + s.hit + ' of ' + s.of + ' views have a same-class nearest neighbour. Each class has pulled into one tight group, and the three groups push apart.';
};
