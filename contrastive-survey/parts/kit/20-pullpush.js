/* ch-compare toy: a shared linear encoder W (2 x D) maps two noisy views of each of 8 "images" (points in R^D) to the plane.
   'pull' trains on the similar-pair term of the margin loss only; 'both' adds the dissimilar-pair term (Hadsell 2006, Eq. 4).
   Pure math in KIT.pp*; settings and the numbers the reader sees are in notes/sim-pullpush.txt. */
KIT.PP = { N: 8, D: 4, NOISE: .25, M: 1, LR: .4, SEED: 11 };
KIT.ppData = () => {
  const { N, D, NOISE, SEED } = KIT.PP, g = PE.gauss(PE.rng(SEED)), X = [], V1 = [], V2 = [];
  for (let i = 0; i < N; i++) { const x = []; for (let d = 0; d < D; d++) x.push(g()); X.push(x); }
  X.forEach(x => { V1.push(x.map(v => v + NOISE * g())); V2.push(x.map(v => v + NOISE * g())); });
  const W = [0, 1].map(() => Array.from({ length: D }, () => .5 * g()));
  return { V1, V2, W };
};
KIT.ppEmbed = (W, v) => [W[0].reduce((s, w, d) => s + w * v[d], 0), W[1].reduce((s, w, d) => s + w * v[d], 0)];
// One gradient step on W for the chosen mode; returns { loss, pairDist, spread } measured before the step.
KIT.ppStep = (st, mode) => {
  const { N, D, M, LR } = KIT.PP, { V1, V2, W } = st, V = V1.concat(V2), img = k => k % N;
  const G = [new Array(D).fill(0), new Array(D).fill(0)];
  let loss = 0, pd = 0, sp = 0, nNeg = 0;
  const add = (u, v, c) => { const diff = u.map((x, d) => x - v[d]), e = KIT.ppEmbed(W, diff);
    for (let r = 0; r < 2; r++) for (let d = 0; d < D; d++) G[r][d] += c * e[r] * diff[d]; };
  for (let a = 0; a < 2 * N; a++) for (let b = a + 1; b < 2 * N; b++) {
    const ea = KIT.ppEmbed(W, V[a]), eb = KIT.ppEmbed(W, V[b]), Dw = Math.hypot(ea[0] - eb[0], ea[1] - eb[1]);
    if (img(a) === img(b)) { loss += .5 * Dw * Dw; pd += Dw; add(V[a], V[b], 1 / N); }
    else { sp += Dw; nNeg++;
      if (mode === 'both' && Dw < M) { loss += .5 * (M - Dw) ** 2; add(V[a], V[b], -(M - Dw) / Math.max(Dw, 1e-6) / N); } }
  }
  for (let r = 0; r < 2; r++) for (let d = 0; d < D; d++) W[r][d] -= LR * G[r][d];
  return { loss: loss / N, pairDist: pd / N, spread: sp / nNeg };
};
// Drawing: the embedding plane. Blue dots = view 1, yellow rings = view 2, green line = the pair, red ring = margin m around image 1.
KIT.pullpush = (canvas, opts = {}) => {
  const STEPS = 240, SPAN = 4.6;
  const state = { mode: null, step: 0, st: KIT.ppData(), last: null };
  const glow = opts.fig ? PE.linkPulses(opts.fig, () => cv.redraw()) : () => 0;
  let cancel = () => {};
  const draw = (ctx, W, H) => {
    if (!W) return;
    const B = PE.css('--blue'), Y = PE.css('--yellow'), G = PE.css('--green'), R = PE.css('--red'), M = PE.css('--muted'), RU = PE.css('--rule');
    ctx.clearRect(0, 0, W, H);
    const v = PE.view(W, H, SPAN);
    ctx.save(); ctx.strokeStyle = RU; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(0, v.Y(0)); ctx.lineTo(W, v.Y(0)); ctx.moveTo(v.X(0), 0); ctx.lineTo(v.X(0), H); ctx.stroke(); ctx.restore();
    const { V1, V2, W: Wm } = state.st, e1 = V1.map(x => KIT.ppEmbed(Wm, x)), e2 = V2.map(x => KIT.ppEmbed(Wm, x));
    const gm = glow('margin'), gp = glow('pull'), gd = glow('dist'), gn = glow('push');
    if (state.mode === 'both' || gm) PE.ring(ctx, v.X(e1[0][0]), v.Y(e1[0][1]), KIT.PP.M * v.s, R, 1.5 + 3 * gm, .55 + .45 * gm);
    if (state.mode === 'both' || gn) e1.forEach((p, i) => { if (i) { const dx = p[0] - e1[0][0], dy = p[1] - e1[0][1], d = Math.hypot(dx, dy) || 1;
      if (d < KIT.PP.M * 1.6 || gn) PE.arrow(ctx, v.X(p[0]), v.Y(p[1]), v.X(p[0] + dx / d * .35), v.Y(p[1] + dy / d * .35), R, 1.5, 7, .5 + .5 * gn); } });
    e1.forEach((p, i) => { const q = e2[i]; ctx.save(); ctx.strokeStyle = G; ctx.lineWidth = 2 + 3 * Math.max(gp, gd); ctx.beginPath(); ctx.moveTo(v.X(p[0]), v.Y(p[1])); ctx.lineTo(v.X(q[0]), v.Y(q[1])); ctx.stroke(); ctx.restore(); });
    e2.forEach(q => PE.ring(ctx, v.X(q[0]), v.Y(q[1]), 7, Y, 2.2));
    e1.forEach((p, i) => { PE.dot(ctx, v.X(p[0]), v.Y(p[1]), 4.5, B); PE.label(ctx, String(i + 1), v.X(p[0]) + 8, v.Y(p[1]) - 7, M, 11); });
    const small = KIT.narrow(W);
    PE.label(ctx, state.mode === 'pull' ? 'pull only' : state.mode === 'both' ? 'pull + push (margin m = 1)' : 'before training', 12, 20, M, small ? 12 : 13);
    if (state.step) PE.label(ctx, 'step ' + state.step + ' of ' + STEPS, 12, 38, M, small ? 11 : 12, 'left', 400);
  };
  const cv = PE.canvas(canvas, w => KIT.narrow(w) ? .9 : .5, draw);
  const set = mode => {
    cancel(); state.st = KIT.ppData(); state.mode = mode; state.step = 0; state.last = null;
    if (!mode) { cv.redraw(); return; }
    const run = n => { while (state.step < n) { state.last = KIT.ppStep(state.st, mode); state.step++; } };
    if (PE.instant()) { run(STEPS); cv.redraw(); return; }
    cancel = PE.animate(3000, u => { run(Math.round(u * STEPS)); cv.redraw(); });
  };
  return { set, state, redraw: cv.redraw, resize: cv.resize, destroy: () => { cancel(); cv.destroy(); }, STEPS };
};
// Readout text, measured on the finished run so it is written once per click.
KIT.pullpushText = mode => {
  if (!mode) return 'Before training: a random encoder. Partners sit a little apart; different images are scattered.';
  const st = KIT.ppData(); let r; for (let s = 0; s < 240; s++) r = KIT.ppStep(st, mode);
  const f = x => x.toFixed(2);
  return mode === 'pull'
    ? 'Pull only, after 240 steps: partners ' + f(r.pairDist) + ' apart, different images ' + f(r.spread) + ' apart on average. The encoder shrank to almost nothing, so every image lands on one spot.'
    : 'Pull and push, after 240 steps: partners ' + f(r.pairDist) + ' apart, different images ' + f(r.spread) + ' apart on average, about ' + Math.round(r.spread / r.pairDist) + ' times farther. A linear encoder cannot remove all the view noise, but it can no longer collapse.';
};
