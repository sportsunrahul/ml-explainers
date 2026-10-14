/* ch-negatives toy: one anchor, its positive and 24 negatives on the unit circle, similarity s = cos(angle).
   Four negatives secretly share the anchor's class (false negatives: 4 of 24 = 1/6, as with 6 equal classes).
   Share of the push on negative j: Wang & Liu Eq. 5, with hardness weights e^(beta s / tau) normalised by their mean
   (Robinson et al.; our notation). Debiasing (Chuang et al. Eq. 7-8, M = 1, our notation) changes only the denominator.
   Pure math in KIT.neg*; the numbers the reader sees are in notes/sim-negatives.txt. */
KIT.NEG = { TAU: .5, BETA: 1, TPLUS: 1 / 6, POS: .3, FALSE: [.2, -.3, .42, -.48], HARD: [.6, -.68], REST: 18 };
KIT.negSetup = () => {
  const P = KIT.NEG, a = P.FALSE.map(x => ({ a: x, f: 1 })).concat(P.HARD.map(x => ({ a: x, f: 0 })));
  for (let i = 0; i < P.REST; i++) a.push({ a: 1 + (2 * Math.PI - 2) * i / (P.REST - 1), f: 0 });
  a.forEach(n => { n.s = Math.cos(n.a); });
  return a;
};
// mode: 'plain' | 'hard' | 'debias'. Returns per-negative terms and shares, the denominator pieces and three losses.
KIT.negCompute = mode => {
  const P = KIT.NEG, negs = KIT.negSetup(), N = negs.length, b = mode === 'hard' ? P.BETA : 0;
  const pos = Math.exp(Math.cos(P.POS) / P.TAU);
  const w = negs.map(n => Math.exp(b * n.s / P.TAU)), Zb = w.reduce((t, v) => t + v, 0) / N;
  const e = negs.map((n, j) => w[j] / Zb * Math.exp(n.s / P.TAU)), tot = e.reduce((t, v) => t + v, 0);
  const share = e.map(v => v / tot), sum = f => negs.reduce((t, n, j) => t + (f(n, j) ? share[j] : 0), 0);
  const falseMass = e.reduce((t, v, j) => t + (negs[j].f ? v : 0), 0), trueMass = tot - falseMass, nTrue = negs.filter(n => !n.f).length;
  const estimate = N * P.TPLUS * pos, ng = Math.max((tot - estimate) / (1 - P.TPLUS), N * Math.exp(-1 / P.TAU));
  const L = d => -Math.log(pos / (pos + d));
  return { negs, e, share, pos, tot, falseMass, trueMass, estimate, ng, N,
    shareFalse: sum(n => n.f), shareHard: sum(n => !n.f && Math.abs(n.a) < .9), shareRest: sum(n => Math.abs(n.a) >= .9),
    lossAll: L(tot), lossTrue: L(N * trueMass / nTrue), lossDebiased: L(ng), floored: (tot - estimate) / (1 - P.TPLUS) < N * Math.exp(-1 / P.TAU) };
};

// Drawing: the circle (left, or top when narrow) and three denominator bars (right, or below).
KIT.negatives = (canvas, opts = {}) => {
  const state = { mode: null, fromBars: false, u: 1, from: null, to: KIT.negCompute('plain') };
  const glow = opts.fig ? PE.linkPulses(opts.fig, () => cv.redraw()) : () => 0;
  let cancel = () => {};
  const SH_MAX = .2;
  const draw = (ctx, W, H) => {
    if (!W) return;
    const B = PE.css('--blue'), Y = PE.css('--yellow'), G = PE.css('--green'), R = PE.css('--red'), M = PE.css('--muted'), RU = PE.css('--rule'), FG = PE.css('--fg');
    ctx.clearRect(0, 0, W, H);
    const small = KIT.narrow(W), fs = small ? 11 : 12;
    const cur = state.to, prev = state.from || cur, u = state.u, lerpA = (a, b) => a.map((v, i) => PE.lerp(b ? b[i] : v, v, u));
    const zero = cur.share.map(() => 0), share = lerpA(state.mode ? cur.share : zero, state.fromBars ? prev.share : zero);
    // Circle panel.
    const side = small ? Math.min(W, H * .58) : Math.min(W * .5, H), cx = small ? W / 2 : side / 2 + 4, cy = small ? side / 2 + 6 : H / 2, r = side * .31;
    ctx.save(); ctx.strokeStyle = RU; ctx.lineWidth = 1.5; ctx.beginPath(); ctx.arc(cx, cy, r, 0, 7); ctx.stroke(); ctx.restore();
    const at = (a, rr = r) => [cx + rr * Math.sin(a), cy - rr * Math.cos(a)];
    const gp = glow('push'), gs = glow('share'), gl = glow('pull'), gh = glow('hard');
    const Lmax = side * .16, gb = Math.max(gp, gs, gh);
    cur.negs.forEach((n, j) => {
      const L = Math.min(1, share[j] / SH_MAX) * Lmax, [x0, y0] = at(n.a, r + 7), [x1, y1] = at(n.a, r + 7 + Math.max(L, .5));
      ctx.save(); ctx.strokeStyle = R; ctx.globalAlpha = .85 + .15 * gb; ctx.lineWidth = (small ? 5 : 7) + 3 * gb; ctx.lineCap = 'butt';
      ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x1, y1); ctx.stroke(); ctx.restore();
    });
    cur.negs.forEach(n => { const [x, y] = at(n.a); PE.dot(ctx, x, y, 4.5, R); if (n.f) PE.ring(ctx, x, y, 8, Y, 2); });
    const [ax, ay] = at(0), [px, py] = at(KIT.NEG.POS);
    PE.dot(ctx, px, py, 5.5, Y);
    PE.dot(ctx, ax, ay, 6.5, B);
    // The pull: a green arrow from the positive toward the anchor, just inside the circle.
    const [qx, qy] = at(KIT.NEG.POS, r - 13), [bx, by] = at(.06, r - 13);
    PE.arrow(ctx, qx, qy, bx, by, G, 2.5 + 2 * gl, 8, .9);
    PE.label(ctx, 'blue: the anchor', cx, cy - 14, B, fs, 'center');
    PE.label(ctx, 'yellow: its positive', cx, cy + 4, Y, fs, 'center');
    PE.label(ctx, 'green arrow: the pull', cx, cy + 22, G, fs, 'center');
    const [lx, ly] = at(Math.PI, r + 8);
    PE.label(ctx, 'red bar = share of the push', lx, ly + (small ? 18 : 22), M, fs, 'center', 400);
    PE.label(ctx, 'yellow ring = same class (false negative)', lx, ly + (small ? 32 : 38), M, fs, 'center', 400);
    // Denominator panel: what each loss puts next to the positive term.
    const x0 = small ? 12 : W * .52, x1 = W - 12, top = small ? side + 26 : H * .24, rowH = small ? 46 : Math.min(58, H * .2);
    const scale = (x1 - x0 - 6) / 125;
    PE.label(ctx, 'The denominator: positive term + negative terms', x0, top - 10, FG, fs + 1);
    const rows = [
      { name: 'all 24 negatives', parts: [[cur.pos, Y], [cur.falseMass, R, 1], [cur.trueMass, R]], loss: cur.lossAll, link: 'negsum' },
      { name: 'true negatives only (needs labels)', parts: [[cur.pos, Y], [cur.N * cur.trueMass / (cur.N - KIT.NEG.FALSE.length), R]], loss: cur.lossTrue },
      { name: 'debiased estimate (no labels)', parts: [[cur.pos, Y], [cur.ng, R]], loss: cur.lossDebiased, link: 'subtract', only: 'debias' },
    ];
    rows.forEach((row, i) => {
      const y = top + 8 + i * rowH, off = row.only && state.mode !== row.only, g = row.link ? glow(row.link) : 0;
      PE.label(ctx, row.name + (off ? ' (press Debiased)' : ''), x0, y + 10, off ? M : FG, fs, 'left', 400);
      if (off) return;
      let x = x0;
      row.parts.forEach(([v, c, ring]) => { const w = v * scale; ctx.save(); ctx.globalAlpha = .75 + .25 * g; ctx.fillStyle = c; ctx.fillRect(x, y + 16, Math.max(w - 1, 1), 14);
        if (ring) { ctx.strokeStyle = Y; ctx.lineWidth = 2; ctx.strokeRect(x + 1, y + 17, Math.max(w - 3, 1), 12); } ctx.restore(); x += w; });
      PE.label(ctx, 'loss ' + row.loss.toFixed(2), x1, y + 10, FG, fs, 'right');
      if (i === 0 && (state.mode === 'debias' || glow('subtract'))) {
        const gsub = glow('subtract');
        ctx.save(); ctx.setLineDash([4, 3]); ctx.strokeStyle = G; ctx.lineWidth = 2 + 2 * gsub; ctx.strokeRect(x0 + cur.pos * scale, y + 13, cur.estimate * scale, 20); ctx.restore();
      }
    });
    if (state.mode === 'debias') PE.label(ctx, 'dashed box: the part debiasing subtracts', x0, top + 8 + 3 * rowH - 2, G, fs, 'left', 400);
  };
  const cv = PE.canvas(canvas, w => KIT.narrow(w) ? 1.38 : .56, draw);
  const set = mode => {
    cancel(); state.fromBars = !!state.mode; state.from = state.to; state.to = KIT.negCompute(mode || 'plain'); state.mode = mode; state.u = 0;
    cancel = PE.animate(700, u => { state.u = u; cv.redraw(); });
  };
  return { set, state, redraw: cv.redraw, resize: cv.resize, destroy: () => { cancel(); cv.destroy(); } };
};
KIT.negText = mode => {
  const c = KIT.negCompute(mode || 'plain'), p = x => Math.round(100 * x) + '%', f = x => x.toFixed(2);
  const sh = 'the 4 false negatives take ' + p(c.shareFalse) + ' of the push, the 2 nearest true negatives ' + p(c.shareHard) + ', the other 18 together ' + p(c.shareRest) + '.';
  if (!mode) return 'No bars yet. Press a button to see how the push is shared among the 24 negatives.';
  if (mode === 'plain') return 'Plain InfoNCE at \u03c4 = ' + KIT.NEG.TAU + ': ' + sh;
  if (mode === 'hard') return 'Hardness weights, \u03b2 = ' + KIT.NEG.BETA + ': ' + sh + ' The nearest negatives gain, and here the nearest are mostly false.';
  return 'Debiased, \u03c0\u207a = 1/6: the shares do not move (' + p(c.shareFalse) + ' still on the false negatives). The loss does: ' + f(c.lossAll) + ' with all negatives, ' + f(c.lossTrue) + ' with true negatives only, ' + f(c.lossDebiased) + ' for the debiased estimate.';
};
