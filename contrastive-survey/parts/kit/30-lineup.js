/* ch-infonce toy: one anchor, N candidates (the partner first, then N - 1 negatives), the softmax over sim / tau,
   the loss -log p(partner), and the ceiling log N of the MI bound. Pure math in KIT.lineupCalc (needs only PE.rng and
   PE.softmax); the default state (N = 4, tau = 1, easy) is the chapter's worked example. */
KIT.LINEUP = { POS: .8, EASY: [.3, .1, -.2], HARD: [.7, .6, .5], NMAX: 64, SEED: 7, LMAX: 4.5 };
// Similarities of all NMAX candidates to the anchor. Easy negatives sit in [-0.4, 0.3], hard ones in [0.45, 0.78].
KIT.lineupSims = mode => {
  const { POS, EASY, HARD, NMAX, SEED } = KIT.LINEUP, r = PE.rng(SEED), s = [POS].concat(mode === 'hard' ? HARD : EASY);
  while (s.length < NMAX) { const u = r(); s.push(mode === 'hard' ? .45 + .33 * u : -.4 + .7 * u); }
  return s;
};
KIT.lineupCalc = (mode, N, tau) => {
  const sims = KIT.lineupSims(mode).slice(0, N), p = PE.softmax(sims.map(x => x / tau)), loss = -Math.log(p[0]), logN = Math.log(N);
  return { sims, p, loss, logN, bound: logN - loss };
};
KIT.lineup = (canvas, opts = {}) => {
  const { NMAX, LMAX } = KIT.LINEUP;
  const state = { mode: 'easy', N: 4, tau: 1 };
  const disp = { sim: KIT.lineupSims('easy'), p: new Array(NMAX).fill(0), loss: 0, logN: 0 };
  const glow = opts.fig ? PE.linkPulses(opts.fig, () => cv.redraw()) : () => 0;
  let cancel = () => {};
  const target = () => { const c = KIT.lineupCalc(state.mode, state.N, state.tau), sim = KIT.lineupSims(state.mode), p = new Array(NMAX).fill(0);
    c.p.forEach((x, j) => { p[j] = x; }); return { sim, p, loss: c.loss, logN: c.logN }; };
  Object.assign(disp, target());
  // A small "photo" glyph: the mug for the anchor, the partner and hard negatives; other shapes for easy negatives.
  const thumb = (ctx, cx, cy, s, kind, j, col) => {
    ctx.save(); ctx.strokeStyle = col; ctx.fillStyle = col; ctx.lineWidth = 1.6;
    if (kind === 'mug') {
      const w = s * (.5 + .12 * Math.sin(j * 2.1)), h = s * (.55 + .1 * Math.cos(j * 1.3));
      ctx.strokeRect(cx - w / 2 - s * .08, cy - h / 2, w, h);
      ctx.beginPath(); ctx.arc(cx + w / 2 - s * .08, cy, h * .22, -Math.PI / 2, Math.PI / 2); ctx.stroke();
    } else {
      const k = j % 3; ctx.beginPath();
      if (k === 0) ctx.arc(cx, cy, s * .3, 0, 7);
      else if (k === 1) { ctx.moveTo(cx, cy - s * .32); ctx.lineTo(cx + s * .32, cy + s * .26); ctx.lineTo(cx - s * .32, cy + s * .26); ctx.closePath(); }
      else { ctx.moveTo(cx - s * .34, cy + s * .2); ctx.lineTo(cx - s * .1, cy - s * .2); ctx.lineTo(cx + s * .08, cy + s * .05); ctx.lineTo(cx + s * .34, cy - s * .25); }
      ctx.stroke();
    }
    ctx.restore();
  };
  const card = (ctx, x, y, w, h, col, g = 0) => {
    ctx.save(); ctx.fillStyle = PE.css('--panel'); ctx.strokeStyle = col; ctx.lineWidth = 1.5 + 2.5 * g;
    ctx.beginPath(); ctx.roundRect(x, y, w, h, 6); ctx.fill(); ctx.stroke(); ctx.restore();
  };
  const draw = (ctx, W, H) => {
    if (!W) return;
    const B = PE.css('--blue'), Y = PE.css('--yellow'), G = PE.css('--green'), R = PE.css('--red'), M = PE.css('--muted'), F = PE.css('--fg'), RU = PE.css('--rule');
    ctx.clearRect(0, 0, W, H);
    const small = KIT.narrow(W), N = state.N, fs = small ? 10 : 11;
    const gA = glow('anchor'), gP = glow('partner'), gL = glow('lineup'), gS = glow('sim'), gT = glow('temp'), gLo = glow('loss'), gC = glow('ceiling'), gB = glow('bound');
    // Regions.
    const aw = small ? 0 : Math.min(118, W * .15), gw = small ? 0 : Math.min(150, W * .2);
    const x0 = small ? 24 : aw + 26, x1 = small ? W - 12 : W - gw - 18, colW = (x1 - x0) / N;
    const showCards = colW >= 26;
    const yTop = small ? 78 : 28, ch = showCards ? Math.min(44, colW * .9) : 10;
    const ySim = yTop + ch + 30, gaugeH = small ? 88 : 0, hs = (H - ySim - gaugeH - 64) * .45, yP = ySim + hs + 34, hp = H - yP - gaugeH - 26;
    // Anchor card.
    if (small) {
      card(ctx, 12, 8, 60, 48, B, gA); thumb(ctx, 42, 32, 40, 'mug', 0, B);
      PE.label(ctx, 'anchor u', 82, 26, B, 12); PE.label(ctx, 'which of the ' + N + ' is its partner?', 82, 44, M, 11, 'left', 400);
    } else {
      const ay = yTop, ah = Math.min(aw, 92);
      card(ctx, 12, ay, aw - 12, ah, B, gA); thumb(ctx, 12 + (aw - 12) / 2, ay + ah / 2, ah * .7, 'mug', 0, B);
      PE.label(ctx, 'anchor u', 12 + (aw - 12) / 2, ay + ah + 16, B, 12, 'center');
      PE.label(ctx, 'one view', 12 + (aw - 12) / 2, ay + ah + 31, M, 11, 'center', 400);
      PE.arrow(ctx, aw + 2, yTop + ch / 2, x0 - 4, yTop + ch / 2, M, 1.4, 7, .7);
    }
    // Candidate cards.
    for (let j = 0; j < N; j++) {
      const cx = x0 + colW * (j + .5), col = j === 0 ? Y : R, gj = j === 0 ? Math.max(gP, gL) : gL;
      if (showCards) { const w = Math.min(colW * .86, 52); card(ctx, cx - w / 2, yTop, w, ch, col, gj);
        thumb(ctx, cx, yTop + ch / 2, Math.min(w, ch) * .8, j === 0 || state.mode === 'hard' ? 'mug' : 'shape', j, j === 0 ? Y : M); }
      else { ctx.save(); ctx.globalAlpha = .8; ctx.fillStyle = col; ctx.fillRect(cx - colW * .35, yTop, colW * .7, ch); ctx.restore();
        if (gj) PE.ring(ctx, cx, yTop + ch / 2, colW * .5 + 2, col, 1.5, gj); }
    }
    PE.label(ctx, 'partner', x0 + colW * .5, yTop - 6, Y, fs, colW < 50 ? 'left' : 'center');
    if (N > 1) PE.label(ctx, (state.mode === 'hard' ? 'hard' : 'easy') + ' negatives: ' + (N - 1), x1, yTop - 6, R, fs, 'right', 400);
    // Similarity bars, range -1..1.
    const zy = ySim + hs / 2, sy = v => zy - v * hs / 2;
    PE.label(ctx, 'similarity sim(u, v_j)', x0, ySim - 10, M, fs);
    ctx.save(); ctx.strokeStyle = RU; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(x0, zy); ctx.lineTo(x1, zy);
    ctx.moveTo(x0, sy(1)); ctx.lineTo(x1, sy(1)); ctx.moveTo(x0, sy(-1)); ctx.lineTo(x1, sy(-1)); ctx.stroke(); ctx.restore();
    if (!small || colW >= 30) { PE.label(ctx, '1', x0 - 5, sy(1) + 4, M, 10, 'right', 400); PE.label(ctx, '0', x0 - 5, zy + 4, M, 10, 'right', 400); PE.label(ctx, '-1', x0 - 5, sy(-1) + 4, M, 10, 'right', 400); }
    for (let j = 0; j < N; j++) {
      const cx = x0 + colW * (j + .5), bw = Math.max(2, colW * .62), v = disp.sim[j], col = j === 0 ? Y : R;
      ctx.save(); ctx.globalAlpha = .55 + .45 * Math.max(gS, j === 0 ? gP : gL); ctx.fillStyle = col;
      ctx.fillRect(cx - bw / 2, Math.min(zy, sy(v)), bw, Math.abs(sy(v) - zy)); ctx.restore();
      if (colW >= 34) PE.label(ctx, v.toFixed(2), cx, v >= 0 ? sy(v) - 4 : sy(v) + 12, j === 0 ? Y : M, 10, 'center', 400);
    }
    // Softmax probabilities, range 0..1.
    const py = v => yP + hp - v * hp;
    PE.label(ctx, 'softmax of sim / \u03c4 (\u03c4 = ' + state.tau.toFixed(2) + ')', x0, yP - 10, gT ? F : M, fs + gT * 2, 'left', gT ? 700 : 600);
    ctx.save(); ctx.strokeStyle = RU; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(x0, py(0)); ctx.lineTo(x1, py(0)); ctx.moveTo(x0, py(1)); ctx.lineTo(x1, py(1)); ctx.stroke(); ctx.restore();
    PE.label(ctx, '1', x0 - 5, py(1) + 4, M, 10, 'right', 400); PE.label(ctx, '0', x0 - 5, py(0) + 4, M, 10, 'right', 400);
    for (let j = 0; j < N; j++) {
      const cx = x0 + colW * (j + .5), bw = Math.max(2, colW * .62), v = disp.p[j], col = j === 0 ? Y : R;
      ctx.save(); ctx.globalAlpha = j === 0 ? 1 : .7 + .3 * gL; ctx.fillStyle = col; ctx.fillRect(cx - bw / 2, py(v), bw, py(0) - py(v)); ctx.restore();
      if (j === 0 && gP) PE.ring(ctx, cx, py(v), 8, Y, 2, gP);
      const hi = v > .85, lx = hi ? cx + bw / 2 + 4 : cx + (j === 0 && colW < 34 ? -bw / 2 : 0), ly = hi ? py(v) + 12 : py(v) - 5;
      if (colW >= 34 || j === 0) PE.label(ctx, (j === 0 ? 'p = ' : '') + v.toFixed(2), lx, ly, j === 0 ? Y : M, 10, hi || (j === 0 && colW < 34) ? 'left' : 'center', j === 0 ? 700 : 400);
    }
    // The loss gauge: L, the log N line (chance level and the ceiling of the bound), the bound log N - L.
    const L = disp.loss, lg = disp.logN;
    if (!small) {
      const gx = W - gw + 34, gt = yTop + 14, gb = H - 36, gy = v => gb - Math.min(v, LMAX) / LMAX * (gb - gt), bw = 24;
      PE.label(ctx, 'loss (nats)', gx + bw / 2, yTop - 6, M, fs, 'center');
      ctx.save(); ctx.strokeStyle = RU; ctx.beginPath(); ctx.moveTo(gx - 6, gt); ctx.lineTo(gx - 6, gb); ctx.stroke(); ctx.restore();
      for (let t = 0; t <= 4; t++) PE.label(ctx, String(t), gx - 10, gy(t) + 4, M, 10, 'right', 400);
      ctx.save(); ctx.fillStyle = R; ctx.globalAlpha = .85 + .15 * gLo; ctx.fillRect(gx, gy(L), bw, gb - gy(L)); ctx.restore();
      if (gLo) PE.ring(ctx, gx + bw / 2, gy(L), 10, R, 2, gLo);
      PE.label(ctx, 'L = ' + L.toFixed(2), gx + bw / 2, gb + 18, R, 11, 'center');
      PE.polyline(ctx, [[gx - 6, gy(lg)], [W - 10, gy(lg)]], F, 1.6 + 2 * gC, [5, 4], .8 + .2 * gC);
      PE.label(ctx, 'log N = ' + lg.toFixed(2), W - 10, gy(lg) - 5, F, 10 + gC, 'right');
      if (lg - L > .02) { const bx = gx + bw + 14; ctx.save(); ctx.strokeStyle = G; ctx.lineWidth = 2 + 2 * gB;
        ctx.beginPath(); ctx.moveTo(bx - 5, gy(lg)); ctx.lineTo(bx, gy(lg)); ctx.lineTo(bx, gy(L)); ctx.lineTo(bx - 5, gy(L)); ctx.stroke(); ctx.restore();
        const my = gy(L) - gy(lg) > 30 ? (gy(lg) + gy(L)) / 2 : gy(L) + 12;
        PE.label(ctx, 'bound', bx + 5, my, G, 10); PE.label(ctx, (lg - L).toFixed(2), bx + 5, my + 12, G, 10); }
    } else {
      const gl = 12, gr = W - 12, gy0 = H - 54, gx = v => gl + Math.min(v, LMAX) / LMAX * (gr - gl), bh = 16;
      PE.label(ctx, 'loss (nats): ', gl, gy0 - 22, M, 11); PE.label(ctx, 'L = ' + L.toFixed(2), gl + 72, gy0 - 22, R, 11);
      ctx.save(); ctx.fillStyle = R; ctx.fillRect(gl, gy0 - bh / 2, gx(L) - gl, bh); ctx.restore();
      if (gLo) PE.ring(ctx, gx(L), gy0, 10, R, 2, gLo);
      for (let t = 0; t <= 4; t++) PE.label(ctx, String(t), gx(t), gy0 + 24, M, 10, 'center', 400);
      PE.polyline(ctx, [[gx(lg), gy0 - 18], [gx(lg), gy0 + 12]], F, 1.6 + 2 * gC, [4, 3]);
      PE.label(ctx, 'log N = ' + lg.toFixed(2), PE.clamp(gx(lg), 40, gr - 40), gy0 + 40, F, 10, 'center');
      if (lg - L > .02) { ctx.save(); ctx.strokeStyle = G; ctx.lineWidth = 2 + 2 * gB; ctx.beginPath(); ctx.moveTo(gx(L), gy0 + 11); ctx.lineTo(gx(lg), gy0 + 11); ctx.stroke(); ctx.restore(); }
    }
  };
  const cv = PE.canvas(canvas, w => KIT.narrow(w) ? 1.3 : .52, draw);
  const set = (patch = {}) => {
    cancel(); Object.assign(state, patch);
    const to = target(), from = { sim: disp.sim.slice(), p: disp.p.slice(), loss: disp.loss, logN: disp.logN };
    cancel = PE.animate(600, u => {
      for (let j = 0; j < NMAX; j++) { disp.sim[j] = PE.lerp(from.sim[j], to.sim[j], u); disp.p[j] = PE.lerp(from.p[j], to.p[j], u); }
      disp.loss = PE.lerp(from.loss, to.loss, u); disp.logN = PE.lerp(from.logN, to.logN, u); cv.redraw();
    });
  };
  return { set, state, redraw: cv.redraw, resize: cv.resize, destroy: () => { cancel(); cv.destroy(); } };
};
// Readout text, written once per change.
KIT.lineupText = st => {
  const c = KIT.lineupCalc(st.mode, st.N, st.tau), f = x => x.toFixed(2);
  const where = c.loss > c.logN ? 'The loss is above log N: worse than a blind guess, so the bound gives nothing.'
    : c.bound < .1 * c.logN ? 'The loss sits near log N: the scores barely beat a blind guess.'
    : c.loss < .05 ? 'The loss is almost 0, so the bound is pinned at its ceiling, log N.'
    : 'The bound log N - L is ' + f(c.bound) + ' nats.';
  return 'N = ' + st.N + ', \u03c4 = ' + f(st.tau) + ', ' + st.mode + ' negatives: p(partner) = ' + f(c.p[0]) + ', loss L = ' + f(c.loss) + ' nats, log N = ' + f(c.logN) + '. ' + where;
};
