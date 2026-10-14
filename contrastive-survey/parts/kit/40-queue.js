/* ch-moco toy: MoCo Eq. 2 in a 2D "weight plane". The query encoder's weights theta_q take noisy SGD steps around a
   slowly drifting trend; the key encoder follows by theta_k <- m theta_k + (1 - m) theta_q. Each step the current key
   encoder makes one batch of keys, recorded as its position; the queue keeps the last SLOTS batches (16 x 256 = K 4096).
   Pure math in KIT.q*; settings and the numbers the reader sees are in notes/sim-moco.txt (build/sim/moco.js). */
KIT.Q = { SLOTS: 16, S: 1, V: .0003, DIR: [.94, .34], SEED: 5, BURN: 60000, TRAIL: 32, RUN: 500, MS: [0, .9, .99, .999, .9999] };
KIT.qInit = m => {
  const { V, DIR, SEED, BURN } = KIT.Q, g = PE.gauss(PE.rng(SEED));
  const c = [-BURN * V * DIR[0], -BURN * V * DIR[1]];
  const st = { m, t: 0, g, c, q: c.slice(), k: c.slice(), queue: [], trail: [] };
  for (let i = 0; i < BURN; i++) KIT.qStep(st);
  st.t = 0;
  return st;
};
// One training step: the key encoder encodes this batch (recorded), SGD moves theta_q, Eq. 2 moves theta_k, enqueue/dequeue.
KIT.qStep = st => {
  const { S, V, DIR, SLOTS, TRAIL } = KIT.Q, m = st.m;
  const key = st.k.slice();
  st.c[0] += V * DIR[0]; st.c[1] += V * DIR[1];
  st.q = [st.c[0] + S * st.g(), st.c[1] + S * st.g()];
  st.k = [m * st.k[0] + (1 - m) * st.q[0], m * st.k[1] + (1 - m) * st.q[1]];
  st.queue.unshift(key); if (st.queue.length > SLOTS) st.queue.pop();
  st.trail.unshift(st.q.slice()); if (st.trail.length > TRAIL) st.trail.pop();
  st.t++;
};
// spread: the largest distance between two key-encoder versions in the queue (inconsistency).
// lag: distance from the key encoder to the average of the query encoder's last SLOTS positions.
KIT.qStats = st => {
  const Q = st.queue, n = KIT.Q.SLOTS; let spread = 0;
  for (let a = 0; a < Q.length; a++) for (let b = a + 1; b < Q.length; b++) spread = Math.max(spread, Math.hypot(Q[a][0] - Q[b][0], Q[a][1] - Q[b][1]));
  const tr = st.trail.slice(0, n), avg = [0, 1].map(d => tr.reduce((s, p) => s + p[d], 0) / tr.length);
  return { spread, lag: Math.hypot(st.k[0] - avg[0], st.k[1] - avg[1]), avg };
};
KIT.qHalf = m => m > 0 ? Math.log(2) / -Math.log(m) : 0;

// Drawing. Left (top on phones): the weight plane. Right (below): the two encoders and the FIFO queue strip.
KIT.queue = (canvas, opts = {}) => {
  const state = { m: .999, st: KIT.qInit(.999), slide: 1, prev: null };
  const glow = opts.fig ? PE.linkPulses(opts.fig, () => cv.redraw()) : () => 0;
  let cancel = () => {};
  const fmtM = m => String(m);
  const draw = (ctx, W, H) => {
    if (!W) return;
    const B = PE.css('--blue'), Y = PE.css('--yellow'), R = PE.css('--red'), Br = PE.css('--brown'), F = PE.css('--fg'), M = PE.css('--muted'), RU = PE.css('--rule');
    ctx.clearRect(0, 0, W, H);
    const nar = KIT.narrow(W), st = state.st, u = state.slide, pv = state.prev;
    const gq = glow('query'), gk = glow('key'), gQ = glow('queue'), gm = glow('momentum'), gp = glow('pos');
    // Panel geometry.
    const P = nar ? { x: 8, y: 8, w: W - 16, h: W - 16 } : { x: 8, y: 8, w: W * .5 - 16, h: H - 16 };
    const D = nar ? { x: 8, y: P.y + P.h + 14, w: W - 16, h: H - P.h - 30 } : { x: W * .5 + 8, y: 8, w: W * .5 - 16, h: H - 16 };
    // ---- weight plane ----
    ctx.save(); ctx.strokeStyle = RU; ctx.lineWidth = 1; ctx.beginPath(); ctx.roundRect(P.x, P.y, P.w, P.h, 8); ctx.stroke(); ctx.clip();
    const lerp2 = (a, b) => [PE.lerp(a[0], b[0], u), PE.lerp(a[1], b[1], u)];
    const qNow = pv ? lerp2(pv.q, st.q) : st.q, kNow = pv ? lerp2(pv.k, st.k) : st.k;
    const stats = KIT.qStats(st), ctr = pv ? lerp2(pv.avg, stats.avg) : stats.avg;
    const half = 3.6, s = Math.min(P.w, P.h) / (2 * half), cxp = P.x + P.w / 2, cyp = P.y + P.h / 2;
    const X = p => cxp + (p[0] - ctr[0]) * s, Yp = p => cyp - (p[1] - ctr[1]) * s;
    ctx.strokeStyle = RU; ctx.beginPath();
    for (let gx = Math.ceil(ctr[0] - half * 2); gx <= ctr[0] + half * 2; gx++) { ctx.moveTo(X([gx, 0]), P.y); ctx.lineTo(X([gx, 0]), P.y + P.h); }
    for (let gy = Math.ceil(ctr[1] - half * 2); gy <= ctr[1] + half * 2; gy++) { ctx.moveTo(P.x, Yp([0, gy])); ctx.lineTo(P.x + P.w, Yp([0, gy])); }
    ctx.globalAlpha = .5; ctx.stroke(); ctx.globalAlpha = 1;
    // query trail: where SGD has taken theta_q over the last steps
    st.trail.forEach((p, i) => { if (i) PE.dot(ctx, X(p), Yp(p), 2.6, B, .55 * (1 - i / st.trail.length)); });
    // queue: the key-encoder versions that made the keys now in the queue
    const qd = st.queue;
    qd.forEach((p, i) => PE.dot(ctx, X(p), Yp(p), 3.4 + 2 * gQ, Br, .35 + .55 * (1 - i / qd.length)));
    if (qd.length > 1) {
      const cm = [0, 1].map(d => qd.reduce((a, p) => a + p[d], 0) / qd.length);
      const rr = Math.max(...qd.map(p => Math.hypot(p[0] - cm[0], p[1] - cm[1]))) * s + 6;
      PE.ring(ctx, X(cm), Yp(cm), rr, Br, 1.2 + 2 * gQ, .7);
      PE.label(ctx, 'queue keys', Math.max(P.x + 74, X(cm) - rr * .72 - 4), Math.max(P.y + 54, Yp(cm) - rr * .72 - 4), Br, nar ? 11 : 12, 'right');
    }
    // momentum pull: key encoder moves (1 - m) of the way toward theta_q each step
    PE.polyline(ctx, [[X(kNow), Yp(kNow)], [X(qNow), Yp(qNow)]], Br, 1.2 + 2.5 * gm, [4, 4], .55 + .45 * gm);
    PE.dot(ctx, X(kNow), Yp(kNow), 7 + 3 * gk, Br);
    PE.label(ctx, 'key encoder', X(kNow), Yp(kNow) + 24, Br, nar ? 11 : 12, 'center');
    PE.dot(ctx, X(qNow), Yp(qNow), 6 + 3 * gq, B); PE.ring(ctx, X(qNow), Yp(qNow), 9 + 3 * gq, R, 2);
    PE.label(ctx, 'query encoder', X(qNow) + 13, Yp(qNow) - 10, B, nar ? 11 : 12);
    ctx.restore();
    PE.label(ctx, 'weight plane (toy)', P.x + 10, P.y + 18, M, nar ? 11 : 12);
    PE.label(ctx, 'm = ' + fmtM(state.m) + ',  step ' + st.t, P.x + 10, P.y + 34, F, nar ? 11 : 12, 'left', 400);
    // ---- encoders and queue strip ----
    const fs = nar ? 11 : 12;
    const bw = Math.min(D.w * .36, 170), bh = 46, by = D.y + (nar ? 4 : D.h * .16);
    const qb = { x: D.x + 4, y: by, w: bw, h: bh }, kb = { x: D.x + D.w - bw - 4, y: by, w: bw, h: bh };
    PE.box(ctx, qb.x, qb.y, qb.w, qb.h, B, 'query encoder f_q\ntrained by SGD');
    ctx.save(); ctx.strokeStyle = R; ctx.lineWidth = 2 + 2 * gq; ctx.beginPath(); ctx.roundRect(qb.x - 3, qb.y - 3, qb.w + 6, qb.h + 6, 10); ctx.stroke(); ctx.restore();
    PE.box(ctx, kb.x, kb.y, kb.w, kb.h, Br, 'key encoder f_k\nmomentum copy', 1);
    if (gk) { ctx.save(); ctx.strokeStyle = Br; ctx.lineWidth = 3 * gk; ctx.beginPath(); ctx.roundRect(kb.x - 3, kb.y - 3, kb.w + 6, kb.h + 6, 10); ctx.stroke(); ctx.restore(); }
    PE.arrow(ctx, qb.x + qb.w + 6, by + bh / 2, kb.x - 6, by + bh / 2, Br, 1.6 + 2 * gm, 8, .8 + .2 * gm);
    PE.label(ctx, 'momentum m', (qb.x + qb.w + kb.x) / 2, by + bh / 2 - 8, Br, nar ? 10 : 11, 'center');
    PE.label(ctx, '(no gradient)', (qb.x + qb.w + kb.x) / 2, by + bh / 2 + 16, M, nar ? 10 : 11, 'center', 400);
    // queue strip
    const n = KIT.Q.SLOTS, sy = by + bh + (nar ? 58 : D.h * .28), sw = (D.w - 8) / n, sh = Math.min(30, sw * 1.4), sx = D.x + 4;
    // positive key token from the key encoder: k+ is the current batch's key, enqueued after the step
    const tokX0 = kb.x + kb.w / 2, tokY0 = by + bh + 16, tokX1 = sx + sw / 2, tokY1 = sy + sh / 2;
    const tk = pv ? u : 1;
    PE.polyline(ctx, [[tokX0, by + bh + 4], [tokX0, sy - 12], [sx + sw / 2, sy - 12]], Br, 1.2, null, .6);
    PE.arrow(ctx, sx + sw / 2, sy - 12, sx + sw / 2, sy - 2, Br, 1.2, 6, .6);
    if (pv && u < 1) { const tx = PE.lerp(tokX0, tokX1, tk), ty = PE.lerp(tokY0, tokY1, tk); PE.ring(ctx, tx, ty, 8, Y, 2.4); }
    else PE.ring(ctx, tokX0, tokY0, 8 + 3 * gp, Y, 2.4 + 2 * gp);
    PE.label(ctx, 'k+', tokX0 + 13, tokY0 + 4, Y, fs);
    for (let i = 0; i < n; i++) {
      const off = pv ? (1 - u) * -sw : 0, x = sx + i * sw + off, filled = i < qd.length;
      if (x < sx - sw * .5) continue;
      const a = filled ? .25 + .6 * (1 - i / n) : .08;
      ctx.save(); ctx.globalAlpha = a; ctx.fillStyle = Br; ctx.beginPath(); ctx.roundRect(x + 1.5, sy, sw - 3, sh, 3); ctx.fill(); ctx.restore();
      if (gQ) { ctx.save(); ctx.strokeStyle = Br; ctx.lineWidth = 2 * gQ; ctx.beginPath(); ctx.roundRect(x + 1.5, sy, sw - 3, sh, 3); ctx.stroke(); ctx.restore(); }
    }
    if (pv && u < 1) { ctx.save(); ctx.globalAlpha = 1 - u; ctx.fillStyle = Br; ctx.beginPath(); ctx.roundRect(sx + (n - 1) * sw + u * sw * .3 + 1.5, sy, sw - 3, sh, 3); ctx.fill(); ctx.restore(); }
    PE.label(ctx, 'newest', sx, sy + sh + 15, M, fs);
    PE.label(ctx, 'oldest: leaves next', sx + n * sw, sy + sh + 15, M, fs, 'right');
    PE.label(ctx, 'queue: ' + n + ' batches of keys (negatives)', sx, sy - 22 - (nar ? 0 : 4), Br, fs);
    const sv = KIT.qStats(st);
    const ly = sy + sh + (nar ? 40 : 48);
    PE.label(ctx, 'queue spread ' + sv.spread.toFixed(3), sx, ly, Br, fs);
    PE.label(ctx, 'key lag ' + sv.lag.toFixed(2), sx, ly + 18, Br, fs, 'left', 400);
    PE.label(ctx, 'half catch-up: ' + (state.m ? Math.round(KIT.qHalf(state.m)) + ' steps' : 'instant'), sx, ly + 36, M, fs, 'left', 400);
  };
  const cv = PE.canvas(canvas, w => KIT.narrow(w) ? 1 + 250 / w : .52, draw);
  const snap = st => ({ q: st.q.slice(), k: st.k.slice(), avg: KIT.qStats(st).avg });
  const setM = m => { cancel(); state.m = m; state.prev = null; state.slide = 1; const from = snap(state.st); state.st = KIT.qInit(m);
    state.prev = from; state.slide = 0; cancel = PE.animate(900, v => { state.slide = v; cv.redraw(); }, () => { state.prev = null; cv.redraw(); }); };
  const step = () => { cancel(); state.prev = snap(state.st); KIT.qStep(state.st); state.slide = 0;
    cancel = PE.animate(700, v => { state.slide = v; cv.redraw(); }, () => { state.prev = null; cv.redraw(); }); };
  const run = () => { cancel(); state.prev = null; state.slide = 1; const t0 = state.st.t, N = KIT.Q.RUN;
    const go = n => { while (state.st.t < t0 + n) KIT.qStep(state.st); };
    if (PE.instant()) { go(N); cv.redraw(); return; }
    cancel = PE.animate(2500, v => { go(Math.round(v * N)); cv.redraw(); }); };
  const set = o => { if (o.m !== undefined) setM(o.m); if (o.step) step(); if (o.run) run(); };
  return { set, state, redraw: cv.redraw, resize: cv.resize, destroy: () => { cancel(); cv.destroy(); } };
};
// Readout HTML, written once per click.
KIT.queueText = (st, m, what) => {
  const r = KIT.qStats(st), sp = r.spread.toFixed(3), lg = r.lag.toFixed(2), h = Math.round(KIT.qHalf(m));
  const lead = what === 'step' ? 'One step: the key encoder made this batch\'s keys, they entered the queue, and the oldest batch left. '
    : what === 'run' ? KIT.Q.RUN + ' more steps. ' : '';
  if (m === 0) return lead + 'm = 0: the key encoder is replaced by the query encoder at every step, so the <span data-claim="moco-derived-k4096-batches">16</span> batches in the queue came from <span data-claim="moco-derived-k4096-batches">16</span> different encoders, up to ' + sp + ' apart (toy units). The lineup mixes keys made by different networks.';
  return lead + 'm = ' + m + ': the key encoder closes half of its gap to the query encoder in about ' + h + ' steps. The <span data-claim="moco-derived-k4096-batches">16</span> queued batches came from versions at most ' + sp + ' apart, and the key encoder trails the query encoder\'s recent average by ' + lg + ' (toy units).';
};
