/* ch-beyond toy: SwAV's Sinkhorn-Knopp step on a made-up score matrix (K = 4 prototypes x B = 8 images, the paper's
   K x B orientation of Q). Follows the App. A.1 pseudocode: Q = exp(scores / eps), Q /= sum(Q), then per iteration
   scale rows to 1/K and columns to 1/B. Image 1's code (its column, renormalized to sum 1) is compared with the
   prediction from its second view, p = softmax(scores2 / tau). Numbers the reader sees: notes/sim-beyond.txt. */
KIT.SK = {
  K: 4, B: 8, EPS: .05, TAU: .1,
  // Cosine scores z^T c_k of view 1 (rows = prototypes). Prototype 1 is the top score for 6 of the 8 images.
  S: [[.62, .60, .58, .55, .61, .57, .40, .42],
      [.50, .45, .52, .41, .44, .48, .58, .38],
      [.45, .52, .40, .50, .39, .44, .43, .55],
      [.40, .38, .47, .46, .51, .36, .50, .47]],
  S2: [.58, .52, .47, .41]  // image 1, view 2: its scores against the 4 prototypes
};
// Q after each half-step: [raw, it1 rows, it1 cols, it2 rows, it2 cols, it3 rows, it3 cols].
KIT.skStates = () => {
  const { K, B, EPS, S } = KIT.SK;
  let Q = S.map(r => r.map(v => Math.exp(v / EPS)));
  const tot = Q.flat().reduce((a, b) => a + b, 0);
  Q = Q.map(r => r.map(v => v / tot));
  const out = [Q];
  for (let it = 0; it < 3; it++) {
    const rs = Q.map(r => r.reduce((a, b) => a + b, 0));
    Q = Q.map((r, k) => r.map(v => v * (1 / K) / rs[k])); out.push(Q);
    const cs = Array.from({ length: B }, (_, b) => Q.reduce((s, r) => s + r[b], 0));
    Q = Q.map(r => r.map((v, b) => v * (1 / B) / cs[b])); out.push(Q);
  }
  return out;
};
KIT.skRows = Q => Q.map(r => r.reduce((a, b) => a + b, 0));
KIT.skCols = Q => Q[0].map((_, b) => Q.reduce((s, r) => s + r[b], 0));
KIT.skCode = (Q, b) => { const c = Q.map(r => r[b]), s = c.reduce((a, x) => a + x, 0); return c.map(x => x / s); };
KIT.skPred = () => PE.softmax(KIT.SK.S2.map(v => v / KIT.SK.TAU));
KIT.skCE = (q, p) => -q.reduce((s, qk, k) => s + qk * Math.log(p[k]), 0);
KIT.skTop = Q => Q[0].map((_, b) => { let m = 0; for (let k = 1; k < Q.length; k++) if (Q[k][b] > Q[m][b]) m = k; return m; });

KIT.sinkhorn = (canvas, opts = {}) => {
  const { K, B } = KIT.SK, STATES = KIT.skStates();
  const state = { iters: 0, Q: STATES[0].map(r => r.slice()) };
  const glow = opts.fig ? PE.linkPulses(opts.fig, () => cv.redraw()) : () => 0;
  let cancel = () => {};
  const f3 = x => x.toFixed(3);
  const draw = (ctx, W, H) => {
    if (!W) return;
    const BR = PE.css('--brown'), Y = PE.css('--yellow'), G = PE.css('--green'), R = PE.css('--red'), M = PE.css('--muted'),
      FG = PE.css('--fg'), BG = PE.css('--bg'), RU = PE.css('--rule'), BL = PE.css('--blue');
    ctx.clearRect(0, 0, W, H);
    const small = KIT.narrow(W), fs = small ? 10 : 12;
    // Matrix block: left (wide) or top (narrow).
    const mw = small ? W - 24 : W * .6, x0 = small ? 58 : 84, y0 = small ? 52 : 58;
    const cw = (mw - x0 - (small ? 40 : 66)) / B, ch = Math.min(small ? 30 : 44, cw * .9);
    const Q = state.Q, rows = KIT.skRows(Q), cols = KIT.skCols(Q);
    const gk = glow('kernel'), gc = glow('code'), gr = glow('rows'), gcol = glow('cols'), ge = glow('eps');
    PE.label(ctx, 'codes Q: no gradient', 12, 20, BR, small ? 12 : 13);
    PE.label(ctx, 'eps = 0.05', 12, 36, ge ? FG : M, fs + ge * 3, 'left', 400 + 300 * ge);
    PE.label(ctx, state.iters ? state.iters + (state.iters > 1 ? ' iterations' : ' iteration') : 'before Sinkhorn', mw - 4, 20, M, fs + 1, 'right');
    for (let b = 0; b < B; b++) PE.label(ctx, (small ? '' : 'img ') + (b + 1), x0 + (b + .5) * cw, y0 - 8, b === 0 ? BL : M, fs, 'center');
    for (let k = 0; k < K; k++) {
      PE.label(ctx, (small ? 'p' : 'proto ') + (k + 1), x0 - 8, y0 + (k + .5) * ch + 4, M, fs, 'right');
      for (let b = 0; b < B; b++) {
        const u = PE.clamp(Math.sqrt(Q[k][b] * B), 0, 1), x = x0 + b * cw, y = y0 + k * ch;
        ctx.fillStyle = PE.mix(BG, BR, Math.min(1, u + .35 * Math.max(gk, gc)));
        ctx.fillRect(x + 1, y + 1, cw - 2, ch - 2);
        if (!small && cw > 40) PE.label(ctx, Q[k][b] < .0005 ? '0' : Q[k][b].toFixed(3).slice(1), x + cw / 2, y + ch / 2 + 4, u > .6 ? BG : FG, 11, 'center', 400);
      }
      const tr = Math.abs(rows[k] - 1 / K) < .0015;
      PE.label(ctx, f3(rows[k]), x0 + B * cw + 6, y0 + (k + .5) * ch + 4, gr ? FG : tr ? G : R, fs + 2 * gr, 'left', 600);
    }
    ctx.save(); ctx.strokeStyle = RU; ctx.lineWidth = 1; ctx.strokeRect(x0 + .5, y0 + .5, B * cw, K * ch); ctx.restore();
    const yc = y0 + K * ch + 16;
    for (let b = 0; b < B; b++) { const tc = Math.abs(cols[b] - 1 / B) < .0015;
      PE.label(ctx, small ? f3(cols[b]).slice(1) : f3(cols[b]), x0 + (b + .5) * cw, yc, gcol ? FG : tc ? G : R, fs - (small ? 1 : 0) + 2 * gcol, 'center', 600); }
    PE.label(ctx, small ? 'rows' : 'row sums', x0 + B * cw + 6, y0 - 8, M, Math.max(10, fs - 1), 'left', 400);
    PE.label(ctx, 'column sums (target 0.125); row sums (target 0.250)', small ? 12 : x0, yc + 18, M, Math.max(10, fs - 1), 'left', 400);
    // Image 1 panel: its code q (brown) vs. the prediction p from view 2 (yellow), and the cross-entropy (red).
    const px = small ? 16 : mw + 26, py = small ? yc + 50 : 58, pw = small ? W - 32 : W - mw - 44, ph = small ? H - py - 56 : H - py - 80;
    const q = KIT.skCode(Q, 0), p = KIT.skPred(), ce = KIT.skCE(q, p), gq = gc, gp = glow('pred'), gl = glow('loss');
    PE.label(ctx, 'image 1: code vs. prediction', px, py - 14, FG, fs + 1);
    const bw = pw / K;
    for (let k = 0; k < K; k++) {
      const bx = px + k * bw, base = py + ph;
      const hq = q[k] * ph, hp = p[k] * ph;
      ctx.globalAlpha = 1; ctx.fillStyle = BR; ctx.fillRect(bx + bw * .12, base - hq, bw * .34, hq);
      if (gq) PE.ring(ctx, bx + bw * .29, base - hq - 6, 4, FG, 2, gq);
      ctx.fillStyle = Y; ctx.globalAlpha = .85 + .15 * gp; ctx.fillRect(bx + bw * .52, base - hp, bw * .34, hp); ctx.globalAlpha = 1;
      if (gp) PE.ring(ctx, bx + bw * .69, base - hp - 6, 4, FG, 2, gp);
      PE.label(ctx, (small ? 'p' : 'proto ') + (k + 1), bx + bw / 2, base + 14, M, Math.max(10, fs - 1), 'center', 400);
    }
    ctx.save(); ctx.strokeStyle = RU; ctx.beginPath(); ctx.moveTo(px, py + ph + .5); ctx.lineTo(px + pw, py + ph + .5); ctx.stroke(); ctx.restore();
    const ly = py + ph + 30;
    PE.label(ctx, 'q (code, view 1)', px, ly, BR, fs);
    PE.label(ctx, 'p (prediction, view 2)', px + (small ? pw * .46 : 0), small ? ly : ly + 16, Y, fs);
    PE.label(ctx, 'loss -sum q log p = ' + ce.toFixed(2), px, small ? ly + 17 : ly + 34, R, fs + 1 + 2 * gl, 'left', 600 + 100 * gl);
  };
  const cv = PE.canvas(canvas, w => KIT.narrow(w) ? 1.25 : .52, draw);
  // Walk the half-steps from the current state to the target, one tween per half-step.
  const set = n => {
    cancel();
    const to = 2 * n, from = 2 * state.iters;
    state.iters = n;
    const target = STATES[to];
    if (PE.instant() || from === to) { state.Q = target.map(r => r.slice()); cv.redraw(); return; }
    const path = to > from ? STATES.slice(from + 1, to + 1) : [target];
    const per = Math.min(450, 2700 / path.length), start = state.Q.map(r => r.slice());
    cancel = PE.animate(per * path.length, u => {
      const s = u * path.length, i = Math.min(path.length - 1, Math.floor(s)), f = s - i, A = i ? path[i - 1] : start, Bq = path[i];
      state.Q = A.map((r, k) => r.map((v, b) => v + (Bq[k][b] - v) * f)); cv.redraw();
    }, () => { state.Q = target.map(r => r.slice()); cv.redraw(); });
  };
  return { set, state, redraw: cv.redraw, resize: cv.resize, destroy: () => { cancel(); cv.destroy(); } };
};
KIT.sinkhornText = n => {
  const Q = KIT.skStates()[2 * n], rows = KIT.skRows(Q), cols = KIT.skCols(Q), top = KIT.skTop(Q), q = KIT.skCode(Q, 0), p = KIT.skPred();
  const used = new Set(top).size, f3 = x => x.toFixed(3), pc = x => Math.round(100 * x) + '%';
  const lo = Math.min(...rows), hi = Math.max(...rows), share = top.filter(t => t === 0).length;
  const head = n === 0
    ? 'Before Sinkhorn: prototype 1 holds ' + pc(rows[0]) + ' of the mass and is the top choice for ' + share + ' of the 8 images. Only ' + used + ' of 4 prototypes are anyone\'s top choice.'
    : 'After ' + n + (n > 1 ? ' iterations' : ' iteration') + ': every column sums to 0.125; row sums run from ' + f3(lo) + ' to ' + f3(hi) + ' (target 0.250). ' + used + ' of 4 prototypes are now someone\'s top choice.';
  return head + ' Image 1\'s code puts ' + pc(q[0]) + ' on prototype 1; the loss for predicting it from view 2 is ' + KIT.skCE(q, p).toFixed(2) + '.';
};
