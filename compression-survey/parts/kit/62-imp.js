/* Iterative magnitude pruning (Frankle and Carbin 1803.03635v5, Sec. 1) on a toy 32 x 32 weight mask.
   Each round: "train" (toy: theta_0 plus a fixed drift plus round noise, no real training), prune the 20% of survivors
   with the smallest trained magnitude (the LeNet rate), reset survivors to theta_0 (or draw a new random init: the
   control). Only the findings Frankle states in the text are plotted (LeNet on MNIST, Sec. 2); the paper's curves have
   no tabulated values. Mask sizes simulated in notes/sim-imp.txt. */
KIT.IMP = { G: 32, rate: .2, rounds: [0, 3, 7, 9, 15, 16], seed: 31 };
KIT.impRun = () => {
  if (KIT.IMP.run) return KIT.IMP.run;
  const P = KIT.IMP, n = P.G * P.G, g0 = PE.gauss(PE.rng(P.seed)), gd = PE.gauss(PE.rng(P.seed + 1)), gr = PE.gauss(PE.rng(P.seed + 2));
  const th0 = Array.from({ length: n }, () => g0()), drift = Array.from({ length: n }, () => .9 * gd()), re = Array.from({ length: n }, () => gr());
  const masks = [new Array(n).fill(1)], trained = [];
  const last = Math.max(...P.rounds);
  for (let r = 1; r <= last; r++) {
    const gn = PE.gauss(PE.rng(1000 + r)), th = th0.map((v, i) => v + drift[i] + .35 * gn()), m = masks[r - 1].slice();
    const alive = m.map((x, i) => x ? i : -1).filter(i => i >= 0).sort((a, b) => Math.abs(th[a]) - Math.abs(th[b]));
    alive.slice(0, Math.round(P.rate * alive.length)).forEach(i => (m[i] = 0));
    trained[r] = th; masks.push(m);
  }
  const counts = masks.map(m => m.reduce((s, x) => s + x, 0));
  return (P.run = { n, th0, re, masks, trained, counts });
};
// Frankle and Carbin, LeNet-300-100 on MNIST, Sec. 2: the only accuracy and speed points the text states.
KIT.IMP_FIND = {
  0: { t: 'The dense network, before any pruning.', r: 'The dense network, before any pruning.' },
  1: { t: 'One round in, 80% of the weights remain; the text states no result at this size.', r: 'One round in, 80% of the weights remain; the text states no result at this size.' },
  3: { t: 'At P<sub>m</sub> = <span data-claim="frankle2019-lenet-pm-51">51.3</span>%, the ticket already learns faster than the dense network.',
    r: 'Reinitialized subnetworks keep the dense accuracy only down to about P<sub>m</sub> = <span data-claim="frankle2019-reinit-dropoff">21.1</span>%.' },
  7: { t: 'Near P<sub>m</sub> = <span data-claim="frankle2019-lenet-fast-pm">21</span>%, tickets learn fastest: early stopping comes <span data-claim="frankle2019-lenet-fast-earlier">38</span>% earlier than for the dense network.',
    r: 'Here the ticket reaches its minimum validation loss <span data-claim="frankle2019-reinit-speed">2.51</span>x faster than the same mask reinitialized, whose accuracy starts to fall below the dense network at P<sub>m</sub> = <span data-claim="frankle2019-reinit-dropoff">21.1</span>%.' },
  9: { t: 'At P<sub>m</sub> = <span data-claim="frankle2019-lenet-acc-pm">13.5</span>%, the ticket is more than <span data-claim="frankle2019-lenet-acc-gain">0.3</span> points more accurate than the dense network.',
    r: 'The reinitialized network is already below the dense accuracy (its drop-off starts at P<sub>m</sub> = <span data-claim="frankle2019-reinit-dropoff">21.1</span>%).' },
  15: { t: 'At P<sub>m</sub> = <span data-claim="frankle2019-lenet-regress-pm">3.6</span>%, the ticket is back to the dense network\'s accuracy.',
    r: 'The reinitialized network is well below the dense accuracy.' },
  16: { t: 'Below P<sub>m</sub> = <span data-claim="frankle2019-ticket-dropoff">2.9</span>%, even the ticket falls below the dense network.',
    r: 'The reinitialized network is well below the dense accuracy.' },
};
KIT.IMP_PHASE = { train: '1. train', prune: '2. prune 20%', reset: '3. reset' };
KIT.impText = s => {
  const R = KIT.impRun(), r = s.round, c = R.counts[r], pm = (100 * c / R.n).toFixed(1), F = KIT.IMP_FIND[r];
  const ph = s.round === 0 ? (s.phase === 'train' ? 'The dense network after training. ' : '') :
    { train: `Round ${r}, step 1: train the ${R.counts[r - 1]} survivors of round ${r - 1}. `, prune: `Round ${r}, step 2: remove the 20% with the smallest trained magnitude. `,
      reset: s.init === 'ticket' ? `Round ${r}, step 3: reset the survivors to their initial values \u03b8<sub>0</sub>. ` : `Round ${r}, step 3: control, the same mask with a new random initialization. ` }[s.phase];
  return `${ph}Toy mask: <b>${c} of ${R.n} weights, P<sub>m</sub> = ${pm}%</b>. ` +
    (s.init === 'ticket' ? `<b class="c-green">Ticket</b> (Frankle's LeNet): ${F.t}` : `<b class="c-brown">Reinitialized</b> (Frankle's LeNet): ${F.r}`);
};
KIT.imp = (el, opts = {}) => {
  const P = KIT.IMP, R = KIT.impRun(), n = R.n;
  const S = { round: 0, phase: 'reset', init: 'ticket', v: {}, k: {} };
  const values = () => {
    const r = S.round, m = S.phase === 'train' && r > 0 ? R.masks[r - 1] : R.masks[r];
    const th = S.phase === 'reset' ? (S.init === 'ticket' || r === 0 ? R.th0 : R.re) : R.trained[Math.max(1, r)];
    const v = {}, k = {}; for (let i = 0; i < n; i++) { v[i] = Math.min(1, Math.abs(th[i]) / 2.6); k[i] = S.phase === 'train' && r === 0 ? 1 : m[i]; }
    return { v, k };
  };
  Object.assign(S, values());
  const glow = opts.fig ? PE.linkPulses(opts.fig, () => cv.redraw()) : () => 0;
  let stop = () => {};
  function draw(ctx, w, h) {
    ctx.clearRect(0, 0, w, h);
    const narrow = w < 560, cs = k => PE.css(k), pad = opts.bottomPad || 0, H = h - pad;
    const C = { fg: cs('--fg'), mut: cs('--muted'), green: cs('--green'), brown: cs('--brown'), blue: cs('--blue'), yellow: cs('--yellow'), red: cs('--red'), rule: cs('--rule') };
    const gMask = glow('mask'), gInit = glow('init'), gNet = glow('net'), gPm = glow('pm');
    const side = narrow ? Math.min(w - 24, H * .5) : Math.min(H - 40, w * .44), gx = narrow ? (w - side) / 2 : 16, gy = 28, cell = side / P.G;
    const cnt = R.counts[S.round], pm = 100 * cnt / n;
    PE.label(ctx, `toy layer, ${n} weights: ${cnt} left (P_m ${pm.toFixed(1)}%)`, gx, 16, C.fg, narrow ? 11 : 12, 'left', 600);
    for (let i = 0; i < n; i++) {
      const x = gx + (i % P.G) * cell, y = gy + Math.floor(i / P.G) * cell, k = S.k[i];
      ctx.globalAlpha = k * (.15 + .85 * S.v[i]); ctx.fillStyle = C.blue; ctx.fillRect(x + .5, y + .5, cell - 1, cell - 1);
      ctx.globalAlpha = (1 - k) * (.2 + .35 * gMask); ctx.fillStyle = C.brown; ctx.fillRect(x + .5, y + .5, cell - 1, cell - 1);
    }
    ctx.globalAlpha = 1;
    if (gNet > 0 || gInit > 0) { ctx.save(); ctx.strokeStyle = gInit > gNet ? C.green : C.fg; ctx.globalAlpha = Math.max(gNet, gInit); ctx.lineWidth = 3; ctx.strokeRect(gx - 3, gy - 3, side + 6, side + 6); ctx.restore(); }
    // the loop
    const lx = narrow ? 12 : gx + side + 30, lw = narrow ? w - 24 : w - lx - 16, ly = narrow ? gy + side + 26 : 44;
    const bw = Math.min(narrow ? (lw - 24) / 3 : 132, (lw - 40) / 3), bh = 44, gap = (lw - 3 * bw) / 2;
    const resetName = S.init === 'ticket' ? '3. reset to\n\u03b8\u2080 (initial)' : '3. new random\ninit (control)';
    const names = ['1. train', '2. prune 20%\nof survivors', resetName], keys = ['train', 'prune', 'reset'];
    PE.label(ctx, 'iterative magnitude pruning: repeat each round', lx, ly - 12, C.fg, narrow ? 11 : 12, 'left', 600);
    keys.forEach((key, j) => {
      const x = lx + j * (bw + gap), on = S.phase === key && !(S.round === 0 && key === 'reset');
      const col = on ? (key === 'reset' && S.init !== 'ticket' ? C.brown : C.green) : C.mut, gl = key === 'reset' ? gInit : key === 'prune' ? gMask : 0;
      PE.box(ctx, x, ly, bw, bh, gl > .2 ? C.green : col, narrow ? names[j].replace('of survivors', '').replace(' (initial)', '').replace(' (control)', '') : names[j], 1);
      if (on || gl > 0) { ctx.save(); ctx.strokeStyle = col; ctx.globalAlpha = Math.max(on ? 1 : 0, gl); ctx.lineWidth = 2.5; ctx.beginPath(); ctx.roundRect(x - 2, ly - 2, bw + 4, bh + 4, 9); ctx.stroke(); ctx.restore(); }
      if (j < 2) PE.arrow(ctx, x + bw + 3, ly + bh / 2, x + bw + gap - 3, ly + bh / 2, C.mut, 1.5, 7);
    });
    const back = [[lx + 2 * (bw + gap) + bw / 2, ly + bh + 2], [lx + 2 * (bw + gap) + bw / 2, ly + bh + 14], [lx + bw / 2, ly + bh + 14]];
    PE.polyline(ctx, back, C.mut, 1.5); PE.arrow(ctx, lx + bw / 2, ly + bh + 14, lx + bw / 2, ly + bh + 3, C.mut, 1.5, 7);
    PE.label(ctx, 'next round', lx + bw + gap + bw / 2, ly + bh + 26, C.mut, 10, 'center', 400);
    // findings strip: P_m on a log axis, 100% on the left
    const sx = lx + 6, sw = lw - 12, sy = narrow ? ly + bh + 124 : Math.max(ly + bh + 120, (ly + bh + H) / 2 + 30), X = p => sx + (2 - Math.log10(p)) / 2 * sw;
    PE.label(ctx, "Frankle's stated LeNet findings, by P_m (% of weights left)", lx, sy - 74, C.fg, narrow ? 10.5 : 12, 'left', 600);
    PE.label(ctx, 'green: winning ticket; brown: same mask, random reinit', lx, sy - 58, C.mut, 10, 'left', 400);
    PE.polyline(ctx, [[sx, sy], [sx + sw, sy]], C.mut, 1);
    [100, 50, 20, 10, 5, 2, 1].forEach(p => { PE.polyline(ctx, [[X(p), sy - 3], [X(p), sy + 3]], C.mut, 1); PE.label(ctx, p + '%', X(p), sy + 15, C.mut, 10, 'center', 400); });
    const tk = [[51.3, 'faster', 0], [21, 'fastest', 0], [13.5, '+0.3 pts', 1], [3.6, '= dense', 0], [2.9, 'drops', 1]];
    tk.forEach(([p, t, row]) => { PE.dot(ctx, X(p), sy - 6, 4, C.green); PE.label(ctx, t, X(p), sy - 14 - row * 13, C.green, 10, 'center', 600); });
    PE.dot(ctx, X(21.1), sy + 24, 4, C.brown);
    PE.label(ctx, narrow ? 'drops; 2.51x slower' : 'accuracy drops; 2.51x slower than the ticket', X(21.1) + 8, sy + 28, C.brown, 10, 'left', 600);
    const cx = X(Math.max(1, pm));
    PE.polyline(ctx, [[cx, sy - 36], [cx, sy + 6]], C.fg, 1.5 + 2.5 * gPm, [3, 3], .9);
    PE.label(ctx, 'toy', cx, sy - 40, C.fg, 10, 'center', 600);
  }
  const cv = PE.canvas(el, opts.aspect || (w => w < 560 ? 1.62 : .56), draw);
  function set(next) {
    if (next.round != null) { S.round = next.round; if (next.phase == null) S.phase = 'reset'; }
    if (next.phase) { if (next.phase === 'prune' && S.round === 0) S.round = 1; S.phase = next.phase; }
    if (next.init) S.init = next.init;
    const t = values(); stop();
    const to = {}; for (let i = 0; i < n; i++) { to['v' + i] = t.v[i]; to['k' + i] = t.k[i]; }
    const from = {}; for (let i = 0; i < n; i++) { from['v' + i] = S.v[i]; from['k' + i] = S.k[i]; }
    stop = KIT.tween(from, to, 700, () => { for (let i = 0; i < n; i++) { S.v[i] = from['v' + i]; S.k[i] = from['k' + i]; } cv.redraw(); });
  }
  return { set, state: S, text: () => KIT.impText(S), redraw: cv.redraw, resize: cv.resize, destroy: () => { stop(); cv.destroy(); } };
};
