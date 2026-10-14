/* ch-clip toy: an 8 x 8 table of cosines between 8 made-up photo embeddings and 8 caption embeddings (12-d, unit length).
   'softmax' turns each row into CLIP-style probabilities (each row sums to 1); 'sigmoid' scores every cell alone,
   sigma(t cos + b), as in SigLIP's Algorithm 1. t = 10 for both (SigLIP's starting t). The batch panel counts, for a
   batch of N pairs over D devices, how many scores one probability needs (our arithmetic). Values: notes/sim-clipsig.txt. */
KIT.CS = { NAMES: ['dog', 'cat', 'car', 'truck', 'pizza', 'cake', 'tree', 'flower'], D: 12, GROUP: .62, OWN: .78, NOISE: .1, SEED: 3, T: 10 };
KIT.csCos = () => {
  const { NAMES, D, GROUP, OWN, NOISE, SEED } = KIT.CS, g = PE.gauss(PE.rng(SEED)), n = NAMES.length;
  // Pairs (dog, cat), (car, truck), ... share one direction, so each photo has one confusable caption.
  const mk = i => { const v = new Array(D).fill(0); v[i >> 1] += GROUP; v[4 + i] += OWN; for (let d = 0; d < D; d++) v[d] += NOISE * g();
    const r = Math.hypot(...v); return v.map(x => x / r); };
  const I = [], T = []; for (let i = 0; i < n; i++) { I.push(mk(i)); T.push(mk(i)); }
  return I.map(a => T.map(b => a.reduce((s, x, d) => s + x * b[d], 0)));
};
KIT.csNum = x => x >= 1e12 ? (x / 1e12).toFixed(2) + ' trillion' : x >= 1e9 ? (x / 1e9).toFixed(2) + ' billion' : x >= 1e6 ? (x / 1e6).toFixed(2) + ' million' : Math.round(x).toLocaleString('en-US');
KIT.csSig = u => 1 / (1 + Math.exp(-u));
KIT.csProbs = (C, mode, b) => mode === 'softmax' ? C.map(r => PE.softmax(r.map(c => KIT.CS.T * c)))
  : mode === 'sigmoid' ? C.map(r => r.map(c => KIT.csSig(KIT.CS.T * c + b))) : C.map(r => r.slice());
// One probability, batch N over D devices: how many similarity scores feed it, and how many of those live on other devices.
KIT.csCounts = (N, D) => ({ N, D, perDev: N / D, rowTerms: N, remote: N - N / D, cells: N * N, block: (N / D) * (N / D) });
KIT.clipsig = (canvas, opts = {}) => {
  const n = KIT.CS.NAMES.length, C = KIT.csCos();
  const state = { mode: null, b: -10, N: 32768, Dv: 8, shown: KIT.csProbs(C, null, -10) };
  const glow = opts.fig ? PE.linkPulses(opts.fig, () => cv.redraw()) : () => 0;
  let cancel = () => {};
  const layout = (W, H) => {
    const small = KIT.narrow(W), lw = small ? 44 : 52, th = small ? 50 : 40, mw = small ? W - 16 : W * .56;
    const cell = Math.floor(Math.min((mw - lw - 8) / (n + 1.3), ((small ? H * .62 : H) - th - 30) / n));
    const mx = small ? 8 + lw : 16 + lw, my = th + 16;
    const panel = small ? { x: 14, y: my + n * cell + 22, w: W - 28 } : { x: W * .6 + 10, y: my + 6, w: W * .4 - 28 };
    return { small, cell, mx, my, panel };
  };
  const draw = (ctx, W, H) => {
    if (!W) return;
    const B = PE.css('--blue'), Y = PE.css('--yellow'), G = PE.css('--green'), R = PE.css('--red'), F = PE.css('--fg'), Mu = PE.css('--muted'), RU = PE.css('--rule'), P2 = PE.css('--panel2'), BG = PE.css('--bg');
    ctx.clearRect(0, 0, W, H);
    const { small, cell, mx, my, panel } = layout(W, H), mode = state.mode, fs = small ? 10 : 11;
    const gRow = glow('row'), gCol = glow('col'), gDiag = glow('diag'), gCell = glow('cell'), gT = glow('scale'), gB = glow('bias');
    // Column labels (captions, yellow) tilted so they fit narrow cells; row labels (photos, blue).
    KIT.CS.NAMES.forEach((nm, j) => { ctx.save(); ctx.translate(mx + (j + .5) * cell, my - 6); ctx.rotate(-Math.PI / 4);
      ctx.font = `600 ${fs}px system-ui, sans-serif`; ctx.fillStyle = Y; ctx.fillText('"' + nm + '"', 0, 0); ctx.restore(); });
    KIT.CS.NAMES.forEach((nm, i) => PE.label(ctx, nm, mx - 6, my + (i + .5) * cell + 4, B, fs, 'right'));
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
      const v = state.shown[i][j], x = mx + j * cell, y = my + i * cell;
      // Cosines are shaded by size; probabilities by value. Green = agreement.
      const a = mode ? PE.clamp(v, 0, 1) : PE.clamp(v, 0, 1) * .8;
      ctx.fillStyle = P2; ctx.fillRect(x + 1, y + 1, cell - 2, cell - 2);
      ctx.save(); ctx.globalAlpha = a; ctx.fillStyle = G; ctx.fillRect(x + 1, y + 1, cell - 2, cell - 2); ctx.restore();
      const txt = mode ? (v < .005 ? '0' : v > .995 ? '1' : v.toFixed(2).replace(/^0/, '')) : v.toFixed(2).replace(/^(-?)0/, '$1');
      PE.label(ctx, txt, x + cell / 2, y + cell / 2 + 4, a > .55 ? BG : F, cell < 34 ? 10 : 10.5, 'center', 500);
      if (i === j && gDiag) { ctx.save(); ctx.globalAlpha = gDiag; ctx.strokeStyle = G; ctx.lineWidth = 3; ctx.strokeRect(x + 1.5, y + 1.5, cell - 3, cell - 3); ctx.restore(); }
    }
    // What the probability in cell (dog, "dog") depends on: its whole row for the softmax, only itself for the sigmoid.
    ctx.save(); ctx.strokeStyle = R; ctx.lineWidth = 2.5;
    if (mode === 'softmax' || gRow) { ctx.globalAlpha = mode === 'softmax' ? Math.max(.9, gRow) : gRow; ctx.strokeRect(mx + 1, my + 1, n * cell - 2, cell - 2); }
    if (gCol) { ctx.globalAlpha = gCol; ctx.setLineDash([5, 4]); ctx.strokeRect(mx + 1, my + 1, cell - 2, n * cell - 2); ctx.setLineDash([]); }
    if (mode === 'sigmoid' || gCell) { ctx.globalAlpha = mode === 'sigmoid' ? Math.max(.9, gCell) : gCell; ctx.strokeRect(mx + 1, my + 1, cell - 2, cell - 2); }
    // t and b act on every cell at once, so their terms light the whole table.
    if (gT) { ctx.globalAlpha = gT; ctx.strokeStyle = G; ctx.lineWidth = 3; ctx.strokeRect(mx - 2, my - 2, n * cell + 4, n * cell + 4); }
    if (gB) { ctx.globalAlpha = gB; ctx.strokeStyle = R; ctx.lineWidth = 3; ctx.setLineDash([6, 4]); ctx.strokeRect(mx - 2, my - 2, n * cell + 4, n * cell + 4); ctx.setLineDash([]); }
    ctx.restore();
    // Row sums: exactly 1 under the softmax, anything under the sigmoid.
    if (mode) { const sx = mx + n * cell + cell * .25;
      PE.label(ctx, 'sum', sx + cell / 2, my - 6, Mu, fs, 'center');
      state.shown.forEach((r, i) => PE.label(ctx, r.reduce((s, v) => s + v, 0).toFixed(2), sx + cell / 2, my + (i + .5) * cell + 4, mode === 'softmax' ? F : Mu, cell < 34 ? 10 : 10.5, 'center', 500)); }
    const head = mode === 'softmax' ? 'row softmax of t \u00b7 cos, t = 10' : mode === 'sigmoid' ? 'sigmoid(t \u00b7 cos + b), t = 10, b = ' + state.b : 'cosine similarity (photo \u00b7 caption)';
    PE.label(ctx, head, 12, 16, Mu, small ? 11 : 12);
    if (gT || gB) PE.label(ctx, gB ? 'b shifts every cell the same way' : 't stretches cosines into scores', 12, 31, gB ? R : G, small ? 11 : 12, 'left', 600);
    // Batch panel: one row of the real table at batch N, split over D devices.
    const c = KIT.csCounts(state.N, Math.min(state.Dv, state.N)), { x, y, w } = panel, bh = 16;
    PE.label(ctx, 'one row at batch ' + KIT.csNum(c.N) + ', ' + c.D + ' device' + (c.D > 1 ? 's' : ''), x, y, F, small ? 11 : 12);
    const by = y + 10, seg = w / c.D, gap = c.D <= 32 ? 2 : 0;
    for (let d = 0; d < c.D; d++) {
      const own = d === 0, sx = x + d * seg;
      ctx.fillStyle = mode === 'softmax' ? (own ? G : R) : P2;
      ctx.globalAlpha = mode === 'softmax' && !own ? .7 : 1; ctx.fillRect(sx, by, Math.max(1, seg - gap), bh); ctx.globalAlpha = 1;
      if (gap) { ctx.strokeStyle = Mu; ctx.lineWidth = 1; ctx.strokeRect(sx + .5, by + .5, Math.max(1, seg - gap) - 1, bh - 1); }
    }
    PE.label(ctx, 'device 1', x, by + bh + 13, Mu, 10, 'left', 400);
    if (c.D > 1) PE.label(ctx, c.D > 2 ? 'devices 2 to ' + c.D : 'device 2', x + w, by + bh + 13, Mu, 10, 'right', 400);
    if (mode === 'sigmoid') { ctx.fillStyle = G; ctx.fillRect(x, by, Math.max(3, w / c.N), bh); }
    const lines = mode === 'softmax'
      ? [['this cell needs all ' + KIT.csNum(c.rowTerms) + ' scores in its row', F], [KIT.csNum(c.remote) + ' of them sit on other devices (red)', R], ['full table: ' + KIT.csNum(c.cells) + ' cells', Mu]]
      : mode === 'sigmoid'
        ? [['this cell needs 1 score: its own', F], ['0 from other devices for this cell', G], ['per device at a time: ' + KIT.csNum(c.block) + ' cells', Mu]]
        : [['press a button to see what', Mu], ['one probability needs', Mu]];
    lines.forEach(([t, col], k) => PE.label(ctx, t, x, by + bh + 32 + 16 * k, col, small ? 11 : 12, 'left', k ? 400 : 600));
  };
  const cv = PE.canvas(canvas, w => KIT.narrow(w) ? 1.1 : .54, draw);
  const set = (mode, b = state.b) => {
    cancel(); const from = state.shown.map(r => r.slice()), to = KIT.csProbs(C, mode, b);
    state.mode = mode; state.b = b;
    cancel = PE.animate(600, u => { state.shown = from.map((r, i) => r.map((v, j) => PE.lerp(v, to[i][j], u))); cv.redraw(); });
  };
  const setBatch = (N, Dv) => { state.N = N; state.Dv = Dv; cv.redraw(); };
  return { set, setBatch, state, C, redraw: cv.redraw, resize: cv.resize, destroy: () => { cancel(); cv.destroy(); } };
};
KIT.clipsigText = (mode, b, N, Dv) => {
  const C = KIT.csCos(), P = KIT.csProbs(C, mode, b), c = KIT.csCounts(N, Math.min(Dv, N)), f = x => x.toFixed(2);
  const tail = ' At batch ' + KIT.csNum(N) + ' over ' + c.D + ' devices: ';
  if (!mode) return 'Raw cosines. Each photo is closest to its own caption (the diagonal, ' + f(C[0][0]) + ' for dog) and fairly close to one look-alike (' + f(C[0][1]) + ' for "cat").';
  if (mode === 'softmax') return 'Softmax: row dog gives "dog" ' + f(P[0][0]) + ' and "cat" ' + f(P[0][1]) + ', and every row sums to 1. Change any score in the row and all eight numbers move.' + tail + 'one row needs ' + KIT.csNum(c.rowTerms) + ' scores, ' + KIT.csNum(c.remote) + ' of them from other devices.';
  const yes = P.reduce((s, r) => s + r.filter(v => v > .5).length, 0);
  return 'Sigmoid with b = ' + b + ': (dog, "dog") is ' + f(P[0][0]) + ', (dog, "cat") is ' + f(P[0][1]) + '. ' + yes + ' of 64 cells say yes (above 0.5); a cell flips at cosine ' + f(-b / KIT.CS.T) + '.' + tail + 'each cell needs only its own score.';
};
