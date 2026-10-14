/* FIG.cl: confident learning on our run C (ch-confident; atlas "find"). 4-fold out-of-sample probabilities on CIFAR-10
   with asymmetric noise (cat<->dog, automobile->truck labels... see runC). Panels: per-class thresholds t_j (Eq 2), the
   confident joint C (Eq 1) or the true count matrix, and eight real images from a chosen cell with their truth. */
FIG.cl = (el, o = {}) => {
  const D = DATA.runC, N = D.names, m = N.length, SH = { cat: 'cat', automobile: 'car', truck: 'truck', airplane: 'plane' };
  const short = n => n === 'automobile' ? 'auto' : n === 'airplane' ? 'plane' : n;
  // true counts from the true joint
  const Ctrue = D.Qtrue.map(r => r.map(v => Math.round(v * D.n)));
  const S = { view: 'joint', cell: 0, hover: null }; let im = null;
  const cv = PE.canvas(el, o.aspect || (w => w < 620 ? 1.75 : .56), draw);
  let L = null;
  function draw(c, W, H) {
    const C = { fg: PE.css('--fg'), mu: PE.css('--muted'), blue: PE.css('--blue'), yel: PE.css('--yellow'), red: PE.css('--red'), green: PE.css('--green'), rule: PE.css('--rule'), bg: PE.css('--bg') };
    c.clearRect(0, 0, W, H); H = Math.max(140, H - (o.padBottom || 0));
    const narrow = W < 620, side = narrow ? Math.min(W - 70, H * .5) : Math.min(H - 70, W * .46);
    const M0 = { x: 58, y: 46, s: side }, cs = side / m;
    const mat = S.view === 'true' ? Ctrue : D.C;
    PE.label(c, S.view === 'true' ? 'true counts: given label (row) vs real class (column)' : 'confident joint C: given label (row) vs likely class (column)', 8, 18, S.view === 'true' ? C.yel : C.blue, 12);
    let mxOff = 1; mat.forEach((r, i) => r.forEach((v, j) => { if (i !== j) mxOff = Math.max(mxOff, v); }));
    for (let i = 0; i < m; i++) for (let j = 0; j < m; j++) {
      const v = mat[i][j], x = M0.x + j * cs, y = M0.y + i * cs;
      c.fillStyle = i === j ? PE.mix(C.bg, C.green, .25 + .5 * Math.min(1, v / 5000)) : PE.mix(C.bg, C.red, Math.min(1, Math.sqrt(v / mxOff)));
      c.fillRect(x + .5, y + .5, cs - 1, cs - 1);
      if (cs > 26 && v > 0) PE.label(c, v >= 1000 ? (v / 1000).toFixed(1) + 'k' : String(v), x + cs / 2, y + cs / 2 + 4, C.fg, 9, 'center');
    }
    N.forEach((n, k) => { PE.label(c, short(n), M0.x - 4, M0.y + k * cs + cs / 2 + 4, C.mu, 10, 'right');
      c.save(); c.translate(M0.x + k * cs + cs / 2 + 3, M0.y - 4); c.rotate(-Math.PI / 3); PE.label(c, short(n), 0, 0, C.mu, 10); c.restore(); });
    const cell = D.cells[S.cell], hx = M0.x + cell.guess * cs, hy = M0.y + cell.given * cs;
    c.strokeStyle = C.yel; c.lineWidth = 2.5; c.strokeRect(hx + 1, hy + 1, cs - 2, cs - 2);
    // thresholds and tiles
    const R = narrow ? { x: 8, y: M0.y + side + 18, w: W - 16 } : { x: M0.x + side + 34, y: 30, w: W - M0.x - side - 46 };
    PE.label(c, 'per-class threshold t_j (average self-confidence)', R.x, R.y + 6, C.mu, 11);
    const bw = R.w / m;
    D.t.forEach((t, k) => { const bh = 54 * t; c.fillStyle = C.blue; c.fillRect(R.x + k * bw + 3, R.y + 74 - bh, bw - 6, bh); PE.label(c, short(N[k]), R.x + k * bw + bw / 2, R.y + 88, C.mu, 9, 'center');
      PE.label(c, t.toFixed(2), R.x + k * bw + bw / 2, R.y + 70 - bh, C.fg, 9, 'center'); });
    const ty = R.y + 112;
    PE.label(c, `labeled "${N[cell.given]}", likely "${N[cell.guess]}": ${cell.n.toLocaleString('en-US')} images, ${(100 * cell.prec).toFixed(1)}% really mislabeled`, R.x, ty, C.yel, 11);
    const ts = Math.min(56, (R.w - 7 * 8) / 8);
    cell.tiles.forEach((t, k) => { const x = R.x + k * (ts + 8), y = ty + 12;
      KIT.tile(c, im, S.cell * 8 + k, x, y, ts);
      if (t.err) KIT.strike(c, x + ts / 2, y + ts / 2, ts * .9, C.red, { seed: k }); else KIT.tick(c, x + ts / 2, y + ts + 12, ts * .35, C.green);
      PE.label(c, short(N[t.true]), x + ts / 2, y + ts + 26, C.mu, 9, 'center'); });
    PE.label(c, 'red strike: really mislabeled; green tick: a false flag; caption: true class', R.x, ty + ts + 48, C.mu, 10);
    L = { M0, cs };
    o.say && o.say(`${S.view === 'true' ? 'True counts.' : 'Confident joint.'} Selected cell: ${cell.n.toLocaleString('en-US')} images labeled ${N[cell.given]} that the model, never trained on them, puts under ${N[cell.guess]}; ${(100 * cell.prec).toFixed(1)}% of them really are mislabeled. Estimated errors overall: ${D.estErr.toLocaleString('en-US')} (true: ${D.nErr.toLocaleString('en-US')}).`);
  }
  KIT.sprite('tiles2', i => { im = i; cv.redraw(); });
  PE.onTheme(() => cv.redraw());
  return { view(v) { S.view = v; cv.redraw(); }, cell(k) { S.cell = k; cv.redraw(); }, reset() { S.view = 'joint'; S.cell = 0; cv.redraw(); }, redraw: () => cv.redraw(), destroy() { cv.destroy && cv.destroy(); } };
};
