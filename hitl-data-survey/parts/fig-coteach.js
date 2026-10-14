/* FIG.coteach: small-loss selection with two networks (ch-coteach; atlas "split" step 1). One real mini-batch of 32 CIFAR-10
   images (13 with flipped labels) from runs A and B: each network sorts the batch by its own loss at the chosen epoch and
   keeps the smallest R(T) share; in Co-teaching each network's pick trains the other. Lower panel: label precision of the
   kept set and test accuracy per epoch (run B), against plain training (run A). Computed from our runs. */
FIG.coteach = (el, o = {}) => {
  const B = DATA.runB.batch, SB = DATA.runB.series, SA = DATA.runA.series, n = B.idx.length;
  const S = { e: 10, reveal: false, mode: 'two' }; let im = null;
  const cv = PE.canvas(el, o.aspect || (w => w < 620 ? 1.6 : .62), draw);
  function rowFor(key) { const L = (key === 'ce' ? B.ce : B[key])[String(S.e)]; return L.map((v, i) => [v, i]).sort((a, b) => a[0] - b[0]).map(x => x[1]); }
  function draw(c, W, H) {
    const C = { fg: PE.css('--fg'), mu: PE.css('--muted'), blue: PE.css('--blue'), yel: PE.css('--yellow'), red: PE.css('--red'), green: PE.css('--green'), brown: PE.css('--brown'), rule: PE.css('--rule') };
    c.clearRect(0, 0, W, H); H = Math.max(140, H - (o.padBottom || 0));
    const narrow = W < 620, perRow = narrow ? 16 : 32, ts = Math.min(30, (W - 24 - (perRow - 1) * 2) / perRow);
    const R = SB[S.e - 1].R, keep = Math.round(R * n);
    const rows = S.mode === 'two' ? [['f', 'network f, sorted by its own loss'], ['g', 'network g, sorted by its own loss']] : [['ce', 'one network, sorted by its own loss (plain training, run A)']];
    let y = 22; const prec = [];
    rows.forEach(([key, lab]) => {
      PE.label(c, lab, 10, y, C.mu, 11);
      const ord = rowFor(key); let clean = 0;
      ord.forEach((i, k) => { const r = Math.floor(k / perRow), cc = k % perRow, x = 10 + cc * (ts + 2), yy = y + 8 + r * (ts + 18), kept = k < keep;
        c.globalAlpha = kept ? 1 : .32; KIT.tile(c, im, 32 + i, x, yy, ts); c.globalAlpha = 1;
        if (kept) { if (!B.noisy[i]) clean++; c.strokeStyle = C.green; c.lineWidth = 1.5; c.strokeRect(x - .5, yy - .5, ts + 1, ts + 1); }
        if (S.reveal && B.noisy[i]) KIT.strike(c, x + ts / 2, yy + ts / 2, ts * .9, C.red, { seed: i, lw: 2 });
        if (k === keep - 1) { c.strokeStyle = C.fg; c.setLineDash([3, 3]); c.beginPath(); c.moveTo(x + ts + 1, yy - 4); c.lineTo(x + ts + 1, yy + ts + 4); c.stroke(); c.setLineDash([]); } });
      prec.push(clean / keep);
      y += 8 + Math.ceil(n / perRow) * (ts + 18) + 14;
    });
    if (S.mode === 'two') PE.label(c, "f's kept images train g; g's kept images train f", 10, y - 4, C.yel, 12);
    else PE.label(c, 'its kept images train itself', 10, y - 4, C.yel, 12);
    // lower panel
    const b = { x: 46, y: y + 20, w: W - 60, h: H - y - 52 };
    if (b.h > 40) {
      const ax = PE.axes(c, b, [1, 200], [.4, 1], { logx: true, xt: [1, 3, 10, 30, 100, 200], yt: [.4, .6, .8, 1], yf: v => Math.round(v * 100) + '%', xl: 'epoch (log scale)' });
      PE.polyline(c, SB.map(s => [ax.X(s.e), ax.Y(s.p1)]), C.green, 2.2);
      PE.polyline(c, SA.map(s => [ax.X(s.e), ax.Y(s.test)]), C.mu, 1.8, [4, 3]);
      PE.polyline(c, SB.map(s => [ax.X(s.e), ax.Y(s.a1)]), C.fg, 2.2);
      c.strokeStyle = C.yel; c.setLineDash([3, 3]); c.beginPath(); c.moveTo(ax.X(S.e), b.y); c.lineTo(ax.X(S.e), b.y + b.h); c.stroke(); c.setLineDash([]);
      PE.label(c, 'clean share of the kept images (Co-teaching)', ax.X(1.05), ax.Y(.97), C.green, 11);
      PE.label(c, 'test accuracy, Co-teaching', ax.X(narrow ? 12 : 22), ax.Y(.83), C.fg, 11);
      PE.label(c, 'test accuracy, plain training', ax.X(narrow ? 12 : 22), ax.Y(.52), C.mu, 11);
    }
    o.say && o.say(`Epoch ${S.e}: keep the ${keep} smallest losses of ${n} (R(T) = ${R.toFixed(2)}). In this batch the kept images are ${prec.map(p => (100 * p).toFixed(0) + '%').join(' and ')} clean; ${B.noisy.reduce((a, b2) => a + b2, 0)} of the ${n} labels are flipped. Over all of epoch ${S.e}, network f's kept set was ${(100 * SB[S.e - 1].p1).toFixed(1)}% clean.`);
  }
  KIT.sprite('tiles2', i => { im = i; cv.redraw(); });
  PE.onTheme(() => cv.redraw());
  return { epoch(e) { S.e = e; cv.redraw(); }, reveal(on) { S.reveal = on; cv.redraw(); }, mode(m) { S.mode = m; cv.redraw(); },
    reset() { S.e = 10; S.reveal = false; S.mode = 'two'; cv.redraw(); }, redraw: () => cv.redraw(), destroy() { cv.destroy && cv.destroy(); } };
};
