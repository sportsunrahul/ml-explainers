/* FIG.ridge: memorization in run A (ch-memorize; atlas "memorize"). 3D: per-image training-loss histograms (log10 loss,
   clean images green, mislabeled red) stacked over epochs. 2D (optional second canvas): test accuracy, the share of
   mislabeled images predicted as their wrong label, and as their true label, per epoch. Computed from our CIFAR-10 run. */
FIG.ridge = (el, el2, o = {}) => {
  const A = DATA.runA, R = A.ridge, [lo, hi, nb] = A.edges, nC = A.nTrain * (1 - A.noiseFrac), nN = A.nTrain * A.noiseFrac;
  const S = { e: 12 };
  const xOf = b => -2 + 4 * (b + .5) / nb, zOf = k => 2.2 - 4.4 * k / (R.length - 1);
  const yOf = (v, n) => -1 + 2.4 * Math.sqrt(v / n);
  const idx = e => { let k = 0; R.forEach((r, i) => { if (Math.abs(r.e - e) < Math.abs(R[k].e - e)) k = i; }); return k; };
  const s3 = PE3.scene(el, { aspect: o.aspect || (w => w < 620 ? .95 : .5), fov: 34, dist: 8.6, yaw: .38, pitch: .42 });
  function build() {
    const items = [], k0 = idx(S.e);
    R.forEach((r, k) => {
      const on = k === k0, a = on ? 1 : (k < 60 && k % 2 ? 0 : .2);
      if (!on && a === 0) return;
      items.push({ type: 'line', pts: r.c.map((v, b) => [xOf(b), yOf(v, nC), zOf(k)]), color: '--green', width: on ? 3 : 1, alpha: a, link: 'clean' });
      items.push({ type: 'line', pts: r.n.map((v, b) => [xOf(b), yOf(v, nN), zOf(k)]), color: '--red', width: on ? 3 : 1, alpha: a, link: 'noisy' });
    });
    items.push({ type: 'line', pts: [[-2, -1, 2.2], [2, -1, 2.2]], color: '--muted', width: 1 });
    [-4, -2, 0, 1].forEach(t => { const x = -2 + 4 * (t - lo) / (hi - lo); items.push({ type: 'label', at: [x, -1, 2.2], text: t === 0 ? 'loss 1' : `1e${t}`, color: '--muted', dy: 16, align: 'center' }); });
    [1, 12, 30, 60, 200].forEach(e => items.push({ type: 'label', at: [2.05, -1, zOf(idx(e))], text: `epoch ${e}`, color: '--muted', dx: 6 }));
    const r = R[k0];
    items.push({ type: 'label', at: [xOf(r.c.indexOf(Math.max(...r.c))), yOf(Math.max(...r.c), nC), zOf(k0)], text: 'correct labels', color: '--green', dy: -10 });
    items.push({ type: 'label', at: [xOf(r.n.indexOf(Math.max(...r.n))), yOf(Math.max(...r.n), nN), zOf(k0)], text: 'flipped labels', color: '--red', dy: -10 });
    s3.items = items; s3.redraw(); cv2 && cv2.redraw();
    const sr = A.series[S.e - 1];
    o.say && o.say(`Epoch ${S.e}: test accuracy ${(100 * sr.test).toFixed(1)}%. Of the images with a flipped label, ${(100 * sr.asTrue).toFixed(1)}% are predicted as their true class and ${(100 * sr.fitWrong).toFixed(1)}% as their wrong label. Keeping the 60% of images with the smallest loss keeps ${(100 * sr.prec60).toFixed(1)}% correct labels.`);
  }
  const cv2 = el2 ? PE.canvas(el2, w => w < 620 ? .62 : .26, (c, W, H) => {
    const C = { fg: PE.css('--fg'), mu: PE.css('--muted'), blue: PE.css('--blue'), red: PE.css('--red'), green: PE.css('--green') };
    c.clearRect(0, 0, W, H);
    const b = { x: 42, y: 14, w: W - 56, h: H - 46 }, ax = PE.axes(c, b, [1, 200], [0, 1], { logx: true, xt: [1, 3, 10, 30, 100, 200], yt: [0, .5, 1], yf: v => (v * 100) + '%', xl: 'epoch (log scale)' });
    const ser = A.series, line = (k, col, lab) => { PE.polyline(c, ser.map(s => [ax.X(s.e), ax.Y(s[k])]), col, 2); };
    line('test', C.blue); line('fitWrong', C.red); line('asTrue', C.green);
    c.strokeStyle = C.fg; c.setLineDash([3, 3]); c.beginPath(); c.moveTo(ax.X(S.e), b.y); c.lineTo(ax.X(S.e), b.y + b.h); c.stroke(); c.setLineDash([]);
    const narrow = W < 620;
    PE.label(c, 'test accuracy', ax.X(narrow ? 1.2 : 1.6), ax.Y(.4) - 6, C.blue, 11);
    PE.label(c, 'flipped images fit to the wrong label', ax.X(narrow ? 22 : 34), ax.Y(.93), C.red, 11);
    PE.label(c, 'flipped images predicted as the true class', ax.X(1.1), ax.Y(.86), C.green, 11);
  }) : null;
  build(); PE.onTheme(() => { s3.redraw(); cv2 && cv2.redraw(); });
  return { epoch(e) { S.e = PE.clamp(Math.round(e), 1, 200); build(); }, reset() { S.e = 12; s3.setView({ yaw: .38, pitch: .42, dist: 8.6 }, 0); build(); },
    redraw: () => { s3.redraw(); cv2 && cv2.redraw(); }, destroy() { s3.destroy(); cv2 && cv2.destroy && cv2.destroy(); } };
};
