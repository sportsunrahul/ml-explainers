/* FIG.gmm: DivideMix's split (ch-dividemix; atlas "split" step 2). Per-image training losses of 1,500 images from run A at
   a chosen epoch (stacked histogram: correct labels green, flipped red), a two-component Gaussian mixture fitted by EM in
   the page on min-max normalized loss as DivideMix does, the clean probability w_i, and a threshold tau. Right: label
   co-refinement on one image (toy numbers): y-bar = w y + (1 - w) p, then sharpening with temperature T. */
FIG.gmm = (el, o = {}) => {
  const G = DATA.runA.gmm, noisy = G.noisy;
  const S = { e: 10, tau: .5, w: .3, T: .5 };
  let fit = null, key = -1;
  const refit = () => { if (key === S.e) return; key = S.e; fit = KIT.gmm2(G.loss[String(S.e)]); };
  const cv = PE.canvas(el, o.aspect || (w => w < 620 ? 1.95 : .48), draw);
  function draw(c, W, H) {
    refit();
    const C = { fg: PE.css('--fg'), mu: PE.css('--muted'), blue: PE.css('--blue'), yel: PE.css('--yellow'), red: PE.css('--red'), green: PE.css('--green'), rule: PE.css('--rule') };
    c.clearRect(0, 0, W, H); H = Math.max(140, H - (o.padBottom || 0));
    const narrow = W < 620, L = G.loss[String(S.e)], mx = Math.min(8, Math.max(...L));
    const A = narrow ? { x: 40, y: 26, w: W - 54, h: H * .55 - 50 } : { x: 44, y: 26, w: W * .64 - 60, h: H - 66 };
    const nb = 40, hc = new Array(nb).fill(0), hn = new Array(nb).fill(0);
    L.forEach((v, i) => { const b = Math.min(nb - 1, Math.floor(Math.min(v, mx) / mx * nb)); (noisy[i] ? hn : hc)[b]++; });
    const top = Math.max(...hc.map((v, i) => v + hn[i]));
    const ax = PE.axes(c, A, [0, mx], [0, top * 1.1], { xt: [0, mx / 4, mx / 2, 3 * mx / 4, mx].map(v => +v.toFixed(1)), xl: `per-image loss at epoch ${S.e}` });
    const bw = A.w / nb;
    for (let b = 0; b < nb; b++) { const x = A.x + b * bw;
      c.fillStyle = C.green; c.fillRect(x + .5, ax.Y(hc[b]), bw - 1, A.y + A.h - ax.Y(hc[b]));
      c.fillStyle = C.red; c.fillRect(x + .5, ax.Y(hc[b] + hn[b]), bw - 1, ax.Y(hc[b]) - ax.Y(hc[b] + hn[b])); }
    // the two fitted bumps, scaled to counts
    const scale = L.length * (mx / nb), pts0 = [], pts1 = [];
    for (let k = 0; k <= 160; k++) { const v = mx * k / 160, d = fit.dens(v); pts0.push([ax.X(v), ax.Y(d[fit.lowIs0 ? 0 : 1] * scale)]); pts1.push([ax.X(v), ax.Y(d[fit.lowIs0 ? 1 : 0] * scale)]); }
    PE.polyline(c, pts0, C.blue, 2.4); PE.polyline(c, pts1, C.blue, 2.4, [5, 4]);
    // threshold: smallest loss whose clean probability falls below tau
    const order = L.map((v, i) => [v, fit.clean[i]]).sort((a, b) => a[0] - b[0]); let cut = mx; for (const [v, w] of order) if (w < S.tau) { cut = v; break; }
    c.strokeStyle = C.yel; c.lineWidth = 2; c.beginPath(); c.moveTo(ax.X(Math.min(cut, mx)), A.y); c.lineTo(ax.X(Math.min(cut, mx)), A.y + A.h); c.stroke();
    PE.label(c, 'keep the label', ax.X(Math.min(cut, mx)) - 6, A.y + 12, C.green, 11, 'right');
    PE.label(c, 'drop the label, keep the image', ax.X(Math.min(cut, mx)) + 6, A.y + 12, C.red, 11);
    PE.label(c, 'fitted low-loss bump (solid) and high-loss bump (dashed)', A.x, A.y - 10, C.blue, 11);
    let tp = 0, kept = 0, nc = 0; L.forEach((v, i) => { const k = fit.clean[i] >= S.tau; if (k) { kept++; if (!noisy[i]) tp++; } if (!noisy[i]) nc++; });
    // co-refinement dial (toy)
    const P = narrow ? { x: 40, y: H * .55 + 16, w: W - 54, h: H * .45 - 30 } : { x: W * .64 + 26, y: 26, w: W * .36 - 40, h: H - 66 };
    const y = [0, 1, 0], p = [.15, .25, .6], yb = y.map((v, k) => S.w * v + (1 - S.w) * p[k]), sh = yb.map(v => Math.pow(v, 1 / S.T)), z = sh.reduce((a, b) => a + b, 0), yh = sh.map(v => v / z);
    PE.label(c, 'one image: refine its label (toy numbers)', P.x, P.y - 10, C.mu, 11);
    const rows = [['given label y', y, C.yel], ['prediction p', p, C.blue], [`blend, w = ${S.w.toFixed(2)}`, yb, C.fg], [`sharpened, T = ${S.T.toFixed(2)}`, yh, C.green]];
    const rh = P.h / rows.length, cw = (P.w - 6) / 3;
    rows.forEach(([lab, v, col], r) => { PE.label(c, lab, P.x, P.y + r * rh + 12, col, 11);
      v.forEach((q, k) => { const bh = (rh - 40) * q, x = P.x + k * cw, yy = P.y + r * rh + rh - 16; c.fillStyle = col; c.globalAlpha = .85; c.fillRect(x + 4, yy - bh, cw - 10, bh); c.globalAlpha = 1;
        PE.label(c, ['cat', 'dog', 'deer'][k], x + cw / 2, yy + 11, C.mu, 9, 'center'); }); });
    o.say && o.say(`Epoch ${S.e}, tau = ${S.tau.toFixed(2)}: ${kept} of ${L.length} images keep their label; ${(100 * tp / Math.max(kept, 1)).toFixed(1)}% of those labels are correct, and ${(100 * tp / nc).toFixed(1)}% of all correct labels are kept. The others become unlabeled images.`);
  }
  PE.onTheme(() => cv.redraw());
  return { epoch(e) { S.e = e; cv.redraw(); }, tau(v) { S.tau = v; cv.redraw(); }, w(v) { S.w = v; cv.redraw(); }, T(v) { S.T = v; cv.redraw(); },
    reset() { S.e = 10; S.tau = .5; S.w = .3; S.T = .5; cv.redraw(); }, redraw: () => cv.redraw(), destroy() { cv.destroy && cv.destroy(); } };
};
