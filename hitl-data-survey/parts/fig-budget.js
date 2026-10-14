/* FIG.budget: why random is hard to beat (ch-random; atlas "budget").
   Left: TypiClust's two-region model (Hacohen et al. 2022, Eq 2) with E(m) = e^-m. The curve is the benefit of tilting a
   batch toward the easy, typical region R1: g(m) = p e^(-p m) - (1-p) alpha e^(-alpha (1-p) m), the negative derivative of
   the expected error with respect to the bias Delta at Delta = 0. Positive: label typical images. Negative: label atypical
   (uncertain) ones. Right: Munjal et al. Table 4, each AL method minus random on ResNet-18, plain vs strongly regularized. */
FIG.budget = (el, o = {}) => {
  const S = { p: .8, a: .1, sr: false };
  const T4 = {   // CIFAR-10 test accuracy at 20 / 30 / 40% labels (Munjal et al. Table 4, R18 and R18-SR columns)
    plain: { RSB: [75.73, 79.69, 80.69], Coreset: [79.27, 82.13, 85.09], DBAL: [75.05, 80.69, 82.95], QBC: [76.61, 81.77, 84.41], VAAL: [74.87, 79.96, 82.24] },
    sr: { RSB: [88.69, 90.44, 91.8], Coreset: [88.48, 91.72, 92.94], DBAL: [89.41, 92.23, 93.34], QBC: [88.34, 91.37, 92.62], VAAL: [87.11, 90.43, 92.19] } };
  const g = m => S.p * Math.exp(-S.p * m) - (1 - S.p) * S.a * Math.exp(-S.a * (1 - S.p) * m);
  const m0 = () => Math.log(S.p / (S.a * (1 - S.p))) / (S.p - S.a * (1 - S.p));
  const cv = PE.canvas(el, o.aspect || (w => w < 620 ? 1.3 : .46), draw);
  function draw(c, W, H) {
    const C = { fg: PE.css('--fg'), mu: PE.css('--muted'), blue: PE.css('--blue'), green: PE.css('--green'), brown: PE.css('--brown'), rule: PE.css('--rule') };
    c.clearRect(0, 0, W, H); H = Math.max(140, H - (o.padBottom || 0));
    const narrow = W < 620;
    const A = narrow ? { x: 46, y: 26, w: W - 60, h: H * .44 - 40 } : { x: 50, y: 30, w: W * .52 - 60, h: H - 76 };
    const Bx = narrow ? { x: 46, y: H * .5 + 30, w: W - 60, h: H * .5 - 66 } : { x: W * .52 + 46, y: 30, w: W * .48 - 60, h: H - 76 };
    // panel A: benefit of oversampling typical images vs budget
    const M = 30, ms = Array.from({ length: 241 }, (_, i) => i * M / 240), vs = ms.map(g), mx = Math.max(...vs.map(Math.abs));
    const ax = PE.axes(c, A, [0, M], [-1.05, 1.05], { xt: [0, 10, 20, 30], yt: [0], yf: () => '0', xl: 'labels per region (toy units)' });
    const z = ax.Y(0), cross = m0();
    c.save(); c.globalAlpha = .12; c.fillStyle = C.green; c.fillRect(A.x, A.y, ax.X(Math.min(cross, M)) - A.x, z - A.y); c.fillStyle = C.blue; if (cross < M) c.fillRect(ax.X(cross), z, A.x + A.w - ax.X(cross), A.y + A.h - z); c.restore();
    PE.polyline(c, ms.map((m, i) => [ax.X(m), ax.Y(vs[i] / mx)]), C.fg, 2.4);
    if (cross > 0 && cross < M) { c.save(); c.strokeStyle = C.mu; c.setLineDash([4, 4]); c.beginPath(); c.moveTo(ax.X(cross), A.y); c.lineTo(ax.X(cross), A.y + A.h); c.stroke(); c.restore();
      PE.label(c, `switch at m = ${cross.toFixed(2)}`, ax.X(cross) + 6, A.y + A.h - 8, C.fg, 12); }
    PE.label(c, 'label typical images', A.x + 6, A.y + 14, C.green, 12);
    PE.label(c, 'label uncertain images', A.x + A.w - 6, A.y + A.h - 26, C.blue, 12, 'right');
    PE.label(c, 'benefit of tilting the batch toward typical images', A.x, A.y - 12, C.mu, 11);
    // panel B: Munjal Table 4, method minus random
    const T = S.sr ? T4.sr : T4.plain, meth = ['Coreset', 'DBAL', 'QBC', 'VAAL'], col = [C.blue, C.blue, C.blue, C.blue];
    const bx = PE.axes(c, Bx, [0, 1], [-2, 5], { yt: [-2, 0, 2, 4], yf: v => (v > 0 ? '+' : '') + v });
    const gw = Bx.w / 3, bw = gw / (meth.length + 1.5);
    ['20%', '30%', '40%'].forEach((lab, k) => {
      PE.label(c, lab + ' labeled', Bx.x + gw * k + gw / 2, Bx.y + Bx.h + 14, C.mu, 11, 'center');
      meth.forEach((m, j) => { const d = T[m][k] - T.RSB[k], x = Bx.x + gw * k + bw * (j + .75), y0 = bx.Y(0), y1 = bx.Y(d);
        c.fillStyle = d >= 0 ? C.blue : C.brown; c.fillRect(x, Math.min(y0, y1), bw * .8, Math.abs(y1 - y0));
        });
    });
    c.strokeStyle = C.fg; c.beginPath(); c.moveTo(Bx.x, bx.Y(0)); c.lineTo(Bx.x + Bx.w, bx.Y(0)); c.stroke();
    PE.label(c, S.sr ? 'ResNet-18 with RandAugment + SWA: AL minus random' : 'plain ResNet-18: AL minus random (points)', Bx.x, Bx.y - 12, C.mu, 11);
    PE.label(c, 'bars, left to right: Coreset, DBAL, QBC, VAAL', Bx.x + 4, Bx.y + 12, C.mu, 10);
    const best = k => Math.max(...meth.map(m => T[m][k] - T.RSB[k]));
    o.say && o.say(`Toy: with p = ${S.p.toFixed(2)} and alpha = ${S.a.toFixed(2)}, typical images are the better pick until m = ${cross.toFixed(2)}, uncertain ones after. Munjal Table 4 (${S.sr ? 'strongly regularized' : 'plain'} ResNet-18): best AL method minus random = ${[0, 1, 2].map(k => '+' + best(k).toFixed(2)).join(', ')} points at 20 / 30 / 40% labels.`);
  }
  PE.onTheme(() => cv.redraw());
  return { p(v) { S.p = v; cv.redraw(); }, a(v) { S.a = v; cv.redraw(); }, sr(on) { S.sr = on; cv.redraw(); },
    reset() { S.p = .8; S.a = .1; S.sr = false; cv.redraw(); }, redraw: () => cv.redraw(), destroy() { cv.destroy && cv.destroy(); } };
};
