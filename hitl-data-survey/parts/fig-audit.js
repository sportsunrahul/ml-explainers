/* FIG.audit: how noisy real test sets are (ch-benchmarks; atlas "audit"). Sourced from Northcutt, Athalye and Mueller
   (2021): Table 1 (validated test-set error rate per dataset), the ImageNet funnel (Sec. 3-4), and Table S1 (accuracy of
   ImageNet models on the 1,428 correctable images with original vs corrected labels). */
FIG.audit = (el, o = {}) => {
  const T1 = [['QuickDraw', 10.12], ['CIFAR-100', 5.85], ['ImageNet', 5.83], ['Amazon Reviews', 3.90], ['IMDB', 2.90], ['Caltech-256', 1.54], ['AudioSet', 1.35], ['20news', 1.09], ['CIFAR-10', 0.54], ['MNIST', 0.15]];
  const FUN = [['ImageNet validation images', 50000], ['flagged by confident learning', 5440], ['confirmed errors (fewer than 3 of 5 workers accept the label)', 2916], ['correctable (workers agree on another class)', 1428]];
  const S1 = [['ResNet-18', 6.51, 82.42, 34, 1], ['ResNet-50', 13.52, 73.74, 20, 2], ['VGG-11 (bn)', 11.13, 72.97, 30, 4], ['DenseNet-169', 14.15, 72.55, 16, 6], ['Xception', 17.65, 68.28, 2, 24], ['NASNet-Large', 19.61, 66.32, 1, 29], ['AlexNet', 11.06, 58.96, 31, 34]];
  const S = { corrected: false, u: 1 }; let stop = () => {};
  const cv = PE.canvas(el, o.aspect || (w => w < 620 ? 1.9 : .5), draw);
  function draw(c, W, H) {
    const C = { fg: PE.css('--fg'), mu: PE.css('--muted'), blue: PE.css('--blue'), yel: PE.css('--yellow'), red: PE.css('--red'), green: PE.css('--green'), rule: PE.css('--rule') };
    c.clearRect(0, 0, W, H); H = Math.max(140, H - (o.padBottom || 0));
    const narrow = W < 620, P1 = narrow ? { x: 104, y: 22, w: W - 140, h: H * .3 } : { x: 112, y: 24, w: W * .3 - 112, h: H * .54 };
    const P2 = narrow ? { x: 8, y: H * .34 + 18, w: W - 16, h: H * .18 } : { x: 8, y: H * .62 + 10, w: W * .3 + 10, h: H * .36 };
    const P3 = narrow ? { x: 96, y: H * .56 + 24, w: W - 150, h: H * .44 - 40 } : { x: W * .34 + 110, y: 24, w: W * .66 - 170, h: H - 50 };
    // 1. Table 1 bars
    PE.label(c, 'test-set labels confirmed wrong', P1.x - 104, P1.y - 8, C.mu, 11);
    const bh = P1.h / T1.length;
    T1.forEach(([n, v], i) => { const y = P1.y + i * bh; PE.label(c, n, P1.x - 6, y + bh * .7, C.fg, 11, 'right');
      c.fillStyle = C.red; c.fillRect(P1.x, y + bh * .18, P1.w * v / 11, bh * .64); PE.label(c, v.toFixed(2) + '%', P1.x + P1.w * v / 11 + 4, y + bh * .7, C.mu, 10); });
    // 2. funnel
    PE.label(c, 'ImageNet validation set', P2.x, P2.y, C.mu, 11);
    FUN.forEach(([n, v], i) => { const y = P2.y + 8 + i * (P2.h - 8) / FUN.length, w = Math.max(3, (P2.w - 10) * Math.sqrt(v / 50000));
      c.fillStyle = i ? (i === 1 ? C.blue : C.red) : C.rule; c.fillRect(P2.x, y, w, (P2.h - 8) / FUN.length * .55);
      PE.label(c, `${v.toLocaleString('en-US')}  ${narrow || i < 2 ? n : n.split(' (')[0]}`, P2.x + Math.max(w, 60) + 6, y + 10, C.fg, 10); });
    // 3. re-ranking on the correctable set
    const key = S.corrected ? 2 : 1, rk = S.corrected ? 4 : 3, k0 = S.corrected ? 1 : 2;
    PE.label(c, `accuracy on the 1,428 correctable ImageNet images, ${S.corrected ? 'corrected' : 'original'} labels`, P3.x - 96, P3.y - 8, C.mu, 11);
    const rows = S1.slice().sort((a, b) => a[rk] - b[rk]), rh = P3.h / rows.length;
    rows.forEach((r, i) => { const yNew = P3.y + i * rh, iOld = S1.slice().sort((a, b) => a[rk === 4 ? 3 : 4] - b[rk === 4 ? 3 : 4]).indexOf(r), yOld = P3.y + iOld * rh, y = PE.lerp(yOld, yNew, S.u);
      const v = PE.lerp(r[k0], r[key], S.u);
      PE.label(c, r[0], P3.x - 6, y + rh * .65, C.fg, 11, 'right');
      c.fillStyle = S.corrected ? C.green : C.yel; c.fillRect(P3.x, y + rh * .18, (P3.w - 120) * v / 100, rh * .62);
      PE.label(c, `${v.toFixed(2)}%  rank ${r[rk]} of 34`, P3.x + (P3.w - 120) * v / 100 + 4, y + rh * .65, C.mu, 10); });
    o.say && o.say(S.corrected ? 'Corrected labels: ResNet-18 ranks 1st of 34 models on these images and NASNet-Large 29th. On the original labels, the larger models had matched the wrong labels more often.' : 'Original labels: on the images whose label is wrong, the big models score highest, because they reproduce the original (wrong) label. NASNet-Large ranks 1st of 34, ResNet-18 34th.');
  }
  function set(corr) { if (corr === S.corrected) return; S.corrected = corr; stop(); if (PE.instant()) { S.u = 1; cv.redraw(); return; } S.u = 0; stop = PE.animate(800, u => { S.u = u; cv.redraw(); }); }
  PE.onTheme(() => cv.redraw());
  return { corrected: set, reset() { stop(); S.corrected = false; S.u = 1; cv.redraw(); }, redraw: () => cv.redraw(), destroy() { stop(); cv.destroy && cv.destroy(); } };
};
