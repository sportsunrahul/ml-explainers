/* FIG.hero: one round of the loop with real CIFAR-10 tiles (page hero; atlas "loop" stage).
   The model scores a stream (blue bars), the three least-confident images get a yellow ring and go to a labeler,
   the tagged images join the training set, the loss exposes the one wrong tag (red strike, back to review). */
FIG.hero = (el, o = {}) => {
  const H0 = SPRITES.hero;
  // Sprite tile 8 carries a flipped label (run A's noisy labels); each layout routes it through the labeler.
  const PLAN = {
    wide: { stream: [2, 3, 4, 5, 6, 7, 8, 9, 10, 11], conf: [.93, .38, .88, .95, .31, .9, .27, .86, .97, .84], pick: [1, 4, 6], train: [12, 13, 14, 15, 16, 17, 18] },
    narrow: { stream: [2, 3, 4, 5, 8, 9], conf: [.93, .38, .88, .31, .27, .95], pick: [1, 3, 4], train: [12, 13, 14] } };
  const S = { u: 1 }; let im = null, stop = () => {};
  const seg = (u, a, b) => PE.clamp((u - a) / (b - a), 0, 1);
  const cv = PE.canvas(el, o.aspect || (w => w < 620 ? 1.08 : .44), draw);
  function draw(c, W, H) {
    const C = { fg: PE.css('--fg'), mu: PE.css('--muted'), rule: PE.css('--rule'), blue: PE.css('--blue'), yel: PE.css('--yellow'), green: PE.css('--green'), red: PE.css('--red') };
    c.clearRect(0, 0, W, H);
    H = Math.max(140, H - (o.padBottom || 0));
    const narrow = W < 620, L = narrow ? PLAN.narrow : PLAN.wide, u = S.u, n = L.stream.length;
    const s = Math.round(PE.clamp(W / (narrow ? 7.4 : 13.5), 30, 58));
    const y1 = H * (narrow ? .1 : .13), y2 = H * (narrow ? .43 : .44), y3 = H * (narrow ? .74 : .72);
    const x0 = W * .03, step = (W * .97 - x0 - s) / (n - 1);
    const slide = seg(u, 0, .16), bars = seg(u, .1, .26), ringU = seg(u, .26, .38), drop = seg(u, .38, .52), join = seg(u, .52, .66);
    const lossU = seg(u, .66, .78), markU = seg(u, .78, .9), capU = seg(u, .9, 1);
    const boxW = 3 * s * 1.9 + 16, bx = (W - boxW) / 2, labX = k => bx + 12 + k * s * 1.9;
    c.save(); c.strokeStyle = C.yel; c.lineWidth = 1.5; c.setLineDash([5, 4]); c.strokeRect(bx, y2 - 22, boxW, s + 34); c.restore();
    PE.label(c, 'labeler', bx, y2 - 28, C.yel, 12);
    PE.label(c, 'stream: the model scores each image', x0, y1 - 10, C.blue, 12);
    PE.label(c, 'training set', x0, y3 - 22, C.mu, 12);
    const nT = L.train.length + 3, tstep = Math.min(step, (W * .97 - x0 - s) / (nT - 1)), trainX = k => x0 + k * tstep;
    let wrongAt = null;
    L.stream.forEach((t, k) => {
      const pi = L.pick.indexOf(k);
      let x = x0 + k * step - (1 - slide) * (W * .5 + k * 8), y = y1;
      if (pi >= 0) { x = PE.lerp(x, labX(pi), drop); y = PE.lerp(y, y2, drop); x = PE.lerp(x, trainX(L.train.length + pi), join); y = PE.lerp(y, y3, join); }
      KIT.tile(c, im, t, x, y, s);
      if (pi < 0 || drop === 0) { c.fillStyle = C.rule; c.fillRect(x, y + s + 5, s, 6); c.fillStyle = C.blue; c.fillRect(x, y + s + 5, s * L.conf[k] * bars, 6); }
      if (pi < 0) return;
      KIT.ring(c, x + s / 2, y + s / 2, s * .72, C.yel, { u: ringU, seed: 11 + k, alpha: 1 - join * .85 });
      const wrong = H0.given[t] !== H0.true[t];
      if (drop > .6) { c.globalAlpha = Math.min(1, (drop - .6) / .4); PE.label(c, KIT.NAMES[H0.given[t]], x + s / 2, y - 6, wrong && markU > 0 ? C.red : C.yel, 11, 'center'); c.globalAlpha = 1; }
      if (join < 1) return;
      const lh = (wrong ? .95 : .12) * s * lossU;
      c.fillStyle = C.red; c.globalAlpha = .85; c.fillRect(x + s + 3, y + s - lh, 5, lh); c.globalAlpha = 1;
      if (wrong) { KIT.strike(c, x + s / 2, y + s / 2, s * 1.05, C.red, { u: markU, seed: 5 }); wrongAt = [x + s / 2, y - 18]; }
      else KIT.tick(c, x + s / 2, y + s + 14, s * .4, C.green, { u: markU });
    });
    L.train.forEach((t, k) => { const x = trainX(k); KIT.tile(c, im, t, x, y3, s); KIT.tick(c, x + s / 2, y3 + s + 14, s * .4, C.green, { alpha: .55 }); PE.label(c, KIT.NAMES[H0.given[t]], x + s / 2, y3 - 6, C.yel, 11, 'center'); });
    if (wrongAt && markU > .5) {   // back to review: a dashed arc from the struck tile to the labeler
      const a = (markU - .5) / .5, ex = bx + boxW, ey = y2 + s / 2;
      c.save(); c.strokeStyle = C.red; c.lineWidth = 2; c.setLineDash([5, 4]); c.globalAlpha = a; c.beginPath(); c.moveTo(wrongAt[0], wrongAt[1]);
      c.quadraticCurveTo(Math.max(wrongAt[0], ex) + 30, (wrongAt[1] + ey) / 2, ex + 2, ey); c.stroke(); c.restore();
      if (a > .9) PE.label(c, 'back to review', W - 8, (wrongAt[1] + ey) / 2 - 10, C.red, 11, 'right');
    }
    if (capU > 0) {
      c.globalAlpha = capU;
      PE.label(c, 'Part one: which images get a label?', bx, y2 + s + 30, C.yel, narrow ? 12 : 13);
      PE.label(c, 'Part two: which labels do we believe?', x0, Math.min(H - 6, y3 + s + 36), C.red, narrow ? 12 : 13);
      c.globalAlpha = 1;
    }
  }
  function play() { stop(); if (PE.instant()) { S.u = 1; cv.redraw(); return; } S.u = 0; stop = PE.animate(7000, u => { S.u = u; o.onFrame && o.onFrame(u); cv.redraw(); }); }
  KIT.sprite('hero', i => { im = i; cv.redraw(); });
  PE.onTheme(() => cv.redraw());
  play();
  return { replay: play, redraw: () => cv.redraw(), destroy() { stop(); cv.destroy && cv.destroy(); } };
};
