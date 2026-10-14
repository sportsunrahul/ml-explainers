/* ch-sam-arch intuition: compute per click. SAM runs the image encoder once, then each prompt costs only the
   prompt encoder + mask decoder (less than 1% of the encoder's compute, App. A). The "no split" mode re-runs a
   whole model per click. Bars are compute, not time; the 1% ratio is the paper's bound, the rest is toy. */
SCENES.costsplit = ctx => {
  const el = mkCanvas(ctx.stage);
  const S = { split: true, clicks: 3, u: 1 };
  const splitB = ctx.qa('[data-csplit]'), addB = ctx.q('[data-cadd]');
  const pr = splitB.find(b => b.getAttribute('aria-pressed') === 'true'); if (pr) S.split = pr.dataset.csplit === 'sam';
  let cancel = () => {}, prevClicks = S.clicks;
  const cv = PE.canvas(el, S3ASPECT(ctx.stage), draw);
  const sync = () => press(splitB, b => (b.dataset.csplit === 'sam') === S.split);
  const anim = () => { cancel(); cancel = PE.animate(600, t => { S.u = t; cv.redraw(); }, () => { prevClicks = S.clicks; }); };
  const offs = [...splitB.map(b => listen(b, 'click', () => { S.split = b.dataset.csplit === 'sam'; prevClicks = 0; sync(); readout(); anim(); }))];
  if (addB) offs.push(listen(addB, 'click', () => { if (S.clicks < 12) { prevClicks = S.clicks; S.clicks++; readout(); anim(); } }));
  // Long-page Reset remounts the scene: clear the pressed state first so the remount starts fresh.
  ctx.qa('[data-cue-reset]').forEach(b => offs.push(listen(b, 'click', () => ctx.qa('[aria-pressed]').forEach(x => x.removeAttribute('aria-pressed')))));
  const DEC = .01; // decoder compute as a fraction of the encoder (paper: "less than 1%")
  function draw(c, W, H) {
    c.clearRect(0, 0, W, H);
    const narrow = W < 620, pad = 16, top = 46, bottom = ctx.reserve || 0, labW = narrow ? 0 : 150;
    const x0 = pad + labW, x1 = W - pad, rowH = narrow ? 34 : 42, gap = narrow ? 30 : 26;
    const B = PE.css('--blue'), G = PE.css('--green'), M = PE.css('--muted'), F = PE.css('--fg'), Rd = PE.css('--red');
    const total = S.split ? 1 + S.clicks * DEC : S.clicks * (1 + DEC), maxT = 12 * (1 + DEC);
    const unit = (x1 - x0) / (S.split ? 1.6 : maxT * 1.02);
    const yE = top + (narrow ? 16 : 0), yD = yE + rowH + gap;
    if (narrow) { PE.label(c, 'image encoder', pad, yE - 4, B, 11); PE.label(c, 'prompt encoder + mask decoder', pad, yD - 4, G, 11); }
    else { PE.label(c, 'image encoder', pad, yE + rowH / 2 + 4, B, 12); PE.label(c, 'prompt encoder', pad, yD + rowH / 2 - 4, G, 12); PE.label(c, '+ mask decoder', pad, yD + rowH / 2 + 12, G, 12); }
    let x = x0;
    for (let k = 0; k < S.clicks; k++) {
      const fresh = k >= prevClicks, a = fresh ? S.u : 1;
      if (!S.split || k === 0) { const w = unit * a; c.fillStyle = B; c.globalAlpha = S.split || k === 0 ? .8 : .5; c.fillRect(x, yE, w, rowH); c.globalAlpha = 1;
        if (!S.split && k > 0) { c.save(); c.strokeStyle = Rd; c.setLineDash([4, 3]); c.lineWidth = 1.5; c.strokeRect(x, yE, unit, rowH); c.restore(); }
        x += unit; }
      const dw = Math.max(3, unit * DEC); c.fillStyle = G; c.globalAlpha = a; c.fillRect(x, yD, dw * a, rowH); c.globalAlpha = 1;
      TOY.point(c, x + dw / 2, yD + rowH + 10, 'pos', a, 4);
      x += dw + (S.split ? Math.max(4, (x1 - x0 - unit) / 14) : 0);
    }
    PE.label(c, S.split ? 'once per image' : 'again for every click', x0 + 6, yE + rowH / 2 + 4, PE.css('--bg'), 11, 'left', 700);
    const tY = yD + rowH + 34;
    PE.label(c, `total compute: ${total.toFixed(2)} encoder runs for ${S.clicks} click${S.clicks > 1 ? 's' : ''}`, x0, tY, S.split ? G : Rd, narrow ? 11 : 13, 'left', 700);
    PE.label(c, 'bars are compute, not time; decoder drawn at 1% of the encoder', x0, tY + 18, M, 11, 'left', 400);
  }
  function readout() {
    ctx.hud(badge('toy', S.split
      ? `SAM: the image encoder runs once; each of the ${S.clicks} clicks adds a decoder pass of less than 1% of the encoder's compute (App. A), about 50 ms in a web browser on a CPU.`
      : `Without the split, every click would run the whole network again: ${S.clicks} clicks cost ${S.clicks} full passes. Interaction would wait on the heavy part each time.`));
  }
  sync(); readout();
  return { flat: true, resize: () => cv.resize(), destroy: () => { cancel(); offs.forEach(f => f()); cv.destroy(); } };
};
