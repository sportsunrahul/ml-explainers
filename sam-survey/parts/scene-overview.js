/* Hero / atlas overview: a click on the mug's logo; the image is encoded once; three cutouts lift off with
   their predicted scores (logo, mug, mug and saucer). Toy scores. */
SCENES.overview = ctx => {
  const el = mkCanvas(ctx.stage), click = [1.115, .7], outs = TOY.three(click[0], click[1]), scores = [.79, .93, .88];
  let u = 1, cancel = () => {};
  const sm = (a, b, t) => PE.clamp((t - a) / (b - a), 0, 1);
  const cv = PE.canvas(el, S3ASPECT(ctx.stage), draw);
  function draw(c, W, H) {
    c.clearRect(0, 0, W, H);
    const narrow = W < 620, pad = 12, bottom = ctx.reserve || 0;
    const mainW = narrow ? W - 2 * pad : W * .6, mainH = narrow ? (H - bottom) * .56 : H - bottom - 2 * pad;
    const V = TOY.fit(pad, pad, mainW, mainH);
    const enc = sm(.25, .45, u), drop = sm(.12, .22, u);
    TOY.drawWorld(c, V);
    if (enc > 0) { c.save(); c.globalAlpha = .22 * (enc < 1 ? 1 : .55); c.fillStyle = PE.css('--blue'); c.fillRect(V.ox, V.oy, V.w * enc, V.h); c.restore();
      PE.label(c, 'image encoder: runs once', V.ox + 8, V.oy + 18, PE.css('--blue'), 12, 'left', 600); }
    // cursor glide, then the yellow dot
    const cx = V.X(click[0]), cy = V.Y(click[1]), g = sm(0, .12, u);
    if (u < .22) { const x = PE.lerp(V.X(.9), cx, g), y = PE.lerp(V.Y(.25), cy, g); drawCursor(c, x, y); }
    // three panels
    const panels = outs.map((n, i) => {
      const t = sm(.45 + i * .14, .59 + i * .14, u);
      let px, py, pw, ph;
      if (narrow) { pw = (W - 2 * pad - 16) / 3; ph = (H - bottom) - mainH - 3 * pad - 18; px = pad + i * (pw + 8); py = mainH + 2 * pad + 10; }
      else { const top = 40; px = mainW + 2 * pad + 6; pw = W - px - pad; ph = (H - bottom - top - 3 * pad) / 3; py = top + i * (ph + pad); }
      return { n, t, px, py, pw, ph, i };
    });
    const shown = panels.filter(p => p.t > 0), focus = u >= 1 ? panels[1] : shown[shown.length - 1];
    if (focus) TOY.drawMask(c, V, focus.n, { alpha: focus.t });
    panels.forEach(p => { if (p.t <= 0) return;
      c.save(); c.globalAlpha = p.t; c.strokeStyle = PE.css('--rule'); c.lineWidth = 1; c.beginPath(); c.roundRect(p.px, p.py, p.pw, p.ph, 8); c.stroke();
      const [x0, y0, x1, y1] = TOY.bbox(p.n), m = .06, bw = x1 - x0 + 2 * m, bh = y1 - y0 + 2 * m, lab = 22;
      const s = Math.min((p.pw - 12) / bw, (p.ph - lab - 18) / bh), V2 = { s, ox: p.px + (p.pw - bw * s) / 2 - (x0 - m) * s, oy: p.py + 6 - (y0 - m) * s };
      V2.ox = Math.min(p.px, Math.max(p.px + p.pw - TOY.WW * s, V2.ox));
      V2.X = x => V2.ox + x * s; V2.Y = y => V2.oy + y * s; V2.w = TOY.WW * s; V2.h = TOY.WH * s;
      c.save(); c.beginPath(); c.rect(p.px + 1, p.py + 1, p.pw - 2, p.ph - lab - 8); c.clip(); TOY.drawWorld(c, V2); TOY.drawMask(c, V2, p.n, { lift: 2 }); TOY.point(c, V2.X(click[0]), V2.Y(click[1]), 'pos', 1, 4); c.restore();
      const by = p.py + p.ph - lab + 2, bwid = p.pw - 16;
      c.fillStyle = PE.css('--panel2'); c.fillRect(p.px + 8, by + 10, bwid, 5); c.fillStyle = PE.css('--green'); c.fillRect(p.px + 8, by + 10, bwid * scores[p.i] * p.t, 5);
      PE.label(c, narrow ? (p.n.short || p.n.name) : p.n.name, p.px + 8, by + 4, PE.css('--fg'), narrow ? 10 : 12, 'left', 600);
      c.font = `600 ${narrow ? 10 : 12}px system-ui, sans-serif`; const nm = narrow ? (p.n.short || p.n.name) : p.n.name, fits = c.measureText(nm + '  0.00').width < p.pw - 16;
      PE.label(c, scores[p.i].toFixed(2), p.px + p.pw - 8, fits ? by + 4 : by - 10, PE.css('--green'), narrow ? 10 : 12, 'right', 600);
      c.restore(); });
    if (drop > 0) TOY.point(c, cx, cy, 'pos', drop, 7);
    if (u >= 1) { PE.label(c, 'one click', cx, V.Y(.5), PE.css('--yellow'), 12, 'center', 600);
      if (!narrow) PE.label(c, 'three valid masks, each with a predicted score', panels[0].px, H - bottom - 4, PE.css('--muted'), 12, 'left', 400); }
  }
  function drawCursor(c, x, y) { c.save(); c.translate(x, y); c.fillStyle = PE.css('--fg'); c.strokeStyle = PE.css('--bg'); c.lineWidth = 1.5;
    c.beginPath(); c.moveTo(0, 0); c.lineTo(0, 16); c.lineTo(4.5, 12); c.lineTo(8, 19); c.lineTo(10.5, 18); c.lineTo(7, 11); c.lineTo(12.5, 11); c.closePath(); c.fill(); c.stroke(); c.restore(); }
  function readout() { ctx.hud(badge('toy', u < .25 ? 'You click once on the logo of the mug.' : u < .45 ? 'The heavy image encoder runs once per image.' : u < 1 ? 'The light decoder returns three masks for the one click.' : 'One click, three valid answers: the logo, the mug, and the mug with its saucer. The scores are toy values.')); }
  function play() { cancel(); if (PE.instant()) { u = 1; cv.redraw(); readout(); return; }
    let last = -1; cancel = PE.animate(6000, t => { u = t; cv.redraw(); const k = u < .25 ? 0 : u < .45 ? 1 : u < 1 ? 2 : 3; if (k !== last) { last = k; readout(); } }); }
  play();
  return { flat: true, replay: play, resize: () => cv.resize(), destroy: () => { cancel(); cv.destroy(); } };
};
