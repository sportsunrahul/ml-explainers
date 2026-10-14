/* ch-inspection (toy): a machined plate with a bolt and three defects of different kinds: a dent (clear edges), a
   thin scratch, and a faint stain (low contrast). Tabs: 0 one click per defect, as SAM answers it; 1 a box per
   defect from a detector; 2 SAM's box masks used as noisy labels to train a small student segmenter. The
   behaviours follow the failure modes the inspection papers report; the drawing and the outcomes are ours. */
SCENES.inspect = ctx => {
  const el = mkCanvas(ctx.stage), S = { step: 0, u: 1 };
  const WW = 1.6, WH = 1;
  const plate = (() => { const p = new Path2D(); p.roundRect(.12, .14, 1.36, .74, .06); return p; })();
  const ell = (cx, cy, rx, ry, a = 0) => { const p = new Path2D(); p.ellipse(cx, cy, rx, ry, a, 0, 7); return p; };
  const DEF = [
    { id: 'dent', name: 'dent', at: [.42, .55], shape: ell(.42, .55, .075, .055, .3) },
    { id: 'scratch', name: 'thin scratch', at: [.86, .42], pts: [[.66, .58], [.78, .47], [.9, .4], [1.05, .3]] },
    { id: 'stain', name: 'faint stain', at: [1.2, .66], shape: (() => { const p = new Path2D(); p.moveTo(1.12, .62); p.bezierCurveTo(1.15, .55, 1.27, .56, 1.3, .63); p.bezierCurveTo(1.33, .7, 1.22, .76, 1.16, .72); p.bezierCurveTo(1.11, .7, 1.1, .66, 1.12, .62); return p; })() }];
  const scratchPath = (a = 0, b = 1, w = .012) => { const P = DEF[1].pts, n = 40, top = [], bot = [];
    const at = t => { const s = t * (P.length - 1), i = Math.min(P.length - 2, Math.floor(s)), f = s - i; return [PE.lerp(P[i][0], P[i + 1][0], f), PE.lerp(P[i][1], P[i + 1][1], f)]; };
    for (let k = 0; k <= n; k++) { const t = PE.lerp(a, b, k / n), p = at(t), q = at(Math.min(1, t + .01)), dx = q[0] - p[0], dy = q[1] - p[1], L = Math.hypot(dx, dy) || 1; top.push([p[0] - dy / L * w, p[1] + dx / L * w]); bot.push([p[0] + dy / L * w, p[1] - dx / L * w]); }
    const pth = new Path2D(); [...top, ...bot.reverse()].forEach(([x, y], i) => i ? pth.lineTo(x, y) : pth.moveTo(x, y)); pth.closePath(); return pth; };
  // what each method returns per defect: [path, verdict, ok?]
  const OUT = [
    [[DEF[0].shape, 'good', true], [plate, 'wrong: the whole plate', false], [plate, 'wrong: the whole plate', false]],
    [[DEF[0].shape, 'good', true], [scratchPath(.3, .62, .016), 'partial: misses the thin ends', false], [ell(1.2, .655, .055, .035, .2), 'found, but the extent is wrong', false]],
    [[DEF[0].shape, 'good', true], [scratchPath(.04, .97, .014), 'closer', true], [DEF[2].shape, 'closer', true]]];
  let cancel = () => {};
  const cv = PE.canvas(el, S3ASPECT(ctx.stage), draw);
  function fit(x, y, w, h) { const s = Math.min(w / WW, h / WH); return { s, ox: x + (w - WW * s) / 2, oy: y + (h - WH * s) / 2 }; }
  function draw(c, W, H) {
    c.clearRect(0, 0, W, H); const narrow = W < 620, bottom = ctx.reserve || 0, top = 34, V = fit(10, top, W - 20, H - bottom - top - (narrow ? 70 : 10)), u = S.u;
    const tone = t => PE.mix(PE.css('--bg'), PE.css('--fg'), t), G = PE.css('--green');
    c.save(); c.translate(V.ox, V.oy); c.scale(V.s, V.s);
    c.fillStyle = tone(.06); c.fillRect(0, 0, WW, WH);
    c.fillStyle = tone(.3); c.fill(plate); c.save(); c.clip(plate); c.strokeStyle = tone(.34); c.lineWidth = 1 / V.s; for (let y = .15; y < .9; y += .012) { c.beginPath(); c.moveTo(.1, y); c.lineTo(1.5, y + .01); c.stroke(); } c.restore();
    c.fillStyle = tone(.45); c.fill(ell(1.32, .27, .055, .055)); c.fillStyle = tone(.25); c.fill(ell(.25, .27, .03, .03));
    c.fillStyle = tone(.16); c.fill(DEF[0].shape); c.strokeStyle = tone(.5); c.lineWidth = 1.5 / V.s; c.stroke(DEF[0].shape);
    c.strokeStyle = tone(.5); c.lineWidth = 2.2 / V.s; c.beginPath(); DEF[1].pts.forEach(([x, y], i) => i ? c.lineTo(x, y) : c.moveTo(x, y)); c.stroke();
    c.fillStyle = tone(.275); c.fill(DEF[2].shape);
    OUT[S.step].forEach(([p, v, ok], i) => { const a = PE.clamp(u * 1.6 - i * .25, 0, 1); if (!a) return; c.save(); c.globalAlpha = a * (p === plate ? .26 : .45); c.fillStyle = G; c.fill(p); c.globalAlpha = a; c.strokeStyle = G; c.lineWidth = 2 / V.s; c.stroke(p); c.restore(); });
    c.restore();
    const X = x => V.ox + x * V.s, Y = y => V.oy + y * V.s;
    DEF.forEach((d, i) => { const [p, v, ok] = OUT[S.step][i], a = PE.clamp(u * 1.6 - i * .25, 0, 1);
      if (S.step === 0) TOY.point(c, X(d.at[0]), Y(d.at[1]), 'pos', 1, 5);
      if (S.step >= 1) { const r = { dent: [.33, .48, .51, .62], scratch: [.64, .27, 1.07, .61], stain: [1.09, .54, 1.33, .78] }[d.id]; TOY.boxPrompt(c, X(r[0]), Y(r[1]), X(r[2]), Y(r[3]), S.step === 2 ? .45 : 1); }
      const lx = narrow ? 14 : X(d.at[0]), ly = narrow ? H - bottom - 58 + i * 18 : Y(d.id === 'scratch' ? .2 : .86) + (d.id === 'scratch' ? 0 : 26);
      c.save(); c.globalAlpha = a; PE.label(c, `${d.name}: ${v}`, lx, ly, ok ? G : PE.css('--red'), narrow ? 11 : 12, narrow ? 'left' : 'center', 600); c.restore(); });
  }
  const text = ['One click on each defect. SAM was trained on masks of objects, so a click on a defect without an object-like boundary can return the object around it: the plate (our illustration). The clear dent works.',
    'A box from a defect detector fixes most of the "which object" problem. Thin and low-contrast defects still come back partial or with the wrong extent.',
    'Use SAM\'s box masks as noisy training labels for a small, fast segmenter, with a loss that tolerates label noise. The student gets closer, not perfect; at run time only the student runs.'];
  function readout() { ctx.hud(badge('toy', text[S.step])); }
  function animateIn() { cancel(); cancel = PE.animate(800, t => { S.u = t; cv.redraw(); }); }
  readout();
  return { flat: true, step(n) { S.step = n; readout(); animateIn(); }, resize: () => cv.resize(), destroy: () => { cancel(); cv.destroy(); } };
};
