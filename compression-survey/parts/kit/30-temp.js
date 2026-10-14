/* Temperature toy for knowledge distillation. The logits are ours (one "truck" image, six classes); the formulas are
   Hinton 2015 Eq. 1 (softmax at temperature T), Eq. 2 (soft-loss gradient (q - p) / T) and Eq. 4 (its high-T limit,
   computed on zero-meaned logits, since the softmax ignores a shift). Glow words: teacher, student, logits, temp, gap,
   kl, hard, tsq, plain, limit. */
KIT.TEMP_TOY = { classes: ['truck', 'bus', 'car', 'motorbike', 'person', 'cat'],
  v: [9, 6, 4.5, 2, 0.5, -2], z: [7, 2.5, 4.5, 3, 2, 2.5], label: 0 };
KIT.tempSoft = (a, T) => { const m = Math.max(...a), e = a.map(x => Math.exp((x - m) / T)), s = e.reduce((u, x) => u + x, 0); return e.map(x => x / s); };
KIT.tempCore = (T, toy = KIT.TEMP_TOY) => {
  const { v, z } = toy, N = v.length, p = KIT.tempSoft(v, T), q = KIT.tempSoft(z, T), q1 = KIT.tempSoft(z, 1);
  const norm = a => Math.sqrt(a.reduce((s, x) => s + x * x, 0)), mean = a => a.reduce((s, x) => s + x, 0) / a.length;
  const kl = p.reduce((s, pi, i) => s + pi * Math.log(pi / q[i]), 0);
  const gap = q.map((qi, i) => qi - p[i]), g = norm(gap) / T;
  const hard = norm(q1.map((x, i) => x - (i === toy.label ? 1 : 0)));
  const mz = mean(z), mv = mean(v), lim = norm(z.map((x, i) => (x - mz) - (v[i] - mv))) / N;
  return { T, p, q, q1, kl, gap, g, gT2: g * T * T, hard, limit: lim / (T * T), limitT2: lim };
};
KIT.temp = (el, opts = {}) => {
  const toy = KIT.TEMP_TOY, N = toy.classes.length;
  const S = { T: 1, Tt: 1, k: 0, t2: false };
  const glow = opts.fig ? PE.linkPulses(opts.fig, () => cv.redraw()) : () => 0;
  const curve = []; for (let T = 1; T <= 20.001; T += .25) curve.push(KIT.tempCore(T, toy));
  const f2 = x => x >= .1 ? x.toFixed(2) : x >= .001 ? x.toFixed(3) : '<.001';
  function draw(ctx, w, h) {
    ctx.clearRect(0, 0, w, h);
    const C = n => PE.css(n), B = C('--blue'), Y = C('--yellow'), R = C('--red'), F = C('--fg'), M = C('--muted');
    const narrow = w < 560, H = h - (opts.bottomPad || 0), c = KIT.tempCore(S.T, toy), fs = narrow ? 10 : 11;
    // panel boxes: probabilities (b1) and gradient size vs T (b2)
    let b1, b2;
    if (narrow) {
      const h1 = Math.round(H * .56);
      b1 = { x: 44, y: 46, w: w - 54, h: h1 - 46 - 52 };
      b2 = { x: 52, y: h1 + 44, w: w - 66, h: H - h1 - 44 - 40 };
    } else {
      const wl = Math.round(w * .56);
      b1 = { x: 58, y: 50, w: wl - 70, h: H - 50 - 62 };
      b2 = { x: wl + 64, y: 50, w: w - wl - 64 - 16, h: H - 50 - 46 };
    }
    // left: teacher (blue) and student (yellow) softmax at temperature T
    const A = PE.axes(ctx, b1, [0, 1], [0, 1], { yt: [0, .5, 1], yl: 'probability', ylo: narrow ? 30 : 38 });
    const gt = glow('temp');
    ctx.save(); if (gt > 0) { ctx.globalAlpha = .25 * gt; ctx.fillStyle = Y; ctx.fillRect(b1.x + 2, b1.y - 36, 92, 24); } ctx.restore();
    PE.label(ctx, 'T = ' + S.T.toFixed(1), b1.x + 6, b1.y - 18, F, narrow ? 14 : 16, 'left', 700);
    const gk = glow('kl');
    PE.label(ctx, 'KL(teacher || student) = ' + c.kl.toFixed(3), b1.x + b1.w, b1.y - 18, R, (narrow ? 11 : 12) + 2 * gk, 'right', 600 + (gk > .3 ? 100 : 0));
    const gw = b1.w / N, bw = Math.min(gw * .3, 28), gl = glow('logits'), gh = glow('hard');
    if (gl > 0) { ctx.save(); ctx.globalAlpha = .2 * gl; ctx.fillStyle = Y; ctx.fillRect(b1.x - 40, b1.y + b1.h + 22, b1.w + 40, 30); ctx.restore(); }
    PE.label(ctx, 'v/T', b1.x - 6, b1.y + b1.h + 34, B, fs, 'right');
    PE.label(ctx, 'z/T', b1.x - 6, b1.y + b1.h + 48, Y, fs, 'right');
    toy.classes.forEach((name, i) => {
      const cx = b1.x + gw * (i + .5), xt = cx - bw - 1, xs = cx + 1;
      const yp = A.Y(c.p[i]), yq = A.Y(c.q[i]), y0 = A.Y(0);
      const gT = glow('teacher'), gS = glow('student');
      ctx.fillStyle = B; ctx.fillRect(xt, yp, bw, y0 - yp);
      if (gT > 0) { ctx.save(); ctx.strokeStyle = B; ctx.lineWidth = 3 * gT; ctx.strokeRect(xt - 2, yp - 2, bw + 4, y0 - yp + 2); ctx.restore(); }
      ctx.save(); ctx.globalAlpha = .9; ctx.fillStyle = Y; ctx.fillRect(xs, yq, bw, y0 - yq); ctx.restore();
      if (gS > 0) { ctx.save(); ctx.strokeStyle = Y; ctx.lineWidth = 3 * gS; ctx.strokeRect(xs - 2, yq - 2, bw + 4, y0 - yq + 2); ctx.restore(); }
      if (!narrow) PE.label(ctx, f2(c.p[i]), xt + bw / 2, yp - 5, B, 10, 'center', 500);
      // red arrow: the soft loss pulls the student's probability toward the teacher's
      const ga = glow('gap');
      if (Math.abs(yp - yq) > 7) PE.arrow(ctx, xs + bw / 2, yq, xs + bw / 2, yp, R, 2 + 2 * ga, 7);
      const isL = i === toy.label;
      PE.label(ctx, isL ? name + ' (y)' : name, cx, b1.y + b1.h + 17, isL ? F : M, fs, 'center', isL ? 700 : 500);
      if (isL && gh > 0) PE.ring(ctx, cx, b1.y + b1.h + 13, 30, R, 2, gh);
      PE.label(ctx, (toy.v[i] / S.T).toFixed(narrow && S.T > 1.5 ? 1 : 2), cx, b1.y + b1.h + 34, B, fs, 'center', 500);
      PE.label(ctx, (toy.z[i] / S.T).toFixed(narrow && S.T > 1.5 ? 1 : 2), cx, b1.y + b1.h + 48, Y, fs, 'center', 500);
    });
    // right: size of the soft-loss gradient against T, without and with the T^2 factor (log scale)
    PE.label(ctx, 'soft-loss gradient size |dC/dz| (log scale)', b2.x, b2.y - 18, M, fs, 'left', 500);
    const A2 = PE.axes(ctx, b2, [1, 20], [.001, 3], { logy: true, xt: [1, 5, 10, 15, 20], yt: [.001, .01, .1, 1],
      xl: 'temperature T', yl: 'gradient size', ylo: narrow ? 38 : 44 });
    const cl = y => Math.max(.001, Math.min(3, y));
    const gp = glow('plain'), g2 = glow('tsq'), gL = glow('limit');
    const aP = 1 - .7 * S.k, a2 = .3 + .7 * S.k;
    PE.polyline(ctx, curve.map(d => [A2.X(d.T), A2.Y(cl(d.limit))]), M, 1.3 + 2 * gL, [2, 4], (.35 + .5 * (1 - S.k)) + gL * .3);
    PE.polyline(ctx, curve.map(d => [A2.X(d.T), A2.Y(cl(d.limitT2))]), M, 1.3 + 2 * gL, [2, 4], (.35 + .5 * S.k) + gL * .3);
    PE.label(ctx, 'Eq. 4 limit', A2.X(1) + 4, A2.Y(c.limitT2) - 6, M, 10, 'left', 500);
    PE.polyline(ctx, [[A2.X(1), A2.Y(c.hard)], [A2.X(20), A2.Y(c.hard)]], M, 1.4 + 2 * gh, [8, 5], .9);
    PE.label(ctx, 'hard-label gradient', A2.X(20) - 2, A2.Y(c.hard) + 13, gh > 0 ? R : M, 10, 'right', 500);
    PE.polyline(ctx, curve.map(d => [A2.X(d.T), A2.Y(cl(d.g))]), R, 2.2 + 2 * gp, S.k > .5 ? [5, 4] : null, PE.clamp(aP + gp, 0, 1));
    PE.polyline(ctx, curve.map(d => [A2.X(d.T), A2.Y(cl(d.gT2))]), R, 2.2 + 2 * g2, S.k > .5 ? null : [5, 4], PE.clamp(a2 + g2, 0, 1));
    const last = curve[curve.length - 1];
    PE.label(ctx, 'with T\u00b2', A2.X(20) - 2, A2.Y(last.gT2) + 15, R, fs, 'right', S.k > .5 ? 700 : 500);
    PE.label(ctx, 'without T\u00b2', A2.X(20) - 2, A2.Y(last.g) - 8, R, fs, 'right', S.k > .5 ? 500 : 700);
    const xm = A2.X(S.T);
    PE.polyline(ctx, [[xm, b2.y], [xm, b2.y + b2.h]], M, 1 + 2 * gt, [3, 3], .8);
    PE.dot(ctx, xm, A2.Y(cl(c.g)), 4 + 2 * (1 - S.k), R, aP);
    PE.dot(ctx, xm, A2.Y(cl(c.gT2)), 4 + 2 * S.k, R, a2);
    const sel = S.k > .5 ? c.gT2 : c.g, ys = A2.Y(cl(sel));
    PE.label(ctx, f2(sel), xm + (S.T > 14 ? -8 : 8), ys + (S.k > .5 ? -8 : 16), F, 12, S.T > 14 ? 'right' : 'left', 700);
  }
  const cv = PE.canvas(el, opts.aspect || (w => w < 560 ? 1.5 : .5), draw);
  let stopT = () => {}, stopK = () => {};
  function set(next) {
    if ('T' in next) { S.Tt = PE.clamp(+next.T, 1, 20); stopT(); stopT = KIT.tween(S, { T: S.Tt }, 450, cv.redraw); }
    if ('t2' in next) { S.t2 = !!next.t2; stopK(); stopK = KIT.tween(S, { k: S.t2 ? 1 : 0 }, 650, cv.redraw); }
    cv.redraw();
  }
  const info = () => ({ ...KIT.tempCore(S.Tt, toy), t2: S.t2, classes: toy.classes });
  return { set, info, state: S, redraw: cv.redraw, resize: cv.resize, destroy: () => { stopT(); stopK(); cv.destroy(); } };
};
KIT.tempText = i => {
  const f = x => x >= .1 ? x.toFixed(2) : x.toFixed(3), r = x => x >= 1 ? x.toFixed(1) : x.toFixed(2);
  const sel = i.t2 ? i.gT2 : i.g, oth = i.t2 ? i.g : i.gT2;
  return `<b>T = ${i.T.toFixed(1)}</b>. Teacher: truck ${f(i.p[0])}, bus ${f(i.p[1])}, cat ${f(i.p[5])}. ` +
    `Student: truck ${f(i.q[0])}, bus ${f(i.q[1])}, cat ${f(i.q[5])}. KL(teacher || student) = ${i.kl.toFixed(3)}. ` +
    `Soft-loss gradient size ${i.t2 ? 'with' : 'without'} T<sup>2</sup>: <b class="c-red">${f(sel)}</b>, ` +
    `${r(sel / i.hard)}${KIT.times} the hard-label gradient (${f(i.hard)}); ${i.t2 ? 'without' : 'with'} T<sup>2</sup> it would be ${f(oth)} (${r(oth / i.hard)}${KIT.times}).`;
};
