/* Hero: 12 images x 2 views on the unit circle, then real NT-Xent steps (tau .3, lr .25, 480 steps; build/sim/hero2.js). */
KIT.hero = canvas => {
  const N = 12, TAU = .3, LR = .25, STEPS = 480, SEED = 7;
  const T_PULL = 1400, T_PUSH = 2900, T_TRAIN = 3200, T_END = 7400;
  let th, t = 0, stepsDone = 0, raf = 0;
  const init = () => {
    const r = PE.rng(SEED); th = [];
    for (let i = 0; i < N; i++) th.push(r() * 6.283);
    for (let i = 0; i < N; i++) th.push(th[i] + (r() < .5 ? -1 : 1) * (0.5 + r() * 0.9));
    stepsDone = 0;
  };
  const runTo = n => { while (stepsDone < n) { KIT.ntxentStep(th, TAU, LR); stepsDone++; } };
  const draw = (ctx, W, H) => {
    if (!W) return;
    const B = PE.css('--blue'), Y = PE.css('--yellow'), G = PE.css('--green'), R = PE.css('--red'), M = PE.css('--muted'), RU = PE.css('--rule');
    ctx.clearRect(0, 0, W, H);
    const narrow = KIT.narrow(W), cx = narrow ? W / 2 : W * .36, cy = narrow ? H * .58 : H / 2, rad = Math.min(narrow ? W * .36 : W * .26, H * (narrow ? .33 : .40));
    PE.ring(ctx, cx, cy, rad, RU, 1.5);
    const P = k => [cx + rad * Math.cos(th[k]), cy - rad * Math.sin(th[k])];
    const uPull = PE.clamp(t / 900, 0, 1), uPush = PE.clamp((t - T_PULL) / 900, 0, 1), training = t >= T_TRAIN;
    // Pull: green springs between the two views of each image.
    if (!training || stepsDone < STEPS) for (let i = 0; i < N; i++) {
      const [x1, y1] = P(i), [x2, y2] = P(i + N);
      ctx.save(); ctx.strokeStyle = G; ctx.lineWidth = 2; ctx.globalAlpha = uPull * (training ? .6 : 1);
      ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(PE.lerp(x1, x2, uPull), PE.lerp(y1, y2, uPull)); ctx.stroke(); ctx.restore();
    }
    // Push: from image 0's anchor, red arrows away from every other image's views.
    if (t >= T_PULL && !training) {
      const [ax, ay] = P(0);
      for (let k = 1; k < 2 * N; k++) {
        if (k === N) continue;
        const [x, y] = P(k), dx = ax - x, dy = ay - y, d = Math.hypot(dx, dy) || 1, L = 26 * uPush;
        PE.arrow(ctx, ax, ay, ax + dx / d * L, ay + dy / d * L, R, 1.6, 7, .55 + .45 * uPush);
      }
    }
    for (let k = 0; k < 2 * N; k++) { const [x, y] = P(k); if (k >= N) PE.ring(ctx, x, y, 8.5, Y, 2.5); else PE.dot(ctx, x, y, 5.5, B); }
    const tx = narrow ? 16 : W * .66, ty = narrow ? 26 : H * .3, fs = narrow ? 13 : 15;
    const lines = [
      [B, 'blue dot: one view of an image'], [Y, 'yellow ring: the other view'],
      [G, t > 200 ? 'green: pull each pair together' : ''], [R, t >= T_PULL ? 'red: push other images away' : ''],
      [M, training ? 'InfoNCE gradient steps: ' + stepsDone + ' of ' + STEPS : '']];
    if (narrow) lines.splice(0, 2);
    lines.forEach(([c, s], i) => s && PE.label(ctx, s, tx, ty + i * (fs + 9), c, fs, 'left', 600));
    if (stepsDone >= STEPS) PE.label(ctx, 'pull the pair, push the rest', narrow ? W / 2 : cx, narrow ? cy + rad + 30 : cy + 6, PE.css('--fg'), narrow ? 15 : 17, 'center', 700);
  };
  const cv = PE.canvas(canvas, w => KIT.narrow(w) ? 1.05 : .44, draw);
  const frame = now => {
    if (frame.t0 === undefined) frame.t0 = now;
    t = now - frame.t0;
    if (t >= T_TRAIN) runTo(Math.min(STEPS, Math.round((t - T_TRAIN) / (T_END - T_TRAIN) * STEPS)));
    cv.redraw();
    if (stepsDone < STEPS) raf = requestAnimationFrame(frame);
  };
  const replay = () => {
    cancelAnimationFrame(raf); init();
    if (PE.instant()) { t = T_END; runTo(STEPS); cv.redraw(); return; }
    frame.t0 = undefined; raf = requestAnimationFrame(frame);
  };
  init(); replay();
  return { replay, redraw: cv.redraw };
};
