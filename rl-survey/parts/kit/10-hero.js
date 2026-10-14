/* Hero: the yard learns. A real seeded Q-learning run (150 episodes, alpha .3, gamma .9, epsilon .2, seed 7; its
   greedy policy agrees with value iteration on 15 of 16 cells) plays back, slowly at first and then fast. Blue fill
   is max_a Q(s, a); green arrows are the greedy actions. It ends on the learned path and the agent-environment loop. */
KIT.hero = canvas => {
  const W = KIT.makeGrid(KIT.YARD), RUN = KIT.qlearnRun(W, { episodes: 150, alpha: .3, gamma: .9, eps: .2, seed: 7, maxSteps: 60 });
  const F = RUN.steps.length, T_IN = 500, T_LEARN = 5200, T_END = 6900;
  let t = 0, raf = 0, Q, applied;
  const reset = () => { Q = Array.from({ length: W.n }, () => [0, 0, 0, 0]); applied = 0; };
  const applyTo = k => { while (applied < k) { const st = RUN.steps[applied]; Q[st.s][st.a] = st.q; applied++; } };
  const greedyPath = () => {
    const path = [W.start]; let s = W.start;
    while (!W.isTerminal(s) && path.length < 30) { s = W.move(s, Q[s].indexOf(Math.max(...Q[s]))); path.push(s); }
    return path;
  };
  const loop = (ctx, box, u, narrow) => {
    if (u <= 0) return;
    const G = PE.css('--green'), BR = PE.css('--brown'), Y = PE.css('--yellow'), FG = PE.css('--fg'), fs = narrow ? 12 : 14;
    const bw = narrow ? box.w * .3 : Math.min(190, box.w * .62), bh = narrow ? 40 : 46;
    let ag, en;
    if (narrow) { ag = { x: box.x, y: box.y + box.h / 2 - bh / 2 }; en = { x: box.x + box.w - bw, y: ag.y }; }
    else { ag = { x: box.x + (box.w - bw) / 2, y: box.y + box.h * .12 }; en = { x: ag.x, y: box.y + box.h * .88 - bh }; }
    PE.box(ctx, ag.x, ag.y, bw, bh, G, 'agent', Math.min(1, u * 3));
    PE.box(ctx, en.x, en.y, bw, bh, BR, 'environment', Math.min(1, u * 3));
    const v = PE.clamp((u - .25) / .75, 0, 1); if (!v) return;
    if (narrow) {
      const x1 = ag.x + bw, x2 = en.x, yA = ag.y - 10, yS = ag.y + bh + 12, yR = ag.y + bh + 30;
      PE.arrow(ctx, x1 + 4, ag.y + 10, PE.lerp(x1 + 4, x2 - 4, v), en.y + 10, G, 2.2, 8);
      PE.label(ctx, 'action A\u209c', (x1 + x2) / 2, yA, G, fs, 'center');
      PE.arrow(ctx, x2 - 4, en.y + bh - 10, PE.lerp(x2 - 4, x1 + 4, v), ag.y + bh - 10, FG, 2, 8, v);
      PE.label(ctx, 'next state S\u209c\u208a\u2081', (x1 + x2) / 2, yS, FG, fs, 'center');
      PE.label(ctx, 'reward R\u209c\u208a\u2081', (x1 + x2) / 2, yR, Y, fs, 'center');
    } else {
      const xr = ag.x + bw + 30, xl = ag.x - 30, ya = ag.y + bh / 2, ye = en.y + bh / 2;
      PE.polyline(ctx, [[ag.x + bw, ya], [xr, ya], [xr, PE.lerp(ya, ye, v)]], G, 2.2);
      if (v === 1) PE.arrow(ctx, xr, ye, ag.x + bw + 4, ye, G, 2.2, 9);
      PE.label(ctx, 'action A\u209c', xr + 8, (ya + ye) / 2, G, fs, 'left');
      PE.polyline(ctx, [[en.x, ye - 8], [xl, ye - 8], [xl, PE.lerp(ye - 8, ya - 8, v)]], FG, 2);
      if (v === 1) PE.arrow(ctx, xl, ya - 8, ag.x - 4, ya - 8, FG, 2, 9);
      PE.polyline(ctx, [[en.x, ye + 8], [xl - 16, ye + 8], [xl - 16, PE.lerp(ye + 8, ya + 8, v)]], Y, 2);
      if (v === 1) PE.arrow(ctx, xl - 16, ya + 8, ag.x - 4, ya + 8, Y, 2, 9);
      PE.label(ctx, 'next state S\u209c\u208a\u2081', xl - 26, (ya + ye) / 2 - 10, FG, fs, 'right');
      PE.label(ctx, 'reward R\u209c\u208a\u2081', xl - 26, (ya + ye) / 2 + 12, Y, fs, 'right');
    }
  };
  const draw = (ctx, w, h) => {
    if (!w) return;
    ctx.clearRect(0, 0, w, h);
    const narrow = KIT.narrow(w), M = PE.css('--muted');
    const gbox = narrow ? { x: 8, y: 8, w: w - 16, h: h * .62 } : { x: 8, y: 14, w: w * .5, h: h - 44 };
    const L = KIT.gridLayout(W, gbox), uIn = PE.clamp(t / T_IN, 0, 1), done = t >= T_LEARN;
    const V = Q.map((q, s) => q.some(x => x !== 0) ? Math.max(...q) : null);
    // Arrows only where one action is strictly best: a tie means the agent has no preference there yet.
    const pi = Q.map(q => { const m = Math.max(...q), top = q.filter(x => x === m).length;
      return q.some(x => x !== 0) && top === 1 ? q.map(x => x === m ? 1 : 0) : null; });
    const cur = applied ? RUN.steps[applied - 1] : null, speed = applied / F;
    const agentAlpha = done ? 0 : PE.clamp(1 - (speed - .15) / .25, 0, 1);
    KIT.drawGrid(ctx, W, L, { V, vRange: [-.2, 1], pi, alpha: uIn,
      agent: { s: cur && !W.isTerminal(cur.s2) ? cur.s2 : W.start, alpha: agentAlpha * uIn },
      path: done ? greedyPath() : null, pathAlpha: PE.clamp((t - T_LEARN) / 600, 0, 1) });
    const ep = cur ? cur.ep + 1 : 0, ty = L.y0 + L.h + (narrow ? 18 : 22);
    PE.label(ctx, done ? 'after 150 episodes: the greedy path to the charger' : 'Q-learning, episode ' + ep + ' of 150', L.x0, ty, M, narrow ? 12 : 13, 'left', 600);
    const lbox = narrow ? { x: 16, y: h * .62 + 34, w: w - 32, h: h * .38 - 44 } : { x: w * .56, y: 14, w: w * .42, h: h - 28 };
    loop(ctx, lbox, PE.clamp((t - T_LEARN) / (T_END - T_LEARN), 0, 1), narrow);
  };
  const cv = PE.canvas(canvas, w => KIT.narrow(w) ? 1.35 : .46, draw);
  const frame = now => {
    if (frame.t0 === undefined) frame.t0 = now;
    t = now - frame.t0;
    const u = PE.clamp((t - T_IN) / (T_LEARN - T_IN), 0, 1);
    applyTo(Math.round(F * u * u * u));
    cv.redraw();
    if (t < T_END) raf = requestAnimationFrame(frame);
  };
  const replay = () => {
    cancelAnimationFrame(raf); reset();
    if (PE.instant()) { t = T_END; applyTo(F); cv.redraw(); return; }
    frame.t0 = undefined; raf = requestAnimationFrame(frame);
  };
  reset(); replay();
  return { replay, redraw: cv.redraw };
};
