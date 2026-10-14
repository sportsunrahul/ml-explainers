/* Chapter 2: the 10-armed testbed of S&B section 2.3, computed in the browser. Each task draws 10 means from
   N(0, 1); a pull pays N(mean, 1). Methods: greedy, epsilon .01, epsilon .1 (sample averages, Q starts at 0) and
   UCB with c = 2 (an unpulled arm counts as maximizing). Runs are added in chunks so the curves grow; seeded. */
KIT.TESTBED = { K: 10, STEPS: 1000, RUNS: 2000, SEED: 2018,
  methods: [{ key: 'greedy', label: 'greedy', eps: 0, dash: [2, 4] }, { key: 'e01', label: '\u03b5 = 0.01', eps: .01, dash: [8, 5] },
    { key: 'e1', label: '\u03b5 = 0.1', eps: .1, dash: null }, { key: 'ucb', label: 'UCB, c = 2', ucb: 2, dash: [1, 0], thick: true }] };
KIT.testbedRun = (m, r, g, means, best, rew, opt) => {
  const { K, STEPS } = KIT.TESTBED, Q = new Float64Array(K), N = new Float64Array(K);
  for (let t = 0; t < STEPS; t++) {
    let a;
    if (!m.ucb && r() < m.eps) a = Math.floor(r() * K);
    else {
      let bv = -Infinity, ties = 0;
      for (let i = 0; i < K; i++) {
        const v = m.ucb ? (N[i] === 0 ? Infinity : Q[i] + m.ucb * Math.sqrt(Math.log(t + 1) / N[i])) : Q[i];
        if (v > bv) { bv = v; a = i; ties = 1; } else if (v === bv && r() * ++ties < 1) a = i;
      }
    }
    const R = means[a] + g();
    N[a]++; Q[a] += (R - Q[a]) / N[a];
    rew[t] += R; if (a === best) opt[t] += 1;
  }
};
KIT.testbed = (canvas, o = {}) => {
  const TB = KIT.TESTBED, M = TB.methods;
  const st = { runs: 0, running: false };
  let raf = 0, r, g;
  const glow = o.fig ? PE.linkPulses(o.fig, () => cv.redraw()) : () => 0;
  const reset = () => {
    cancelAnimationFrame(raf); st.runs = 0; st.running = false; r = PE.rng(TB.SEED); g = PE.gauss(r);
    st.rew = M.map(() => new Float64Array(TB.STEPS)); st.opt = M.map(() => new Float64Array(TB.STEPS));
  };
  const chunk = n => {
    for (let k = 0; k < n && st.runs < TB.RUNS; k++, st.runs++) {
      const means = Array.from({ length: TB.K }, () => g()), best = means.indexOf(Math.max(...means));
      M.forEach((m, j) => KIT.testbedRun(m, r, g, means, best, st.rew[j], st.opt[j]));
    }
  };
  const run = done => {
    if (st.running || st.runs >= TB.RUNS) return;
    st.running = true;
    if (PE.instant()) { chunk(TB.RUNS); st.running = false; cv.redraw(); done && done(); return; }
    const f = () => { chunk(50); cv.redraw(); if (st.runs < TB.RUNS) raf = requestAnimationFrame(f); else { st.running = false; done && done(); } };
    raf = requestAnimationFrame(f);
  };
  const smooth = (arr, n, w = 5) => { const out = []; for (let t = 0; t < arr.length; t += w) { let s = 0, c = 0; for (let u = t; u < Math.min(arr.length, t + w); u++) { s += arr[u]; c++; } out.push([t + c / 2, s / c / n]); } return out; };
  const draw = (ctx, w, h) => {
    if (!w) return;
    ctx.clearRect(0, 0, w, h);
    const narrow = KIT.narrow(w), Y = PE.css('--yellow'), G = PE.css('--green'), FG = PE.css('--fg'), MU = PE.css('--muted');
    const pads = narrow ? { l: 44, r: 70, t: 16, b: 38 } : { l: 52, r: 96, t: 18, b: 40 };
    const panels = narrow
      ? [{ x: pads.l, y: pads.t, w: w - pads.l - pads.r, h: h / 2 - pads.t - pads.b }, { x: pads.l, y: h / 2 + pads.t, w: w - pads.l - pads.r, h: h / 2 - pads.t - pads.b }]
      : [{ x: pads.l, y: pads.t, w: w / 2 - pads.l - pads.r + 30, h: h - pads.t - pads.b }, { x: w / 2 + pads.l - 10, y: pads.t, w: w / 2 - pads.l - pads.r + 30, h: h - pads.t - pads.b }];
    const specs = [{ data: st.rew, yr: [0, 1.6], yt: [0, .5, 1, 1.5], yl: 'average reward', col: Y, link: 'reward', f: v => v.toFixed(1) },
      { data: st.opt, yr: [0, 1], yt: [0, .25, .5, .75, 1], yl: 'best arm chosen', col: G, link: 'choice', f: v => Math.round(v * 100) + '%' }];
    specs.forEach((sp, i) => {
      const b = panels[i], ax = PE.axes(ctx, b, [0, TB.STEPS], sp.yr, { xt: [0, 250, 500, 750, 1000], yt: sp.yt, yf: sp.f, xl: 'step', yl: sp.yl, ylo: narrow ? 34 : 40 });
      if (!st.runs) { PE.label(ctx, 'press Run', b.x + b.w / 2, b.y + b.h / 2, MU, 13, 'center'); return; }
      const gl = glow(sp.link);
      M.forEach((m, j) => {
        const pts = smooth(sp.data[j], st.runs).map(([t, v]) => [ax.X(t), ax.Y(PE.clamp(v, sp.yr[0], sp.yr[1]))]);
        const gm = glow(m.key);
        PE.polyline(ctx, pts, sp.col, (m.thick ? 2.6 : 1.8) + gl + 2.5 * gm, m.dash, m.key === 'greedy' ? .75 + .25 * gm : 1);
        const end = pts[pts.length - 1];
        PE.label(ctx, m.label, end[0] + 6, end[1] + 4 + (m.key === 'e01' ? 9 : m.key === 'ucb' ? -9 : 0), sp.col, narrow ? 10 : 12, 'left', 600);
      });
    });
    if (st.running) PE.label(ctx, 'tasks averaged: ' + st.runs + ' of ' + TB.RUNS, panels[0].x + panels[0].w / 2, panels[0].y + panels[0].h * .85, MU, 12, 'center', 600);
  };
  const cv = PE.canvas(canvas, w => KIT.narrow(w) ? 1.15 : .42, draw);
  reset();
  return { run, reset: () => { reset(); cv.redraw(); }, state: st, redraw: cv.redraw, resize: cv.resize, destroy: cv.destroy,
    final: () => M.map((m, j) => {
      const L = 100, n = st.runs || 1; let a = 0, b = 0;
      for (let t = TB.STEPS - L; t < TB.STEPS; t++) { a += st.rew[j][t]; b += st.opt[j][t]; }
      return { label: m.label, reward: a / L / n, best: b / L / n };
    }) };
};
KIT.testbedText = (st, fin) => {
  if (!st.runs) return 'Press Run: four methods each play the same 2000 ten-armed tasks for 1000 steps.';
  if (st.running) return 'Running: ' + st.runs + ' of 2000 tasks averaged.';
  return 'Steps 901 to 1000, averaged over ' + st.runs + ' tasks: ' + fin.map(x => x.label + ' earns ' + x.reward.toFixed(2) + ' and picks the best arm ' + Math.round(x.best * 100) + '% of the time').join('; ') + '.';
};
