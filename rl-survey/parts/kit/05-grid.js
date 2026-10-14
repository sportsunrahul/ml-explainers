/* Grid worlds: one engine for every grid on the page (the yard in the hero and chapters 3-5, the cliff, the maze),
   and one drawing language. Cells are indexed s = r * cols + c with row 0 at the top. Actions: 0 up, 1 right,
   2 down, 3 left. A move into a wall or off the grid leaves the agent where it is. Rewards are paid on entering a
   cell: a special cell's reward if it has one, else the step reward. Entering a terminal cell ends the episode,
   and terminal cells have value 0. With slip p, the move goes the intended way with probability 1 - p and to each
   side with probability p / 2. */
KIT.ACTIONS = [[0, -1], [1, 0], [0, 1], [-1, 0]];
KIT.ACTION_NAMES = ['up', 'right', 'down', 'left'];
KIT.makeGrid = spec => {
  const { cols, rows } = spec, n = cols * rows;
  const idx = (c, r) => r * cols + c, cr = s => [s % cols, Math.floor(s / cols)];
  const walls = new Set((spec.walls || []).map(([c, r]) => idx(c, r)));
  const special = {}; (spec.special || []).forEach(o => { special[idx(o.c, o.r)] = o; });
  const isTerminal = s => !!(special[s] && special[s].terminal);
  const move = (s, a) => {
    const [c, r] = cr(s), [dx, dy] = KIT.ACTIONS[a], c2 = c + dx, r2 = r + dy;
    return c2 < 0 || c2 >= cols || r2 < 0 || r2 >= rows || walls.has(idx(c2, r2)) ? s : idx(c2, r2);
  };
  const reward = s2 => special[s2] && special[s2].reward !== undefined ? special[s2].reward : spec.stepReward;
  const slip = spec.slip || 0;
  const transitions = (s, a) => {
    const out = [];
    const add = (p, a2) => {
      if (p <= 0) return;
      const s2 = move(s, a2), hit = out.find(o => o.s2 === s2);
      if (hit) hit.p += p; else out.push({ p, s2, r: reward(s2), done: isTerminal(s2) });
    };
    add(1 - slip, a); add(slip / 2, (a + 1) % 4); add(slip / 2, (a + 3) % 4);
    return out;
  };
  const sample = (s, a, u) => {
    const T = transitions(s, a); let acc = 0;
    for (const t of T) { acc += t.p; if (u < acc) return t; }
    return T[T.length - 1];
  };
  const states = [...Array(n).keys()].filter(s => !walls.has(s));
  const live = states.filter(s => !isTerminal(s));
  const start = idx(...spec.start);
  // One-step look-ahead: the expected reward plus discounted value of where action a lands.
  const qFrom = (V, s, a, g) => transitions(s, a).reduce((q, t) => q + t.p * (t.r + (t.done ? 0 : g * V[t.s2])), 0);
  return { spec, cols, rows, n, idx, cr, walls, special, isTerminal, move, transitions, sample, states, live, start, qFrom };
};

// Synchronous sweeps (every state backs up from the previous sweep's values), so information travels one cell
// per sweep. pi[s] is an array of four action probabilities. Each returns { V, delta }.
KIT.evalSweep = (W, V, pi, g) => {
  const out = V.slice(); let delta = 0;
  for (const s of W.live) {
    let v = 0;
    for (let a = 0; a < 4; a++) if (pi[s][a]) v += pi[s][a] * W.qFrom(V, s, a, g);
    out[s] = v; delta = Math.max(delta, Math.abs(v - V[s]));
  }
  return { V: out, delta };
};
KIT.optSweep = (W, V, g) => {
  const out = V.slice(); let delta = 0;
  for (const s of W.live) {
    let best = -Infinity;
    for (let a = 0; a < 4; a++) best = Math.max(best, W.qFrom(V, s, a, g));
    out[s] = best; delta = Math.max(delta, Math.abs(best - V[s]));
  }
  return { V: out, delta };
};
KIT.solveOpt = (W, g, tol = 1e-10) => {
  let V = new Array(W.n).fill(0), sweeps = 0;
  for (;;) { const r = KIT.optSweep(W, V, g); V = r.V; sweeps++; if (r.delta < tol) return { V, sweeps }; }
};
// Greedy action per state from values (ties go to the lowest action index, so results are reproducible).
KIT.greedy = (W, V, g) => {
  const pi = [];
  for (let s = 0; s < W.n; s++) {
    pi.push([0, 0, 0, 0]);
    if (!W.live.includes(s)) continue;
    let best = 0, bq = -Infinity;
    for (let a = 0; a < 4; a++) { const q = W.qFrom(V, s, a, g); if (q > bq + 1e-12) { bq = q; best = a; } }
    pi[s][best] = 1;
  }
  return pi;
};
// Epsilon-greedy over a row of Q values with random tie-breaking; r is a uniform RNG.
KIT.epsGreedy = (q, eps, r) => {
  if (r() < eps) return Math.floor(r() * q.length);
  const m = Math.max(...q), best = q.map((v, i) => v === m ? i : -1).filter(i => i >= 0);
  return best[Math.floor(r() * best.length)];
};
// A seeded Q-learning run that records every update, for playback figures (the hero, chapter 7).
KIT.qlearnRun = (W, o) => {
  const r = PE.rng(o.seed), Q = Array.from({ length: W.n }, () => [0, 0, 0, 0]), steps = [];
  for (let ep = 0; ep < o.episodes; ep++) {
    let s = W.start;
    for (let t = 0; t < o.maxSteps; t++) {
      const a = KIT.epsGreedy(Q[s], o.eps, r), tr = W.sample(s, a, r());
      const target = tr.r + (tr.done ? 0 : o.gamma * Math.max(...Q[tr.s2]));
      const q = Q[s][a] + o.alpha * (target - Q[s][a]);
      Q[s][a] = q; steps.push({ ep, s, a, s2: tr.s2, r: tr.r, q });
      if (tr.done) break;
      s = tr.s2;
    }
  }
  return { steps, Q };
};
// The yard: 5 x 4, robot starts bottom-left, charger (+1) top-right, stairwell (-1) just below it, one wall.
KIT.YARD = {
  cols: 5, rows: 4, start: [0, 3], walls: [[1, 1]], stepReward: -0.04, slip: 0.2,
  special: [{ c: 4, r: 0, reward: 1, terminal: true, kind: 'goal', label: '+1' },
            { c: 4, r: 1, reward: -1, terminal: true, kind: 'hazard', label: '\u22121' }],
};

// Drawing. KIT.gridLayout fits square cells into a box; KIT.drawGrid draws the shared visual language:
// value = blue fill (opacity from o.vRange), rewards = yellow ring + label, hazards = brown hatching,
// walls = solid brown, policy = green arrows (length = probability), agent = fg disc, o.flash[s] = red border.
KIT.gridLayout = (W, box) => {
  const size = Math.floor(Math.min(box.w / W.cols, box.h / W.rows));
  const x0 = Math.round(box.x + (box.w - size * W.cols) / 2), y0 = Math.round(box.y + (box.h - size * W.rows) / 2);
  const cell = s => { const [c, r] = W.cr(s); return { x: x0 + c * size, y: y0 + r * size }; };
  return { size, x0, y0, w: size * W.cols, h: size * W.rows, cell, center: s => { const p = cell(s); return [p.x + size / 2, p.y + size / 2]; } };
};
KIT.drawGrid = (ctx, W, L, o = {}) => {
  const B = PE.css('--blue'), Y = PE.css('--yellow'), G = PE.css('--green'), BR = PE.css('--brown'), FG = PE.css('--fg'), R = PE.css('--red'), P2 = PE.css('--panel2');
  const gap = Math.max(1.5, L.size * .03), rad = Math.max(3, L.size * .1), [lo, hi] = o.vRange || [-1, 1], nums = o.showNums && o.V;
  const fs = Math.max(10, Math.min(15, L.size * .2));
  ctx.save(); ctx.globalAlpha = o.alpha === undefined ? 1 : o.alpha;
  for (let s = 0; s < W.n; s++) {
    const p = L.cell(s), x = p.x + gap, y = p.y + gap, w = L.size - 2 * gap, sp = W.special[s];
    ctx.beginPath(); ctx.roundRect(x, y, w, w, rad);
    if (W.walls.has(s)) { ctx.fillStyle = BR; ctx.fill(); continue; }
    ctx.fillStyle = P2; ctx.fill();
    if (o.V && o.V[s] !== null && o.V[s] !== undefined && !W.isTerminal(s)) {
      ctx.save(); ctx.globalAlpha *= .06 + .79 * PE.clamp((o.V[s] - lo) / (hi - lo), 0, 1); ctx.fillStyle = B; ctx.fill(); ctx.restore();
    }
    if (sp && sp.kind === 'hazard') {
      ctx.save(); ctx.clip(); ctx.strokeStyle = BR; ctx.lineWidth = 2; ctx.globalAlpha *= .7; ctx.beginPath();
      for (let k = -w; k < w; k += 9) { ctx.moveTo(x + k, y + w); ctx.lineTo(x + k + w, y); }
      ctx.stroke(); ctx.restore();
    }
    if (sp && sp.terminal && sp.reward !== undefined) {
      ctx.save(); ctx.strokeStyle = Y; ctx.lineWidth = 2.5; ctx.beginPath(); ctx.roundRect(x + 3, y + 3, w - 6, w - 6, Math.max(2, rad - 2)); ctx.stroke(); ctx.restore();
      PE.label(ctx, sp.label, x + w / 2, y + w / 2 + fs * .38, Y, fs * 1.15, 'center', 700);
    }
    const fl = o.flash && o.flash[s];
    if (fl) { ctx.save(); ctx.globalAlpha *= fl; ctx.strokeStyle = R; ctx.lineWidth = 3; ctx.beginPath(); ctx.roundRect(x + 1, y + 1, w - 2, w - 2, rad); ctx.stroke(); ctx.restore(); }
    if (nums && !W.isTerminal(s) && o.V[s] !== null && o.V[s] !== undefined) {
      const v = o.V[s], t = (Math.abs(v) < 0.005 ? 0 : v).toFixed(o.digits === undefined ? 2 : o.digits).replace('-', '\u2212');
      PE.label(ctx, t, x + w / 2, y + (o.pi ? w * .36 : w / 2 + fs * .35), FG, fs, 'center', 600);
    }
  }
  if (o.pi) for (const s of W.live) {
    const probs = o.pi[s]; if (!probs) continue;
    const [cx, cy0] = L.center(s), cy = nums ? cy0 + L.size * .17 : cy0, m = Math.max(...probs);
    for (let a = 0; a < 4; a++) {
      const pr = probs[a]; if (pr < .02) continue;
      const len = L.size * (nums ? .2 : .3) * (m === 1 ? 1 : .35 + .65 * pr / m), [dx, dy] = KIT.ACTIONS[a];
      PE.arrow(ctx, cx - dx * len * .45, cy - dy * len * .45, cx + dx * len * .55, cy + dy * len * .55, G, 2.2, 7, m === 1 ? 1 : .4 + .6 * pr / m);
    }
  }
  if (o.path && o.path.length > 1) {
    PE.polyline(ctx, o.path.map(s => L.center(s)), G, 3, [6, 5], o.pathAlpha === undefined ? .9 : o.pathAlpha);
  }
  if (o.agent) {
    const [ax, ay] = o.agent.xy || L.center(o.agent.s);
    PE.dot(ctx, ax, ay, L.size * .17, FG, o.agent.alpha === undefined ? 1 : o.agent.alpha);
  }
  ctx.restore();
};
