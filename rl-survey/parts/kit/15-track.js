/* Chapter 1, "be the agent": a 7-cell track, the robot starts in the middle, a charger sits at one end, 5 moves.
   Supervised mode: a teacher shows the right move before every press and grades it. Reinforcement mode: no teacher,
   one score (+1 or 0) when the episode ends, and the charger stays hidden until the robot reaches it. Each mode has
   its own fixed world (supervised: charger on the right; reinforcement: on the left), so what the teacher showed in
   one world cannot answer the other.
   The strip below counts how often each cell was visited: the data the reader's own choices produced. */
KIT.track = (canvas, o = {}) => {
  const CELLS = 7, START = 3, BUDGET = 5;
  const st = { mode: null, side: 6, pos: START, x: START, moves: [], done: false, score: null, visits: [], scores: [] };
  let stopTween = () => {};
  const glow = o.fig ? PE.linkPulses(o.fig, () => cv.redraw()) : () => 0;
  const reset = () => {
    stopTween(); st.mode = null; st.side = 6; st.scores = []; st.visits = new Array(CELLS).fill(0); newEpisode();
  };
  const newEpisode = () => {
    stopTween(); st.pos = START; st.x = START; st.moves = []; st.done = false; st.score = null;
    st.visits[START]++; cv && cv.redraw();
  };
  const teacher = () => st.side > st.pos ? 1 : 0;
  const act = a => {
    if (!st.mode || st.done) return;
    const correct = a === teacher();
    st.pos = PE.clamp(st.pos + (a ? 1 : -1), 0, CELLS - 1); st.visits[st.pos]++;
    st.moves.push({ a, correct });
    const reached = st.pos === st.side;
    if (reached || st.moves.length === BUDGET) { st.done = true; st.score = reached ? 1 : 0; st.scores.push(st.score); }
    const from = st.x; stopTween();
    stopTween = PE.animate(260, u => { st.x = PE.lerp(from, st.pos, u); cv.redraw(); });
  };
  const draw = (ctx, w, h) => {
    if (!w) return;
    ctx.clearRect(0, 0, w, h);
    const FG = PE.css('--fg'), M = PE.css('--muted'), G = PE.css('--green'), Y = PE.css('--yellow'), R = PE.css('--red'), P2 = PE.css('--panel2'), RU = PE.css('--rule');
    const narrow = KIT.narrow(w), pad = 12, size = Math.min((w - 2 * pad) / CELLS, narrow ? 56 : 84), x0 = (w - size * CELLS) / 2;
    const y0 = narrow ? 34 : 40, cx = i => x0 + (i + .5) * size, fs = narrow ? 12 : 14;
    const shown = st.mode === 'sup' || (st.done && st.score === 1);
    for (let i = 0; i < CELLS; i++) {
      ctx.beginPath(); ctx.roundRect(x0 + i * size + 2, y0, size - 4, size - 4, 8); ctx.fillStyle = P2; ctx.fill();
      if ((i === 0 || i === CELLS - 1) && st.mode) {
        if (shown && i === st.side) {
          ctx.save(); ctx.strokeStyle = Y; ctx.lineWidth = 2.5; ctx.beginPath(); ctx.roundRect(x0 + i * size + 5, y0 + 3, size - 10, size - 10, 6); ctx.stroke(); ctx.restore();
          PE.label(ctx, '+1', cx(i), y0 + size / 2 + 4, Y, fs + 2, 'center', 700);
        } else if (!shown) PE.label(ctx, '?', cx(i), y0 + size / 2 + 4, M, fs + 4, 'center', 700);
      }
    }
    const gp = glow('policy');
    if (gp) { ctx.save(); ctx.globalAlpha = gp; ctx.strokeStyle = G; ctx.lineWidth = 3; ctx.beginPath(); ctx.arc(cx(0) + (st.x) * size, y0 + size / 2 - 2, size * .3, 0, 7); ctx.stroke(); ctx.restore(); }
    PE.dot(ctx, cx(0) + st.x * size, y0 + size / 2 - 2, size * .2, FG);
    if (st.mode === 'sup' && !st.done) {
      const dir = teacher() ? 1 : -1, ax = cx(st.pos), ay = y0 - 12;
      PE.arrow(ctx, ax - dir * 14, ay, ax + dir * 16, ay, G, 2.4, 8);
      PE.label(ctx, 'teacher', ax + dir * 22, ay + 4, G, fs - 1, dir > 0 ? 'left' : 'right');
    }
    // Move log: one chip per move. Supervised chips carry a grade; reinforcement chips carry none.
    const ly = y0 + size + (narrow ? 18 : 22), chip = narrow ? 30 : 36;
    PE.label(ctx, 'moves', x0, ly + chip * .62, M, fs - 1, 'left');
    const lx = x0 + (narrow ? 50 : 58);
    for (let k = 0; k < BUDGET; k++) {
      const m = st.moves[k], x = lx + k * (chip + 6);
      ctx.beginPath(); ctx.roundRect(x, ly, chip, chip, 6); ctx.strokeStyle = gp && m ? G : RU; ctx.lineWidth = gp && m ? 2 : 1; ctx.stroke();
      if (!m) continue;
      PE.label(ctx, m.a ? '\u2192' : '\u2190', x + chip / 2, ly + chip * .64, FG, fs + 2, 'center', 700);
      if (st.mode === 'sup') PE.label(ctx, m.correct ? '\u2713' : '\u2717', x + chip / 2, ly + chip + 14, m.correct ? G : R, fs, 'center', 700);
      else if (st.done && st.score === 0) PE.label(ctx, '?', x + chip / 2, ly + chip + 14, M, fs, 'center', 700);
    }
    const sx = lx + BUDGET * (chip + 6) + 8, gr = glow('return');
    if (st.done) {
      if (gr) { ctx.save(); ctx.globalAlpha = gr; ctx.strokeStyle = Y; ctx.lineWidth = 2; ctx.beginPath(); ctx.roundRect(sx - 6, ly - 2, narrow ? 76 : 96, chip + 4, 6); ctx.stroke(); ctx.restore(); }
      PE.label(ctx, 'score ' + (st.score ? '+1' : '0'), sx, ly + chip * .64, Y, fs + 1, 'left', 700);
    } else if (gr) PE.label(ctx, 'score: at the end', sx, ly + chip * .64, Y, fs, 'left', 600);
    // Visits: which cells the reader's own choices have shown them, over all episodes since Reset.
    const vy = ly + chip + (narrow ? 42 : 46), vs = size * .62, total = Math.max(1, ...st.visits);
    PE.label(ctx, 'cells you have seen (all episodes)', x0, vy - 8, M, fs - 1, 'left');
    for (let i = 0; i < CELLS; i++) {
      const x = cx(i) - vs / 2;
      ctx.beginPath(); ctx.roundRect(x, vy, vs, vs * .55, 5);
      ctx.save(); ctx.fillStyle = FG; ctx.globalAlpha = st.visits[i] ? .12 + .5 * st.visits[i] / total : 0; ctx.fill(); ctx.restore();
      ctx.strokeStyle = RU; ctx.lineWidth = 1; ctx.stroke();
      PE.label(ctx, st.visits[i] ? String(st.visits[i]) : 'never', cx(i), vy + vs * .38, st.visits[i] ? FG : M, fs - 2, 'center', 600);
    }
    const ga = glow('avg');
    const hy = vy + vs * .55 + (narrow ? 22 : 26);
    const hist = st.scores.length ? st.scores.slice(-8).map(s => s ? '+1' : '0').join('  ') : 'none yet';
    PE.label(ctx, 'episode scores: ' + hist + (st.scores.length ? '   average ' + (st.scores.reduce((a, b) => a + b, 0) / st.scores.length).toFixed(2) : ''), x0, hy, ga ? Y : M, fs - 1, 'left', ga ? 700 : 600);
  };
  const cv = PE.canvas(canvas, w => KIT.narrow(w) ? .74 : .35, draw);
  reset();
  return {
    set: mode => { if (mode === st.mode) return; st.mode = mode; st.side = mode === 'rl' ? 0 : 6; st.scores = []; st.visits = new Array(CELLS).fill(0); newEpisode(); }, act, newEpisode, reset: () => { reset(); cv.redraw(); },
    state: st, redraw: cv.redraw, resize: cv.resize, destroy: cv.destroy,
  };
};
KIT.trackText = st => {
  if (!st.mode) return 'Pick a kind of feedback, then move the robot with Left and Right. It has 5 moves to reach the charger.';
  const n = st.moves.length, last = st.moves[n - 1];
  if (st.mode === 'sup') {
    if (st.done) return 'Episode over: ' + (st.score ? 'the robot reached the charger.' : 'out of moves.') + ' The teacher graded every move, so you know exactly which ones were wrong.';
    return n ? 'Move ' + n + ' of 5: the teacher says ' + (last.correct ? 'that move was right.' : 'that move was wrong.') + ' You are told the answer after every move.' : 'Supervised: the green arrow is the teacher showing the right move.';
  }
  if (st.done) return st.score ? 'Episode over: score +1 in ' + n + ' moves. You found the charger, but only because you tried that way.' : 'Episode over: score 0. Which of your 5 moves was the mistake? The score does not say.';
  return n ? 'Move ' + n + ' of 5: no feedback. The score comes only when the episode ends.' : 'Reinforcement: no teacher, and the charger could be at either end.';
};
