<script>
(() => {
  // Toy MPC: a 2-room world (wall at x = 0 with a doorway), true dynamics as the "world model", CEM over 2D actions.
  const fig = document.getElementById('viz-mpc'), el = document.getElementById('mpc-canvas'), out = document.getElementById('mpc-readout');
  const GOALS = { a: [1.6, -1.2], b: [.5, 1.3] }, START = [-1.5, 1.2];
  const S = { pos: START.slice(), goal: 'a', seed: 1, cands: [], best: null, steps: 0, running: false };
  const blocked = (p, q) => { // wall at x in [-.08,.08], y in [-2,2] except a door at y in [-.35,.35]
    if ((p[0] < 0) === (q[0] < 0)) return false; const t = (0 - p[0]) / (q[0] - p[0]), y = p[1] + t * (q[1] - p[1]); return Math.abs(y) > .35; };
  const step = (p, a) => { const n = Math.hypot(a[0], a[1]), s = Math.min(.42, n), q = [p[0] + a[0] / (n || 1) * s, p[1] + a[1] / (n || 1) * s]; q[0] = Math.max(-2, Math.min(2, q[0])); q[1] = Math.max(-1.7, Math.min(1.7, q[1])); return blocked(p, q) ? p.slice() : q; };
  const H = 6, K = 80, E = 8;
  function plan(iters) { const g = GOALS[S.goal]; let mu = Array.from({ length: H }, () => [0, 0]), sig = Array.from({ length: H }, () => [1, 1]);
    // Seeds derive from the step count, so pressing plan twice (or a cue replay) reproduces the same candidates.
    for (let it = 0; it < iters; it++) { const r = PE.rng(1000 * S.steps + 7 * it + 1), gs = PE.gauss(r); S.cands = [];
      for (let k = 0; k < K; k++) { const acts = mu.map((m, t) => [m[0] + sig[t][0] * gs(), m[1] + sig[t][1] * gs()]); const path = [S.pos]; let p = S.pos; for (const a of acts) { p = step(p, a); path.push(p); } S.cands.push({ acts, path, e: Math.hypot(p[0] - g[0], p[1] - g[1]) }); }
      S.cands.sort((a, b) => a.e - b.e); const el8 = S.cands.slice(0, E);
      mu = mu.map((_, t) => [el8.reduce((s, c) => s + c.acts[t][0], 0) / E, el8.reduce((s, c) => s + c.acts[t][1], 0) / E]);
      sig = sig.map((_, t) => [0, 1].map(i => Math.max(.1, Math.sqrt(el8.reduce((s, c) => s + (c.acts[t][i] - mu[t][i]) ** 2, 0) / E)))); }
    S.best = S.cands[0]; }
  const s3 = PE3.scene(el, { aspect: w => w < 560 ? .9 : .55, fov: 34, dist: 6.4, yaw: .15, pitch: .95 });
  const P = p => [p[0], 0, -p[1]];
  function scene() {
    const g = GOALS[S.goal], items = [{ type: 'grid', y: 0, size: 4, step: .5, color: '--rule' }];
    for (const [y0, y1] of [[-1.7, -.35], [.35, 1.7]]) items.push({ type: 'quad', pts: [[0, 0, -y0], [0, .35, -y0], [0, .35, -y1], [0, 0, -y1]], fill: '--brown', alpha: .7 });
    S.cands.slice(1, 40).forEach(c => items.push({ type: 'line', pts: c.path.map(P), color: '--green', width: 1, dash: [3, 4], alpha: .2 }));
    if (S.best) items.push({ type: 'line', pts: S.best.path.map(P), color: '--green', width: 3, arrow: true, link: 'plan' });
    items.push({ type: 'points', pts: [P(S.pos)], color: '--blue', size: 9, link: 'state' }, { type: 'points', pts: [P(g)], color: '--yellow', size: 9, link: 'goal' }, { type: 'label', at: P(g), text: 'goal', color: '--yellow', dy: -10 }, { type: 'label', at: [0, .4, 0], text: 'wall with a doorway', color: '--brown', dy: -6 });
    s3.items = items; s3.redraw();
    const d = Math.hypot(S.pos[0] - g[0], S.pos[1] - g[1]);
    out.textContent = S.best ? `Step ${S.steps}: best of ${K} imagined ${H}-step futures ends ${S.best.e.toFixed(2)} from the goal; the agent is ${d.toFixed(2)} away.` + (d < .25 ? ' Goal reached.' : '') : 'Press "Plan" to imagine futures from the current state.';
  }
  function doStep() { if (S.busy) return Promise.resolve(); S.busy = true; if (!S.best) plan(4); const to = S.best.path[1], from = S.pos.slice(); S.steps++;
    return new Promise(res => s3.tween(350, u => { S.pos = [from[0] + (to[0] - from[0]) * u, from[1] + (to[1] - from[1]) * u]; scene(); }, () => { S.pos = to; plan(4); scene(); S.busy = false; res(); })); }
  document.getElementById('btn-mpc-plan').addEventListener('click', () => { plan(4); scene(); });
  document.getElementById('btn-mpc-step').addEventListener('click', () => { if (!S.running) doStep(); });
  document.getElementById('btn-mpc-run').addEventListener('click', async () => { if (S.running) return; S.running = true; const g = GOALS[S.goal];
    for (let i = 0; i < 30 && Math.hypot(S.pos[0] - g[0], S.pos[1] - g[1]) > .25; i++) await doStep(); scene(); S.running = false; });
  fig.querySelectorAll('[data-goal]').forEach(b => b.addEventListener('click', () => { S.goal = b.dataset.goal; fig.querySelectorAll('[data-goal]').forEach(x => x.setAttribute('aria-pressed', String(x === b))); S.cands = []; S.best = null; scene(); }));
  document.getElementById('btn-mpc-reset').addEventListener('click', () => { S.pos = START.slice(); S.seed = 1; S.steps = 0; S.busy = false; S.running = false; S.cands = []; S.best = null; S.goal = 'a'; fig.querySelectorAll('[data-goal]').forEach(x => x.setAttribute('aria-pressed', String(x.dataset.goal === 'a'))); scene(); });
  scene();
})();
</script>
