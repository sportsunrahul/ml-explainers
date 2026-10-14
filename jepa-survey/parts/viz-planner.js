<script>
(() => {
  // Toy CEM planner over a hand-written latent dynamics f(z, a) = z + R(z) a with a small curl, in 3D.
  const fig = document.getElementById('viz-planner'), el = document.getElementById('planner-canvas'), out = document.getElementById('planner-readout'), hz = document.getElementById('planner-horizon');
  const GOAL = [1.3, .55, -.9]; const S = { z: [-1.3, .1, .9], seed: 1, mu: null, sig: null, cands: [], best: null, iter: 0, u: 1 };
  const dyn = (z, a) => [z[0] + .32 * a[0] - .06 * z[2] * a[1], z[1] + .3 * a[1] + .04 * (z[0] * a[2]), z[2] + .32 * a[2] + .05 * z[1] * a[0]];
  const roll = (z, acts) => { const p = [z]; let c = z; for (const a of acts) { c = dyn(c, a); p.push(c); } return p; };
  const dist = (p, q) => Math.hypot(p[0] - q[0], p[1] - q[1], p[2] - q[2]);
  const H = () => +hz.value;
  function initDist() { S.mu = Array.from({ length: H() }, () => [0, 0, 0]); S.sig = Array.from({ length: H() }, () => [1, 1, 1]); S.iter = 0; }
  function sample() { const r = PE.rng(S.seed++), g = PE.gauss(r); S.cands = [];
    for (let k = 0; k < 60; k++) { const acts = S.mu.map((m, t) => m.map((v, i) => Math.max(-1.5, Math.min(1.5, v + S.sig[t][i] * g())))); const path = roll(S.z, acts); S.cands.push({ acts, path, e: dist(path[path.length - 1], GOAL) }); }
    S.cands.sort((a, b) => a.e - b.e); S.best = S.cands[0]; }
  function refine() { if (!S.cands.length) sample(); const el6 = S.cands.slice(0, 6);
    S.mu = S.mu.map((m, t) => m.map((_, i) => el6.reduce((s, c) => s + c.acts[t][i], 0) / 6));
    S.sig = S.sig.map((s, t) => s.map((_, i) => Math.max(.08, Math.sqrt(el6.reduce((q, c) => q + (c.acts[t][i] - S.mu[t][i]) ** 2, 0) / 6)))); S.iter++; sample(); }
  const s3 = PE3.scene(el, { aspect: w => w < 560 ? .95 : .58, fov: 36, dist: 6.2, yaw: .55, pitch: .38 });
  function scene() {
    const items = [{ type: 'grid', y: -1.4, size: 3.6, step: .6, color: '--rule' }];
    S.cands.slice(1).forEach(c => items.push({ type: 'line', pts: c.path, color: '--green', width: 1, dash: [3, 4], alpha: .22 }));
    if (S.best) items.push({ type: 'line', pts: S.best.path, color: '--green', width: 3, arrow: true, link: 'plan' });
    items.push({ type: 'points', pts: [S.z], color: '--blue', size: 9, link: 'state' }, { type: 'label', at: S.z, text: 'current embedding', color: '--blue', dy: 18 });
    items.push({ type: 'points', pts: [GOAL], color: '--yellow', size: 9, link: 'goal' }, { type: 'label', at: GOAL, text: 'goal embedding', color: '--yellow', dy: -10 });
    if (S.best) items.push({ type: 'line', pts: [S.best.path[S.best.path.length - 1], GOAL], color: '--red', width: 1.5, dash: [4, 3], link: 'energy' });
    s3.items = items; s3.redraw();
    out.textContent = S.best ? `Horizon ${H()}, refinement ${S.iter}: best of ${S.cands.length} imagined futures ends ${S.best.e.toFixed(2)} from the goal (red). Distance from the current state: ${dist(S.z, GOAL).toFixed(2)}.` : 'Press "Sample" to imagine candidate futures.';
  }
  document.getElementById('btn-planner-sample').addEventListener('click', () => { if (!S.mu || S.mu.length !== H()) initDist(); sample(); scene(); });
  document.getElementById('btn-planner-refine').addEventListener('click', () => { if (!S.mu || S.mu.length !== H()) initDist(); refine(); scene(); });
  document.getElementById('btn-planner-step').addEventListener('click', () => { if (!S.best) { initDist(); for (let i = 0; i < 4; i++) refine(); }
    const from = S.z.slice(), to = dyn(S.z, S.best.acts[0]); initDist();
    s3.tween(500, u => { S.z = from.map((v, i) => v + (to[i] - v) * u); scene(); }, () => { S.z = to; for (let i = 0; i < 3; i++) refine(); scene(); }); });
  hz.addEventListener('input', () => { initDist(); S.cands = []; S.best = null; scene(); });
  document.getElementById('btn-planner-reset').addEventListener('click', () => { S.z = [-1.3, .1, .9]; S.seed = 1; hz.value = 3; initDist(); S.cands = []; S.best = null; scene(); });
  initDist(); scene();
})();
</script>
