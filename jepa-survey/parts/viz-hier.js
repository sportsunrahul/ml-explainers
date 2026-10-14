<script>
(() => {
  const fig = document.getElementById('viz-hier'), el = document.getElementById('hier-canvas'), out = document.getElementById('hier-readout');
  const S = { level: -1, u: 1 };
  const YS = [-1.2, 0, 1.2];                                    // plane heights: level 1 bottom, level 3 top
  const top = [[-1.5, 1.2], [-.2, .9], [1.4, -.9]];             // level-3 plan: start, mid, goal
  const mid = [[-1.5, 1.2], [-1.1, 1.25], [-.6, 1.0], [-.2, .9], [.4, .5], [1.0, -.3], [1.4, -.9]];
  const low = []; for (let i = 0; i < mid.length - 1; i++) { low.push(mid[i]); low.push([(mid[i][0] + mid[i + 1][0]) / 2 + .12 * Math.sin(i * 2.1), (mid[i][1] + mid[i + 1][1]) / 2 + .12 * Math.cos(i * 1.7)]); } low.push(mid[mid.length - 1]);
  const flat = [[-1.5, 1.2], [-1.0, 1.4], [-.4, 1.5], [.3, 1.3], [.7, .7], [.6, .1], [1.1, -.2], [1.5, -.6], [1.0, -1.2]];
  const s3 = PE3.scene(el, { aspect: w => w < 560 ? 1 : .6, fov: 34, dist: 7.4, yaw: .55, pitch: .35 });
  const P = (p, y) => [p[0], y, -p[1]];
  function scene() {
    const items = []; const L = S.level, u = S.u;
    YS.forEach((y, k) => { items.push({ type: 'quad', pts: [[-2, y, 2], [2, y, 2], [2, y, -2], [-2, y, -2]], fill: '--panel2', stroke: '--rule', alpha: .55, link: 'level' + (k + 1) }); items.push({ type: 'label', at: [-2, y, 2.1], text: `level ${k + 1}: step = ${5 * 2 ** k} env steps`, color: '--muted', dy: 12 }); });
    const trail = (pts, y, color, n, link) => { const m = Math.max(2, Math.round(pts.length * n)); items.push({ type: 'line', pts: pts.slice(0, m).map(p => P(p, y)), color, width: 3, arrow: true, link }); items.push({ type: 'points', pts: pts.slice(0, m).map(p => P(p, y)), color, size: 5 }); };
    items.push({ type: 'points', pts: [P(top[0], YS[2]), P(top[0], YS[1]), P(top[0], YS[0])], color: '--blue', size: 8 });
    items.push({ type: 'points', pts: [P(top[2], YS[2]), P(top[2], YS[1]), P(top[2], YS[0])], color: '--yellow', size: 8 }, { type: 'label', at: P(top[2], YS[2]), text: 'goal', color: '--yellow', dy: -10 });
    if (L >= 3) trail(top, YS[2], '--green', L === 3 ? u : 1, 'plan');
    if (L >= 2 && L <= 3) { [top[1]].forEach(p => items.push({ type: 'line', pts: [P(p, YS[2]), P(p, YS[1])], color: '--yellow', width: 1.5, dash: [3, 3], alpha: L === 2 ? u : 1 }, { type: 'label', at: P(p, YS[1]), text: 'subgoal', color: '--yellow', dx: 8, alpha: L === 2 ? u : 1 })); if (L === 2) trail(mid, YS[1], '--green', u, 'plan'); }
    if (L === 1) { trail(top, YS[2], '--green', 1, 'plan'); trail(mid, YS[1], '--green', 1, 'plan'); items.push({ type: 'line', pts: [P(top[1], YS[2]), P(top[1], YS[1])], color: '--yellow', width: 1.5, dash: [3, 3] }); mid.slice(1, -1).forEach(p => items.push({ type: 'line', pts: [P(p, YS[1]), P(p, YS[0])], color: '--yellow', width: 1, dash: [2, 3], alpha: .6 })); trail(low, YS[0], '--green', u, 'plan'); }
    if (L === 0) { trail(flat, YS[0], '--red', u, 'plan'); S.cands = S.cands || PE3.gaussian(12, { scale: [.8, 0, .8], seed: 9 }); S.cands.forEach((c, i) => items.push({ type: 'line', pts: flat.slice(0, 9).map((p, k) => P([p[0] + c[0] * k / 8 * .6, p[1] + c[2] * k / 8 * .6], YS[0])), color: '--red', width: 1, dash: [3, 4], alpha: .25 })); }
    s3.items = items; s3.redraw();
    out.textContent = { '-1': 'Three latent spaces, coarse at the top. Start blue, goal yellow, at every level.', 3: 'Top level: two macro-steps reach the goal in the coarse latent; fast details (leg pose, distractors) are not even represented here.', 2: 'The first top prediction is lifted down as a subgoal; the middle level plans a short path to it, scored in the upper latent.', 1: 'Level 1 plans primitive actions toward the middle level\'s subgoals; only the first block is executed before replanning.', 0: 'Flat: one planner must search eight fine steps in the detailed latent at once; the fan of candidates is wide and the cost signal weak far from the goal.' }[String(L)];
  }
  function set(L) { S.level = L; fig.querySelectorAll('[data-level]').forEach(b => b.setAttribute('aria-pressed', String(+b.dataset.level === L))); S.u = 0; s3.tween(700, u => { S.u = u; scene(); }); }
  fig.querySelectorAll('[data-level]').forEach(b => b.addEventListener('click', () => set(+b.dataset.level)));
  document.getElementById('btn-hier-reset').addEventListener('click', () => { S.level = -1; fig.querySelectorAll('[data-level]').forEach(b => b.setAttribute('aria-pressed', 'false')); S.u = 1; scene(); });
  scene();
})();
</script>
