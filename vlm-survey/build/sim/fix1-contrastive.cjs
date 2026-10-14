// Re-run of parts/viz-contrastive.js trainFrames (seed 65, 400 steps, lr 0.05) for fact-check items 1, 2, 22.
require('./pe.cjs');
const names = ['dog', 'cat', 'car', 'pizza', 'tree'], N = 5, TWO = 2 * Math.PI;
const start = seed => { const r = PE.rng(seed); return { th: names.map(() => r() * TWO), ph: names.map(() => r() * TWO) }; };
const wrap = a => ((a % TWO) + TWO) % TWO;
function train(th, ph, tau, steps = 400, lr = .05, log) {
  for (let k = 0; k < steps; k++) { const g = PE.contrast(th, ph, tau);
    if (log) log(k, g, th, ph);
    th = th.map((v, i) => v - lr * g.gth[i]); ph = ph.map((v, i) => v - lr * g.gph[i]); }
  return [th, ph];
}
const argmaxCount = S => S.filter((r, i) => r.indexOf(Math.max(...r)) === i).length;
const strictCount = S => S.filter((r, i) => r.every((s, j) => j === i || r[i] - s > 1e-9)).length;
const deg = x => (x * 180 / Math.PI).toFixed(1);
function describe(label, th, ph, tau) {
  const g = PE.contrast(th, ph, tau), S = g.S;
  const diag = S.map((r, i) => r[i]);
  // Spacing: mean direction of each pair, sorted around the circle, gaps between neighbours.
  const mid = th.map((t, i) => Math.atan2(Math.sin(t) + Math.sin(ph[i]), Math.cos(t) + Math.cos(ph[i])));
  const order = mid.map((a, i) => [wrap(a), i]).sort((a, b) => a[0] - b[0]);
  const gaps = order.map((o, k) => [names[o[1]] + '->' + names[order[(k + 1) % N][1]], ((order[(k + 1) % N][0] - o[0] + TWO) % TWO).toFixed(2)]);
  console.log(label, 'tau', tau, 'loss', g.L.toFixed(3), 'argmax', argmaxCount(S), 'strict', strictCount(S));
  console.log('  diag cos', diag.map(d => d.toFixed(3)).join(' '), ' pair gap deg', th.map((t, i) => deg(Math.acos(Math.min(1, diag[i])))).join(' '));
  console.log('  best caption per photo', S.map(r => names[r.indexOf(Math.max(...r))]).join(','));
  console.log('  neighbour gaps (rad, even = 1.26)', gaps.map(g => g.join(' ')).join(' | '));
  console.log('  photo angles', th.map(a => wrap(a).toFixed(3)).join(' '), ' caption angles', ph.map(a => wrap(a).toFixed(3)).join(' '));
  return g;
}
const s0 = start(65);
describe('START', s0.th, s0.ph, .07);
describe('START', s0.th, s0.ph, 1);
for (const tau of [1, .07]) {
  const L = [];
  const [th, ph] = train(s0.th.slice(), s0.ph.slice(), tau, 400, .05, (k, g) => { if ([0, 10, 20, 30, 40, 50, 60, 80, 100, 150, 200, 300, 399].includes(k)) L.push(k + ':' + g.L.toFixed(3)); });
  console.log('\n=== TRAIN tau', tau, 'loss by step', L.join(' '));
  const g = describe('END', th, ph, tau);
  const gn = Math.hypot(...g.gth, ...g.gph); console.log('  |grad| at end', gn.toExponential(2));
  // Step at which each photo's own caption first wins strictly and stays
  let th2 = s0.th.slice(), ph2 = s0.ph.slice(), first = -1;
  for (let k = 0; k <= 400; k++) { const gg = PE.contrast(th2, ph2, tau); if (strictCount(gg.S) === 5 && first < 0) first = k; if (strictCount(gg.S) < 5) first = -1;
    th2 = th2.map((v, i) => v - .05 * gg.gth[i]); ph2 = ph2.map((v, i) => v - .05 * gg.gph[i]); }
  console.log('  strict 5/5 from step', first);
  // Loss thresholds
  let th3 = s0.th.slice(), ph3 = s0.ph.slice(), hit = {};
  for (let k = 0; k <= 400; k++) { const gg = PE.contrast(th3, ph3, tau); for (const t of [.5, .1, .05, .01]) if (gg.L < t && hit[t] === undefined) hit[t] = k;
    th3 = th3.map((v, i) => v - .05 * gg.gth[i]); ph3 = ph3.map((v, i) => v - .05 * gg.gph[i]); }
  console.log('  first step with loss below', JSON.stringify(hit));
}
// Zero-shot probe (tau from the slider default 0.07): readout percentage at the end of training.
{ const [th, ph] = train(s0.th.slice(), s0.ph.slice(), .07); const pr = PE.softmax(ph.map(p => Math.cos(th[0] + .35 - p) / .07));
  console.log('\nzero-shot probe at tau 0.07:', pr.map((p, j) => names[j] + ' ' + (100 * p).toFixed(0) + '%').join(', ')); }

// Item 22: over seeds 1-200, how often does training end with at least one pair not strictly matched, and are those jams?
for (const tau of [.07, 1]) {
  let notAll = 0, jam = 0, argNotAll = 0; const hist = [0, 0, 0, 0, 0, 0], ahist = [0, 0, 0, 0, 0, 0];
  for (let s = 1; s <= 200; s++) { const st = start(s); const [th, ph] = train(st.th, st.ph, tau); const S = PE.contrast(th, ph, tau).S;
    const c = strictCount(S), a = argmaxCount(S); hist[c]++; ahist[a]++;
    if (c < 5) { notAll++;
      // a jam: two photos within 0.05 rad of each other
      let j = false; for (let i = 0; i < N; i++) for (let k = i + 1; k < N; k++) if (Math.abs(Math.atan2(Math.sin(th[i] - th[k]), Math.cos(th[i] - th[k]))) < .05) j = true;
      if (j) jam++; }
    if (a < 5) argNotAll++; }
  console.log(`tau ${tau}: seeds 1-200, strict <5: ${notAll} (${(notAll / 2).toFixed(1)}%), of which with two photos jammed within 0.05 rad: ${jam}; argmax <5: ${argNotAll}; strict hist ${hist}; argmax hist ${ahist}`);
}
