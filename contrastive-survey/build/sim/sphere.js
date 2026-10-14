// Runs the ch-sphere toy with the kit's own code: node build/sim/sphere.js ['{"SEED":2}']
const fs = require('fs'); const KIT = {};
function rng(seed) { let a = seed >>> 0; return () => { a |= 0; a = a + 0x6D2B79F5 | 0;
  let t = Math.imul(a ^ a >>> 15, 1 | a); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }
function gauss(r) { return () => Math.sqrt(-2 * Math.log(1 - r())) * Math.cos(2 * Math.PI * r()); }
const PE = { rng, gauss };
eval(fs.readFileSync('parts/kit/60-sphere.js', 'utf8'));
if (process.argv[2]) Object.assign(KIT.SPH, JSON.parse(process.argv[2]));
console.log('settings', JSON.stringify(KIT.SPH));
const f = x => x.toFixed(3), show = (name, st) => { const m = KIT.sphMetrics(st); console.log(name.padEnd(12), 'L_align', f(m.align), 'L_uniform', f(m.unif), 'tolerance', f(m.tol)); };
show('init', KIT.sphTrain(null));
show('align', KIT.sphTrain('align'));
for (const tau of [0.05, 0.07, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]) show('tau ' + tau, KIT.sphTrain('both', tau));
// Uniform reference for t = 2 on S^2: log E exp(-2|u-v|^2) = log((1 - e^-8) / 8).
console.log('random-uniform reference L_uniform', f(Math.log((1 - Math.exp(-8)) / 8)));
// Trajectory for the animation (InfoNCE tau 0.3 and align): metrics every 50 steps.
for (const [mode, tau] of [['align', 1], ['both', 0.3]]) { const st = KIT.sphRun(); const out = [];
  for (let s = 0; s <= KIT.SPH.STEPS; s++) { if (s % 50 === 0) { const m = KIT.sphMetrics(st); out.push(s + ':' + f(m.align) + '/' + f(m.unif)); } if (s < KIT.SPH.STEPS) { KIT.sphStep(st, mode, tau); st.step++; } }
  console.log('trajectory', mode, tau, 'step:L_align/L_uniform', out.join(' ')); }
// Seed check: is the effect a property of the setup or of one lucky seed? 'A' = align-only L_uniform, then tau: L_uniform/tolerance.
const seed0 = KIT.SPH.SEED;
for (let seed = 1; seed <= 6; seed++) { KIT.SPH.SEED = seed; const r = ['A ' + KIT.sphMetrics(KIT.sphTrain('align')).unif.toFixed(2)];
  for (const tau of [0.07, 0.3, 1]) { const m = KIT.sphMetrics(KIT.sphTrain('both', tau)); r.push(tau + ' ' + m.unif.toFixed(2) + '/' + m.tol.toFixed(2)); }
  console.log('seed', seed, r.join(' | ')); }
KIT.SPH.SEED = seed0;
