const fs = require('fs'); const KIT = {}; eval(fs.readFileSync('parts/kit/05-circle.js', 'utf8'));
function rng(seed) { let a = seed >>> 0; return () => { a |= 0; a = a + 0x6D2B79F5 | 0;
  let t = Math.imul(a ^ a >>> 15, 1 | a); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }
const [tau, lr, seed] = process.argv.slice(2).map(Number);
const r = rng(seed), N = 12, th = [];
for (let i = 0; i < N; i++) th.push(r() * 6.283);
for (let i = 0; i < N; i++) th.push(th[i] + (r() < .5 ? -1 : 1) * (0.5 + r() * 0.9));
for (let s = 0; s <= 480; s++) { const st = KIT.circleStats(th); if (s % 60 === 0) console.log(`s${s} pairGap=${st.pairGap.toFixed(3)} spread=${st.spread.toFixed(2)}`); KIT.ntxentStep(th, tau, lr); }
