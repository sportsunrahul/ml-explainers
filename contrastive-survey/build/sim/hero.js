const fs = require('fs'); const KIT = {}; eval(fs.readFileSync('parts/kit/05-circle.js', 'utf8'));
function rng(seed) { let a = seed >>> 0; return () => { a |= 0; a = a + 0x6D2B79F5 | 0;
  let t = Math.imul(a ^ a >>> 15, 1 | a); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }
for (const [tau, lr] of [[0.5, 3], [0.3, 2], [0.2, 1]]) {
  const r = rng(7), N = 12, th = [];
  for (let i = 0; i < N; i++) th.push(r() * 6.283);
  for (let i = 0; i < N; i++) th.push(th[i] + (r() - .5) * 2.4);
  let out = [];
  for (let s = 0; s <= 300; s++) { const L = KIT.ntxentStep(th, tau, lr); if ([0, 25, 50, 100, 150, 200, 300].includes(s)) { const st = KIT.circleStats(th); out.push(`s${s} L=${L.toFixed(3)} pairGap=${st.pairGap.toFixed(3)} spread=${st.spread.toFixed(2)}`); } }
  console.log('tau', tau, 'lr', lr, '\n ' + out.join('\n '));
}
