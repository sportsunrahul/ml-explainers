const fs = require('fs'); const KIT = {};
function rng(seed) { let a = seed >>> 0; return () => { a |= 0; a = a + 0x6D2B79F5 | 0;
  let t = Math.imul(a ^ a >>> 15, 1 | a); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }
function gauss(r) { return () => Math.sqrt(-2 * Math.log(1 - r())) * Math.cos(2 * Math.PI * r()); }
const PE = { rng, gauss };
eval(fs.readFileSync('parts/kit/20-pullpush.js', 'utf8'));
if (process.argv[2]) Object.assign(KIT.PP, JSON.parse(process.argv[2]));
console.log('settings', JSON.stringify(KIT.PP));
for (const mode of ['pull', 'both']) {
  const st = KIT.ppData(); let r;
  for (let s = 0; s <= 300; s++) { r = KIT.ppStep(st, mode); if ([0, 30, 60, 120, 200, 300].includes(s)) console.log(mode, 's' + s, 'loss', r.loss.toFixed(3), 'pairDist', r.pairDist.toFixed(3), 'spread', r.spread.toFixed(3)); }
}
for (const mode of ['init', 'both']) { const st = KIT.ppData(); if (mode === 'both') for (let s = 0; s < 240; s++) KIT.ppStep(st, 'both');
  const E = st.V1.concat(st.V2).map(v => KIT.ppEmbed(st.W, v)); console.log(mode, 'max |x|', Math.max(...E.map(e => Math.abs(e[0]))).toFixed(2), 'max |y|', Math.max(...E.map(e => Math.abs(e[1]))).toFixed(2)); }
