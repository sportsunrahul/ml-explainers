const fs = require('fs'); const KIT = {};
function rng(seed) { let a = seed >>> 0; return () => { a |= 0; a = a + 0x6D2B79F5 | 0;
  let t = Math.imul(a ^ a >>> 15, 1 | a); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }
function gauss(r) { return () => Math.sqrt(-2 * Math.log(1 - r())) * Math.cos(2 * Math.PI * r()); }
const softmax = v => { const m = Math.max(...v), e = v.map(x => Math.exp(x - m)), Z = e.reduce((s, x) => s + x, 0); return e.map(x => x / Z); };
const PE = { rng, gauss, softmax };
eval(fs.readFileSync('parts/kit/00-base.js', 'utf8').replace('const KIT = {};', ''));
eval(fs.readFileSync('parts/kit/90-clipsig.js', 'utf8'));
console.log('settings', JSON.stringify(KIT.CS));
const C = KIT.csCos(), show = (t, M) => { console.log(t); M.forEach(r => console.log(r.map(x => x.toFixed(2).padStart(6)).join(''))); };
show('cosines', C); show('row softmax t=10', KIT.csProbs(C, 'softmax')); show('sigmoid t=10 b=-10', KIT.csProbs(C, 'sigmoid', -10)); show('sigmoid t=10 b=-5', KIT.csProbs(C, 'sigmoid', -5));
for (const [m, b] of [[null, -10], ['softmax', -10], ['sigmoid', -10], ['sigmoid', -5]]) console.log(KIT.clipsigText(m, b, 32768, 8));
console.log('counts N=32768 D=8', JSON.stringify(KIT.csCounts(32768, 8)));
// Worked 3 x 3 example in the prose: row cosines 0.8, 0.5, 0.1; t = 10.
const row = [.8, .5, .1], sm = softmax(row.map(c => 10 * c));
console.log('3x3 row softmax', sm.map(x => x.toFixed(4)).join(' '), 'loss', (-Math.log(sm[0])).toFixed(4));
for (const b of [-10, -5]) { const p = row.map(c => KIT.csSig(10 * c + b)), z = [1, -1, -1];
  const L = row.map((c, j) => -Math.log(KIT.csSig(z[j] * (10 * c + b))));
  console.log('3x3 row sigmoid b=' + b, p.map(x => x.toFixed(4)).join(' '), 'cell losses', L.map(x => x.toFixed(4)).join(' '), 'sum', L.reduce((s, x) => s + x, 0).toFixed(4)); }
const row4 = [.8, .5, .1, .7], sm4 = softmax(row4.map(c => 10 * c));
console.log('add 4th caption cos 0.7: softmax', sm4.map(x => x.toFixed(4)).join(' '));
