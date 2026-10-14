// Median curves over 20 pools (embedded in the page) and the pool seed whose curves sit closest to the medians.
import { createRequire } from 'module'; import fs from 'fs'; const require = createRequire(import.meta.url);
const T = require('../parts/toy-pool.js');
const S = ['margin', 'entropy', 'lc', 'random'], curves = Object.fromEntries(S.map(s => [s, []])), R = 12;
for (let seed = 1; seed <= 20; seed++) {
  const pool = T.sample(700, seed), test = T.sample(900, 1000 + seed);
  for (const s of S) curves[s].push(T.campaign(pool, test, s, { b: 5, rounds: R, seed: seed * 31 }).curve.map(x => x.acc));
}
const q = (a, f) => a.slice().sort((x, y) => x - y)[Math.round(f * (a.length - 1))];
const n = R + 1, band = s => ({ q25: [], q50: [], q75: [] });
const out = { labels: Array.from({ length: n }, (_, i) => 3 + 5 * i), pools: 20, b: 5 };
for (const s of S) { out[s] = { q25: [], q50: [], q75: [] }; for (let i = 0; i < n; i++) { const v = curves[s].map(c => c[i]); out[s].q25.push(+q(v, .25).toFixed(4)); out[s].q50.push(+q(v, .5).toFixed(4)); out[s].q75.push(+q(v, .75).toFixed(4)); } }
let best = 1, bd = 1e9; for (let k = 0; k < 20; k++) { let d = 0; for (const s of ['margin', 'random']) for (let i = 0; i < n; i++) d += (curves[s][k][i] - out[s].q50[i]) ** 2; if (d < bd) { bd = d; best = k + 1; } }
out.seed = best;
fs.writeFileSync(new URL('./data/poolsim.json', import.meta.url), JSON.stringify(out));
console.log('chosen pool seed', best); for (const s of S) console.log(s.padEnd(8), out[s].q50.map(v => (100 * v).toFixed(0)).join(' '));
console.log('chosen margin', curves.margin[best - 1].map(v => (100 * v).toFixed(0)).join(' ')); console.log('chosen random', curves.random[best - 1].map(v => (100 * v).toFixed(0)).join(' '));
