import { createRequire } from 'module'; const require = createRequire(import.meta.url);
const T = require('../parts/toy-pool.js');
const S = ['entropy', 'margin', 'lc', 'random'], curves = Object.fromEntries(S.map(s => [s, []]));
for (let seed = 1; seed <= 20; seed++) {
  const pool = T.sample(700, seed), test = T.sample(900, 1000 + seed);
  for (const s of S) curves[s].push(T.campaign(pool, test, s, { b: 5, rounds: 12, seed: seed * 31 }).curve.map(x => x.acc));
}
const med = a => a.slice().sort((x, y) => x - y)[a.length >> 1];
const q = (a, f) => a.slice().sort((x, y) => x - y)[Math.floor(f * (a.length - 1))];
const n = curves.random[0].length;
console.log('labels:', Array.from({ length: n }, (_, i) => 3 + 5 * i).join(' '));
for (const s of S) console.log(s.padEnd(8), Array.from({ length: n }, (_, i) => (100 * med(curves[s].map(c => c[i]))).toFixed(0).padStart(3)).join(''));
for (const s of ['entropy','margin']) console.log(s, 'minus random, 25/50/75%:', Array.from({ length: n }, (_, i) => { const d = curves[s].map((c, k) => c[i] - curves.random[k][i]); return [q(d,.25),q(d,.5),q(d,.75)].map(v => (100*v).toFixed(0)).join('/'); }).join(' '));
console.log('pool1 entropy', curves.entropy[0].map(v => (100*v).toFixed(0)).join(' '));
console.log('pool1 random ', curves.random[0].map(v => (100*v).toFixed(0)).join(' '));
