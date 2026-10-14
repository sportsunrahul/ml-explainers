// Simulate the ch-uncertainty toy over many pools before it ships; print what the reader will see.
import { createRequire } from 'module'; const require = createRequire(import.meta.url);
const T = require('../parts/toy-pool.js');
const rows = [];
for (let seed = 1; seed <= 20; seed++) {
  const pool = T.sample(700, seed), test = T.sample(900, 1000 + seed);
  const res = {}; for (const s of ['entropy', 'margin', 'lc', 'random']) res[s] = T.campaign(pool, test, s, { b: 5, rounds: 12, seed: seed * 31 }).curve;
  rows.push({ seed, res });
}
const fmt = c => c.map(x => (100 * x.acc).toFixed(0)).join(' ');
for (const r of rows.slice(0, 5)) { console.log('pool', r.seed); for (const s in r.res) console.log('  ', s.padEnd(8), fmt(r.res[s]), ' rare:', r.res[s].map(x => (100 * x.rare).toFixed(0)).join(' ')); }
const final = s => rows.map(r => r.res[s].at(-1).acc), med = a => a.slice().sort((x, y) => x - y)[a.length >> 1];
for (const s of ['entropy', 'margin', 'lc', 'random']) console.log(s, 'median final acc', (100 * med(final(s))).toFixed(1), 'median rare recall', (100 * med(rows.map(r => r.res[s].at(-1).rare))).toFixed(0));
console.log('entropy beats random at the end on', rows.filter(r => r.res.entropy.at(-1).acc > r.res.random.at(-1).acc).length, 'of 20 pools');
