// ch-problem: image score (max patch 2D distance) of good and scratched caps against banks of 1, 5, 20 good caps.
import fs from 'fs';
const src = fs.readFileSync(new URL('../parts/lib-toy.js', import.meta.url), 'utf8').replace(/<\/?script>/g, '');
const TOY = new Function(src + '; return TOY;')();
const f2 = f => { const o = []; for (let k = 0; k < f.length / 3; k++) o.push([f[k * 3 + 1], f[k * 3 + 2]]); return o; };
const pool = Array.from({ length: 20 }, (_, i) => f2(TOY.features(TOY.make('cap', 1 + i))));
const score = (bank, f) => Math.max(...f.map(p => Math.min(...bank.map(b => Math.hypot(b[0] - p[0], b[1] - p[1])))));
const goods = [104, 105, 106, 107, 108].map(s => f2(TOY.features(TOY.make('cap', s))));
const bad = f2(TOY.features(TOY.defect(TOY.make('cap', 200), 'scratch', 200)));
for (const n of [1, 2, 5, 20]) {
  const bank = pool.slice(0, n).flat();
  console.log(`bank ${String(n).padStart(2)} caps (${bank.length} pts): good caps`, goods.map(g => score(bank, g).toFixed(2)).join(' '), ' scratched', score(bank, bad).toFixed(2));
}
