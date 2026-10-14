// viz-padim: covariance shape at the hole position (79) and the rim position (71); the "fool Euclidean" probe numbers.
import fs from 'fs';
const src = fs.readFileSync(new URL('../parts/lib-toy.js', import.meta.url), 'utf8').replace(/<\/?script>/g, '');
const TOY = new Function(src + '; return TOY;')();
const G2 = TOY.G * TOY.G, EPS = .01, f2 = f => { const o = []; for (let k = 0; k < G2; k++) o.push([f[k * 3 + 1], f[k * 3 + 2]]); return o; };
const tr = Array.from({ length: 20 }, (_, i) => f2(TOY.features(TOY.make('cap', 1 + i))));
const nt = Array.from({ length: 20 }, (_, i) => f2(TOY.features(TOY.make('cap', 100 + i))));
const hole = f2(TOY.features(TOY.defect(TOY.make('cap', 207), 'hole', 207))), good = f2(TOY.features(TOY.make('cap', 104)));
for (const q of [79, 71]) {
  const m = [0, 0]; tr.forEach(f => { m[0] += f[q][0] / 20; m[1] += f[q][1] / 20; });
  let a = EPS, b = 0, d = EPS; tr.forEach(f => { const u = f[q][0] - m[0], v = f[q][1] - m[1]; a += u * u / 19; b += u * v / 19; d += v * v / 19; });
  const tr_ = a + d, det = a * d - b * b, l1 = tr_ / 2 + Math.sqrt(tr_ * tr_ / 4 - det), l2 = tr_ / 2 - Math.sqrt(tr_ * tr_ / 4 - det);
  const e1 = Math.abs(b) > 1e-12 ? [l1 - d, b] : (a >= d ? [1, 0] : [0, 1]), n1 = Math.hypot(...e1), E1 = e1.map(x => x / n1), E2 = [-E1[1], E1[0]];
  const maha = p => { const u = p[0] - m[0], v = p[1] - m[1]; return Math.sqrt((d * u * u - 2 * b * u * v + a * v * v) / det); };
  const eu = p => Math.hypot(p[0] - m[0], p[1] - m[1]);
  const probe = [m[0] + Math.sqrt(l1) * E2[0], m[1] + Math.sqrt(l1) * E2[1]];
  console.log(`q ${q}: var(diag) ${(a - EPS).toFixed(4)} var(bright) ${(d - EPS).toFixed(4)} cov ${b.toFixed(4)} | sigma long ${Math.sqrt(l1).toFixed(3)} short ${Math.sqrt(l2).toFixed(3)} long axis [${E1.map(x => x.toFixed(2))}]`);
  console.log(`   normal test maha max here ${Math.max(...nt.map(f => maha(f[q]))).toFixed(2)}, normal eu max ${Math.max(...nt.map(f => eu(f[q]))).toFixed(3)} | hole: eu ${eu(hole[q]).toFixed(3)} maha ${maha(hole[q]).toFixed(2)} | good: eu ${eu(good[q]).toFixed(3)} maha ${maha(good[q]).toFixed(2)} | probe: eu ${eu(probe).toFixed(3)} maha ${maha(probe).toFixed(2)}`);
}
