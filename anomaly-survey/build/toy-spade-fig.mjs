// viz-spade numbers: K=3 retrieval of 12 caps by pooled features, per-patch squared 2-d distance to the gallery.
import fs from 'fs';
const src = fs.readFileSync(new URL('../parts/lib-toy.js', import.meta.url), 'utf8').replace(/<\/?script>/g, '');
const TOY = new Function(src + '; return TOY;')();
const G2 = TOY.G * TOY.G, f2 = f => { const o = []; for (let k = 0; k < G2; k++) o.push([f[k * 3 + 1], f[k * 3 + 2]]); return o; };
const pooled = f => { const m = [0, 0, 0, 0]; for (let k = 0; k < G2; k++) for (let d = 0; d < 3; d++) m[d] += f[k * 3 + d] / G2; let v = 0; for (let k = 0; k < G2; k++) v += (f[k * 3 + 2] - m[2]) ** 2 / G2; m[3] = Math.sqrt(v); return m; };
const F12 = Array.from({ length: 12 }, (_, i) => TOY.features(TOY.make('cap', 1 + i)));
const tests = { scratch: TOY.defect(TOY.make('cap', 200), 'scratch', 200), hole: TOY.defect(TOY.make('cap', 207), 'hole', 207), good: TOY.make('cap', 104) };
for (const [k, img] of Object.entries(tests)) {
  const F = TOY.features(img), p = pooled(F), d = F12.map((f, i) => [pooled(f).reduce((a, x, j) => a + (x - p[j]) ** 2, 0), i]).sort((a, b) => a[0] - b[0]);
  const K = d.slice(0, 3), gal = K.flatMap(([, i]) => f2(F12[i])), sc = f2(F).map(q => Math.min(...gal.map(g => (g[0] - q[0]) ** 2 + (g[1] - q[1]) ** 2)));
  const srt = [...sc].sort((a, b) => b - a);
  console.log(k.padEnd(8), 'K', K.map(x => x[1] + 1).join(','), 'image score', (K.reduce((a, x) => a + x[0], 0) / 3).toFixed(4), 'pixel sq max', srt[0].toFixed(3), 'p95', srt[11].toFixed(3), 'hole patch 79', sc[79].toFixed(4));
}
