// Picks the hole example for viz-padim / viz-spade: per-position (2-d) peak on the hole well above the normal maximum,
// kNN at that patch small, and SPADE's nearest gallery patch (K=3 retrieved of 12) sitting on a bolt hole.
import fs from 'fs';
const src = fs.readFileSync(new URL('../parts/lib-toy.js', import.meta.url), 'utf8').replace(/<\/?script>/g, '');
const TOY = new Function(src + '; return TOY;')();
const { N, G, S, P } = TOY, G2 = G * G, EPS = .01;
const f2 = f => { const o = []; for (let k = 0; k < G2; k++) o.push([f[k * 3 + 1], f[k * 3 + 2]]); return o; };
const tr = Array.from({ length: 20 }, (_, i) => f2(TOY.features(TOY.make('cap', 1 + i))));
const gs = []; for (let q = 0; q < G2; q++) { const m = [0, 0]; tr.forEach(f => { m[0] += f[q][0] / 20; m[1] += f[q][1] / 20; });
  let a = EPS, b = 0, d = EPS; tr.forEach(f => { const u = f[q][0] - m[0], v = f[q][1] - m[1]; a += u * u / 19; b += u * v / 19; d += v * v / 19; }); gs.push({ m, a, b, d }); }
const maha = f => f.map((p, q) => { const { m, a, b, d } = gs[q], u = p[0] - m[0], v = p[1] - m[1], det = a * d - b * b; return Math.sqrt((d * u * u - 2 * b * u * v + a * v * v) / det); });
const nM = Array.from({ length: 20 }, (_, i) => Math.max(...maha(f2(TOY.features(TOY.make('cap', 100 + i))))));
const nmax = Math.max(...nM); console.log('normal maha image max: max', nmax.toFixed(2), 'median', [...nM].sort((a, b) => a - b)[10].toFixed(2));
const tr12 = Array.from({ length: 12 }, (_, i) => TOY.make('cap', 1 + i)), F12 = tr12.map(c => TOY.features(c));
const pooled = f => { const m = [0, 0, 0, 0]; for (let k = 0; k < G2; k++) for (let d = 0; d < 3; d++) m[d] += f[k * 3 + d] / G2; let v = 0; for (let k = 0; k < G2; k++) v += (f[k * 3 + 2] - m[2]) ** 2 / G2; m[3] = Math.sqrt(v); return m; };
for (let s = 200; s < 240; s++) {
  const hd = TOY.defect(TOY.make('cap', s), 'hole', s), F = TOY.features(hd), hf = f2(F), hm = maha(hf);
  let q = 0; hm.forEach((v, k) => { if (v > hm[q]) q = k; });
  let inMask = 0; for (let y = Math.floor(q / G) * S; y < Math.floor(q / G) * S + P; y++) for (let x = (q % G) * S; x < (q % G) * S + P; x++) inMask += hd.mask[y * N + x];
  const tp = pooled(F), K = F12.map((f, i) => [pooled(f).reduce((a, x, j) => a + (x - tp[j]) ** 2, 0), i]).sort((a, b) => a[0] - b[0]).slice(0, 3);
  let bd = 1e9, bi = 0, bk = 0; K.forEach(([, i]) => f2(F12[i]).forEach((p, k) => { const d = (p[0] - hf[q][0]) ** 2 + (p[1] - hf[q][1]) ** 2; if (d < bd) { bd = d; bi = i; bk = k; } }));
  const cx = (bk % G) * S + P / 2, cy = Math.floor(bk / G) * S + P / 2, bolt = Math.min(...tr12[bi].bolts.map(([x, y]) => Math.hypot(x - cx, y - cy)));
  const hx = (q % G) * S + P / 2, hy = Math.floor(q / G) * S + P / 2, hb = Math.min(...hd.bolts.map(([x, y]) => Math.hypot(x - hx, y - hy))), hr = Math.hypot(hx - hd.cx, hy - hd.cy);
  if (inMask > 20 && hm[q] > 1.4 * nmax && hb > 14) console.log('seed', s, 'q', q, 'maha', hm[q].toFixed(1), 'sq nn', bd.toFixed(4), 'nn cap', bi + 1, 'patch', bk, 'nn-bolt', bolt.toFixed(1), 'test-bolt', hb.toFixed(1), 'r', hr.toFixed(1), 'K', K.map(k => k[1] + 1).join(','));
}
