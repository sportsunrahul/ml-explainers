// 2-d (axes 1,2) checks for viz-padim / viz-spade: per-position Gaussian vs kNN, with and without random rotation (+-10 deg,
// the Rd-MVTec range), and where SPADE's pixel correspondence for a hole patch lands.
import fs from 'fs';
const src = fs.readFileSync(new URL('../parts/lib-toy.js', import.meta.url), 'utf8').replace(/<\/?script>/g, '');
const TOY = new Function(src + '; return TOY;')();
const { N, G, S, P } = TOY, G2 = G * G, EPS = .01;
const rotate = (img, a) => { const px = new Float32Array(N * N), c = N / 2 - .5, ca = Math.cos(a), sa = Math.sin(a);
  for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) { const u = ca * (x - c) + sa * (y - c) + c, v = -sa * (x - c) + ca * (y - c) + c;
    const x0 = Math.floor(u), y0 = Math.floor(v), fx = u - x0, fy = v - y0, at = (i, j) => (i < 0 || j < 0 || i >= N || j >= N) ? .07 : img.px[j * N + i];
    px[y * N + x] = (1 - fx) * (1 - fy) * at(x0, y0) + fx * (1 - fy) * at(x0 + 1, y0) + (1 - fx) * fy * at(x0, y0 + 1) + fx * fy * at(x0 + 1, y0 + 1); }
  return { ...img, px }; };
const f2 = f => { const o = []; for (let k = 0; k < G2; k++) o.push([f[k * 3 + 1], f[k * 3 + 2]]); return o; };
const auroc = (pos, neg) => pos.reduce((a, v) => a + neg.filter(n => n < v).length + .5 * neg.filter(n => n === v).length, 0) / (pos.length * neg.length);
function run(rot) {
  const R = s => rot ? (TOY.rng(s * 7 + 5)() - .5) * 2 * rot : 0;
  const tr = Array.from({ length: 20 }, (_, i) => f2(TOY.features(rotate(TOY.make('cap', 1 + i), R(1 + i)))));
  const gs = []; for (let q = 0; q < G2; q++) { const m = [0, 0]; tr.forEach(f => { m[0] += f[q][0] / 20; m[1] += f[q][1] / 20; });
    let a = EPS, b = 0, d = EPS; tr.forEach(f => { const u = f[q][0] - m[0], v = f[q][1] - m[1]; a += u * u / 19; b += u * v / 19; d += v * v / 19; }); gs.push({ m, a, b, d }); }
  const maha = f => f.map((p, q) => { const { m, a, b, d } = gs[q], u = p[0] - m[0], v = p[1] - m[1], det = a * d - b * b; return Math.sqrt((d * u * u - 2 * b * u * v + a * v * v) / det); });
  const bank = tr.flat(), knn = f => f.map(p => Math.min(...bank.map(m => Math.hypot(m[0] - p[0], m[1] - p[1]))));
  const nf = Array.from({ length: 20 }, (_, i) => f2(TOY.features(rotate(TOY.make('cap', 100 + i), R(100 + i)))));
  const nK = nf.map(f => Math.max(...knn(f))), nM = nf.map(f => Math.max(...maha(f)));
  const out = {};
  for (const t of ['scratch', 'hole', 'stain', 'missing']) {
    const df = Array.from({ length: 20 }, (_, i) => f2(TOY.features(rotate(TOY.defect(TOY.make('cap', 200 + i), t, 200 + i), R(200 + i)))));
    out[t] = [auroc(df.map(f => Math.max(...knn(f))), nK).toFixed(3), auroc(df.map(f => Math.max(...maha(f))), nM).toFixed(3)];
  }
  return { out, gs, maha, knn, nM };
}
for (const rot of [0, 10 * Math.PI / 180]) { const r = run(rot); console.log(rot ? 'rotation +-10 deg' : 'aligned', ' kNN vs per-position (2-d):', JSON.stringify(r.out)); }
const r0 = run(0), hd = TOY.defect(TOY.make('cap', 203), 'hole', 203), hf = f2(TOY.features(hd)), hm = r0.maha(hf), hk = r0.knn(hf);
let q = 0; hm.forEach((v, k) => { if (v > hm[q]) q = k; });
console.log('hole 203: max maha patch', q, 'row', Math.floor(q / G), 'col', q % G, 'maha', hm[q].toFixed(2), 'knn', hk[q].toFixed(2), 'normal maha max', Math.max(...r0.nM).toFixed(2));
const g = r0.gs[q]; console.log('gaussian at q: mean', g.m.map(v => v.toFixed(2)), 'cov', [g.a, g.b, g.d].map(v => v.toFixed(4)));
// elongation per position: ratio of covariance eigenvalues, find a strongly elongated position
let best = 0, bq = 0; r0.gs.forEach((g, k) => { const tr = g.a + g.d, det = g.a * g.d - g.b * g.b, l1 = tr / 2 + Math.sqrt(tr * tr / 4 - det), l2 = tr / 2 - Math.sqrt(tr * tr / 4 - det); if (l1 / l2 > best && l1 > .02) { best = l1 / l2; bq = k; } });
console.log('most elongated position', bq, 'row', Math.floor(bq / G), 'col', bq % G, 'ratio', best.toFixed(1));
// SPADE: K=3 retrieved of 12 by pooled 4-d features; nearest gallery patch for the hole patch; is it a bolt hole?
const tr12 = Array.from({ length: 12 }, (_, i) => TOY.make('cap', 1 + i)), F12 = tr12.map(c => TOY.features(c));
const pooled = f => { const m = [0, 0, 0, 0]; for (let k = 0; k < G2; k++) for (let d = 0; d < 3; d++) m[d] += f[k * 3 + d] / G2; let v = 0; for (let k = 0; k < G2; k++) v += (f[k * 3 + 2] - m[2]) ** 2 / G2; m[3] = Math.sqrt(v); return m; };
const tp = pooled(TOY.features(hd)), ds = F12.map((f, i) => [pooled(f).reduce((a, x, j) => a + (x - tp[j]) ** 2, 0), i]).sort((a, b) => a[0] - b[0]).slice(0, 3);
console.log('retrieved', ds.map(d => `cap${d[1] + 1}:${d[0].toFixed(4)}`).join(' '));
let bd = 1e9, bi = 0, bk = 0; ds.forEach(([, i]) => f2(F12[i]).forEach((p, k) => { const d = (p[0] - hf[q][0]) ** 2 + (p[1] - hf[q][1]) ** 2; if (d < bd) { bd = d; bi = i; bk = k; } }));
const cx = (bk % G) * S + P / 2, cy = Math.floor(bk / G) * S + P / 2, bolt = Math.min(...tr12[bi].bolts.map(([x, y]) => Math.hypot(x - cx, y - cy)));
console.log('hole patch nearest gallery: cap', bi + 1, 'patch', bk, 'sq dist', bd.toFixed(4), 'patch centre to nearest bolt', bolt.toFixed(1), 'px');
