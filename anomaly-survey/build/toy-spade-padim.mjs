// ch-spade / ch-padim toy checks. (1) PaDiM: per-position Gaussian (3-d toy features, eps I) vs position-free kNN, per defect.
// (2) SPADE: image score from pooled features (mean sq. distance to K nearest images) vs the max of the pixel map.
import fs from 'fs';
const src = fs.readFileSync(new URL('../parts/lib-toy.js', import.meta.url), 'utf8').replace(/<\/?script>/g, '');
const TOY = new Function(src + '; return TOY;')();
const G2 = TOY.G * TOY.G, NTR = 20, EPS = .01;
const train = Array.from({ length: NTR }, (_, i) => TOY.features(TOY.make('cap', 1 + i)));
// per-position mean and covariance
const mu = [], inv = [];
for (let q = 0; q < G2; q++) {
  const m = [0, 0, 0]; train.forEach(f => { for (let d = 0; d < 3; d++) m[d] += f[q * 3 + d] / NTR; });
  const C = [[EPS, 0, 0], [0, EPS, 0], [0, 0, EPS]];
  train.forEach(f => { for (let a = 0; a < 3; a++) for (let b = 0; b < 3; b++) C[a][b] += (f[q * 3 + a] - m[a]) * (f[q * 3 + b] - m[b]) / (NTR - 1); });
  const [[a, b, c], [d, e, g], [h, i, k]] = C, det = a * (e * k - g * i) - b * (d * k - g * h) + c * (d * i - e * h);
  inv.push([[(e * k - g * i) / det, (c * i - b * k) / det, (b * g - c * e) / det], [(g * h - d * k) / det, (a * k - c * h) / det, (c * d - a * g) / det], [(d * i - e * h) / det, (b * h - a * i) / det, (a * e - b * d) / det]]);
  mu.push(m);
}
const maha = f => { const s = new Float32Array(G2); for (let q = 0; q < G2; q++) { const v = [0, 1, 2].map(d => f[q * 3 + d] - mu[q][d]), I = inv[q];
  s[q] = Math.sqrt(v.reduce((acc, x, a) => acc + x * v.reduce((t, y, b) => t + I[a][b] * y, 0), 0)); } return s; };
const B = TOY.bank('cap', NTR, 1);
const auroc = (pos, neg) => pos.reduce((a, v) => a + neg.filter(n => n < v).length + .5 * neg.filter(n => n === v).length, 0) / (pos.length * neg.length);
const normals = Array.from({ length: 20 }, (_, i) => TOY.features(TOY.make('cap', 100 + i)));
const nK = normals.map(f => Math.max(...TOY.nnScores(f, B))), nM = normals.map(f => Math.max(...maha(f)));
console.log('PaDiM toy (cap, 20 train images): image AUROC  kNN vs per-position Gaussian');
for (const t of ['scratch', 'hole', 'stain', 'missing']) {
  const fs_ = Array.from({ length: 20 }, (_, i) => TOY.features(TOY.defect(TOY.make('cap', 200 + i), t, 200 + i)));
  console.log(' ', t.padEnd(8), auroc(fs_.map(f => Math.max(...TOY.nnScores(f, B))), nK).toFixed(3), auroc(fs_.map(f => Math.max(...maha(f))), nM).toFixed(3));
}
// a single hole example for the figure: where is its max, and how do the two scores compare there?
const hd = TOY.defect(TOY.make('cap', 203), 'hole', 203), hf = TOY.features(hd), hk = TOY.nnScores(hf, B), hm = maha(hf);
let q = 0; for (let k = 1; k < G2; k++) if (hm[k] > hm[q]) q = k;
console.log('hole seed 203: maha max at', q, hm[q].toFixed(2), 'kNN there', hk[q].toFixed(2), '| normal maha max range', Math.min(...nM).toFixed(2), Math.max(...nM).toFixed(2), '| normal kNN max range', Math.min(...nK).toFixed(2), Math.max(...nK).toFixed(2));
// (2) SPADE toy: pooled global feature = mean of the 3 features over patches + brightness std
const pooled = f => { const m = [0, 0, 0, 0]; for (let q = 0; q < G2; q++) for (let d = 0; d < 3; d++) m[d] += f[q * 3 + d] / G2;
  let v = 0; for (let q = 0; q < G2; q++) v += (f[q * 3 + 2] - m[2]) ** 2 / G2; m[3] = Math.sqrt(v); return m; };
const trP = train.map(pooled), K = 3;
const imgScore = f => { const p = pooled(f), d = trP.map(t => t.reduce((a, x, i) => a + (x - p[i]) ** 2, 0)).sort((a, b) => a - b); return d.slice(0, K).reduce((a, b) => a + b, 0) / K; };
const nI = normals.map(imgScore);
console.log('SPADE toy (cap): image AUROC  pooled-feature kNN (K=3) vs max of pixel map');
for (const t of ['scratch', 'hole', 'stain']) {
  const fs_ = Array.from({ length: 20 }, (_, i) => TOY.features(TOY.defect(TOY.make('cap', 200 + i), t, 200 + i)));
  console.log(' ', t.padEnd(8), auroc(fs_.map(imgScore), nI).toFixed(3), auroc(fs_.map(f => Math.max(...TOY.nnScores(f, B))), nK).toFixed(3));
}
