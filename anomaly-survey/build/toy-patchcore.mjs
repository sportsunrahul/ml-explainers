// ch-patchcore: greedy k-center coreset on toy cap features (3-d). Prints coverage radius and toy image AUROC per coreset size.
// Seeding follows the official code (sampler.py): initial "distance to coreset" = mean distance to 10 random points.
import fs from 'fs';
const src = fs.readFileSync(new URL('../parts/lib-toy.js', import.meta.url), 'utf8').replace(/<\/?script>/g, '');
const TOY = new Function(src + '; return TOY;')();
const NB = +(process.argv[2] || 8);
const B = TOY.bank('cap', NB, 1), n = B.length / 3;
const d2 = (i, j, A = B, C = B) => (A[i * 3] - C[j * 3]) ** 2 + (A[i * 3 + 1] - C[j * 3 + 1]) ** 2 + (A[i * 3 + 2] - C[j * 3 + 2]) ** 2;
function greedy(seed) {
  const r = TOY.rng(seed), anchors = Array.from({ length: 10 }, () => Math.floor(r() * n));
  const md = new Float64Array(n); for (let i = 0; i < n; i++) { let s = 0; for (const a of anchors) s += Math.sqrt(d2(i, a)); md[i] = s / 10; }
  const order = [], radius = [];
  for (let k = 0; k < n; k++) {
    let bi = 0; for (let i = 1; i < n; i++) if (md[i] > md[bi]) bi = i;
    radius.push(md[bi]); order.push(bi);
    for (let i = 0; i < n; i++) { const d = Math.sqrt(d2(i, bi)); if (d < md[i]) md[i] = d; }
  }
  return { order, radius };
}
const t0 = Date.now(), G = greedy(7);
console.log(`bank ${NB} caps = ${n} patches; greedy full order in ${Date.now() - t0} ms`);
// true coverage radius of the first l picks (max over bank of distance to nearest pick), checked against the recorded value
function cover(l) { const P = G.order.slice(0, l); let worst = 0; for (let i = 0; i < n; i++) { let b = Infinity; for (const p of P) { const d = d2(i, p); if (d < b) b = d; } if (b > worst) worst = b; } return Math.sqrt(worst); }
const scoreImg = (img, P) => { const f = TOY.features(img); let mx = 0; for (let q = 0; q < TOY.G * TOY.G; q++) { let b = Infinity; for (const p of P) { const d = (f[q*3]-B[p*3])**2 + (f[q*3+1]-B[p*3+1])**2 + (f[q*3+2]-B[p*3+2])**2; if (d < b) b = d; } mx = Math.max(mx, Math.sqrt(b)); } return mx; };
const auroc = (pos, neg) => pos.reduce((a, v) => a + neg.filter(x => x < v).length + .5 * neg.filter(x => x === v).length, 0) / (pos.length * neg.length);
const normals = Array.from({ length: 20 }, (_, i) => TOY.make('cap', 100 + i));
const defects = {}; for (const t of ['scratch', 'hole', 'stain', 'missing']) defects[t] = Array.from({ length: 20 }, (_, i) => TOY.defect(TOY.make('cap', 200 + i), t, 200 + i));
for (const pct of [.5, 1, 2, 5, 10, 25, 50, 100]) {
  const l = Math.max(1, Math.floor(n * pct / 100)), P = G.order.slice(0, l);
  const ns = normals.map(x => scoreImg(x, P));
  const row = Object.entries(defects).map(([t, xs]) => `${t} ${auroc(xs.map(x => scoreImg(x, P)), ns).toFixed(3)}`).join('  ');
  console.log(`${String(pct).padStart(4)}%  l=${String(l).padStart(4)}  radius(greedy) ${(l < n ? G.radius[l] : 0).toFixed(3)}  radius(check) ${cover(l).toFixed(3)}  normal max ${Math.max(...ns).toFixed(2)}  ${row}`);
}
// random subsampling at 1% and 10% for comparison (same l)
const r = TOY.rng(99); for (const pct of [1, 10]) { const l = Math.floor(n * pct / 100); const P = [...Array(n).keys()].sort(() => r() - .5).slice(0, l);
  let worst = 0; for (let i = 0; i < n; i++) { let b = Infinity; for (const p of P) b = Math.min(b, d2(i, p)); worst = Math.max(worst, b); }
  console.log(`random ${pct}%: radius ${Math.sqrt(worst).toFixed(3)}`); }
