// WinCLIP toy: window "damaged" probability = logistic of mean |diagonal-edge| feature over the window's patches.
// Prints window scores and toy pixel AUROC for small (2x2), mid (3x3), whole-image and all-three, harmonic vs arithmetic.
import fs from 'fs';
const src = fs.readFileSync(new URL('../parts/lib-toy.js', import.meta.url), 'utf8').replace(/<\/?script>/g, '');
const TOY = new Function(src + '; return TOY;')();
const { G, N, S, P } = TOY;
const M0 = +(process.argv[2] || 0.6), SC = +(process.argv[3] || 0.12);
const sig = z => 1 / (1 + Math.exp(-z));
function run(img) {
  const f = TOY.features(img), d = []; for (let k = 0; k < G * G; k++) d.push(Math.abs(f[k * 3 + 1]));
  const win = k => { const out = []; for (let j = 0; j + k <= G; j++) for (let i = 0; i + k <= G; i++) { let m = 0; for (let b = 0; b < k; b++) for (let a = 0; a < k; a++) m += d[(j + b) * G + i + a]; out.push({ i, j, k, p: sig((m / (k * k) - M0) / SC) }); } return out; };
  const W = { small: win(2), mid: win(3), img: win(G) };
  const map = (ws, harm) => { const acc = new Float64Array(N * N), cnt = new Float64Array(N * N);
    for (const w of ws) { const x0 = w.i * S, y0 = w.j * S, x1 = (w.i + w.k - 1) * S + P, y1 = (w.j + w.k - 1) * S + P;
      for (let y = y0; y < y1; y++) for (let x = x0; x < x1; x++) { acc[y * N + x] += harm ? 1 / Math.max(w.p, 1e-6) : w.p; cnt[y * N + x]++; } }
    const o = new Float64Array(N * N); for (let q = 0; q < N * N; q++) o[q] = cnt[q] ? (harm ? cnt[q] / acc[q] : acc[q] / cnt[q]) : 0; return o; };
  return { d, W, map };
}
function auroc(map, mask) { const pos = [], neg = []; for (let q = 0; q < N * N; q++) (mask[q] ? pos : neg).push(map[q]);
  neg.sort((a, b) => a - b); let s = 0; const lo = v => { let a = 0, b = neg.length; while (a < b) { const m = (a + b) >> 1; if (neg[m] < v) a = m + 1; else b = m; } return a; };
  const hi = v => { let a = 0, b = neg.length; while (a < b) { const m = (a + b) >> 1; if (neg[m] <= v) a = m + 1; else b = m; } return a; };
  for (const v of pos) { const l = lo(v), h = hi(v); s += l + .5 * (h - l); } return s / (pos.length * neg.length); }
for (const [kind, seed] of [['mesh', 200], ['mesh', 203], ['cap', 200]]) {
  const bad = TOY.defect(TOY.make(kind, seed), 'scratch', seed), R = run(bad), good = run(TOY.make(kind, 104));
  const mx = ws => Math.max(...ws.map(w => w.p)).toFixed(3);
  console.log(`\n${kind} seed ${seed}: patch |d| normal-image mean ${(good.d.reduce((a, b) => a + b) / good.d.length).toFixed(2)} max ${Math.max(...good.d).toFixed(2)}; scratch-image max ${Math.max(...R.d).toFixed(2)}`);
  console.log(`  max window p  bad: small ${mx(R.W.small)} mid ${mx(R.W.mid)} img ${mx(R.W.img)} | good: small ${mx(good.W.small)} mid ${mx(good.W.mid)} img ${mx(good.W.img)}`);
  for (const harm of [true, false]) {
    const ms = R.map(R.W.small, harm), mm = R.map(R.W.mid, harm), mi = R.map(R.W.img, harm);
    const all = new Float64Array(N * N); for (let q = 0; q < N * N; q++) all[q] = harm ? 3 / (1 / Math.max(ms[q], 1e-6) + 1 / Math.max(mm[q], 1e-6) + 1 / Math.max(mi[q], 1e-6)) : (ms[q] + mm[q] + mi[q]) / 3;
    console.log(`  ${harm ? 'harmonic ' : 'arithmetic'} pixel AUROC: small ${auroc(ms, bad.mask).toFixed(3)} mid ${auroc(mm, bad.mask).toFixed(3)} img ${auroc(mi, bad.mask).toFixed(3)} all ${auroc(all, bad.mask).toFixed(3)}`);
  }
}
// Flagged area (p > 0.5) vs mask area, harmonic vs arithmetic, mesh 200.
{ const bad = TOY.defect(TOY.make('mesh', 200), 'scratch', 200), R = run(bad), ma = bad.mask.reduce((a, b) => a + b, 0);
  for (const harm of [true, false]) for (const k of ['small', 'mid']) { const m = R.map(R.W[k], harm); let a = 0, hit = 0; for (let q = 0; q < N * N; q++) if (m[q] > .5) { a++; hit += bad.mask[q]; }
    console.log(`  mesh200 ${harm ? 'harm' : 'arith'} ${k}: flagged ${a} px, of which in mask ${hit}; mask ${ma} px`); } }
