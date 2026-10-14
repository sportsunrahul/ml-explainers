// Checks a 2D PCA view of the cap bank: ranges, where scratch patches land, and 2D vs 3D nearest-neighbour agreement.
import fs from 'fs';
const src = fs.readFileSync(new URL('../parts/lib-toy.js', import.meta.url), 'utf8').replace(/<\/?script>/g, '');
const TOY = new Function(src + '; return TOY;')();
const B = TOY.bank('cap', 4, 1), n = B.length / 3;
const mu = [0, 1, 2].map(d => { let s = 0; for (let t = 0; t < n; t++) s += B[t * 3 + d]; return s / n; });
const C = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
for (let t = 0; t < n; t++) for (let a = 0; a < 3; a++) for (let b = 0; b < 3; b++) C[a][b] += (B[t * 3 + a] - mu[a]) * (B[t * 3 + b] - mu[b]) / n;
const pow = (M, skip) => { let v = [1, .5, .25]; for (let it = 0; it < 200; it++) { if (skip) { const d = v.reduce((s, x, i) => s + x * skip[i], 0); v = v.map((x, i) => x - d * skip[i]); }
  v = [0, 1, 2].map(i => M[i].reduce((s, x, j) => s + x * v[j], 0)); const L = Math.hypot(...v); v = v.map(x => x / L); } return v; };
const e1 = pow(C), e2 = pow(C, e1);
console.log('mu', mu.map(v => v.toFixed(2)), 'e1', e1.map(v => v.toFixed(2)), 'e2', e2.map(v => v.toFixed(2)));
const proj = (f, k) => [0, 1].map(a => [e1, e2][a].reduce((s, x, i) => s + x * (f[k * 3 + i] - mu[i]), 0));
let lo = [9, 9], hi = [-9, -9];
for (let t = 0; t < n; t++) { const p = proj(B, t); for (let a = 0; a < 2; a++) { lo[a] = Math.min(lo[a], p[a]); hi[a] = Math.max(hi[a], p[a]); } }
console.log('bank 2D range', lo.map(v => v.toFixed(2)), hi.map(v => v.toFixed(2)));
const d = TOY.defect(TOY.make('cap', 204), 'scratch', 204), f = TOY.features(d), s = TOY.nnScores(f, B);
let k = 0; for (let q = 1; q < s.length; q++) if (s[q] > s[k]) k = q;
console.log('top scratch patch', k, 'score3D', s[k].toFixed(2), 'at 2D', proj(f, k).map(v => v.toFixed(2)));
let best = 1e9, bi = 0; for (let t = 0; t < n; t++) { const dd = (B[t*3]-f[k*3])**2 + (B[t*3+1]-f[k*3+1])**2 + (B[t*3+2]-f[k*3+2])**2; if (dd < best) { best = dd; bi = t; } }
const p = proj(B, bi), q = proj(f, k); console.log('3D-nearest bank point at 2D', p.map(v => v.toFixed(2)), '2D length', Math.hypot(p[0]-q[0], p[1]-q[1]).toFixed(2));
let b2 = 1e9; for (let t = 0; t < n; t++) { const pp = proj(B, t); b2 = Math.min(b2, Math.hypot(pp[0]-q[0], pp[1]-q[1])); } console.log('2D-nearest distance', b2.toFixed(2));
