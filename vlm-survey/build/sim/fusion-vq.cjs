require('./pe.cjs');
const S = 32, P = 4, G = S / P;
function pix(x, y) {
  const cx = x + .5, cy = y + .5;
  if ((cx - 24) ** 2 + (cy - 7.5) ** 2 < 16) return .95;
  if (cx > 9 && cx < 11.5 && cy > 20 && cy < 25) return .08;
  if (cx > 5 && cx < 15.5 && cy > 16 && cy < 25) return .72;
  if (cy > 9 && cy <= 16 && Math.abs(cx - 10.25) < (cy - 9) * .8) return .42;
  if (cy >= 25) return .5 + .12 * ((x + y) % 2);
  return .12 + .1 * cy / 25;
}
const rn = PE.rng(3), img = []; for (let y = 0; y < S; y++) for (let x = 0; x < S; x++) img.push(Math.min(1, Math.max(0, pix(x, y) + .06 * (rn() - .5) + .04 * Math.sin(x * .7 + y * .3))));
const patches = []; for (let py = 0; py < G; py++) for (let px = 0; px < G; px++) {
  const v = []; for (let y = 0; y < P; y++) for (let x = 0; x < P; x++) v.push(img[(py * P + y) * S + px * P + x]); patches.push(v); }
const d2 = (a, b) => a.reduce((s, v, i) => s + (v - b[i]) ** 2, 0);
function kmeans(K) {
  const r = PE.rng(7), C = [patches[Math.floor(r() * patches.length)].slice()];
  while (C.length < K) { const d = patches.map(p => Math.min(...C.map(c => d2(p, c)))), T = d.reduce((a, b) => a + b, 0);
    if (T === 0) break; let u = r() * T, i = 0; while ((u -= d[i]) > 0) i++; C.push(patches[i].slice()); }
  let ids = [];
  for (let it = 0; it < 30; it++) {
    ids = patches.map(p => { let b = 0, bd = 1e9; C.forEach((c, k) => { const d = d2(p, c); if (d < bd) { bd = d; b = k; } }); return b; });
    C.forEach((c, k) => { const m = patches.filter((_, i) => ids[i] === k); if (m.length) for (let j = 0; j < c.length; j++) c[j] = m.reduce((s, p) => s + p[j], 0) / m.length; });
  }
  const mse = patches.reduce((s, p, i) => s + d2(p, C[ids[i]]), 0) / img.length;
  return { C, ids, mse };
}
const uniq = new Set(patches.map(p => p.join(','))).size;
console.log('unique patches', uniq);
for (const K of [1, 2, 4, 8, 12, 16, 24, 32, 64]) { const r = kmeans(K); console.log(K, r.C.length, r.mse.toFixed(5), 'rmse', Math.sqrt(r.mse).toFixed(3), 'used', new Set(r.ids).size); }
const r2 = kmeans(2);
let s=''; for (let py=0;py<G;py++){ s+=r2.ids.slice(py*G,py*G+G).join('')+'\n'; } console.log(s, r2.C.map(c=>(c.reduce((a,b)=>a+b)/16).toFixed(2)));
const r8 = kmeans(8); console.log('rmse2', Math.sqrt(r2.mse), 'rmse8', Math.sqrt(r8.mse), 'rmse32', Math.sqrt(kmeans(32).mse));
