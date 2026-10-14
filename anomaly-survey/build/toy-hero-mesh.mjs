// Picks the hero scratch on the mesh: 2D gap (axes 1,2) against a bank of the three normal hero meshes.
import fs from 'fs';
const src = fs.readFileSync(new URL('../parts/lib-toy.js', import.meta.url), 'utf8').replace(/<\/?script>/g, '');
const TOY = new Function(src + '; return TOY;')();
const f2 = f => { const o = []; for (let k = 0; k < f.length / 3; k++) o.push([f[k * 3 + 1], f[k * 3 + 2]]); return o; };
const bank = [101, 102, 103].flatMap(s => f2(TOY.features(TOY.make('mesh', s))));
const nn = p => Math.min(...bank.map(b => Math.hypot(b[0] - p[0], b[1] - p[1])));
const normal = f2(TOY.features(TOY.make('mesh', 104))).map(nn).sort((a, b) => a - b);
console.log('normal cap 104 patch 2D scores: p99', normal[Math.floor(normal.length * .99)].toFixed(2), 'max', normal.at(-1).toFixed(2));
for (let s = 200; s < 216; s++) {
  const d = TOY.defect(TOY.make('mesh', s), 'scratch', s), sc = f2(TOY.features(d)).map(nn);
  let k = 0; sc.forEach((v, q) => { if (v > sc[k]) k = q; });
  const inMask = (() => { const i = k % TOY.G, j = Math.floor(k / TOY.G); let m = 0; for (let y = j * 8; y < j * 8 + 16; y++) for (let x = i * 8; x < i * 8 + 16; x++) m += d.mask[y * 128 + x]; return m; })();
  console.log('seed', s, 'top', k, 'gap', sc[k].toFixed(2), 'mask px', inMask, 'second', [...sc].sort((a, b) => b - a)[1].toFixed(2));
}
