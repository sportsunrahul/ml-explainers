// Simulates the toy kit in node: feature statistics, normal vs defect image scores, and whether the top patch hits the mask.
import fs from 'fs';
const src = fs.readFileSync(new URL('../parts/lib-toy.js', import.meta.url), 'utf8').replace(/<\/?script>/g, '');
const TOY = new Function(src + '; return TOY;')();
const { G, S, P, N } = TOY;
for (const kind of ['cap', 'mesh']) {
  const B = TOY.bank(kind, 20, 1);
  const nb = B.length / 3, mu = [0, 0, 0], sd = [0, 0, 0];
  for (let t = 0; t < nb; t++) for (let d = 0; d < 3; d++) mu[d] += B[t * 3 + d] / nb;
  for (let t = 0; t < nb; t++) for (let d = 0; d < 3; d++) sd[d] += (B[t * 3 + d] - mu[d]) ** 2 / nb;
  console.log(`\n${kind}: bank ${nb} patches; z-feature mean ${mu.map(v => v.toFixed(2))} sd ${sd.map(v => Math.sqrt(v).toFixed(2))}`);
  const img = s => Math.max(...TOY.nnScores(TOY.features(s), B));
  const normals = []; for (let s = 100; s < 120; s++) normals.push(img(TOY.make(kind, s)));
  console.log(`  normal image scores: min ${Math.min(...normals).toFixed(2)} mean ${(normals.reduce((a, b) => a + b) / normals.length).toFixed(2)} max ${Math.max(...normals).toFixed(2)}`);
  for (const type of ['scratch', 'hole', 'stain', 'missing']) {
    if (type === 'missing' && kind !== 'cap') continue;
    const sc = [], hit = [];
    for (let s = 200; s < 220; s++) {
      const d = TOY.defect(TOY.make(kind, s), type, s), sp = TOY.nnScores(TOY.features(d), B);
      let k = 0; for (let q = 1; q < sp.length; q++) if (sp[q] > sp[k]) k = q;
      const i = k % G, j = Math.floor(k / G); let inMask = 0;
      for (let y = j * S; y < j * S + P; y++) for (let x = i * S; x < i * S + P; x++) inMask += d.mask[y * N + x];
      sc.push(sp[k]); hit.push(inMask > 0);
    }
    const auroc = sc.reduce((a, v) => a + normals.filter(n => n < v).length + .5 * normals.filter(n => n === v).length, 0) / (sc.length * normals.length);
    console.log(`  ${type.padEnd(8)} scores: min ${Math.min(...sc).toFixed(2)} mean ${(sc.reduce((a, b) => a + b) / sc.length).toFixed(2)}  top patch in mask ${hit.filter(Boolean).length}/20  image AUROC ${auroc.toFixed(3)}`);
  }
}
