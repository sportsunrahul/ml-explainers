// Toy UniAD check: error inside the scratch vs the largest error outside it, for full attention, neighbour mask, learned query.
import fs from 'fs';
const strip = f => fs.readFileSync(new URL('../parts/' + f, import.meta.url), 'utf8').replace(/<\/?script>/g, '');
const TOY = new Function(strip('lib-toy.js') + '; return TOY;')();
const U = new Function('TOY', strip('lib-uniad.js') + '; return UNIAD_KIT;')(TOY);
const G = 15;
const inMask = (img, q) => { const i = q % G, j = Math.floor(q / G); let m = 0; for (let y = j * 8; y < j * 8 + 16; y++) for (let x = i * 8; x < i * 8 + 16; x++) m += img.mask[y * 128 + x]; return m > 20; };
U.learnQueries();
for (const mode of ['full', 'mask', 'query']) {
  const rows = [];
  for (let s = 200; s < 208; s++) {
    const img = TOY.defect(TOY.make('cap', s), 'scratch', s), x = U.feats(img), r = U.reconstruct(x, mode);
    let inMax = 0, outMax = 0; for (let q = 0; q < 225; q++) { if (inMask(img, q)) inMax = Math.max(inMax, r.err[q]); else outMax = Math.max(outMax, r.err[q]); }
    rows.push([inMax, outMax, Math.max(...r.selfW)]);
  }
  const good = U.reconstruct(U.feats(TOY.make('cap', 700)), mode).err;
  console.log(mode.padEnd(6), 'defect max', rows.map(r => r[0].toFixed(2)).join(' '), '| outside max', rows.map(r => r[1].toFixed(2)).join(' '), '| good cap max', Math.max(...good).toFixed(2), '| max self weight', Math.max(...rows.map(r => r[2])).toFixed(2));
}
