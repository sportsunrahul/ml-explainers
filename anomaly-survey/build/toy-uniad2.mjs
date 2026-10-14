// Toy UniAD on the mesh and the cap: defect vs outside error for each mode, with tau and window radius from argv.
import fs from 'fs';
const strip = f => fs.readFileSync(new URL('../parts/' + f, import.meta.url), 'utf8').replace(/<\/?script>/g, '');
const TOY = new Function(strip('lib-toy.js') + '; return TOY;')();
const tau = process.argv[2] || '.08', R = process.argv[3] || '3';
const src = strip('lib-uniad.js').replace('TAU = .08, R = 3', `TAU = ${tau}, R = ${R}`).replace(/T\.make\('cap', 600 \+ s\)/, "T.make(KIND, 600 + s)");
for (const kind of ['mesh', 'cap']) {
  const U = new Function('TOY', 'KIND', src + '; return UNIAD_KIT;')(TOY, kind);
  const G = 15, inMask = (img, q) => { const i = q % G, j = Math.floor(q / G); let m = 0; for (let y = j * 8; y < j * 8 + 16; y++) for (let x = i * 8; x < i * 8 + 16; x++) m += img.mask[y * 128 + x]; return m > 20; };
  U.learnQueries();
  for (const mode of ['full', 'mask', 'query']) {
    const ratios = [];
    for (let s = 200; s < 206; s++) { const img = TOY.defect(TOY.make(kind, s), 'scratch', s), r = U.reconstruct(U.feats(img), mode);
      let a = 0, b = 0; for (let q = 0; q < 225; q++) { if (inMask(img, q)) a = Math.max(a, r.err[q]); else b = Math.max(b, r.err[q]); } ratios.push(`${a.toFixed(2)}/${b.toFixed(2)}`); }
    const good = Math.max(...U.reconstruct(U.feats(TOY.make(kind, 700)), mode).err);
    console.log(`tau ${tau} R ${R}`, kind.padEnd(4), mode.padEnd(5), 'defect/outside max', ratios.join(' '), '| good max', good.toFixed(2));
  }
}
