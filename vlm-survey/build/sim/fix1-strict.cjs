// Item 2: strict match count (own caption strictly closest; ties are not matches) for the SigLIP toy and the CLIP toy.
// Same start, step count, lr and TAU as parts/viz-siglip.js trainFrames and parts/viz-contrastive.js trainFrames.
require('./pe.cjs');
const N = 5, TWO = 2 * Math.PI, names = ['dog', 'cat', 'car', 'pizza', 'tree'];
const start = seed => { const r = PE.rng(seed); return { th: names.map(() => r() * TWO), ph: names.map(() => r() * TWO) }; };
const strictRow = (row, i, tol) => row.every((s, j) => j === i || row[i] - s > tol);
const strict = (S, tol = 1e-9) => S.filter((row, i) => strictRow(row, i, tol)).length;
const argmax = S => S.filter((row, i) => row.indexOf(Math.max(...row)) === i).length;

function sig(seed, b0, tol) {
  let { th, ph } = start(seed), b = b0; const TAU = .1, lr = .05;
  for (let k = 0; k < 400; k++) { const g = PE.contrast(th, ph, TAU, 'sigmoid', b);
    th = th.map((v, i) => v - lr * g.gth[i]); ph = ph.map((v, i) => v - lr * g.gph[i]); b -= lr * g.gb; }
  const g = PE.contrast(th, ph, TAU, 'sigmoid', b);
  return { th, ph, b, L: g.L, S: g.S, strict: strict(g.S, tol), argmax: argmax(g.S) };
}
function clip(seed, tau) {
  let { th, ph } = start(seed); const lr = .05;
  for (let k = 0; k < 400; k++) { const g = PE.contrast(th, ph, tau);
    th = th.map((v, i) => v - lr * g.gth[i]); ph = ph.map((v, i) => v - lr * g.gph[i]); }
  const g = PE.contrast(th, ph, tau);
  return { th, ph, L: g.L, S: g.S, strict: strict(g.S), argmax: argmax(g.S) };
}

console.log('== SigLIP toy, seed 65 ==');
for (const b0 of [-10, 0]) {
  const r = sig(65, b0);
  console.log(`b0 ${b0}: strict ${r.strict}/5, argmax ${r.argmax}/5, bias now ${r.b.toFixed(2)}, loss ${r.L.toFixed(2)}`);
  console.log('  own-caption margin per photo (own cos minus best other cos):', r.S.map((row, i) => (row[i] - Math.max(...row.filter((_, j) => j !== i))).toExponential(2)).join(' '));
}
console.log('\n== SigLIP toy, seeds 1-200 ==');
for (const b0 of [-10, 0]) {
  const all = []; for (let s = 1; s <= 200; s++) all.push(sig(s, b0));
  const m = k => (all.reduce((a, r) => a + r[k], 0) / all.length);
  console.log(`b0 ${b0}: mean strict ${m('strict').toFixed(3)}, mean argmax ${m('argmax').toFixed(3)}, all-5 strict ${all.filter(r => r.strict === 5).length}/200`);
  for (const tol of [1e-12, 1e-9, 1e-6, 1e-3, 1e-2]) {
    const a2 = []; for (let s = 1; s <= 200; s++) a2.push(sig(s, b0, tol).strict);
    console.log(`   tol ${tol}: mean strict ${(a2.reduce((a, c) => a + c, 0) / 200).toFixed(3)}`);
  }
}
console.log('\n== CLIP toy, seed 65 ==');
for (const tau of [1, .07]) { const r = clip(65, tau); console.log(`tau ${tau}: strict ${r.strict}/5, argmax ${r.argmax}/5, loss ${r.L.toFixed(3)}`); }
