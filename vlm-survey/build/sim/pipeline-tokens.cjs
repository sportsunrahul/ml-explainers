// Toy batch for viz-pipeline: lengths, padding waste, packing, loss-token fractions, Molmo sharing.
const T = (k, s) => s.split(' ').map(w => ({ k, w }));
const sys = [{ k: 'sys', w: 'SYS' }], stop = [{ k: 'stop', w: '###' }], H = [{ k: 'role', w: 'Human:' }], A_ = [{ k: 'role', w: 'Assistant:' }];
const ph = [{ k: 'ph', w: '<image>' }];
const ex = {
  A: [...sys, ...stop, ...H, ...T('q', 'What is the dog doing ?'), ...ph, ...stop, ...A_, ...T('a', 'Catching a frisbee .'), ...stop,
      ...H, ...T('q', 'What color is it ?'), ...stop, ...A_, ...T('a', 'Red .'), ...stop],
  B: [...sys, ...stop, ...H, ...ph, ...T('q', 'Describe the image in detail .'), ...stop, ...A_,
      ...T('a', 'A gray cat is curled up asleep on a red sofa . Sunlight from a window falls across its fur , and a blue cushion rests beside it .'), ...stop],
  C: [...sys, ...stop, ...H, ...ph, ...T('q', 'What color is the bus ? Answer the question using a single word or phrase .'), ...stop, ...A_, ...T('a', 'Yellow'), ...stop],
};
const splice = (seq, nv) => seq.flatMap(t => t.k === 'ph' ? Array.from({ length: nv }, () => ({ k: 'img', w: '' })) : [t]);
const loss = t => t.k === 'a' || t.k === 'stop';
for (const nv of [9, 576]) {
  const S = Object.fromEntries(Object.entries(ex).map(([n, s]) => [n, splice(s, nv)]));
  const L = Object.values(S).map(s => s.length), Lmax = Math.max(...L), sum = L.reduce((a, b) => a + b, 0);
  const pads = 3 * Lmax - sum, lossN = Object.values(S).flat().filter(loss).length;
  console.log(`nv=${nv}: lengths A/B/C = ${L.join('/')}, Lmax ${Lmax}, padded cells ${3 * Lmax}, pads ${pads} (${(100 * pads / (3 * Lmax)).toFixed(1)}%), packed ${sum}, free after pack ${3 * Lmax - sum}`);
  console.log(`   loss tokens ${lossN} of ${3 * Lmax} cells (${(100 * lossN / (3 * Lmax)).toFixed(1)}%), of real tokens ${(100 * lossN / sum).toFixed(1)}%`);
  Object.entries(S).forEach(([n, s]) => console.log(`   ${n}: len ${s.length}, loss ${s.filter(loss).length}, img ${s.filter(t => t.k === 'img').length}`));
}
// Molmo: three QA annotations on one image (toy). Naive = three sequences each with the image; shared = image once.
const qa = [['What is the dog doing ?', 'Catching a frisbee .'], ['What color is the frisbee ?', 'Red .'], ['Is it indoors ?', 'No .']];
for (const nv of [9, 576]) {
  const ann = qa.map(([q, a]) => [...H, ...T('q', q), ...stop, ...A_, ...T('a', a), ...stop]);
  const naive = ann.map(x => nv + x.length), shared = nv + ann.reduce((s, x) => s + x.length, 0);
  console.log(`Molmo nv=${nv}: annotation lengths ${ann.map(x => x.length).join('/')}; naive rows ${naive.join('/')} total ${naive.reduce((a, b) => a + b)} tokens, 3 image encodes; shared ${shared} tokens, 1 encode; longest-row growth ${(100 * (shared / Math.max(...naive) - 1)).toFixed(0)}%`);
}
console.log('connectors: 224/14 =', 224 / 14, '->', (224 / 14) ** 2, '; 336/14 =', 336 / 14, '->', (336 / 14) ** 2, '; ratio', (576 / 256), '; width ratio', 336 / 224);
