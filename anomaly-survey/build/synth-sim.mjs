// Checks the synthetic kit: Perlin mask coverage per threshold, texture range, CutPaste mask area.
import fs from 'fs';
const load = (f, name) => new Function(fs.readFileSync(new URL('../parts/' + f, import.meta.url), 'utf8').replace(/<\/?script>/g, '') + '; return ' + name + ';')();
const S = load('lib-synthetic.js', 'SYNTH_KIT'), TOY = load('lib-toy.js', 'TOY');
for (const thr of [.4, .5, .55, .6, .65, .7]) console.log('thr', thr, 'mask fraction over seeds 1-6:', [1, 2, 3, 4, 5, 6].map(s => S.frac(S.perlinMask(s, thr)).toFixed(3)).join(' '));
const t = S.texture(3); console.log('texture range', Math.min(...t).toFixed(2), Math.max(...t).toFixed(2));
const img = TOY.make('mesh', 5).px, cp = S.cutpaste(img, [20, 20], [70, 60], 30, 20, 0, .08);
console.log('cutpaste mask px', cp.mask.reduce((a, b) => a + b), 'expected', 30 * 20, 'area ratio', (S.frac(cp.mask) * 100).toFixed(1) + '%');
const sc = S.cutpaste(img, [30, 30], [80, 40], 4, 12, .5, .05); console.log('scar mask px', sc.mask.reduce((a, b) => a + b));
// Does the toy NN detector flag a DRAEM fake on the mesh? (sanity: fakes should be anomalous under the chapter-1 score)
const B = TOY.bank('mesh', 10, 1), m = S.perlinMask(3, .6), fake = { kind: 'mesh', px: S.draem(img, m, t, .6) };
const sc2 = TOY.nnScores(TOY.features(fake), B); console.log('draem fake max patch score', Math.max(...sc2).toFixed(2), 'normal mesh max', Math.max(...TOY.nnScores(TOY.features(TOY.make('mesh', 99)), B)).toFixed(2));
