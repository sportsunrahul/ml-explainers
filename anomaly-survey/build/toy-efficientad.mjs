// Toy EfficientAD check: local (student-teacher) and global (autoencoder) gaps on good, scratched and missing-bolt caps.
import fs from 'fs';
const strip = f => fs.readFileSync(new URL('../parts/' + f, import.meta.url), 'utf8').replace(/<\/?script>/g, '');
const TOY = new Function(strip('lib-toy.js') + '; return TOY;')();
const K = new Function('TOY', strip('lib-efficientad.js') + '; return EAD_KIT;')(TOY);
const k = +(process.argv[2] || 8), lam = +(process.argv[3] || 1);
console.time('fit'); K.fit({ k, nTrain: +(process.argv[3] || 40), kn: +(process.argv[4] || 3) }); console.timeEnd('fit');
const G = 15, inMask = (img, q) => { const i = q % G, j = Math.floor(q / G); let m = 0; for (let y = j * 8; y < j * 8 + 16; y++) for (let x = i * 8; x < i * 8 + 16; x++) m += img.mask[y * 128 + x]; return m > 0; };
const argmax = a => a.reduce((b, v, i) => v > a[b] ? i : b, 0);
const rows = [];
const run = (label, mk) => { const r = []; for (let s = 0; s < 10; s++) { const img = mk(s), o = K.score(img);
  r.push({ st: o.maxSt, ae: o.maxAe, im: o.image, hitSt: img.mask ? inMask(img, argmax(o.nst)) : null, hitAe: img.mask ? inMask(img, argmax(o.nae)) : null }); }
  const m = key => (r.reduce((a, x) => a + x[key], 0) / r.length).toFixed(3), mn = key => Math.min(...r.map(x => x[key])).toFixed(3), mx = key => Math.max(...r.map(x => x[key])).toFixed(3);
  console.log(label.padEnd(9), `ST max mean ${m('st')} [${mn('st')}..${mx('st')}]  AE max mean ${m('ae')} [${mn('ae')}..${mx('ae')}]  image ${m('im')} [${mn('im')}..${mx('im')}]`,
    r[0].hitSt === null ? '' : `ST peak in mask ${r.filter(x => x.hitSt).length}/10, AE peak in mask ${r.filter(x => x.hitAe).length}/10`);
  rows.push([label, r]); };
run('good', s => TOY.make('cap', 500 + s));
run('scratch', s => TOY.defect(TOY.make('cap', 200 + s), 'scratch', 200 + s));
run('missing', s => TOY.defect(TOY.make('cap', 200 + s), 'missing', 200 + s));
run('extra', s => K.extraHole(TOY.make('cap', 200 + s), s));
const good = rows[0][1].map(x => x.im);
for (const [label, r] of rows.slice(1)) { const auc = r.reduce((a, x) => a + good.filter(g => g < x.im).length, 0) / (r.length * good.length); console.log(label, 'toy image AUROC vs good', auc.toFixed(2)); }
const o = K.score(TOY.make('cap', 501)); for (const p of [0, .9, .99, .999, .9999, .99999]) console.log('p_hard', p, 'elements kept', K.hardMask(o, p).n);
console.log('per-seed image scores:'); for (const [label, r] of rows) console.log(' ', label.padEnd(8), r.map(x => x.im.toFixed(2)).join(' '), '| ST', r.map(x => x.st.toFixed(2)).join(' '), '| AE', r.map(x => x.ae.toFixed(2)).join(' '));
