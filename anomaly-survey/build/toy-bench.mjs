// ch-benchmark toy: image AUROC, pixel AUROC and AU-PRO (FPR <= 0.3) for find-all, miss-tiny and miss-large.
import fs from 'fs';
const load = f => fs.readFileSync(new URL('../parts/' + f, import.meta.url), 'utf8').replace(/<\/?script>/g, '');
const TOY = new Function(load('lib-toy.js') + '; return TOY;')();
const BENCH_KIT = new Function('TOY', load('lib-benchmark.js') + '; return BENCH_KIT;')(TOY);
const { items, featured } = BENCH_KIT.build();
const base = items.map(it => BENCH_KIT.pixels(it));
console.log('image scores good', base.slice(0, 8).map(m => Math.max(...m.patch).toFixed(2)).join(' '));
console.log('image scores bad ', base.slice(8).map(m => Math.max(...m.patch).toFixed(2)).join(' '));
const f = items[featured]; console.log('featured region pixels', [1, 2].map(l => f.img.lab.filter(v => v === l).length));
for (const [name, miss] of [['find all', 0], ['miss tiny', 2], ['miss large', 1]]) {
  const maps = base.slice(); maps[featured] = BENCH_KIT.pixels(f, miss);
  const g = BENCH_KIT.grade(items, maps);
  console.log(name.padEnd(10), 'imgAUROC', g.imgAuroc.toFixed(3), 'pixAUROC', g.pixAuroc.toFixed(4), 'AU-PRO', g.aupro.toFixed(3), 'regions', g.regions, 'P', g.P, 'N', g.Nn, 'top', g.top.toFixed(2));
  for (const t of [.1, .2, .3, .5]) { const a = BENCH_KIT.at(items, maps, t); console.log('   t', t, 'tpr', a.tpr.toFixed(3), 'fpr', a.fpr.toFixed(4), 'pro', a.pro.toFixed(3), 'rejBad', a.rejBad, 'rejGood', a.rejGood); }
}
