// node build/sim/beyond.js > notes/sim-beyond.txt
// Loads the ch-beyond kits with the page's PE helpers (rng, gauss, softmax, clamp) and prints what the reader sees.
const fs = require('fs'), path = require('path'), vm = require('vm');
const P = path.join(__dirname, '..', '..');
const head = fs.readFileSync(path.join(P, 'parts/head.html'), 'utf8');
const peSrc = head.slice(head.indexOf('const PE = (() => {'), head.indexOf('</script>', head.indexOf('const PE = (() => {')));
const ctx = { console, Math, Float64Array, Array, Set, Object, String, Number, matchMedia: () => ({ matches: false, addEventListener() {} }), getComputedStyle: () => ({ getPropertyValue: () => '' }), document: { documentElement: { dataset: {} } } };
vm.createContext(ctx);
vm.runInContext(peSrc + '\nthis.PE = PE;', ctx);
const kit = ['00-base.js', '75-sinkhorn.js', '78-barlow.js', '81-dimcollapse.js'].map(f => fs.readFileSync(path.join(P, 'parts/kit', f), 'utf8')).join('\n');
vm.runInContext(kit + '\nthis.KIT = KIT;', ctx);
const { KIT } = ctx, f3 = x => x.toFixed(3);

console.log('== viz-sinkhorn: K = 4 prototypes x B = 8 images, eps 0.05, tau 0.1 ==');
KIT.skStates().forEach((Q, i) => {
  const lab = i === 0 ? 'raw      ' : 'it' + Math.ceil(i / 2) + (i % 2 ? ' rows  ' : ' cols  ');
  console.log(lab, 'rows', KIT.skRows(Q).map(f3).join(' '), '| cols', KIT.skCols(Q).map(f3).join(' '), '| top', KIT.skTop(Q).map(t => t + 1).join(''));
});
[0, 1, 2, 3].forEach(n => console.log('readout', n, ':', KIT.sinkhornText(n)));

console.log('\n== viz-barlow: gradient check (central differences, lambda_toy 3) ==');
{ const st = KIT.btData(), r = KIT.btLoss(st, 3); let worst = 0;
  for (let x = 0; x < st.W.length; x += 7) { const h = 1e-5, w = st.W[x]; st.W[x] = w + h; const a = KIT.btLoss(st, 3, true).L; st.W[x] = w - h; const b = KIT.btLoss(st, 3, true).L; st.W[x] = w;
    worst = Math.max(worst, Math.abs((a - b) / 2 / h - r.GW[x])); }
  console.log('max |fd - analytic| over 19 weights:', worst.toExponential(2)); }
console.log('settings', JSON.stringify(KIT.BT));
console.log('readout init:', KIT.barlowText(null));
for (const lam of [0, .002, .005, .02]) { const t0 = Date.now(), r = KIT.btRun(lam);
  console.log('lambda_paper', lam, 'toy', f3(lam * KIT.BT.SCALE), '| final L', f3(r.L), 'diag', f3(r.diag), 'offRMS', f3(r.off), '| ms', Date.now() - t0);
  console.log('  readout:', KIT.barlowText(lam)); }

console.log('\n== viz-dimcollapse ==');
console.log('settings', JSON.stringify(KIT.DC));
console.log('readout init:', KIT.dimcollapseText(null));
for (const m of ['healthy', 'dimensional', 'complete']) { const t0 = Date.now(), r = KIT.dcRun(m);
  const traj = [0, 20, 60].map(i => KIT.dcSpectrum(r.snaps[i], KIT.dcHeld()).map(v => v.toExponential(2)).join(' '));
  console.log(m, 'spectrum at steps 0/100/300:', traj.join(' | '), '| ms', Date.now() - t0);
  console.log('  readout:', KIT.dimcollapseText(m)); }
