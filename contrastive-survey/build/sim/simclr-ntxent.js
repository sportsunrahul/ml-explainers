// Checks the ch-simclr NT-Xent figure: KIT.ntxentRows (paper order) against the shared KIT.ntxentAngles (k, k+N order).
const fs = require('fs'), P = __dirname + '/../../parts/kit/';
const src = ['00-base.js', '05-circle.js', '52-ntxent.js'].map(f => fs.readFileSync(P + f, 'utf8')).join('\n');
const KIT = new Function('PE', src + '\nreturn KIT;')({});
const rad = N => KIT.NTX_DEG[N].map(d => d * Math.PI / 180);
for (const N of [2, 4]) for (const t of [0.1, 0.5, 1]) {
  const th = rad(N), r = KIT.ntxentRows(th, t), re = [...th.filter((_, i) => i % 2 === 0), ...th.filter((_, i) => i % 2 === 1)];
  const ref = KIT.ntxentAngles(re, t).loss;
  if (Math.abs(ref - r.mean) > 1e-9) throw new Error('mismatch');
  console.log(`N=${N} tau=${t}: row1 p(partner)=${r.P[0][1].toFixed(4)} loss=${r.loss[0].toFixed(4)} mean=${r.mean.toFixed(4)} (shared kit mean ${ref.toFixed(4)})`);
}
console.log('N=2 cosine matrix:\n' + KIT.ntxentRows(rad(2), .5).S.map(r => r.map(v => v.toFixed(3).padStart(7)).join('')).join('\n'));
console.log('readout N=2 tau=0.5: ' + KIT.ntxentText(2, .5));
