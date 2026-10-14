// Prints the ch-negatives toy numbers with the kit's own code: node build/sim/negatives.js
const fs = require('fs'); const KIT = {}; const PE = {};
eval(fs.readFileSync('parts/kit/65-negatives.js', 'utf8'));
console.log('settings', JSON.stringify(KIT.NEG));
for (const mode of ['plain', 'hard', 'debias']) {
  const c = KIT.negCompute(mode), f = x => x.toFixed(3);
  console.log(mode.padEnd(7), 'share false', f(c.shareFalse), 'share 2 nearest true', f(c.shareHard), 'share other 18', f(c.shareRest),
    'max share', f(Math.max(...c.share)), 'min share', f(Math.min(...c.share)));
  console.log('        pos term', f(c.pos), 'false mass', f(c.falseMass), 'true mass', f(c.trueMass), 'estimate N tau+ e^(s+/tau)', f(c.estimate),
    'Ng', f(c.ng), 'floored', c.floored, 'loss all', f(c.lossAll), 'loss true only', f(c.lossTrue), 'loss debiased', f(c.lossDebiased));
  console.log('        readout:', KIT.negText(mode));
}
// Worked example for Eq. 5: three negatives with s = 0.9, 0.5, 0.0.
for (const tau of [0.5, 0.1]) { const e = [0.9, 0.5, 0].map(s => Math.exp(s / tau)), Z = e.reduce((a, v) => a + v, 0);
  console.log('worked tau', tau, 'terms', e.map(v => v.toFixed(2)).join(' '), 'shares', e.map(v => (v / Z).toFixed(4)).join(' ')); }
