const fs = require('fs'); const KIT = {};
function rng(seed) { let a = seed >>> 0; return () => { a |= 0; a = a + 0x6D2B79F5 | 0;
  let t = Math.imul(a ^ a >>> 15, 1 | a); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }
const PE = { rng };
eval(fs.readFileSync('parts/kit/05-circle.js', 'utf8'));
eval(fs.readFileSync('parts/kit/85-supcon.js', 'utf8'));
if (process.argv[2]) Object.assign(KIT.SC, JSON.parse(process.argv[2]));
console.log('settings', JSON.stringify(KIT.SC));
// The 'self' mode must equal NT-Xent exactly.
{ const th = KIT.scInit(), a = KIT.supconAngles(th, 'self', KIT.SC.TAU), b = KIT.ntxentAngles(th, KIT.SC.TAU);
  console.log('self vs ntxent: loss diff', Math.abs(a.loss - b.loss).toExponential(1), 'max grad diff', Math.max(...a.g.map((x, k) => Math.abs(x - b.g[k]))).toExponential(1)); }
const wrap = a => Math.atan2(Math.sin(a), Math.cos(a));
const classSpread = th => { // mean angular distance of each view from its class's circular mean, in degrees
  const out = []; for (let c = 0; c < KIT.SC.C; c++) { const ks = th.map((t, k) => k).filter(k => KIT.scClass(k) === c);
    const m = Math.atan2(ks.reduce((s, k) => s + Math.sin(th[k]), 0), ks.reduce((s, k) => s + Math.cos(th[k]), 0));
    out.push(ks.reduce((s, k) => s + Math.abs(wrap(th[k] - m)), 0) / ks.length * 180 / Math.PI); }
  return out.map(x => x.toFixed(0)).join('/'); };
for (const mode of ['self', 'sup']) {
  const th = KIT.scInit(); let L;
  for (let s = 0; s <= KIT.SC.STEPS; s++) {
    if ([0, 30, 60, 100, 150, 200, 300].includes(s)) { const st = KIT.scStats(th), cs = KIT.circleStats(th);
      console.log(mode, 's' + s, 'loss', KIT.supconAngles(th, mode, KIT.SC.TAU).loss.toFixed(3), 'nn-same-class', st.hit + '/' + st.of, 'class spread deg', classSpread(th), 'twin gap', cs.pairGap.toFixed(3)); }
    if (s < KIT.SC.STEPS) KIT.scStep(th, mode);
  }
}
