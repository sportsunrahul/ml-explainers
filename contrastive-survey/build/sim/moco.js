// node build/sim/moco.js > notes/sim-moco.txt   (cwd = project root)
const fs = require('fs'); const KIT = {};
function rng(seed) { let a = seed >>> 0; return () => { a |= 0; a = a + 0x6D2B79F5 | 0;
  let t = Math.imul(a ^ a >>> 15, 1 | a); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }
function gauss(r) { return () => Math.sqrt(-2 * Math.log(1 - r())) * Math.cos(2 * Math.PI * r()); }
const PE = { rng, gauss };
eval(fs.readFileSync('parts/kit/40-queue.js', 'utf8').split('// Drawing.')[0]);
console.log('viz-queue simulation (build/sim/moco.js), 2026-10-13');
console.log('settings', JSON.stringify(KIT.Q));
console.log('Readout after a chip click (steady state after BURN steps), then after 1 step, then after RUN more steps:');
for (const m of KIT.Q.MS) {
  const st = KIT.qInit(m), a = KIT.qStats(st); KIT.qStep(st); const b = KIT.qStats(st);
  for (let i = 0; i < KIT.Q.RUN; i++) KIT.qStep(st); const c = KIT.qStats(st);
  console.log(`m=${m}  half-life ${KIT.qHalf(m).toFixed(1)} steps | init spread ${a.spread.toFixed(2)} lag ${a.lag.toFixed(2)} | +1 spread ${b.spread.toFixed(2)} lag ${b.lag.toFixed(2)} | +${KIT.Q.RUN} spread ${c.spread.toFixed(2)} lag ${c.lag.toFixed(2)}`);
}
console.log('Averages over 2000 later steps (spread, lag):');
for (const m of KIT.Q.MS) { const st = KIT.qInit(m); let s = 0, l = 0; for (let i = 0; i < 2000; i++) { KIT.qStep(st); const r = KIT.qStats(st); s += r.spread; l += r.lag; }
  console.log(`m=${m}  mean spread ${(s / 2000).toFixed(3)}  mean lag ${(l / 2000).toFixed(3)}`); }
