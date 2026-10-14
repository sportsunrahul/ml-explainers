// Trace of the cold (tau 0.07) and hot (tau 1) runs from seed 65: gradient norm and the car / pizza pair gaps.
require('./pe.cjs');
const TWO = 2 * Math.PI, r = PE.rng(65), th0 = [0, 1, 2, 3, 4].map(() => r() * TWO), ph0 = [0, 1, 2, 3, 4].map(() => r() * TWO);
for (const tau of [.07, 1]) {
  let th = th0.slice(), ph = ph0.slice(); const rows = [];
  for (let k = 0; k <= 400; k++) { const g = PE.contrast(th, ph, tau);
    if ([0, 10, 25, 40, 60, 100, 200, 300, 400].includes(k)) rows.push(`${k}: L ${g.L.toFixed(3)} |g| ${Math.hypot(...g.gth, ...g.gph).toFixed(3)} gaps(deg) ${th.map((t, i) => (Math.acos(Math.min(1, Math.cos(t - ph[i]))) * 180 / Math.PI).toFixed(1)).join(' ')}`);
    th = th.map((v, i) => v - .05 * g.gth[i]); ph = ph.map((v, i) => v - .05 * g.gph[i]); }
  console.log('tau', tau); rows.forEach(x => console.log('  ' + x));
}
