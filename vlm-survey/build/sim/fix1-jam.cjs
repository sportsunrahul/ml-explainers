// Item 22: how often does cold (tau 0.07) and hot (tau 1) training end with pairs jammed on top of each other? Seeds 1-200.
require('./pe.cjs');
const N = 5, TWO = 2 * Math.PI, names = ['dog', 'cat', 'car', 'pizza', 'tree'];
const start = seed => { const r = PE.rng(seed); return { th: names.map(() => r() * TWO), ph: names.map(() => r() * TWO) }; };
const d = (a, b) => Math.abs(Math.atan2(Math.sin(a - b), Math.cos(a - b)));
const strictRow = (row, i) => row.every((s, j) => j === i || row[i] - s > 1e-9);
for (const tau of [.07, 1]) {
  let notStrict = 0, photoJam = 0, pairJam = 0, pairJam2 = 0, argmaxFail = 0;
  for (let s = 1; s <= 200; s++) {
    let { th, ph } = start(s);
    for (let k = 0; k < 400; k++) { const g = PE.contrast(th, ph, tau); th = th.map((v, i) => v - .05 * g.gth[i]); ph = ph.map((v, i) => v - .05 * g.gph[i]); }
    const S = PE.contrast(th, ph, tau).S;
    if (S.some((row, i) => !strictRow(row, i))) notStrict++;
    if (S.some((row, i) => row.indexOf(Math.max(...row)) !== i)) argmaxFail++;
    let pj = false, ppj = false, pp2 = false;
    for (let i = 0; i < N; i++) for (let k = i + 1; k < N; k++) {
      if (d(th[i], th[k]) < .05) pj = true;
      if (d(th[i], th[k]) < .1 && d(ph[i], ph[k]) < .1) ppj = true;
      if (d(th[i], th[k]) < .3 && d(ph[i], ph[k]) < .3) pp2 = true;
    }
    if (pj) photoJam++; if (ppj) pairJam++; if (pp2) pairJam2++;
  }
  console.log(`tau ${tau}: not strictly matched ${notStrict}/200; argmax fails ${argmaxFail}/200; two photos within .05 rad ${photoJam}; two whole pairs (photo and caption) within .1 rad ${pairJam}, within .3 rad ${pairJam2}`);
}
