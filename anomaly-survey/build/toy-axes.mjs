// For each pair of feature axes: bank range and the 2D gap of the top scratch patch (cap and mesh), several seeds.
import fs from 'fs';
const src = fs.readFileSync(new URL('../parts/lib-toy.js', import.meta.url), 'utf8').replace(/<\/?script>/g, '');
const TOY = new Function(src + '; return TOY;')();
for (const kind of ['cap', 'mesh']) {
  const B = TOY.bank(kind, 4, 1), n = B.length / 3;
  for (const [a, b] of [[0, 1], [0, 2], [1, 2]]) {
    const gaps = [];
    for (let s = 200; s < 208; s++) {
      const d = TOY.defect(TOY.make(kind, s), 'scratch', s), f = TOY.features(d), sc = TOY.nnScores(f, B);
      let k = 0; for (let q = 1; q < sc.length; q++) if (sc[q] > sc[k]) k = q;
      let g = 1e9; for (let t = 0; t < n; t++) g = Math.min(g, Math.hypot(B[t*3+a] - f[k*3+a], B[t*3+b] - f[k*3+b]));
      gaps.push(g.toFixed(2));
    }
    const r = [a, b].map(x => { let lo = 9, hi = -9; for (let t = 0; t < n; t++) { lo = Math.min(lo, B[t*3+x]); hi = Math.max(hi, B[t*3+x]); } return `${lo.toFixed(1)}..${hi.toFixed(1)}`; });
    console.log(kind, `axes ${a},${b}`, 'range', r.join(' '), 'scratch 2D gaps', gaps.join(' '));
  }
}
