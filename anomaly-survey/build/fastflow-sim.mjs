// Checks the toy flow: invariance of log p_X(x) across stages, total mass of each stage's density, ranges, ASCII plots.
import fs from 'fs';
const src = fs.readFileSync(new URL('../parts/lib-fastflow.js', import.meta.url), 'utf8').replace(/<\/?script>/g, '');
const F = new Function(src + '; return FASTFLOW_KIT;')();
const ext = 3.4, n = 120, dx = 2 * ext / n;
for (const st of [0, 0.5, 1, 2, 2.5, 3, 4]) {
  let mass = 0, mx = 0;
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) { const p = Math.exp(F.logDensity([-ext + (i + .5) * dx, -ext + (j + .5) * dx], st)); mass += p * dx * dx; mx = Math.max(mx, p); }
  const pts = F.sample(2000).map(x => F.push(x, st).h), xs = pts.map(p => p[0]), ys = pts.map(p => p[1]);
  const q = a => { const s = [...a].sort((u, v) => u - v); return [s[10], s[1990]].map(v => v.toFixed(2)).join('..'); };
  console.log(`stage ${st}: mass in box ${mass.toFixed(3)}  peak ${mx.toFixed(3)}  sample x ${q(xs)} y ${q(ys)}`);
}
for (const [name, x] of Object.entries(F.TEST)) {
  const vals = [0, .3, 1, 1.7, 2, 3, 3.5, 4].map(st => { const r = F.push(x, st); return (F.logDensity(r.h, st) + r.logdet).toFixed(4); });
  const z = F.push(x, 4).h;
  console.log(`${name}: log pX across stages ${vals.join(' ')} | z = ${z.map(v => v.toFixed(2))} | log N(z) ${F.logN(z).toFixed(2)} logdet ${F.push(x, 4).logdet.toFixed(2)}`);
}
// percentile of each test point's log-density among training samples
const train = F.sample(4000, 11).map(x => F.logDensity(x, 0)).sort((a, b) => a - b);
for (const [name, x] of Object.entries(F.TEST)) { const lp = F.logDensity(x, 0); console.log(name, 'log pX', lp.toFixed(2), 'fraction of training points with lower log p', (train.filter(v => v < lp).length / train.length).toFixed(4)); }
const plot = st => { const rows = []; for (let j = 15; j >= 0; j--) { let r = ''; for (let i = 0; i < 48; i++) { const p = Math.exp(F.logDensity([-ext + (i + .5) * 2 * ext / 48, -ext + (j + .5) * 2 * ext / 16], st)); r += p > .2 ? '#' : p > .08 ? '+' : p > .02 ? '.' : ' '; } rows.push(r); } return rows.join('\n'); };
console.log('stage 0\n' + plot(0)); console.log('stage 2\n' + plot(2)); console.log('stage 4\n' + plot(4));
