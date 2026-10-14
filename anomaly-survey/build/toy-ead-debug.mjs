// Debug: fit quality of the ridge student and PCA autoencoder on normal caps.
import fs from 'fs';
const strip = f => fs.readFileSync(new URL('../parts/' + f, import.meta.url), 'utf8').replace(/<\/?script>/g, '');
const TOY = new Function(strip('lib-toy.js') + '; return TOY;')();
const K = new Function('TOY', strip('lib-efficientad.js') + '; return EAD_KIT;')(TOY);
const lam = +(process.argv[2] || 1), k = +(process.argv[3] || 8);
K.fit({ k });
for (const [label, img] of [['train-like good', TOY.make('cap', 301)], ['new good', TOY.make('cap', 501)], ['scratch', TOY.defect(TOY.make('cap', 200), 'scratch', 200)]]) {
  const o = K.score(img); let vt = 0, es = 0, ea = 0, mu = [0, 0, 0];
  for (let q = 0; q < 225; q++) for (let d = 0; d < 3; d++) mu[d] += o.t[q * 3 + d] / 225;
  for (let q = 0; q < 225; q++) for (let d = 0; d < 3; d++) { vt += (o.t[q * 3 + d] - mu[d]) ** 2; es += (o.t[q * 3 + d] - o.s[q * 3 + d]) ** 2; ea += (o.t[q * 3 + d] - o.a[q * 3 + d]) ** 2; }
  const top = a => { const s = Array.from(a).map((v, i) => [v, i]).sort((x, y) => y[0] - x[0]).slice(0, 3); return s.map(([v, i]) => `${i}:${v.toFixed(3)}`).join(' '); };
  console.log(label.padEnd(16), 'T var', (vt / 675).toFixed(3), 'S mse', (es / 675).toFixed(3), 'A mse', (ea / 675).toFixed(3), '| top st', top(o.st), '| top ae', top(o.ae));
}
console.log('q', JSON.stringify(K.get().q));
