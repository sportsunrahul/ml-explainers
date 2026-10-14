// ch-patchcore figure: image scores of the two displayed test caps (good 104, scratched 200) against greedy coresets of a 20-cap bank.
import fs from 'fs';
const src = fs.readFileSync(new URL('../parts/lib-toy.js', import.meta.url), 'utf8').replace(/<\/?script>/g, '');
const TOY = new Function(src + '; return TOY;')();
const B = TOY.bank('cap', 20, 1), n = B.length / 3;
const d2 = (i, j) => (B[i*3]-B[j*3])**2 + (B[i*3+1]-B[j*3+1])**2 + (B[i*3+2]-B[j*3+2])**2;
const r = TOY.rng(7), anchors = Array.from({ length: 10 }, () => Math.floor(r() * n));
const md = new Float64Array(n); for (let i = 0; i < n; i++) { let s = 0; for (const a of anchors) s += Math.sqrt(d2(i, a)); md[i] = s / 10; }
const order = []; for (let k = 0; k < n; k++) { let bi = 0; for (let i = 1; i < n; i++) if (md[i] > md[bi]) bi = i; order.push(bi); for (let i = 0; i < n; i++) { const d = Math.sqrt(d2(i, bi)); if (d < md[i]) md[i] = d; } }
const fg = TOY.features(TOY.make('cap', 104)), fs_ = TOY.features(TOY.defect(TOY.make('cap', 200), 'scratch', 200));
const sc = (f, P) => { let mx = 0, arg = 0; for (let q = 0; q < TOY.G * TOY.G; q++) { let b = Infinity; for (const p of P) { const d = (f[q*3]-B[p*3])**2 + (f[q*3+1]-B[p*3+1])**2 + (f[q*3+2]-B[p*3+2])**2; if (d < b) b = d; } if (Math.sqrt(b) > mx) { mx = Math.sqrt(b); arg = q; } } return [mx, arg]; };
for (const pct of [.5, 1, 2, 5, 10, 25, 50, 100]) { const P = order.slice(0, Math.max(1, Math.floor(n * pct / 100)));
  const g = sc(fg, P), s = sc(fs_, P); console.log(`${pct}%: good ${g[0].toFixed(2)} (patch ${g[1]})  scratched ${s[0].toFixed(2)} (patch ${s[1]})  ${s[0] > g[0] ? 'separated' : 'NOT separated'}`); }
