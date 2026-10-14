// How far do the points move if the cold run (tau 0.07, seed 65) trains 400 more steps, and 3600 more?
require('./pe.cjs');
const names = ['dog', 'cat', 'car', 'pizza', 'tree'], TWO = 2 * Math.PI;
const r = PE.rng(65); let th = names.map(() => r() * TWO), ph = names.map(() => r() * TWO);
const step = (tau) => { const g = PE.contrast(th, ph, tau); th = th.map((v, i) => v - .05 * g.gth[i]); ph = ph.map((v, i) => v - .05 * g.gph[i]); };
const deg = x => (x * 180 / Math.PI);
const gap = i => deg(Math.acos(Math.min(1, Math.cos(th[i] - ph[i]))));
for (let k = 0; k < 400; k++) step(.07);
const a = th.concat(ph), gaps400 = names.map((_, i) => gap(i).toFixed(1)).join(' ');
for (let k = 0; k < 400; k++) step(.07);
const b = th.concat(ph);
console.log('gaps (deg) at 400:', gaps400, ' at 800:', names.map((_, i) => gap(i).toFixed(1)).join(' '));
console.log('max point movement 400 -> 800 (deg):', Math.max(...a.map((v, i) => deg(Math.abs(Math.atan2(Math.sin(b[i] - v), Math.cos(b[i] - v)))))).toFixed(2));
for (let k = 0; k < 3600; k++) step(.07);
console.log('gaps (deg) at 4400:', names.map((_, i) => gap(i).toFixed(1)).join(' '), ' loss', PE.contrast(th, ph, .07).L.toFixed(4));
