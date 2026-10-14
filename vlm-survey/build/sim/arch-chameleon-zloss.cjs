// Checks the two derived gradient sums used in the chameleonflow backward text.
const V = 64, y = 5, lam = 1e-5;
const o = Array.from({ length: V }, (_, i) => Math.sin(i * 1.3) * 3 + 2);
function f(o) {
  const Z = o.reduce((s, v) => s + Math.exp(v), 0), lZ = Math.log(Z);
  return { ce: lZ - o[y], z: lam * lZ * lZ, lZ, p: o.map(v => Math.exp(v) / Z) };
}
const b = f(o);
const gce = b.p.map((p, i) => p - (i === y ? 1 : 0)), gz = b.p.map(p => 2 * lam * b.lZ * p);
const sum = a => a.reduce((s, v) => s + v, 0);
console.log('sum dCE/do =', sum(gce).toExponential(3), ' sum dz/do =', sum(gz).toExponential(4), ' 2*lam*logZ =', (2 * lam * b.lZ).toExponential(4));
// Shift every logit by c: CE unchanged, z-loss changes.
for (const c of [0, 5, -5]) { const s = f(o.map(v => v + c)); console.log('shift', c, 'CE', s.ce.toFixed(6), 'zloss', s.z.toExponential(4), 'logZ', s.lZ.toFixed(3)); }
// Numerical derivative wrt shift.
const e = 1e-4, d = (f(o.map(v => v + e)).ce + f(o.map(v => v + e)).z - b.ce - b.z) / e;
console.log('d(total)/dc numeric', d.toExponential(4));
// Derived step counts.
console.log('7B steps', 9.2e12 / 2 ** 23, ' 34B steps', 9.2e12 / (3 * 2 ** 22), ' 600k/0.55', 6e5 / .55, ' 600k/0.8', 6e5 / .8);
console.log('seqs/batch at 4096: 7B', 2 ** 23 / 4096, ' 34B', 3 * 2 ** 22 / 4096, ' images per 4096 ctx', 4096 / 1024);
console.log('Florence steps B', 3e9 / 2048, ' L', 3e9 / 3072, ' hi-res B', 0.5e9 / 2048, ' L', 0.1e9 / 3072);
console.log('params B', 90 + 140, ' L', 360 + 410);
