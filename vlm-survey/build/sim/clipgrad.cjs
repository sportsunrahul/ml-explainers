const deg = d => d * Math.PI / 180;
const names = ['dog', 'cat', 'car', 'pizza', 'tree'], N = 5;
const PH = [90, 162, 234, 306, 18].map(deg);          // captions, evenly spread
const OTH = [null, 168, 228, 300, 24].map(d => d === null ? null : deg(d)); // other photos near their captions
function grad(a, tau) {
  const th = OTH.map((t, i) => i === 0 ? a : t);
  const cos = (x, y) => Math.cos(x - y);
  const sm = v => { const m = Math.max(...v), e = v.map(x => Math.exp(x - m)), Z = e.reduce((s, x) => s + x); return e.map(x => x / Z); };
  const p = sm(PH.map(ph => cos(th[0], ph) / tau));                 // row of photo 0
  const q = PH.map((ph, j) => sm(th.map(t => cos(t, ph) / tau))[0]); // photo 0's share of column j
  const c = PH.map((_, j) => (p[j] + q[j]) / 2 - (j === 0 ? 1 : 0));
  // gradient wrt u (2D), without 1/(N tau) prefactor
  let gx = 0, gy = 0; PH.forEach((ph, j) => { gx += c[j] * Math.cos(ph); gy += c[j] * Math.sin(ph); });
  const ux = Math.cos(a), uy = Math.sin(a), rad = gx * ux + gy * uy, tx = gx - rad * ux, ty = gy - rad * uy;
  return { p, q, c, g: [gx, gy], rad, tan: Math.hypot(tx, ty), tdir: Math.sign(-(tx * -uy + ty * ux)) };
}
for (const tau of [0.07, 0.2]) for (const [lab, a] of [['far (on tree side, 350deg)', deg(330)], ['on cat caption', deg(160)], ['between', deg(125)], ['on own', deg(90)], ['near own', deg(100)]]) {
  const r = grad(a, tau);
  console.log(tau, lab.padEnd(28), 'c=', r.c.map(x => x.toFixed(3)).join(' '), '| rad', r.rad.toFixed(3), 'tan', r.tan.toFixed(3), 'dir', r.tdir);
}
for (const [lab, a0] of [['far', deg(330)], ['oncat', deg(160)]]) { let a = a0; const tr = [];
  for (let k = 0; k < 200; k++) { const r = grad(a, .07); const dLda = r.g[0] * -Math.sin(a) + r.g[1] * Math.cos(a); a -= .08 * dLda; if (k % 20 === 0) tr.push((a * 180 / Math.PI % 360 + 360) % 360 | 0); }
  console.log(lab, tr.join(' ')); }
