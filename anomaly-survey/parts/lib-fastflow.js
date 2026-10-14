<script>
/* Toy 2D normalizing flow for ch-fastflow (and the atlas). DOM-free; node runs it in build/fastflow-sim.mjs.
   The data distribution is DEFINED as the pushforward of N(0, I) through four affine couplings g1..g4 (generative
   direction, form of FastFlow Fig. 2b: b' = b * exp(s(a)) + t(a)). The flow f = g1^-1 o g2^-1 o g3^-1 o g4^-1 is
   therefore exact, every intermediate density is exact, and a layer applied at strength u (s -> u s, t -> u t) is
   itself an affine coupling, so the animation between stages shows exact densities too. */
const FASTFLOW_KIT = (() => {
  // Generative couplings: c = index of the conditioning coordinate (unchanged), the other one is scaled and shifted.
  const G = [
    { c: 0, s: a => -0.9, t: a => 0 },                                          // squeeze into a thin band
    { c: 0, s: a => 0, t: a => 0.45 * a * a - 1.1 },                              // bend it into a crescent
    { c: 1, s: a => 0.3 * Math.tanh(a), t: a => 0.55 * Math.sin(1.4 * a) },     // twist it sideways
    { c: 0, s: a => 0.12, t: a => 0.25 * a },                                     // tilt
  ];
  const K = G.length;
  // Flow layer j (1..K) is the inverse of generative layer K + 1 - j.
  const flowLayer = j => G[K - j];
  // Apply flow layer j at strength u to point h. Inverse of g: b = (b' - t(a)) * exp(-s(a)); at strength u the coupling is
  // b_new = b * exp(-u s(a)) - u t(a) exp(-u s(a)), whose log|det| is -u s(a).
  function fwd(j, h, u = 1) {
    const L = flowLayer(j), a = h[L.c], b = h[1 - L.c], s = L.s(a), t = L.t(a), e = Math.exp(-u * s);
    const out = h.slice(); out[1 - L.c] = (b - u * t) * e; return [out, -u * s];
  }
  function inv(j, h, u = 1) {
    const L = flowLayer(j), a = h[L.c], b = h[1 - L.c], s = L.s(a), t = L.t(a);
    const out = h.slice(); out[1 - L.c] = b * Math.exp(u * s) + u * t; return [out, u * s];
  }
  const logN = z => -Math.log(2 * Math.PI) - .5 * (z[0] * z[0] + z[1] * z[1]);
  // Position after `stage` layers (stage may be fractional: k + u means layers 1..k, then layer k+1 at strength u).
  function push(x, stage) {
    let h = x.slice(), ld = 0; const k = Math.floor(stage), u = stage - k;
    for (let j = 1; j <= k; j++) { const r = fwd(j, h); h = r[0]; ld += r[1]; }
    if (u > 0 && k < K) { const r = fwd(k + 1, h, u); h = r[0]; ld += r[1]; }
    return { h, logdet: ld };
  }
  // log density of the intermediate variable at `stage`, evaluated at h: undo the partial layer, then run the rest to z.
  function logDensity(h, stage) {
    const k = Math.floor(stage), u = stage - k; let p = h.slice(), ld = 0;
    if (u > 0 && k < K) { const r = inv(k + 1, p, u); p = r[0]; ld += r[1]; }   // p(h) = p_k(h_k) |dh_k/dh|
    for (let j = k + 1; j <= K; j++) { const r = fwd(j, p); p = r[0]; ld += r[1]; }
    return logN(p) + ld;
  }
  // Samples of the data distribution: z ~ N(0, I) pushed through g1..g4.
  function sample(n, seed = 7) {
    let a = seed >>> 0; const r = () => { a |= 0; a = a + 0x6D2B79F5 | 0; let t = Math.imul(a ^ a >>> 15, 1 | a); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; };
    const g = () => Math.sqrt(-2 * Math.log(1 - r())) * Math.cos(2 * Math.PI * r());
    const out = [];
    for (let i = 0; i < n; i++) { let h = [g(), g()]; for (let j = K; j >= 1; j--) h = inv(j, h)[0]; out.push(h); }
    return out;
  }
  const TEST = { anomaly: [1.25, 0.95], normal: [-0.15, -1.05] };
  return { K, push, logDensity, logN, sample, TEST };
})();
if (typeof module !== 'undefined') module.exports = FASTFLOW_KIT;
</script>
