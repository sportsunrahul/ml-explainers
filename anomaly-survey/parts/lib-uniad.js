<script>
/* Toy UniAD on the toy cap (DOM-free; build/toy-uniad.mjs runs it in node). Tokens = the toy patch features on the
   15 x 15 grid. One attention layer reconstructs every token as a softmax-weighted average of tokens, with similarity
   -||query - key||^2 / tau. Three variants:
   full  - the query is the token itself and every key is visible: the token attends to itself, so it is copied.
   mask  - same query, but keys inside the token's own 7 x 7 neighbourhood are hidden (UniAD's neighbour-masked attention).
   query - a learned query per position (the mean good-cap feature there) replaces the token as the query, no mask:
           a stand-in for UniAD's learnable query embeddings, fit on good caps only.
   Error map = ||token - reconstruction||. */
const UNIAD_KIT = (() => {
  const T = typeof TOY !== 'undefined' ? TOY : null;
  const G = 15, NP = G * G, TAU = .08, R = 3;   // R = 3 gives the 7 x 7 window
  let Q = null;
  const feats = img => Float64Array.from(T.features(img));
  function learnQueries(n = 20) {
    const q = new Float64Array(NP * 3);
    for (let s = 0; s < n; s++) { const f = feats(T.make('cap', 600 + s)); for (let i = 0; i < NP * 3; i++) q[i] += f[i] / n; }
    return Q = q;
  }
  const masked = (i, j) => Math.max(Math.abs(i % G - j % G), Math.abs(Math.floor(i / G) - Math.floor(j / G))) <= R;
  function attend(x, mode, i) {               // attention weights of output token i over all keys
    const q = mode === 'query' ? (Q || learnQueries()) : x, w = new Float64Array(NP); let mx = -Infinity;
    for (let j = 0; j < NP; j++) {
      if (mode === 'mask' && masked(i, j)) { w[j] = -Infinity; continue; }
      let d = 0; for (let c = 0; c < 3; c++) d += (q[i * 3 + c] - x[j * 3 + c]) ** 2;
      w[j] = -d / TAU; if (w[j] > mx) mx = w[j];
    }
    let z = 0; for (let j = 0; j < NP; j++) { w[j] = w[j] === -Infinity ? 0 : Math.exp(w[j] - mx); z += w[j]; }
    for (let j = 0; j < NP; j++) w[j] /= z; return w;
  }
  function reconstruct(x, mode) {
    const y = new Float64Array(NP * 3), err = new Float64Array(NP), selfW = new Float64Array(NP);
    for (let i = 0; i < NP; i++) { const w = attend(x, mode, i); selfW[i] = w[i];
      for (let j = 0; j < NP; j++) if (w[j]) for (let c = 0; c < 3; c++) y[i * 3 + c] += w[j] * x[j * 3 + c];
      let e = 0; for (let c = 0; c < 3; c++) e += (x[i * 3 + c] - y[i * 3 + c]) ** 2; err[i] = Math.sqrt(e); }
    return { y, err, selfW };
  }
  return { G, NP, TAU, R, feats, learnQueries, attend, reconstruct, masked };
})();
if (typeof module !== 'undefined') module.exports = UNIAD_KIT;
</script>
