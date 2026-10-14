/* ToyPool: the 2D pool, the classifier and the scores behind the uncertainty-sampling figure (ch-uncertainty).
   Pure functions, no DOM, so the same code runs in the page, in the atlas and in node (build/sim-pool.mjs). */
const ToyPool = (() => {
  function rng(seed) { let a = seed >>> 0; return () => { a |= 0; a = a + 0x6D2B79F5 | 0;
    let t = Math.imul(a ^ a >>> 15, 1 | a); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }
  const gauss = r => () => Math.sqrt(-2 * Math.log(1 - r())) * Math.cos(2 * Math.PI * r());
  // Three classes as interleaved spiral arms: long curved boundaries, so where you label matters.
  function sample(n, seed) {
    const r = rng(seed), g = gauss(r), pts = [];
    for (let i = 0; i < n; i++) {
      const c = Math.floor(r() * 3), t = Math.sqrt(r());
      const ang = c * 2 * Math.PI / 3 + t * 3.4, rad = .25 + 2.55 * t;
      pts.push({ x: rad * Math.cos(ang) + .17 * g(), y: rad * Math.sin(ang) + .17 * g(), c });
    }
    return pts;
  }
  const K = 3, G = 13, LO = -3.2, HI = 3.2, SIG = .42;
  const centers = []; for (let i = 0; i < G; i++) for (let j = 0; j < G; j++) centers.push([LO + (HI - LO) * i / (G - 1), LO + (HI - LO) * j / (G - 1)]);
  function feat(x, y) { const f = new Float64Array(centers.length + 1); f[0] = 1;
    for (let k = 0; k < centers.length; k++) { const dx = x - centers[k][0], dy = y - centers[k][1]; f[k + 1] = Math.exp(-(dx * dx + dy * dy) / (2 * SIG * SIG)); } return f; }
  function softmax(z) { const m = Math.max(...z), e = z.map(v => Math.exp(v - m)), s = e.reduce((a, b) => a + b, 0); return e.map(v => v / s); }
  // Softmax regression on RBF features; full-batch gradient descent from zero, so a fit is deterministic.
  function train(X, Y, o = {}) {
    const D = X[0].length, W = Array.from({ length: K }, () => new Float64Array(D)), lr = o.lr || 2, lam = o.lam || 1e-3, it = o.iters || 300;
    for (let s = 0; s < it; s++) {
      const gW = Array.from({ length: K }, () => new Float64Array(D));
      for (let n = 0; n < X.length; n++) {
        const p = softmax(W.map(w => { let z = 0; for (let d = 0; d < D; d++) z += w[d] * X[n][d]; return z; }));
        for (let k = 0; k < K; k++) { const e = p[k] - (Y[n] === k ? 1 : 0); for (let d = 0; d < D; d++) gW[k][d] += e * X[n][d]; }
      }
      for (let k = 0; k < K; k++) for (let d = 0; d < D; d++) W[k][d] -= lr * (gW[k][d] / X.length + lam * W[k][d]);
    }
    return W;
  }
  function prob(W, f) { return softmax(W.map(w => { let z = 0; for (let d = 0; d < f.length; d++) z += w[d] * f[d]; return z; })); }
  const score = {
    lc: p => 1 - Math.max(...p),
    margin: p => { const s = [...p].sort((a, b) => b - a); return 1 - (s[0] - s[1]); },   // larger = more uncertain
    entropy: p => -p.reduce((a, v) => a + (v > 0 ? v * Math.log(v) : 0), 0) / Math.log(K),
  };
  // A labeling campaign one round at a time: start from one labeled point per class, then rounds of b picks.
  function session(pool, test, strategy, o = {}) {
    const b = o.b || 5, r = rng(o.seed || 7), F = pool.map(p => feat(p.x, p.y)), Ft = test.map(p => feat(p.x, p.y));
    const labeled = []; for (let c = 0; c < K; c++) labeled.push(pool.findIndex(p => p.c === c));
    let W = null, P = null;
    const fit = () => { W = train(labeled.map(i => F[i]), labeled.map(i => pool[i].c)); P = F.map(f => prob(W, f)); };
    const acc = () => { let ok = 0; test.forEach((p, i) => { const q = prob(W, Ft[i]); ok += q.indexOf(Math.max(...q)) === p.c; }); return ok / test.length; };
    fit(); const curve = [{ n: labeled.length, acc: acc() }];
    function next(measure) {   // indices this strategy would pick now
      const cand = []; for (let i = 0; i < pool.length; i++) if (!labeled.includes(i)) cand.push(i);
      if (strategy === 'random') { const c2 = cand.slice(); for (let i = c2.length - 1; i > 0; i--) { const j = Math.floor(r() * (i + 1)); [c2[i], c2[j]] = [c2[j], c2[i]]; } return c2.slice(0, b); }
      const m = measure || strategy;
      return cand.map(i => [score[m](P[i]), i]).sort((a, b2) => b2[0] - a[0] || a[1] - b2[1]).slice(0, b).map(x => x[1]);
    }
    function round(measure) { const pick = next(measure); labeled.push(...pick); fit(); curve.push({ n: labeled.length, acc: acc() }); return pick; }
    return { labeled, curve, round, probs: () => P, model: () => W, F };
  }
  function campaign(pool, test, strategy, o = {}) {
    const s = session(pool, test, strategy, o); for (let t = 0; t < (o.rounds || 8); t++) s.round(); return { curve: s.curve, labeled: s.labeled };
  }
  return { rng, sample, feat, train, prob, score, session, campaign, K };
})();
if (typeof module !== 'undefined') module.exports = ToyPool;
