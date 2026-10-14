<script>
/* Toy EfficientAD on the toy cap (DOM-free; build/toy-efficientad.mjs runs it in node).
   Teacher T = the toy patch features (3 per patch). Student S = a nearest-neighbour regressor: it finds the most similar
   good training patch (16 pooled pixels) and outputs that patch's teacher features, so like a small network trained only
   on good parts it imitates well only near what it has seen (a linear student could not fit T even on good caps: error
   1.0 vs variance 1.36). Autoencoder A = PCA of the whole 15x15x3 teacher map (the optimal linear autoencoder), fit on
   good caps. Student head S2 = the same neighbour's autoencoder output. Maps are calibrated
   with EfficientAD's quantile rule (q_a = 0.9 -> 0, q_b = 0.995 -> 0.1) on held-out good caps. */
const EAD_KIT = (() => {
  const T = typeof TOY !== 'undefined' ? TOY : null;
  let K = null;
  const G = 15, NP = G * G;
  // Toy teacher: the toy kit's three patch features (edge balance, diagonal edges, brightness), computed with a larger
  // stabiliser in the edge ratios (0.3 instead of 0.001). On flat metal the kit's raw ratios are pure noise, which no
  // student can predict; with 0.3 they fade to zero there and stay near +-1 on real edges, scratches and bolts.
  const EPS = .3;
  function teacher(img) {
    const N = 128, P = 16, S = 8, p = img.px, f = new Float64Array(NP * 3);
    for (let j = 0; j < G; j++) for (let i = 0; i < G; i++) {
      let xx = 0, yy = 0, xy = 0, m = 0;
      for (let y = j * S; y < j * S + P; y++) for (let x = i * S; x < i * S + P; x++) {
        const gx = x > 0 && x < N - 1 ? (p[y * N + x + 1] - p[y * N + x - 1]) / 2 : 0, gy = y > 0 && y < N - 1 ? (p[(y + 1) * N + x] - p[(y - 1) * N + x]) / 2 : 0;
        xx += gx * gx; yy += gy * gy; xy += gx * gy; m += p[y * N + x]; }
      const e = xx + yy + EPS, k = (j * G + i) * 3;
      f[k] = (xx - yy) / e / .35; f[k + 1] = 2 * xy / e / .35; f[k + 2] = (m / (P * P) - .42) / .18;
    }
    return f;
  }
  const D = 16;
  function desc(img) {
    const N = 128, out = new Float32Array(NP * D);
    for (let j = 0; j < G; j++) for (let i = 0; i < G; i++) {
      const o = (j * G + i) * D;
      for (let b = 0; b < 4; b++) for (let a = 0; a < 4; a++) {
        let v = 0; for (let dy = 0; dy < 4; dy++) for (let dx = 0; dx < 4; dx++) v += img.px[(j * 8 + b * 4 + dy) * N + i * 8 + a * 4 + dx];
        out[o + b * 4 + a] = v / 16;
      }
    }
    return out;
  }
  // Index of the nearest training patch (squared distance over the 16 pooled pixels) for every patch of X.
  function nearest(X, bank, kn) {               // the kn nearest training patches of every patch of X
    const nb = bank.length / D, idx = new Int32Array(NP * kn);
    for (let q = 0; q < NP; q++) { const bd = new Float64Array(kn).fill(Infinity), bi = new Int32Array(kn); const o = q * D;
      for (let t = 0; t < nb; t++) { let d = 0; const p = t * D; for (let a = 0; a < D; a++) { const e = X[o + a] - bank[p + a]; d += e * e; if (d >= bd[kn - 1]) break; }
        if (d < bd[kn - 1]) { let r = kn - 1; while (r > 0 && bd[r - 1] > d) { bd[r] = bd[r - 1]; bi[r] = bi[r - 1]; r--; } bd[r] = d; bi[r] = t; } }
      idx.set(bi, q * kn); }
    return idx;
  }
  function pca(maps, k) {                     // maps: list of Float64Array(675); returns mean + k unit components (via the Gram matrix)
    const n = maps.length, d = maps[0].length, mu = new Float64Array(d);
    maps.forEach(m => m.forEach((v, i) => { mu[i] += v / n; }));
    const Z = maps.map(m => m.map((v, i) => v - mu[i])), Gm = [];
    for (let a = 0; a < n; a++) { Gm.push([]); for (let b = 0; b < n; b++) { let s = 0; for (let i = 0; i < d; i++) s += Z[a][i] * Z[b][i]; Gm[a].push(s); } }
    const comps = [];
    for (let c = 0; c < k; c++) {
      let v = Array.from({ length: n }, (_, i) => Math.sin(i * 1.7 + c + 1));
      for (let it = 0; it < 300; it++) {
        let w = Gm.map(row => row.reduce((s, x, j) => s + x * v[j], 0));
        comps.forEach(u => { const dd = u.a.reduce((s, x, j) => s + x * w[j], 0); w = w.map((x, j) => x - dd * u.a[j]); });
        const L = Math.hypot(...w) || 1; v = w.map(x => x / L);
      }
      const vec = new Float64Array(d); Z.forEach((z, j) => z.forEach((x, i) => { vec[i] += v[j] * x; }));
      const L = Math.hypot(...vec) || 1; comps.push({ a: v, u: vec.map(x => x / L) });
    }
    return { mu, U: comps.map(c => c.u) };
  }
  function recon(P, m) { const y = Float64Array.from(P.mu); P.U.forEach(u => { let s = 0; for (let i = 0; i < m.length; i++) s += (m[i] - P.mu[i]) * u[i]; for (let i = 0; i < m.length; i++) y[i] += s * u[i]; }); return y; }
  // 3x3 mean filter over the patch grid: real defects span neighbouring patches, edge-alignment noise does not.
  const smooth = m => { const o = new Float64Array(NP); for (let j = 0; j < G; j++) for (let i = 0; i < G; i++) { let s = 0, n = 0;
    for (let b = -1; b <= 1; b++) for (let a = -1; a <= 1; a++) { const J = j + b, I = i + a; if (J >= 0 && I >= 0 && J < G && I < G) { s += m[J * G + I]; n++; } } o[j * G + i] = s / n; } return o; };
  const gap = (A, B) => { const o = new Float64Array(NP); for (let q = 0; q < NP; q++) { let s = 0; for (let k = 0; k < 3; k++) s += (A[q * 3 + k] - B[q * 3 + k]) ** 2; o[q] = s / 3; } return o; };
  const quant = (arr, p) => { const s = Array.from(arr).sort((a, b) => a - b); return s[Math.min(s.length - 1, Math.floor(p * s.length))]; };

  function fit(cfg = {}) {
    const nTrain = cfg.nTrain || 40, nVal = cfg.nVal || 4, k = cfg.k || 8, kn = cfg.kn || 3;
    const train = Array.from({ length: nTrain }, (_, i) => T.make('cap', 300 + i)), val = Array.from({ length: nVal }, (_, i) => T.make('cap', 400 + i));
    const tf = teacher;
    const Tt = train.map(tf), P = pca(Tt, k), At = Tt.map(m => recon(P, m));
    const bank = new Float32Array(nTrain * NP * D), tOut = new Float64Array(nTrain * NP * 3), aOut = new Float64Array(nTrain * NP * 3);
    train.forEach((img, s) => { bank.set(desc(img), s * NP * D); tOut.set(Tt[s], s * NP * 3); aOut.set(At[s], s * NP * 3); });
    K = { P, bank, tOut, aOut, k, kn, nTrain };
    const vr = val.map(raw), cat = key => { const all = []; vr.forEach(r => all.push(...r[key])); return all; };
    K.q = { st: [quant(cat('st'), .9), quant(cat('st'), .995)], ae: [quant(cat('ae'), .9), quant(cat('ae'), .995)] };
    return K;
  }
  function raw(img) {
    const kn = K.kn, idx = nearest(desc(img), K.bank, kn), t = teacher(img), a = recon(K.P, t);
    const s = new Float64Array(NP * 3), s2 = new Float64Array(NP * 3);
    for (let q = 0; q < NP; q++) for (let r = 0; r < kn; r++) { const n = idx[q * kn + r];
      for (let c = 0; c < 3; c++) { s[q * 3 + c] += K.tOut[n * 3 + c] / kn; s2[q * 3 + c] += K.aOut[n * 3 + c] / kn; } }
    return { t, s, a, s2, st: smooth(gap(t, s)), ae: smooth(gap(a, s2)), stRaw: gap(t, s) };
  }
  function score(img) {
    if (!K) fit();
    const r = raw(img), nst = new Float64Array(NP), nae = new Float64Array(NP), comb = new Float64Array(NP);
    for (let i = 0; i < NP; i++) { nst[i] = .1 * (r.st[i] - K.q.st[0]) / (K.q.st[1] - K.q.st[0]); nae[i] = .1 * (r.ae[i] - K.q.ae[0]) / (K.q.ae[1] - K.q.ae[0]); comb[i] = .5 * nst[i] + .5 * nae[i]; }
    return { ...r, nst, nae, comb, image: Math.max(...comb), maxSt: Math.max(...nst), maxAe: Math.max(...nae) };
  }
  // Elements the hard feature loss would train on this image: squared errors at or above the p_hard quantile (of 675).
  function hardMask(r, p) { const D = []; for (let q = 0; q < NP; q++) for (let k = 0; k < 3; k++) D.push((r.t[q * 3 + k] - r.s[q * 3 + k]) ** 2);
    const th = quant(D, p), keep = new Uint8Array(NP); let n = 0; D.forEach((v, i) => { if (v >= th) { keep[Math.floor(i / 3)] = 1; n++; } }); return { keep, n }; }
  // Logical defect for this chapter: a surplus bolt hole halfway between two bolts. Locally it is an ordinary hole; only
  // its position is wrong (EfficientAD's "missing, misplaced, or surplus objects").
  function extraHole(img, which = 0) {
    const N = 128, px = Float32Array.from(img.px), mask = new Uint8Array(N * N);
    const [b0, b1] = [img.bolts[which % 6], img.bolts[(which + 1) % 6]];
    const a0 = Math.atan2(b0[1] - img.cy, b0[0] - img.cx), a1 = Math.atan2(b1[1] - img.cy, b1[0] - img.cx);
    let am = (a0 + a1) / 2; if (Math.abs(a1 - a0) > Math.PI) am += Math.PI;
    const R = Math.hypot(b0[0] - img.cx, b0[1] - img.cy), bx = img.cx + R * Math.cos(am), by = img.cy + R * Math.sin(am), hr = img.hr;
    for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) { const e = Math.hypot(x - bx, y - by);
      if (e < hr) { px[y * N + x] = .1; mask[y * N + x] = 1; } else if (e < hr + 1.4) { px[y * N + x] = .82; mask[y * N + x] = 1; } }
    return { ...img, px, mask, defect: 'extra' };
  }
  return { G, NP, fit, score, hardMask, extraHole, get: () => K };
})();
if (typeof module !== 'undefined') module.exports = EAD_KIT;
</script>
