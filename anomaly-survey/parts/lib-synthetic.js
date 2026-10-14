<script>
/* Synthetic-defect kit for ch-synthetic (and the atlas). DOM-free; node runs it in build/synth-sim.mjs.
   Works on TOY images (128 x 128 grayscale Float32Array). Perlin noise is classic gradient noise; the foreign texture is
   a procedural stand-in for the Describable Textures Dataset that DRAEM samples from. */
const SYNTH_KIT = (() => {
  const N = 128;
  function rng(seed) { let a = seed >>> 0; return () => { a |= 0; a = a + 0x6D2B79F5 | 0;
    let t = Math.imul(a ^ a >>> 15, 1 | a); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }
  const fade = t => t * t * t * (t * (t * 6 - 15) + 10);
  // Perlin noise on an N x N grid with `cells` lattice cells per side; values roughly in [-0.7, 0.7].
  function perlin(seed, cells = 4) {
    const r = rng(seed * 2654435761 + 5), G = [];
    for (let i = 0; i <= cells; i++) { G.push([]); for (let j = 0; j <= cells; j++) { const a = r() * 2 * Math.PI; G[i].push([Math.cos(a), Math.sin(a)]); } }
    const out = new Float32Array(N * N);
    for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
      const gx = x / N * cells, gy = y / N * cells, i = Math.floor(gx), j = Math.floor(gy), fx = gx - i, fy = gy - j;
      const dot = (di, dj) => { const g = G[i + di][j + dj]; return g[0] * (fx - di) + g[1] * (fy - dj); };
      const u = fade(fx), v = fade(fy);
      const a = dot(0, 0) + u * (dot(1, 0) - dot(0, 0)), b = dot(0, 1) + u * (dot(1, 1) - dot(0, 1));
      out[y * N + x] = a + v * (b - a);
    }
    return out;
  }
  // Mask = Perlin noise above a threshold. DRAEM samples the threshold at random; the slider makes it a control.
  // `thr` is a quantile-free raw threshold on noise rescaled to [0, 1].
  function perlinMask(seed, thr, cells = 4) {
    const p = perlin(seed, cells), m = new Uint8Array(N * N);
    let lo = Infinity, hi = -Infinity; for (const v of p) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
    for (let k = 0; k < N * N; k++) m[k] = (p[k] - lo) / (hi - lo) > thr ? 1 : 0;
    return m;
  }
  // Foreign texture: marbled stripes, unlike either toy product.
  function texture(seed) {
    const p = perlin(seed + 101, 3), q = perlin(seed + 202, 8), out = new Float32Array(N * N);
    for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
      const k = y * N + x; out[k] = Math.min(1, Math.max(0, .5 + .42 * Math.sin(.32 * (x + .6 * y) + 7 * p[k]) + .25 * q[k]));
    }
    return out;
  }
  // DRAEM Eq. 4: I_a = (1 - M) I + (1 - beta) M I + beta M A.
  function draem(img, mask, tex, beta) {
    const out = new Float32Array(N * N);
    for (let k = 0; k < N * N; k++) out[k] = mask[k] ? (1 - beta) * img[k] + beta * tex[k] : img[k];
    return out;
  }
  // CutPaste: copy a rectangle of the same image to another place, with a brightness jitter. Scar: a thin rotated strip.
  // src/dst are top-left corners; w, h in pixels; angle in radians (0 for the large patch).
  function cutpaste(img, src, dst, w, h, angle = 0, jitter = 0) {
    const out = Float32Array.from(img), mask = new Uint8Array(N * N);
    const cx = dst[0] + w / 2, cy = dst[1] + h / 2, sx = src[0] + w / 2, sy = src[1] + h / 2, c = Math.cos(angle), s = Math.sin(angle);
    const R = Math.ceil(Math.hypot(w, h) / 2) + 1;
    for (let y = Math.floor(cy - R); y <= cy + R; y++) for (let x = Math.floor(cx - R); x <= cx + R; x++) {
      if (x < 0 || y < 0 || x >= N || y >= N) continue;
      const dx = x - cx, dy = y - cy, u = c * dx + s * dy, v = -s * dx + c * dy;          // patch-local coordinates
      if (Math.abs(u) > w / 2 || Math.abs(v) > h / 2) continue;
      const X = Math.round(sx + c * u - s * v), Y = Math.round(sy + s * u + c * v);       // same offset in the source, same rotation
      if (X < 0 || Y < 0 || X >= N || Y >= N) continue;
      out[y * N + x] = Math.min(1, Math.max(0, img[Y * N + X] + jitter)); mask[y * N + x] = 1;
    }
    return { img: out, mask };
  }
  const frac = m => { let s = 0; for (const v of m) s += v; return s / m.length; };
  return { N, perlin, perlinMask, texture, draem, cutpaste, frac };
})();
if (typeof module !== 'undefined') module.exports = SYNTH_KIT;
</script>
