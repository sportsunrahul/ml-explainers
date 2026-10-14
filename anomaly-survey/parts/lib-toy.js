<script>
/* Toy world shared by every toy figure: two products (a woven mesh = texture, a machined cap = object), defects with
   ground-truth masks, and a patch feature extractor (structure tensor + brightness) so feature clouds are computed
   from the toy images. Core functions are DOM-free so `node` can run them (see build/toy-sim.mjs). */
const TOY = (() => {
  const N = 128, P = 16, S = 8, G = (N - P) / S + 1;
  function rng(seed) { let a = seed >>> 0; return () => { a |= 0; a = a + 0x6D2B79F5 | 0;
    let t = Math.imul(a ^ a >>> 15, 1 | a); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }
  const clamp01 = v => v < 0 ? 0 : v > 1 ? 1 : v;

  function mesh(seed) {
    const r = rng(seed * 7919 + 1), px = new Float32Array(N * N), T = 16, h = T / 2;
    const ox = Math.floor(r() * T), oy = Math.floor(r() * T);
    const thread = []; for (let i = 0; i < 64; i++) thread.push(.9 + .2 * r());
    for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
      const X = x + ox, Y = y + oy, ci = Math.floor(X / h), cj = Math.floor(Y / h);
      const fx = (X % h) / h, fy = (Y % h) / h, horiz = (ci + cj) % 2 === 0;
      const prof = Math.pow(Math.sin(Math.PI * (horiz ? fy : fx)), .8);
      const k = horiz ? thread[cj % 64] : thread[(ci + 17) % 64];
      px[y * N + x] = clamp01(.16 + .5 * prof * k + .04 * (r() - .5));
    }
    return { kind: 'mesh', px };
  }

  function cap(seed) {
    const r = rng(seed * 104729 + 3), px = new Float32Array(N * N);
    const cx = N / 2 + (r() - .5) * 3, cy = N / 2 + (r() - .5) * 3, rot = (r() - .5) * .16;
    const R = .42 * N, ring = .27 * N, boltR = .345 * N, hr = .045 * N, boss = .1 * N;
    const bolts = []; for (let k = 0; k < 6; k++) { const a = rot + k * Math.PI / 3; bolts.push([cx + boltR * Math.cos(a), cy + boltR * Math.sin(a)]); }
    for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
      const dx = x - cx, dy = y - cy, d = Math.hypot(dx, dy);
      let v;
      if (d > R + 1) v = .07;
      else {
        v = .52 + .07 * Math.cos(Math.atan2(dy, dx) - .8) + .025 * Math.sin(d * .9);
        if (d > R - 1.5) v = .78;                                   // bright chamfer at the rim
        if (Math.abs(d - ring) < 1.1) v -= .22;                     // stamped groove
        if (d < boss) v = .68 + .04 * Math.cos(d * .7);              // centre boss
        for (const [bx, by] of bolts) { const e = Math.hypot(x - bx, y - by);
          if (e < hr) v = .1; else if (e < hr + 1.4) v = .82; }
      }
      px[y * N + x] = clamp01(v + .035 * (r() - .5));
    }
    return { kind: 'cap', px, bolts, cx, cy, hr };
  }

  const make = (kind, seed) => kind === 'mesh' ? mesh(seed) : cap(seed);

  // Defects return a new image and a 0/1 ground-truth mask.
  function defect(img, type, seed) {
    const r = rng(seed * 31337 + 11), px = Float32Array.from(img.px), mask = new Uint8Array(N * N);
    const mark = (x, y) => { if (x >= 0 && y >= 0 && x < N && y < N) mask[y * N + x] = 1; };
    if (type === 'scratch') {
      const a = .45 + r() * .5, L = .26 * N, x0 = N * (.32 + .2 * r()), y0 = N * (.3 + .2 * r());
      for (let s = 0; s < L; s += .35) {
        const wob = 1.2 * Math.sin(s * .21), x = x0 + s * Math.cos(a) - wob * Math.sin(a), y = y0 + s * Math.sin(a) + wob * Math.cos(a);
        for (let oy = -2; oy <= 2; oy++) for (let ox = -2; ox <= 2; ox++) {
          const X = Math.round(x + ox), Y = Math.round(y + oy), d = Math.hypot(X - x, Y - y);
          if (X < 0 || Y < 0 || X >= N || Y >= N) continue;
          if (d < 1.3) px[Y * N + X] = clamp01(Math.max(px[Y * N + X], .93));
          if (d < 2.2) mark(X, Y);
        }
      }
    } else if (type === 'hole' || type === 'stain') {
      const x0 = N * (.3 + .4 * r()), y0 = N * (.3 + .4 * r()), rad = type === 'hole' ? 4.5 : 9;
      for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
        const d = Math.hypot(x - x0, y - y0);
        if (type === 'hole') { if (d < rad) { px[y * N + x] = .03; mark(x, y); } else if (d < rad + 1.3) { px[y * N + x] = .85; mark(x, y); } }
        else { const w = Math.exp(-d * d / (2 * rad * rad)); if (w > .08) { px[y * N + x] = clamp01(px[y * N + x] - .3 * w + .02); if (w > .3) mark(x, y); } }
      }
    } else if (type === 'missing' && img.kind === 'cap') {
      // Logical defect: one bolt hole is filled with plain surface. Locally it looks like normal metal.
      const [bx, by] = img.bolts[Math.floor(r() * 6)], hr = img.hr + 1.6;
      for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
        if (Math.hypot(x - bx, y - by) < hr) { const dx = x - img.cx, dy = y - img.cy, d = Math.hypot(dx, dy);
          px[y * N + x] = clamp01(.52 + .07 * Math.cos(Math.atan2(dy, dx) - .8) + .025 * Math.sin(d * .9) + .035 * (r() - .5)); mark(x, y); }
      }
    }
    return { kind: img.kind, px, mask, bolts: img.bolts, cx: img.cx, cy: img.cy, hr: img.hr, defect: type };
  }

  // Patch features: [horizontal-vs-vertical edges, diagonal edges, brightness], each z-scored with fixed constants.
  const Z = { mu: [0, 0, .42], sd: [.35, .35, .18] };
  function features(img) {
    const p = img.px, gx = new Float32Array(N * N), gy = new Float32Array(N * N);
    for (let y = 1; y < N - 1; y++) for (let x = 1; x < N - 1; x++) {
      gx[y * N + x] = (p[y * N + x + 1] - p[y * N + x - 1]) / 2; gy[y * N + x] = (p[(y + 1) * N + x] - p[(y - 1) * N + x]) / 2; }
    const f = new Float32Array(G * G * 3);
    for (let j = 0; j < G; j++) for (let i = 0; i < G; i++) {
      let xx = 0, yy = 0, xy = 0, m = 0;
      for (let y = j * S; y < j * S + P; y++) for (let x = i * S; x < i * S + P; x++) {
        const a = gx[y * N + x], b = gy[y * N + x]; xx += a * a; yy += b * b; xy += a * b; m += p[y * N + x]; }
      const e = xx + yy + 1e-3, k = (j * G + i) * 3;
      f[k] = ((xx - yy) / e - Z.mu[0]) / Z.sd[0]; f[k + 1] = (2 * xy / e - Z.mu[1]) / Z.sd[1]; f[k + 2] = (m / (P * P) - Z.mu[2]) / Z.sd[2];
    }
    return f;
  }

  // Memory bank of normal patch features from n images of one product.
  function bank(kind, n, seed = 1) { const out = []; for (let s = 0; s < n; s++) out.push(features(make(kind, seed + s)));
    const B = new Float32Array(out.length * G * G * 3); out.forEach((f, i) => B.set(f, i * G * G * 3)); return B; }

  // Per-patch nearest-neighbour distance to the bank (the SPADE / PatchCore score, on toy features).
  function nnScores(f, B) {
    const out = new Float32Array(G * G), nb = B.length / 3;
    for (let q = 0; q < G * G; q++) { let best = Infinity; const a = f[q * 3], b = f[q * 3 + 1], c = f[q * 3 + 2];
      for (let t = 0; t < nb; t++) { const d = (a - B[t * 3]) ** 2 + (b - B[t * 3 + 1]) ** 2 + (c - B[t * 3 + 2]) ** 2; if (d < best) best = d; }
      out[q] = Math.sqrt(best); }
    return out;
  }

  // Patch grid -> pixel map: each pixel averages the patches that cover it (patches overlap by S).
  function toPixels(s) {
    const acc = new Float32Array(N * N), cnt = new Float32Array(N * N);
    for (let j = 0; j < G; j++) for (let i = 0; i < G; i++) { const v = s[j * G + i];
      for (let y = j * S; y < j * S + P; y++) for (let x = i * S; x < i * S + P; x++) { acc[y * N + x] += v; cnt[y * N + x]++; } }
    for (let k = 0; k < N * N; k++) acc[k] /= cnt[k] || 1; return acc;
  }

  // Drawing (browser only). Grayscale camera view, optional red heat overlay (transparent -> red), viewfinder corners.
  const cache = new WeakMap();
  function bitmap(img) {
    if (cache.has(img)) return cache.get(img);
    const c = document.createElement('canvas'); c.width = c.height = N; const g = c.getContext('2d'), d = g.createImageData(N, N);
    for (let k = 0; k < N * N; k++) { const v = Math.round(img.px[k] * 255); d.data[k * 4] = v; d.data[k * 4 + 1] = v; d.data[k * 4 + 2] = v; d.data[k * 4 + 3] = 255; }
    g.putImageData(d, 0, 0); cache.set(img, c); return c;
  }
  function heatBitmap(map, lo, hi, red) {
    const c = document.createElement('canvas'); c.width = c.height = N; const g = c.getContext('2d'), d = g.createImageData(N, N);
    const [R, Gr, B] = [1, 3, 5].map(i => parseInt(red.slice(i, i + 2), 16));
    for (let k = 0; k < N * N; k++) { const u = clamp01((map[k] - lo) / (hi - lo || 1));
      d.data[k * 4] = R; d.data[k * 4 + 1] = Gr; d.data[k * 4 + 2] = B; d.data[k * 4 + 3] = Math.round(220 * u * u); }
    g.putImageData(d, 0, 0); return c;
  }
  function draw(ctx, img, x, y, size, o = {}) {
    ctx.save(); ctx.imageSmoothingEnabled = false; ctx.drawImage(bitmap(img), x, y, size, size);
    if (o.heat) { ctx.imageSmoothingEnabled = true; ctx.globalAlpha = o.heatAlpha ?? 1; ctx.drawImage(o.heat, x, y, size, size); ctx.globalAlpha = 1; }
    if (o.vf !== false) viewfinder(ctx, x, y, size, size, o.vfColor || 'rgba(160,166,172,.9)');
    ctx.restore();
  }
  function viewfinder(ctx, x, y, w, h, color) {
    const L = Math.max(6, Math.min(w, h) * .09), g = 3;
    ctx.save(); ctx.strokeStyle = color; ctx.lineWidth = 1.5; ctx.beginPath();
    [[x - g, y - g, 1, 1], [x + w + g, y - g, -1, 1], [x - g, y + h + g, 1, -1], [x + w + g, y + h + g, -1, -1]].forEach(([a, b, sx, sy]) => {
      ctx.moveTo(a + sx * L, b); ctx.lineTo(a, b); ctx.lineTo(a, b + sy * L); });
    ctx.stroke(); ctx.restore();
  }

  return { N, P, S, G, rng, mesh, cap, make, defect, features, bank, nnScores, toPixels, bitmap, heatBitmap, draw, viewfinder };
})();
if (typeof module !== 'undefined') module.exports = TOY;
</script>
