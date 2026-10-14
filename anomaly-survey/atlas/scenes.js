/* Anomaly atlas scenes. The toy kits below are copied verbatim from parts/lib-*.js (the long page's figure kits), so the
   stage computes the same toy numbers as the long page. Scene code follows the kits. */

/* ---- kit: parts/lib-toy.js */
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

/* ---- kit: parts/lib-benchmark.js */
/* ch-benchmark toy kit: a small toy test set on the mesh (8 good, 8 defective), positioned defects with region labels,
   and the three grading curves: image ROC, pixel ROC, and PRO against pixel FPR up to 0.3. DOM-free (node-runnable).
   Numbers checked in build/toy-bench.mjs. */
const BENCH_KIT = (() => {
  const { N, G, S, P } = TOY;
  const clamp01 = v => v < 0 ? 0 : v > 1 ? 1 : v;
  // Paint defects at fixed spots; each defect is one ground-truth region (label 1, 2, ...). Formulas follow TOY.defect.
  function paint(img, list) {
    const px = Float32Array.from(img.px), lab = new Uint8Array(N * N);
    list.forEach((d, r) => {
      const id = r + 1;
      if (d.type === 'stain') {
        for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) { const e = Math.hypot(x - d.x, y - d.y), w = Math.exp(-e * e / (2 * d.r * d.r));
          if (w > .08) { px[y * N + x] = clamp01(px[y * N + x] - (d.k ?? .3) * w + .02); if (w > .3) lab[y * N + x] = id; } }
      } else if (d.type === 'hole') {
        for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) { const e = Math.hypot(x - d.x, y - d.y);
          if (e < d.r) { px[y * N + x] = .03; lab[y * N + x] = id; } else if (e < d.r + 1.3) { px[y * N + x] = .85; lab[y * N + x] = id; } }
      } else if (d.type === 'scratch') {
        for (let s = 0; s < d.len; s += .35) { const wob = 1.2 * Math.sin(s * .21), x = d.x + s * Math.cos(d.a) - wob * Math.sin(d.a), y = d.y + s * Math.sin(d.a) + wob * Math.cos(d.a);
          for (let oy = -2; oy <= 2; oy++) for (let ox = -2; ox <= 2; ox++) { const X = Math.round(x + ox), Y = Math.round(y + oy), e = Math.hypot(X - x, Y - y);
            if (X < 0 || Y < 0 || X >= N || Y >= N) continue; if (e < 1.3) px[Y * N + X] = Math.max(px[Y * N + X], .93); if (e < 2.2) lab[Y * N + X] = id; } }
      }
    });
    return { kind: img.kind, px, lab, regions: list.length };
  }
  // The toy test set. Index 8 is the featured part: one large stain (region 1) and one tiny hole (region 2).
  const SPEC = [
    [{ type: 'stain', x: 44, y: 52, r: 13 }, { type: 'hole', x: 98, y: 92, r: 2.6 }],
    [{ type: 'scratch', x: 30, y: 40, a: .7, len: 34 }],
    [{ type: 'stain', x: 80, y: 70, r: 9 }],
    [{ type: 'hole', x: 60, y: 50, r: 4.5 }],
    [{ type: 'scratch', x: 50, y: 80, a: -.5, len: 30 }],
    [{ type: 'stain', x: 64, y: 64, r: 7, k: .1 }],
    [{ type: 'hole', x: 40, y: 90, r: 3.5 }, { type: 'stain', x: 90, y: 40, r: 8 }],
    [{ type: 'stain', x: 70, y: 40, r: 10, k: .07 }],
  ];
  function build() {
    const B = TOY.bank('mesh', 10, 1), items = [];
    for (let s = 0; s < 8; s++) { const im = TOY.make('mesh', 300 + s); items.push({ img: { ...im, lab: new Uint8Array(N * N), regions: 0 }, good: true }); }
    SPEC.forEach((list, s) => items.push({ img: paint(TOY.make('mesh', 400 + s), list), good: false }));
    items.forEach(it => { it.patch = TOY.nnScores(TOY.features(it.img), B); });
    return { items, featured: 8 };
  }
  // Pixel map for one item; `miss` (a region label) pulls that region's patches down to the image's median patch score.
  function pixels(it, miss) {
    let p = it.patch;
    if (miss) { p = Float32Array.from(p); const med = [...it.patch].sort((a, b) => a - b)[it.patch.length >> 1];
      for (let j = 0; j < G; j++) for (let i = 0; i < G; i++) { let hit = 0;
        for (let y = j * S; y < j * S + P && !hit; y++) for (let x = i * S; x < i * S + P; x++) if (it.img.lab[y * N + x] === miss) { hit = 1; break; }
        if (hit) p[j * G + i] = med; } }
    return { patch: p, map: TOY.toPixels(p) };
  }
  // All three curves for a given set of pixel maps (one per item). Thresholds run over `bins` levels of [0, top], spaced
  // on a square-root scale so the crowded low scores of normal pixels get fine steps.
  function grade(items, maps, bins = 400) {
    const top = Math.max(...maps.map(m => Math.max(...m.patch))) * 1.001;
    const imgS = maps.map(m => Math.max(...m.patch));
    const pos = new Float64Array(bins), neg = new Float64Array(bins), reg = [];
    items.forEach((it, k) => { const m = maps[k].map, R = it.img.regions;
      const rh = Array.from({ length: R }, () => new Float64Array(bins)), rn = new Array(R).fill(0);
      for (let q = 0; q < N * N; q++) { const b = Math.min(bins - 1, Math.floor(Math.sqrt(m[q] / top) * bins)), l = it.img.lab[q];
        if (l) { pos[b]++; rh[l - 1][b]++; rn[l - 1]++; } else neg[b]++; }
      for (let r = 0; r < R; r++) reg.push({ h: rh[r], n: rn[r], item: k, label: r + 1 }); });
    const P = pos.reduce((a, b) => a + b), Nn = neg.reduce((a, b) => a + b);
    // Sweep from the highest bin down: at threshold index b, everything in bins >= b is flagged.
    const pixRoc = [[0, 0]], pro = [[0, 0]], cov = reg.map(() => 0);
    let tp = 0, fp = 0;
    for (let b = bins - 1; b >= 0; b--) { tp += pos[b]; fp += neg[b]; reg.forEach((r, i) => { cov[i] += r.h[b]; });
      const fpr = fp / Nn; pixRoc.push([fpr, tp / P]); pro.push([fpr, cov.reduce((a, c, i) => a + c / reg[i].n, 0) / reg.length]); }
    const area = (pts, lim = 1) => { let A = 0; for (let i = 1; i < pts.length; i++) { const [x0, y0] = pts[i - 1], [x1, y1] = pts[i];
      if (x0 >= lim) break; const xe = Math.min(x1, lim), ye = x1 > lim ? y0 + (y1 - y0) * (lim - x0) / (x1 - x0) : y1; A += (xe - x0) * (y0 + ye) / 2; } return A / lim; };
    const goods = imgS.filter((_, k) => items[k].good), bads = imgS.filter((_, k) => !items[k].good);
    let wins = 0; bads.forEach(b => goods.forEach(g => { wins += b > g ? 1 : b === g ? .5 : 0; }));
    const sorted = [...new Set(imgS)].sort((a, b) => b - a), imgRoc = [[0, 0]];
    sorted.forEach(t => imgRoc.push([goods.filter(g => g >= t).length / goods.length, bads.filter(b => b >= t).length / bads.length]));
    imgRoc.push([1, 1]);
    return { top, bins, imgS, imgRoc, imgAuroc: wins / (goods.length * bads.length), pixRoc, pixAuroc: area(pixRoc), pro, aupro: area(pro, .3), regions: reg.length, P, Nn };
  }
  // Point on the curves at threshold t (in score units).
  function at(items, maps, t) {
    let tp = 0, fp = 0, P = 0, Nn = 0; const regs = [];
    items.forEach((it, k) => { const m = maps[k].map, c = new Array(it.img.regions).fill(0), n = new Array(it.img.regions).fill(0);
      for (let q = 0; q < N * N; q++) { const l = it.img.lab[q], f = m[q] >= t; if (l) { P++; n[l - 1]++; if (f) { tp++; c[l - 1]++; } } else { Nn++; if (f) fp++; } }
      for (let r = 0; r < c.length; r++) regs.push(c[r] / n[r]); });
    const imgS = maps.map(m => Math.max(...m.patch)), goods = imgS.filter((_, k) => items[k].good), bads = imgS.filter((_, k) => !items[k].good);
    return { tpr: tp / P, fpr: fp / Nn, pro: regs.reduce((a, b) => a + b, 0) / regs.length,
      imgTpr: bads.filter(b => b >= t).length / bads.length, imgFpr: goods.filter(g => g >= t).length / goods.length,
      rejBad: bads.filter(b => b >= t).length, rejGood: goods.filter(g => g >= t).length };
  }
  return { paint, build, pixels, grade, at, SPEC };
})();

/* ---- kit: parts/lib-fastflow.js */
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

/* ---- kit: parts/lib-synthetic.js */
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

/* ---- kit: parts/lib-efficientad.js */
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

/* ---- kit: parts/lib-uniad.js */
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

/* ---- scenes */

// Shared helpers for the scenes below.
const COL = () => ({ blue: PE.css('--blue'), red: PE.css('--red'), yellow: PE.css('--yellow'), green: PE.css('--green'), brown: PE.css('--brown'), muted: PE.css('--muted'), rule: PE.css('--rule'), fg: PE.css('--fg'), bg: PE.css('--bg'), panel: PE.css('--panel2') });
const lazy = make => { let v = null; return () => v || (v = make()); };
const onKey = btns => { const b = btns.find(x => x.getAttribute('aria-pressed') === 'true'); return b || null; };
// The desktop readout box sits over the stage's bottom-left corner; 2D layouts keep this much free there.
const HUD_H = 150;
function pill(c, x, y, word, color) {
  c.save(); c.font = '600 13px system-ui, sans-serif'; const tw = c.measureText(word).width + 18;
  c.strokeStyle = color; c.lineWidth = 1.4; c.beginPath(); c.roundRect(x, y, tw, 22, 11); c.stroke();
  c.fillStyle = color; c.fillText(word, x + 9, y + 15); c.restore(); return tw;
}

/* 1. The problem: nearest-neighbour scoring on toy cap features (port of parts/viz-normal-cloud.js). Checked there
   (build/toy-ch1.mjs): at 0.30, one cap in memory flags every good test cap (0.33-0.40), five or twenty pass them
   (<= 0.26), the scratched cap scores 0.81-0.94. */
const CLOUD = lazy(() => {
  const f2 = f => { const o = []; for (let k = 0; k < f.length / 3; k++) o.push([f[k * 3 + 1], f[k * 3 + 2]]); return o; };
  const pool = Array.from({ length: 20 }, (_, i) => f2(TOY.features(TOY.make('cap', 1 + i))));
  const tests = { good: TOY.make('cap', 104), scratch: TOY.defect(TOY.make('cap', 200), 'scratch', 200) };
  const tf = { good: f2(TOY.features(tests.good)), scratch: f2(TOY.features(tests.scratch)) };
  const all = pool.flat().concat(tf.scratch);
  const lo = [Math.min(...all.map(p => p[0])) - .2, Math.min(...all.map(p => p[1])) - .2], hi = [Math.max(...all.map(p => p[0])) + .2, Math.max(...all.map(p => p[1])) + .2];
  const nearest = (bank, p) => { let b = Infinity, bi = 0; bank.forEach((m, i) => { const d = Math.hypot(m[0] - p[0], m[1] - p[1]); if (d < b) { b = d; bi = i; } }); return [b, bi]; };
  const cache = {};
  function scored(n, test) {
    const key = n + test; if (cache[key]) return cache[key];
    const bank = pool.slice(0, n).flat(), s = tf[test].map(p => nearest(bank, p));
    let w = 0; s.forEach((v, q) => { if (v[0] > s[w][0]) w = q; });
    return cache[key] = { bank, s, worst: w, map: TOY.toPixels(Float32Array.from(s.map(v => v[0]))) };
  }
  return { tests, tf, lo, hi, nearest, scored };
});

SCENES.problem = ctx => {
  const D = CLOUD(), { tests, tf, lo, hi, nearest, scored } = D, el = mkCanvas(ctx.stage);
  // Behind the title screen the stage is decoration: the hero text covers the left (desktop) or most of it (phone).
  const deco = document.body.dataset.chapter === 'title';
  el.tabIndex = 0; el.setAttribute('aria-label', 'Left: the toy cap under inspection with a red heat map. Right: the feature space; drag or use the arrow keys to move a probe patch.');
  const slider = ctx.q('#cloud-tau'), sOut = ctx.q('#cloud-tau-value'), bb = ctx.qa('[data-bank]'), tb = ctx.qa('[data-test]');
  const S = { n: +((onKey(bb) || bb[2]).dataset.bank), test: (onKey(tb) || tb[0]).dataset.test, tau: +slider.value, probe: null, alpha: 1 };
  S.shown = S.n;
  let L = null, heat = null, heatKey = '', cover = null, coverKey = '', cancel = () => {};
  const cv = PE.canvas(el, S3ASPECT(ctx.stage), (c, W, H) => draw(c, W, H));
  function layout(W, H) {
    const narrow = W < 620;
    if (narrow) { const s = Math.min(W * .36, H * .34), img = { x: 16, y: 28, s }, fy = img.y + s + 34; return { narrow, img, fp: { x: 34, y: fy + 6, w: W - 48, h: H - fy - 34 } }; }
    const s = Math.min(W * .32, (H - HUD_H) * .62), img = { x: 26, y: 50, s };
    const fx = Math.max(img.x + s + 60, W * .42);
    return { narrow, img, fp: { x: fx, y: 50, w: W - fx - 24, h: H - 50 - HUD_H } };
  }
  const X = p => L.fp.x + (p[0] - lo[0]) / (hi[0] - lo[0]) * L.fp.w, Y = p => L.fp.y + L.fp.h - (p[1] - lo[1]) / (hi[1] - lo[1]) * L.fp.h;
  const unX = x => lo[0] + (x - L.fp.x) / L.fp.w * (hi[0] - lo[0]), unY = y => lo[1] + (L.fp.y + L.fp.h - y) / L.fp.h * (hi[1] - lo[1]);
  function coverage(W, H, bank, tau, color) {
    const key = [W, H, bank.length, tau, color].join(); if (key === coverKey) return cover;
    const c = document.createElement('canvas'), r = devicePixelRatio || 1; c.width = W * r; c.height = H * r;
    const g = c.getContext('2d'); g.setTransform(r, 0, 0, r, 0, 0); g.fillStyle = color;
    const rx = tau / (hi[0] - lo[0]) * L.fp.w, ry = tau / (hi[1] - lo[1]) * L.fp.h;
    g.beginPath(); bank.forEach(m => { g.moveTo(X(m) + rx, Y(m)); g.ellipse(X(m), Y(m), rx, ry, 0, 0, 7); }); g.fill();
    coverKey = key; return cover = c;
  }
  function draw(c, W, H) {
    if (!W) return; L = layout(W, H); if (L.fp.w <= 0 || L.fp.h <= 0) return;
    const C = COL(), R = scored(S.n, S.test), cur = scored(S.shown, S.test);
    if (deco) { L = { fp: W < 620 ? { x: 16, y: 16, w: W - 32, h: H - 32 } : { x: W * .45, y: 60, w: W * .5, h: H - 120 } }; c.clearRect(0, 0, W, H);
      c.globalAlpha = .1; c.drawImage(coverage(W, H, cur.bank, S.tau, C.blue), 0, 0, W, H); c.globalAlpha = 1;
      cur.bank.forEach(m => PE.dot(c, X(m), Y(m), 1.6, C.blue, W < 620 ? .3 : .55)); return; }
    c.clearRect(0, 0, W, H);
    const hk = S.n + S.test + S.tau + C.red;
    if (hk !== heatKey) { heat = TOY.heatBitmap(R.map, S.tau, S.tau + .45, C.red); heatKey = hk; }
    TOY.draw(c, tests[S.test], L.img.x, L.img.y, L.img.s, { heat });
    const wsz = L.img.s / TOY.N, w = R.worst, wx = L.img.x + (w % TOY.G) * TOY.S * wsz, wy = L.img.y + Math.floor(w / TOY.G) * TOY.S * wsz;
    c.save(); c.strokeStyle = C.yellow; c.lineWidth = 2; c.strokeRect(wx, wy, TOY.P * wsz, TOY.P * wsz); c.restore();
    const score = R.s[w][0], reject = score > S.tau;
    const tx = L.narrow ? L.img.x + L.img.s + 16 : L.img.x, ty = L.narrow ? L.img.y + 10 : L.img.y + L.img.s + 16;
    pill(c, tx, ty, reject ? 'reject' : 'pass', reject ? C.red : C.blue);
    PE.label(c, `image score ${score.toFixed(2)}`, L.narrow ? tx : tx + 78, L.narrow ? ty + 42 : ty + 15, C.muted, 12);
    PE.label(c, L.narrow ? 'camera view' : 'camera view of the cap under inspection', L.img.x, L.img.y - 12, C.muted, 12);
    c.save(); c.strokeStyle = C.rule; c.lineWidth = 1; c.strokeRect(L.fp.x, L.fp.y, L.fp.w, L.fp.h); c.restore();
    PE.label(c, 'diagonal edges', L.fp.x + L.fp.w / 2, L.fp.y + L.fp.h + 18, C.muted, 11, 'center');
    c.save(); c.translate(L.fp.x - 10, L.fp.y + L.fp.h / 2); c.rotate(-Math.PI / 2); PE.label(c, 'brightness', 0, 0, C.muted, 11, 'center'); c.restore();
    c.save(); c.beginPath(); c.rect(L.fp.x, L.fp.y, L.fp.w, L.fp.h); c.clip();
    c.globalAlpha = .16; c.drawImage(coverage(W, H, cur.bank, S.tau, C.blue), 0, 0, W, H); c.globalAlpha = 1;
    cur.bank.forEach(m => PE.dot(c, X(m), Y(m), 1.6, C.blue, .6 * S.alpha));
    if (S.shown !== S.n) R.bank.forEach(m => PE.dot(c, X(m), Y(m), 1.6, C.blue, .6 * (1 - S.alpha)));
    tf[S.test].forEach((p, q) => { PE.dot(c, X(p), Y(p), 2.2, C.yellow, .9); if (R.s[q][0] > S.tau) PE.ring(c, X(p), Y(p), 4.5, C.red, 1.2); });
    const wp = tf[S.test][w], nb = R.bank[R.s[w][1]];
    PE.polyline(c, [[X(nb), Y(nb)], [X(wp), Y(wp)]], C.red, 2);
    PE.dot(c, X(wp), Y(wp), 5, C.yellow); PE.dot(c, X(nb), Y(nb), 3, C.blue);
    if (S.probe) { const [d, i] = nearest(R.bank, S.probe), m = R.bank[i];
      PE.polyline(c, [[X(m), Y(m)], [X(S.probe), Y(S.probe)]], d > S.tau ? C.red : C.blue, 1.5, [4, 3]); PE.ring(c, X(S.probe), Y(S.probe), 7, C.yellow, 2); }
    c.restore();
    PE.label(c, 'feature space: one dot per patch', L.fp.x, L.fp.y - 8, C.muted, 12);
    if (!L.narrow) { const lx = L.img.x, ly = L.img.y + L.img.s + 70; PE.dot(c, lx + 5, ly - 4, 3, C.blue); PE.label(c, 'patches of good caps in memory', lx + 14, ly, C.blue, 12);
      PE.dot(c, lx + 5, ly + 16, 3, C.yellow); PE.label(c, 'patches of the cap under inspection', lx + 14, ly + 20, C.yellow, 12);
      PE.polyline(c, [[lx, ly + 36], [lx + 10, ly + 36]], C.red, 2); PE.label(c, 'worst patch: distance to the nearest normal', lx + 14, ly + 40, C.red, 12); }
  }
  function say() {
    const R = scored(S.n, S.test), sc = R.s[R.worst][0], rej = sc > S.tau, nfar = R.s.filter(v => v[0] > S.tau).length;
    let t = `${S.n} good ${S.n === 1 ? 'cap' : 'caps'} in memory (${(S.n * TOY.G * TOY.G).toLocaleString('en-US')} patch features). The ${S.test === 'good' ? 'good' : 'scratched'} cap's worst patch is ${sc.toFixed(2)} from normal, ${rej ? 'over' : 'under'} the ${S.tau.toFixed(2)} threshold: <b>${rej ? 'reject' : 'pass'}</b>`;
    t += rej ? ` (${nfar} of ${TOY.G * TOY.G} patches flagged${S.test === 'good' ? '; a false alarm' : ''}).` : '.';
    if (S.probe) { const d = nearest(R.bank, S.probe)[0]; t += ` Probe patch: ${d.toFixed(2)} from the nearest normal feature, ${d > S.tau ? 'flagged' : 'looks normal'}.`; }
    ctx.hud(badge('toy', t));
  }
  function setBank(n) {
    if (n === S.n) return; cancel(); const from = S.n; S.n = n; S.shown = from; S.alpha = 1; press(bb, b => +b.dataset.bank === n);
    cancel = PE.animate(500, u => { S.alpha = 1 - u; cv.redraw(); }, () => { S.shown = n; S.alpha = 1; cv.redraw(); }); say();
  }
  function setTest(t) { S.test = t; press(tb, b => b.dataset.test === t); cv.redraw(); say(); }
  function setTau(v) { S.tau = +v; slider.value = v; sOut.textContent = (+v).toFixed(2); cv.redraw(); say(); }
  let drag = false;
  const at = e => { const r = el.getBoundingClientRect(), x = e.clientX - r.left, y = e.clientY - r.top;
    return (L && x >= L.fp.x && x <= L.fp.x + L.fp.w && y >= L.fp.y && y <= L.fp.y + L.fp.h) ? [unX(x), unY(y)] : null; };
  el.addEventListener('pointerdown', e => { const p = at(e); if (!p) return; drag = true; el.setPointerCapture(e.pointerId); S.probe = p; cv.redraw(); say(); });
  el.addEventListener('pointermove', e => { if (!drag) return; const p = at(e); if (p) { S.probe = p; cv.redraw(); } });
  el.addEventListener('pointerup', () => { if (drag) { drag = false; say(); } });
  el.addEventListener('keydown', e => { const k = { ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowUp: [0, 1], ArrowDown: [0, -1] }[e.key]; if (!k) return;
    e.preventDefault(); const p = S.probe || [(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2];
    S.probe = [PE.clamp(p[0] + k[0] * .015 * (hi[0] - lo[0]), lo[0], hi[0]), PE.clamp(p[1] + k[1] * .015 * (hi[1] - lo[1]), lo[1], hi[1])]; cv.redraw(); say(); });
  const offs = [...bb.map(b => listen(b, 'click', () => setBank(+b.dataset.bank))), ...tb.map(b => listen(b, 'click', () => setTest(b.dataset.test))),
    listen(slider, 'input', () => setTau(slider.value)),
    listen(ctx.q('[data-act="reset"]'), 'click', () => { S.probe = null; setBank(20); setTest('good'); setTau(.3); })];
  say();
  return { flat: true, resize: cv.resize, destroy() { cancel(); offs.forEach(f => f()); cv.destroy(); } };
};

/* 2. Benchmark: image ROC, pixel ROC and PRO (to pixel FPR 0.3) on the 16-part toy test set (port of parts/viz-metrics.js).
   Checked there (build/toy-bench.mjs): find both -> image AUROC 0.922, pixel AUROC 0.886, AU-PRO 0.847; miss the tiny hole
   -> pixel AUROC 0.882, AU-PRO 0.752; miss the large stain -> pixel AUROC 0.757, AU-PRO 0.751. The last tab plots MVTec AD 2
   Table 1 (AU-PRO to FPR 0.3) and EfficientAD Table 2 (MVTec LOCO image AUROC): sourced, values as printed in notes/claims. */
const BENCH = lazy(() => {
  const { items, featured } = BENCH_KIT.build(), base = items.map(it => BENCH_KIT.pixels(it)), states = {};
  const state = miss => states[miss] || (states[miss] = (() => { const maps = base.slice(); maps[featured] = BENCH_KIT.pixels(items[featured], miss); return { maps, g: BENCH_KIT.grade(items, maps) }; })());
  const lab = items[featured].img.lab, N = TOY.N, edge = [];
  for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) { const l = lab[y * N + x]; if (!l) continue;
    if (x === 0 || y === 0 || x === N - 1 || y === N - 1 || lab[y * N + x - 1] !== l || lab[y * N + x + 1] !== l || lab[(y - 1) * N + x] !== l || lab[(y + 1) * N + x] !== l) edge.push([x, y, l]); }
  return { items, featured, state, edge };
});
const MAD2 = [['PatchCore', '92.7', '79.7', '53.8'], ['RD', '93.9', '89.0', '53.0'], ['RD++', '95.0', '89.3', '54.3'], ['EfficientAD', '93.5', '94.0', '58.7'],
  ['MSFlow', '91.2', '88.4', '52.7'], ['SimpleNet', '89.6', '68.9', '46.4'], ['DSR', '90.8', '68.1', '49.0'], ['mean', '92.4', '82.4', '52.6']];
const LOCO = [['PatchCore', '84.8', '75.8'], ['EfficientAD-M', '94.7', '86.8']];   // structural, logical

SCENES.benchmark = ctx => {
  const B = BENCH(), { items, featured, state, edge } = B, el = mkCanvas(ctx.stage);
  const slider = ctx.q('#bench-t'), sOut = ctx.q('#bench-t-value'), mb = ctx.qa('[data-miss]');
  const TLO = .03, THI = 1.8, toT = v => TLO * Math.pow(THI / TLO, v / 100), toV = t => 100 * Math.log(t / TLO) / Math.log(THI / TLO);
  const NPT = 160, sub = (pts, n) => { const o = []; for (let i = 0; i < n; i++) o.push(pts[Math.round(i / (n - 1) * (pts.length - 1))]); return o; };
  const curveOf = m => { const g = state(m).g; return { pix: sub(g.pixRoc, NPT), pro: sub(g.pro, NPT), a: [g.imgAuroc, g.pixAuroc, g.aupro] }; };
  const S = { step: 0, miss: +((onKey(mb) || mb[0]).dataset.miss), t: toT(+slider.value), last: 1 };
  S.curve = curveOf(S.miss);
  let heat = null, heatKey = '', cancel = () => {};
  const cv = PE.canvas(el, S3ASPECT(ctx.stage), (c, W, H) => draw(c, W, H));
  function layout(W, H) {
    const narrow = W < 620;
    if (narrow) { const s = Math.min(W * .38, H * .36), part = { x: 14, y: 26, s }, cy = part.y + s + 50;
      return { narrow, part, bars: { x: part.x + s + 22, y: part.y + 8, w: W - (part.x + s + 22) - 14 }, strip: null, chart: { x: 46, y: cy, w: W - 64, h: H - cy - 40 } }; }
    const s = Math.min(W * .3, H - HUD_H - 230), part = { x: 24, y: 44, s }, x0 = Math.max(part.x + s + 70, W * .4), wR = W - x0 - 24;
    const cy = 160, ch = Math.min(wR - 40, H - HUD_H - cy - 30);
    return { narrow, part, bars: { x: part.x, y: part.y + s + 64, w: s }, strip: { x: x0, y: 52, w: wR, h: 44 }, chart: { x: x0 + 40, y: cy, w: Math.min(wR - 40, ch * 1.25), h: ch } };
  }
  function chart(c, b, pts, xr, title, area, cur, C) {
    const xt = xr[1] === 1 ? [0, .5, 1] : [0, .1, .2, .3];
    const A = PE.axes(c, b, xr, [0, 1], { xt, yt: [0, .5, 1], xl: xr[1] === 1 ? 'false positive rate' : 'pixel false positive rate (to 0.3)', yl: xr[1] === 1 ? 'true positive rate' : 'PRO', ylo: 28 });
    const vis = []; for (let i = 0; i < pts.length; i++) { const [x, y] = pts[i]; if (x <= xr[1]) { vis.push([x, y]); continue; }
      const [x0, y0] = pts[i - 1]; vis.push([xr[1], y0 + (y - y0) * (xr[1] - x0) / (x - x0)]); break; }
    c.save(); c.beginPath(); c.moveTo(A.X(xr[0]), A.Y(0)); vis.forEach(([x, y]) => c.lineTo(A.X(x), A.Y(y)));
    c.lineTo(A.X(vis.at(-1)[0]), A.Y(0)); c.closePath(); c.fillStyle = C.red; c.globalAlpha = .14; c.fill(); c.restore();
    PE.polyline(c, vis.map(([x, y]) => [A.X(x), A.Y(y)]), C.red, 2.4);
    if (cur && cur[0] <= xr[1]) PE.dot(c, A.X(cur[0]), A.Y(cur[1]), 5, C.yellow);
    PE.label(c, title, b.x, b.y - 26, C.fg, 14); PE.label(c, area, b.x, b.y - 9, C.red, 13);
  }
  function saturation(c, W, H, C, narrow) {
    const sl = narrow ? { x: 38, y: 30, w: W - 140, h: H * .5 } : { x: 70, y: 70, w: Math.min(W * .4, 420), h: H - HUD_H - 140 };
    const yr = [40, 100], cols = ['MVTec AD', 'VisA', 'MVTec AD 2'], X = i => sl.x + i / 2 * sl.w, Y = v => sl.y + sl.h - (v - yr[0]) / (yr[1] - yr[0]) * sl.h;
    PE.label(c, narrow ? 'AU-PRO, seven methods (MVTec AD 2 Table 1)' : 'AU-PRO to FPR 0.3 of seven methods, re-run by MVTec AD 2 (Table 1)', 14, narrow ? 16 : 32, C.fg, 13);
    for (let v = 40; v <= 100; v += 20) { PE.polyline(c, [[sl.x, Y(v)], [sl.x + sl.w, Y(v)]], C.rule, 1); PE.label(c, v, sl.x - 8, Y(v) + 4, C.muted, 11, 'right', 400); }
    cols.forEach((t, i) => PE.label(c, t, X(i), sl.y + sl.h + 18, C.muted, narrow ? 10 : 12, i === 0 ? 'left' : i === 2 ? 'right' : 'center'));
    const ends = [];
    MAD2.forEach(r => { const mean = r[0] === 'mean', best = r[0] === 'EfficientAD', pts = [1, 2, 3].map(k => [X(k - 1), Y(+r[k])]);
      PE.polyline(c, pts, mean ? C.fg : best ? C.red : C.muted, mean ? 3 : best ? 2.2 : 1.3, mean ? [6, 4] : null, mean || best ? 1 : .7);
      pts.forEach(p => PE.dot(c, p[0], p[1], mean ? 4 : 3, mean ? C.fg : best ? C.red : C.muted));
      ends.push({ y: pts[2][1], t: `${r[0]} ${r[3]}`, col: mean ? C.fg : best ? C.red : C.muted }); });
    ends.sort((a, b) => a.y - b.y); let last = -1e9; ends.forEach(e => { const y = Math.max(e.y + 4, last + 13); last = y; PE.label(c, e.t, X(2) + 10, y, e.col, 11, 'left', 600); });
    // MVTec LOCO bars: structural vs logical anomalies
    const lb = narrow ? { x: 104, y: sl.y + sl.h + 70, w: W - 200 } : { x: Math.max(sl.x + sl.w + 150, W * .62), y: 112, w: W - Math.max(sl.x + sl.w + 150, W * .62) - 110 };
    if (lb.w < 80) return;
    PE.label(c, 'MVTec LOCO image AUROC', lb.x - (narrow ? 90 : 0), lb.y - (narrow ? 18 : 34), C.fg, 13);
    if (!narrow) PE.label(c, 'EfficientAD Table 2 (re-runs)', lb.x, lb.y - 18, C.muted, 11, 'left', 400);
    const Xb = v => lb.x + (v - 50) / 50 * lb.w;
    LOCO.forEach((r, i) => { const y0 = lb.y + i * (narrow ? 44 : 70);
      PE.label(c, r[0], narrow ? lb.x - 90 : lb.x, narrow ? y0 + 14 : y0 + 4, C.fg, 12, 'left', 600);
      [['structural', r[1], C.muted], ['logical', r[2], C.red]].forEach(([nm, v, col], k) => { const y = y0 + (narrow ? 2 : 12) + k * 17;
        c.save(); c.fillStyle = col; c.globalAlpha = .85; c.fillRect(lb.x, y, Xb(+v) - lb.x, 12); c.restore();
        PE.label(c, `${nm} ${v}`, Xb(+v) + 6, y + 10, col, 11, 'left', 600); }); });
    PE.label(c, '(bars start at 50, chance)', narrow ? lb.x - 90 : lb.x, lb.y + (narrow ? 96 : 150), C.muted, 10, 'left', 400);
  }
  function draw(c, W, H) {
    if (!W) return; const L = layout(W, H), C = COL(); c.clearRect(0, 0, W, H);
    if (S.step === 3) { saturation(c, W, H, C, L.narrow); return; }
    const st = state(S.miss), a = BENCH_KIT.at(items, st.maps, S.t), N = TOY.N;
    const hk = S.miss + ':' + S.t.toFixed(4) + C.red;
    if (hk !== heatKey) { heat = TOY.heatBitmap(st.maps[featured].map, S.t * .999, S.t * 2.2, C.red); heatKey = hk; }
    TOY.draw(c, items[featured].img, L.part.x, L.part.y, L.part.s, { heat, heatAlpha: .9 });
    const k = L.part.s / N; c.save(); c.fillStyle = C.fg; c.globalAlpha = .9; edge.forEach(([x, y]) => { if ((x + y) % 3 === 0) c.fillRect(L.part.x + x * k, L.part.y + y * k, k, k); }); c.restore();
    PE.label(c, L.narrow ? 'featured part' : 'featured part: a large stain and a tiny hole', L.part.x, L.part.y - 14, C.muted, 12);
    if (!L.narrow) { PE.label(c, 'dotted: true defect regions', L.part.x, L.part.y + L.part.s + 22, C.fg, 11); PE.label(c, 'red: pixels over the threshold', L.part.x, L.part.y + L.part.s + 38, C.red, 11); }
    // the three areas, side by side, so a change shows on every metric at once
    const A = S.curve.a, names = ['image AUROC', 'pixel AUROC', 'AU-PRO'], bw = L.bars.w - (L.narrow ? 0 : 130);
    names.forEach((nm, i) => { const y = L.bars.y + i * (L.narrow ? 34 : 30), on = i === S.step, col = on ? C.fg : C.muted;
      PE.label(c, nm, L.bars.x, y + (L.narrow ? 0 : 12), col, L.narrow ? 11 : 12, 'left', on ? 700 : 400);
      const bx = L.narrow ? L.bars.x : L.bars.x + 92, by = L.narrow ? y + 6 : y + 2, w = L.narrow ? L.bars.w - 34 : bw - 40;
      c.save(); c.fillStyle = C.panel; c.fillRect(bx, by, w, 12); c.fillStyle = C.red; c.globalAlpha = on ? .95 : .5; c.fillRect(bx, by, w * A[i], 12); c.restore();
      PE.label(c, A[i].toFixed(2), bx + w + 6, by + 11, col, 11, 'left', on ? 700 : 400); });
    if (!L.narrow) PE.label(c, 'toy areas under the three curves', L.bars.x, L.bars.y - 12, C.muted, 11, 'left', 400);
    if (L.strip) {   // image-level strip: one dot per test part at its image score (log axis)
      const sx = v => L.strip.x + Math.log(v / TLO) / Math.log(THI / TLO) * L.strip.w, sy = L.strip.y + L.strip.h / 2, hl = S.step === 0;
      PE.polyline(c, [[L.strip.x, sy], [L.strip.x + L.strip.w, sy]], C.rule, 1);
      st.g.imgS.forEach((v, i) => { const good = items[i].good, x = sx(PE.clamp(v, TLO, THI)), y = sy + (good ? -9 : 9) + ((i * 7) % 5 - 2) * 1.5;
        PE.dot(c, x, y, hl ? 5 : 4, good ? C.blue : C.red, hl ? .95 : .7); if (i === featured) PE.ring(c, x, y, 8, C.fg, 1.2); });
      const tx = sx(S.t); PE.polyline(c, [[tx, L.strip.y - 6], [tx, L.strip.y + L.strip.h + 6]], C.yellow, 2);
      PE.label(c, 'image scores of 16 toy test parts (log scale): blue good, red defective', L.strip.x, L.strip.y - 14, hl ? C.fg : C.muted, 11, 'left', hl ? 600 : 400);
      PE.label(c, `threshold ${S.t.toFixed(2)}`, PE.clamp(tx + 6, L.strip.x, L.strip.x + L.strip.w - 80), L.strip.y + L.strip.h + 18, C.yellow, 11);
    }
    if (S.step === 0) chart(c, L.chart, st.g.imgRoc, [0, 1], 'Image ROC: one point per threshold', `area = image AUROC ${A[0].toFixed(2)}`, [a.imgFpr, a.imgTpr], C);
    if (S.step === 1) chart(c, L.chart, S.curve.pix, [0, 1], 'Pixel ROC', `area = pixel AUROC ${A[1].toFixed(2)}`, [a.fpr, a.tpr], C);
    if (S.step === 2) chart(c, L.chart, S.curve.pro, [0, .3], 'PRO curve', `area = AU-PRO ${A[2].toFixed(2)}`, [a.fpr, a.pro], C);
  }
  function say() {
    if (S.step === 3) { ctx.hud(badge('sourced', 'MVTec AD 2 (2503.21622v1, Table 1): mean AU-PRO of seven methods 92.4 on MVTec AD, 82.4 on VisA, 52.6 on MVTec AD 2; best EfficientAD 58.7. MVTec LOCO image AUROC (EfficientAD 2303.14535v3, Table 2): PatchCore 84.8 structural, 75.8 logical; EfficientAD-M 94.7 and 86.8.')); return; }
    const st = state(S.miss), a = BENCH_KIT.at(items, st.maps, S.t), g = st.g, pct = v => (100 * v).toFixed(v < .1 ? 1 : 0) + '%';
    const what = S.miss === 2 ? 'The tiny hole is missed. ' : S.miss === 1 ? 'The large stain is missed. ' : '';
    const txt = [`Threshold ${S.t.toFixed(2)}: ${a.rejBad} of 8 defective parts and ${a.rejGood} of 8 good parts rejected. Image AUROC ${g.imgAuroc.toFixed(2)}.`,
      `Threshold ${S.t.toFixed(2)}: ${pct(a.tpr)} of defect pixels flagged at a ${pct(a.fpr)} false positive rate. Pixel AUROC ${g.pixAuroc.toFixed(2)}.`,
      `Threshold ${S.t.toFixed(2)}: mean region overlap (PRO) ${a.pro.toFixed(2)} at a ${pct(a.fpr)} pixel false positive rate. AU-PRO ${g.aupro.toFixed(2)}.`][S.step];
    ctx.hud(badge('toy', what + txt));
  }
  const toStep = n => { const t = ctx.q(`[role="tab"][data-step="${n}"]`); if (t) t.click(); };
  function setMiss(m) {
    S.miss = m; press(mb, b => +b.dataset.miss === m); if (S.step === 3) { toStep(S.last); return; }
    cancel(); const from = S.curve, to = curveOf(m);
    cancel = PE.animate(700, u => { S.curve = { pix: from.pix.map((p, i) => [PE.lerp(p[0], to.pix[i][0], u), PE.lerp(p[1], to.pix[i][1], u)]),
      pro: from.pro.map((p, i) => [PE.lerp(p[0], to.pro[i][0], u), PE.lerp(p[1], to.pro[i][1], u)]), a: from.a.map((v, i) => PE.lerp(v, to.a[i], u)) }; cv.redraw(); },
      () => { S.curve = to; cv.redraw(); });
    say();
  }
  function setT(t) { S.t = t; slider.value = toV(t); sOut.textContent = t.toFixed(2); if (S.step === 3) { toStep(S.last); return; } cv.redraw(); say(); }
  const offs = [...mb.map(b => listen(b, 'click', () => setMiss(+b.dataset.miss))), listen(slider, 'input', () => setT(toT(+slider.value))),
    listen(ctx.q('[data-act="reset"]'), 'click', () => { setT(.1); setMiss(0); })];
  return { flat: true, resize: cv.resize,
    step(n) { S.step = n; if (n < 3) S.last = n; S.curve = curveOf(S.miss); cv.redraw(); say(); },
    destroy() { cancel(); offs.forEach(f => f()); cv.destroy(); } };
};

/* 3. Remember normal: SPADE (2D), PaDiM (pe3 bell + position picker) and PatchCore (pe3 cloud + side panel), one sub-scene
   per tab. Ports of parts/viz-spade.js, viz-padim.js and viz-patchcore.js, checked there:
   SPADE (build/toy-spade-fig.mjs): K = 3 of 12 caps; image scores scratched 0.0222, hole 0.0014, good 0.0003; the hole patch
   (79) matches a bolt-hole patch at squared distance 0.0003.
   PaDiM (build/toy-padim2d.mjs): at position 79 the hole is Euclidean 2.63 / Mahalanobis 19.1 from the mean (normal max 0.26 /
   1.91); at the rim position 71 the thin-axis probe is Euclidean 1.38 (normal max 1.91) but Mahalanobis 13.4 (normal max 1.39).
   The scratched cap is new here: its worst Mahalanobis position is 129 (18.8, on the scratch), checked in node.
   PatchCore (build/toy-patchcore.mjs): coverage radius 0.993 at 1%, 0.306 at 10%, 0.174 at 25%; good / scratched caps score
   0.34 / 0.86 at 10% and 1.01 / 0.98 at 1%. The holed cap is new here: 1.02 at 1%, 0.34 at 10%, 0.25 at 25% (checked in node),
   so like SPADE it is not separated from the good cap. */
const SPADE = lazy(() => {
  const G = TOY.G, G2 = G * G, K = 3;
  const f2 = f => { const o = []; for (let k = 0; k < G2; k++) o.push([f[k * 3 + 1], f[k * 3 + 2]]); return o; };
  const pooled = f => { const m = [0, 0, 0, 0]; for (let k = 0; k < G2; k++) for (let d = 0; d < 3; d++) m[d] += f[k * 3 + d] / G2;
    let v = 0; for (let k = 0; k < G2; k++) v += (f[k * 3 + 2] - m[2]) ** 2 / G2; m[3] = Math.sqrt(v); return m; };
  const train = Array.from({ length: 12 }, (_, i) => { const img = TOY.make('cap', 1 + i), F = TOY.features(img); return { img, f: f2(F), p: pooled(F) }; });
  const mk = img => { const F = TOY.features(img), f = f2(F), p = pooled(F);
    const d = train.map((t, i) => [t.p.reduce((a, x, j) => a + (x - p[j]) ** 2, 0), i]);
    const nearK = [...d].sort((a, b) => a[0] - b[0]).slice(0, K).map(x => x[1]);
    const gal = nearK.flatMap(i => train[i].f.map((g, k) => ({ i, k, g })));
    const best = f.map(q => { let b = Infinity, bj = 0; gal.forEach((e, j) => { const s = (e.g[0] - q[0]) ** 2 + (e.g[1] - q[1]) ** 2; if (s < b) { b = s; bj = j; } }); return [b, gal[bj]]; });
    let worst = 0; best.forEach((v, q) => { if (v[0] > best[worst][0]) worst = q; });
    return { img, f, d: d.map(x => x[0]), nearK, gal, best, worst, score: nearK.reduce((a, i) => a + d[i][0], 0) / K, map: TOY.toPixels(Float32Array.from(best.map(v => v[0]))) }; };
  const T = { scratch: mk(TOY.defect(TOY.make('cap', 200), 'scratch', 200)), hole: mk(TOY.defect(TOY.make('cap', 207), 'hole', 207)), good: mk(TOY.make('cap', 104)) };
  const allF = train.flatMap(t => t.f).concat(T.scratch.f);
  const lo = [Math.min(...allF.map(p => p[0])) - .2, Math.min(...allF.map(p => p[1])) - .2], hi = [Math.max(...allF.map(p => p[0])) + .2, Math.max(...allF.map(p => p[1])) + .2];
  return { G, K, train, T, lo, hi, HOLE_PATCH: 79, LO: .06, HI: .6 };
});
const TESTNAME = { scratch: 'scratched cap', hole: 'cap with a hole', good: 'good cap' };

function spadeScene(ctx, M) {
  const D = SPADE(), { G, train, T, lo, hi, LO } = D, el = mkCanvas(ctx.stage);
  el.tabIndex = 0; el.setAttribute('aria-label', 'SPADE: the cap under inspection, twelve remembered caps, the best match and the feature space. Click a patch on the cap, or use the arrow keys.');
  const S = { q: M.test === 'hole' ? D.HOLE_PATCH : T[M.test].worst, mix: M.spade === 'pixel' ? 1 : 0, line: 1 };
  const heats = {}; let heatColor = '', L = null, cancel = () => {}, cancelM = () => {};
  const cv = PE.canvas(el, S3ASPECT(ctx.stage), (c, W, H) => draw(c, W, H));
  function layout(W, H) {
    const narrow = W < 620;
    if (narrow) { const s = Math.min(W * .42, H * .36), img = { x: 12, y: 26, s }, tx = img.x + s + 16, gap = 4, tw = (W - tx - 12 - gap * 3) / 4;
      const lowY = img.y + s + 34, sm = Math.min((W - 36) / 2, H - lowY - 26);
      return { narrow, img, thumbs: { x: tx, y: 26, w: tw, gap, cols: 4, pitch: tw + 10, bar: 7 }, match: { x: 12, y: lowY, s: sm }, fs: { x: 24 + sm, y: lowY, w: W - 36 - sm, h: sm } }; }
    const s = Math.min(W * .36, H - HUD_H - 110), img = { x: 24, y: 44, s }, x0 = img.x + s + 48, rw = W - x0 - 24, gap = 10, tw = Math.min(90, (rw - gap * 5) / 6);
    const thumbs = { x: x0, y: 44, w: tw, gap, cols: 6, pitch: tw + 36, bar: 26 }, lowY = thumbs.y + 2 * thumbs.pitch + 28, sm = Math.min(rw * .45, H - HUD_H - lowY - 10);
    return { narrow, img, thumbs, match: { x: x0, y: lowY, s: sm }, fs: { x: x0 + sm + 18, y: lowY, w: rw - sm - 18, h: sm } };
  }
  const thumbXY = i => [L.thumbs.x + (i % L.thumbs.cols) * (L.thumbs.w + L.thumbs.gap), L.thumbs.y + Math.floor(i / L.thumbs.cols) * L.thumbs.pitch];
  const patchCenter = (box, k) => { const s = box.s / TOY.N; return [box.x + ((k % G) * TOY.S + TOY.P / 2) * s, box.y + (Math.floor(k / G) * TOY.S + TOY.P / 2) * s]; };
  const patchRect = (box, k) => { const s = box.s / TOY.N; return [box.x + (k % G) * TOY.S * s, box.y + Math.floor(k / G) * TOY.S * s, TOY.P * s]; };
  const onBolt = (img, k) => { const cx = (k % G) * TOY.S + TOY.P / 2, cy = Math.floor(k / G) * TOY.S + TOY.P / 2; return Math.min(...img.bolts.map(([x, y]) => Math.hypot(x - cx, y - cy))) < 12; };
  function draw(c, W, H) {
    if (!W) return; L = layout(W, H); const C = COL();
    if (heatColor !== C.red) { for (const k in heats) delete heats[k]; heatColor = C.red; }
    const t = T[M.test], pm = S.mix; c.clearRect(0, 0, W, H);
    heats[M.test] = heats[M.test] || TOY.heatBitmap(t.map, LO, D.HI, C.red);
    TOY.draw(c, t.img, L.img.x, L.img.y, L.img.s, { heat: heats[M.test], heatAlpha: pm });
    PE.label(c, L.narrow ? 'under inspection' : 'cap under inspection (click a patch)', L.img.x, L.img.y - 12, C.muted, 12);
    PE.label(c, '12 good caps in memory', L.thumbs.x, L.thumbs.y - 12, C.muted, 12);
    const dmax = Math.max(...Object.values(T).flatMap(x => x.d));
    train.forEach((tr, i) => { const [x, y] = thumbXY(i), near = t.nearK.indexOf(i);
      c.save(); c.globalAlpha = near >= 0 ? 1 : .45; TOY.draw(c, tr.img, x, y, L.thumbs.w, { vf: false }); c.restore();
      if (near >= 0) { c.save(); c.strokeStyle = C.blue; c.lineWidth = 2; c.strokeRect(x - 2, y - 2, L.thumbs.w + 4, L.thumbs.w + 4); c.restore(); }
      const bh = Math.max(1.5, L.thumbs.bar * Math.sqrt(t.d[i] / dmax)) * (1 - pm);
      if (bh > 1) { c.save(); c.globalAlpha = 1 - pm; c.fillStyle = near >= 0 ? C.blue : C.muted; c.fillRect(x + L.thumbs.w * .25, y + L.thumbs.w + 4, L.thumbs.w * .5, bh); c.restore(); } });
    const sel = t.best[S.q], g = sel[1], mimg = train[g.i].img, flagged = sel[0] > LO;
    if (pm > 0) {
      c.save(); c.globalAlpha = pm;
      TOY.draw(c, mimg, L.match.x, L.match.y, L.match.s);
      PE.label(c, `best match: cap ${g.i + 1}`, L.match.x, L.match.y - 12, C.muted, 12);
      const [mx, my, ms] = patchRect(L.match, g.k); c.strokeStyle = C.blue; c.lineWidth = 2; c.strokeRect(mx, my, ms, ms);
      c.strokeStyle = C.rule; c.lineWidth = 1; c.strokeRect(L.fs.x, L.fs.y, L.fs.w, L.fs.h);
      const X = p => L.fs.x + (p[0] - lo[0]) / (hi[0] - lo[0]) * L.fs.w, Y = p => L.fs.y + L.fs.h - (p[1] - lo[1]) / (hi[1] - lo[1]) * L.fs.h;
      c.save(); c.beginPath(); c.rect(L.fs.x, L.fs.y, L.fs.w, L.fs.h); c.clip();
      t.gal.forEach(e => PE.dot(c, X(e.g), Y(e.g), 1.4, C.blue, .45));
      const qf = t.f[S.q];
      c.strokeStyle = C.red; c.lineWidth = 2; c.beginPath(); c.moveTo(X(g.g), Y(g.g)); c.lineTo(X(g.g) + (X(qf) - X(g.g)) * S.line, Y(g.g) + (Y(qf) - Y(g.g)) * S.line); c.stroke();
      PE.dot(c, X(g.g), Y(g.g), 3.2, C.blue); PE.dot(c, X(qf), Y(qf), 4.5, C.yellow); c.restore();
      PE.label(c, L.narrow ? 'gallery' : 'gallery features of the 3 caps', L.fs.x, L.fs.y - 12, C.muted, 12);
      PE.label(c, 'diagonal edges', L.fs.x + L.fs.w / 2, L.fs.y + L.fs.h + 15, C.muted, 10, 'center');
      const [ax, ay] = patchCenter(L.img, S.q), [bx, by] = patchCenter(L.match, g.k), u = S.line;
      c.strokeStyle = flagged ? C.red : C.blue; c.lineWidth = 1.6; c.setLineDash([5, 4]); c.beginPath();
      const cx1 = ax + (bx - ax) * .5, cy1 = Math.min(ay, by) - 60;
      for (let s = 0; s <= 1.0001 * u; s += .02) { const x = (1 - s) ** 2 * ax + 2 * s * (1 - s) * cx1 + s * s * bx, y = (1 - s) ** 2 * ay + 2 * s * (1 - s) * cy1 + s * s * by; s ? c.lineTo(x, y) : c.moveTo(x, y); }
      c.stroke(); c.setLineDash([]); c.restore();
    }
    if (pm < 1) {
      c.save(); c.globalAlpha = 1 - pm;
      const x0 = L.match.x, y0 = L.match.y, w = L.match.s + 18 + L.fs.w, rows = [['scratched cap', T.scratch], ['cap with a hole', T.hole], ['good cap', T.good]], lw = L.narrow ? 96 : 110;
      PE.label(c, L.narrow ? 'image score (3 nearest caps)' : 'image score: mean squared distance to the 3 nearest caps', x0, y0 - 12, C.muted, 12);
      const smax = Math.max(...rows.map(r => r[1].score));
      rows.forEach(([name, r], j) => { const y = y0 + 16 + j * 34, bw = Math.max(2, (w - lw - 60) * Math.sqrt(r.score / smax)), cur = r === t;
        PE.label(c, name, x0, y + 13, cur ? C.yellow : C.muted, 12);
        c.fillStyle = cur ? C.yellow : C.muted; c.globalAlpha = (1 - pm) * (cur ? 1 : .5); c.fillRect(x0 + lw, y, bw, 16); c.globalAlpha = 1 - pm;
        PE.label(c, r.score.toFixed(4), x0 + lw + 6 + bw, y + 13, cur ? C.fg : C.muted, 11); });
      PE.label(c, '(square-root scale)', x0, y0 + 16 + 3 * 34 + 6, C.muted, 10);
      c.restore();
    }
    const [px, py, ps] = patchRect(L.img, S.q);
    c.save(); c.globalAlpha = .35 + .65 * pm; c.strokeStyle = C.yellow; c.lineWidth = 2; c.strokeRect(px, py, ps, ps); c.restore();
  }
  function say() {
    const t = T[M.test], name = TESTNAME[M.test];
    if (M.spade === 'image') { ctx.hud(badge('toy', `SPADE stage 1: the 3 remembered caps nearest to the ${name} by pooled feature are caps ${t.nearK.map(i => i + 1).join(', ')}. Its image score is ${t.score.toFixed(4)}; scratched cap ${T.scratch.score.toFixed(4)}, cap with a hole ${T.hole.score.toFixed(4)}, good cap ${T.good.score.toFixed(4)}.`)); return; }
    const [s, g] = t.best[S.q], r = Math.floor(S.q / G) + 1, cc = S.q % G + 1;
    let msg = `SPADE stage 2, ${name}, patch at row ${r}, column ${cc}: its nearest gallery feature is in cap ${g.i + 1}, at squared distance ${s.toFixed(4)}`;
    msg += s > LO ? ', so it lights up.' : (onBolt(train[g.i].img, g.k) && M.test === 'hole' && S.q === D.HOLE_PATCH ? '. The match is a patch on a bolt hole, so SPADE sees nothing wrong.' : ', so it looks normal.');
    ctx.hud(badge('toy', msg));
  }
  function select(q, animate = true) { S.q = q; cancel(); if (!animate || PE.instant()) { S.line = 1; cv.redraw(); say(); return; } cancel = PE.animate(450, u => { S.line = u; cv.redraw(); }); say(); }
  function mode(animate = true) { cancelM(); const a = S.mix, b = M.spade === 'pixel' ? 1 : 0; if (!animate) { S.mix = b; cv.redraw(); } else cancelM = PE.animate(500, u => { S.mix = a + (b - a) * u; cv.redraw(); }); say(); }
  el.addEventListener('pointerdown', e => { if (!L) return; const r = el.getBoundingClientRect(), x = e.clientX - r.left - L.img.x, y = e.clientY - r.top - L.img.y;
    if (x < 0 || y < 0 || x > L.img.s || y > L.img.s) return; const s = L.img.s / TOY.N;
    const i = PE.clamp(Math.floor((x / s - TOY.P / 2) / TOY.S + .5), 0, G - 1), j = PE.clamp(Math.floor((y / s - TOY.P / 2) / TOY.S + .5), 0, G - 1);
    if (M.spade !== 'pixel') { M.spade = 'pixel'; press(ctx.qa('[data-spade]'), b => b.dataset.spade === 'pixel'); mode(); } select(j * G + i); });
  el.addEventListener('keydown', e => { const d = { ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowUp: [0, -1], ArrowDown: [0, 1] }[e.key]; if (!d) return;
    e.preventDefault(); const i = PE.clamp(S.q % G + d[0], 0, G - 1), j = PE.clamp(Math.floor(S.q / G) + d[1], 0, G - 1); select(j * G + i, false); });
  say();
  return { resize: cv.resize,
    update(what) { if (what === 'test' || what === 'reset') select(M.test === 'hole' ? D.HOLE_PATCH : T[M.test].worst); if (what === 'spade' || what === 'reset') mode(); },
    replay() { cancelM(); S.mix = 0; cv.redraw(); const was = M.spade; M.spade = 'image'; say(); M.spade = was;
      cancelM = PE.animate(2200, u => { S.mix = PE.clamp((u - .45) / .4, 0, 1); cv.redraw(); }, () => { M.spade = 'pixel'; press(ctx.qa('[data-spade]'), b => b.dataset.spade === 'pixel'); S.mix = 1; cv.redraw(); say(); }); },
    destroy() { cancel(); cancelM(); cv.destroy(); } };
}

const PADIM = lazy(() => {
  const G = TOY.G, G2 = G * G, NTR = 20, EPS = .01;
  const f2 = f => { const o = []; for (let k = 0; k < G2; k++) o.push([f[k * 3 + 1], f[k * 3 + 2]]); return o; };
  const train = Array.from({ length: NTR }, (_, i) => f2(TOY.features(TOY.make('cap', 1 + i))));
  const gs = [];
  for (let q = 0; q < G2; q++) {
    const m = [0, 0]; train.forEach(f => { m[0] += f[q][0] / NTR; m[1] += f[q][1] / NTR; });
    let a = EPS, b = 0, d = EPS; train.forEach(f => { const u = f[q][0] - m[0], v = f[q][1] - m[1]; a += u * u / (NTR - 1); b += u * v / (NTR - 1); d += v * v / (NTR - 1); });
    const tr = a + d, det = a * d - b * b, disc = Math.sqrt(Math.max(0, tr * tr / 4 - det)), l1 = tr / 2 + disc, l2 = tr / 2 - disc;
    const e = Math.abs(b) > 1e-12 ? [l1 - d, b] : (a >= d ? [1, 0] : [0, 1]), n = Math.hypot(...e), e1 = [e[0] / n, e[1] / n];
    const L11 = Math.sqrt(a), L21 = b / L11, L22 = Math.sqrt(d - L21 * L21);
    gs.push({ m, a, b, d, det, l1, l2, e1, e2: [-e1[1], e1[0]], s: Math.sqrt(l1), chol: [L11, L21, L22] });
  }
  const maha = (q, p) => { const { m, a, b, d, det } = gs[q], u = p[0] - m[0], v = p[1] - m[1]; return Math.sqrt(Math.max(0, (d * u * u - 2 * b * u * v + a * v * v) / det)); };
  const eu = (q, p) => Math.hypot(p[0] - gs[q].m[0], p[1] - gs[q].m[1]);
  const tests = { hole: TOY.defect(TOY.make('cap', 207), 'hole', 207), good: TOY.make('cap', 104), scratch: TOY.defect(TOY.make('cap', 200), 'scratch', 200) };
  const tf = Object.fromEntries(Object.keys(tests).map(k => [k, f2(TOY.features(tests[k]))]));
  const sc = Object.fromEntries(Object.keys(tf).map(k => [k, tf[k].map((p, q) => maha(q, p))]));
  const maps = Object.fromEntries(Object.keys(tf).map(k => [k, TOY.toPixels(Float32Array.from(sc[k]))]));
  const worst = k => sc[k].reduce((w, v, q, a) => v > a[w] ? q : w, 0);
  const normals = Array.from({ length: 20 }, (_, i) => f2(TOY.features(TOY.make('cap', 100 + i))));
  const nMax = gs.map((_, q) => Math.max(...normals.map(f => maha(q, f[q])))), nEu = gs.map((_, q) => Math.max(...normals.map(f => eu(q, f[q]))));
  const bank = train.flat(), knn = p => Math.min(...bank.map(m => Math.hypot(m[0] - p[0], m[1] - p[1])));
  return { G, NTR, gs, maha, eu, tests, tf, maps, nMax, nEu, knn, train, POS: { hole: 79, rim: 71, scratch: worst('scratch') } };
});

function padimScene(ctx, M) {
  const D = PADIM(), { G, NTR, gs, maha, eu, tests, tf, maps, nMax, nEu, knn, train, POS } = D, H = 1.3, R = 3;
  const el = mkCanvas(ctx.stage, 'pe3'), pick = mkCanvas(ctx.stage);
  const narrowP = () => ctx.stage.clientWidth < 620;
  pick.style.cssText = `position:absolute;left:12px;top:12px;width:${narrowP() ? 104 : 168}px;border-radius:8px;border:1px solid var(--line);background:var(--surface);cursor:pointer`;
  pick.tabIndex = 0; pick.setAttribute('aria-label', 'Pick a patch position: click the camera view or use the arrow keys.');
  const S = { q: M.test === 'scratch' ? POS.scratch : POS.hole, probe: null, h: 1, ring: 1, tp: null };
  if (M.probe) { const g = gs[POS.rim]; S.q = POS.rim; S.probe = [g.m[0] + g.s * g.e2[0], g.m[1] + g.s * g.e2[1]]; }
  const toWorld = (q, p) => [(p[0] - gs[q].m[0]) / gs[q].s, (p[1] - gs[q].m[1]) / gs[q].s];
  const toFeat = (q, w) => [gs[q].m[0] + w[0] * gs[q].s, gs[q].m[1] + w[1] * gs[q].s];
  const testFeat = () => S.probe || tf[M.test][S.q];
  const clampR = w => { const r = Math.hypot(w[0], w[1]); return r > R - .1 ? [w[0] * (R - .1) / r, w[1] * (R - .1) / r, true] : [w[0], w[1], false]; };
  const height = (q, X, Z) => { const g = gs[q], v = [X * g.s, Z * g.s], m2 = (g.d * v[0] * v[0] - 2 * g.b * v[0] * v[1] + g.a * v[1] * v[1]) / g.det; return H * Math.exp(-m2 / 2); };
  const s3 = PE3.scene(el, { aspect: S3ASPECT(ctx.stage), fov: 36, dist: narrowP() ? 12.5 : 10.5, yaw: .75, pitch: .62, items: [] });
  function build() {
    const q = S.q, g = gs[q], bg = PE.css('--bg'), blue = PE.css('--blue'), items = [], dm = M.dist === 'maha';
    items.push({ type: 'grid', y: 0, size: 2 * R, step: 1, color: '--rule' });
    items.push(...PE3.surface((X, Z) => S.h * height(q, X, Z), { x0: -R, x1: R, z0: -R, z1: R, nx: 30, nz: 30,
      fill: y => PE3.mix(bg, blue, .18 + .55 * y / H), alpha: .9 }).filter(it => Math.max(...it.pts.map(p => p[1])) > .02 * H * Math.max(S.h, .05)));
    items.push({ type: 'points', pts: train.map(f => { const w = toWorld(q, f[q]); return [w[0], .01, w[1]]; }), color: '--blue', size: 4.5 });
    items.push({ type: 'points', pts: [[0, .01, 0]], color: '--blue', size: 8 });
    for (let k = 1; k <= 3; k++) {
      const pts = [];
      for (let t = 0; t <= 64; t++) { const cc = Math.cos(t / 32 * Math.PI), sn = Math.sin(t / 32 * Math.PI);
        if (dm) { const [L11, L21, L22] = g.chol, v = [k * L11 * cc, k * (L21 * cc + L22 * sn)], w = [v[0] / g.s, v[1] / g.s]; pts.push(Math.hypot(...w) <= R ? [w[0], .012, w[1]] : null); }
        else pts.push([k * cc, .012, k * sn]); }
      let run = []; const flush = () => { if (run.length > 1) items.push({ type: 'line', pts: run, color: dm ? '--blue' : '--muted', width: 1.4, alpha: .8 * S.ring }); run = []; };
      pts.forEach(p => { if (p) run.push(p); else flush(); }); flush();
    }
    const [tx, tz, off] = S.tp || clampR(toWorld(q, testFeat())), th = S.h * height(q, tx, tz);
    items.push({ type: 'line', pts: [[0, .015, 0], [tx, .015, tz]], color: '--red', width: 2.6 });
    items.push({ type: 'line', pts: [[tx, 0, tz], [tx, Math.max(th, .02), tz]], color: '--yellow', width: 1.6, dash: [4, 3] });
    items.push({ type: 'points', pts: [[tx, .02, tz]], color: '--yellow', size: 10 });
    items.push({ type: 'label', at: [tx, .02, tz], text: off ? 'test patch (off the floor)' : (S.probe ? 'probe (drag me)' : 'test patch (drag me)'), color: '--yellow', dy: -14, align: 'center' });
    items.push({ type: 'label', at: [0, .02, 0], text: 'mean', color: '--blue', dy: 18, align: 'center' });
    items.push({ type: 'label', at: [R, 0, 0], text: 'diagonal edges', color: '--muted', align: 'right', dy: 16 });
    items.push({ type: 'label', at: [0, 0, R + .3], text: 'brightness', color: '--muted', align: 'center', dy: 14 });
    if (!narrowP()) items.push({ type: 'label', at: [0, H * S.h, 0], text: `Gaussian of 20 good caps at row ${Math.floor(q / G) + 1}, column ${q % G + 1}`, color: '--muted', dy: -16, align: 'center' });
    s3.items = items; s3.redraw();
  }
  let heats = {}, heatColor = '';
  const pv = PE.canvas(pick, 1.12, (c, W, Hh) => {
    if (!W) return; const red = PE.css('--red'); if (red !== heatColor) { heats = {}; heatColor = red; }
    heats[M.test] = heats[M.test] || TOY.heatBitmap(maps[M.test], 3, 16, red);
    c.clearRect(0, 0, W, Hh); const sz = W - 12, x0 = 6, y0 = 18;
    PE.label(c, 'pick a position', x0, 12, PE.css('--muted'), 10);
    TOY.draw(c, tests[M.test], x0, y0, sz, { heat: heats[M.test], vf: false });
    const s = sz / TOY.N, q = S.q; c.save(); c.strokeStyle = PE.css('--yellow'); c.lineWidth = 2;
    c.strokeRect(x0 + (q % G) * TOY.S * s, y0 + Math.floor(q / G) * TOY.S * s, TOY.P * s, TOY.P * s); c.restore();
  });
  function say() {
    const q = S.q, p = testFeat(), e = eu(q, p), mh = maha(q, p), r = Math.floor(q / G) + 1, cc = q % G + 1;
    let t = `PaDiM, position row ${r}, column ${cc}: a Gaussian fitted to ${NTR} good caps. `;
    t += S.probe ? `Probe patch: Euclidean distance ${e.toFixed(2)}, Mahalanobis distance <b>${mh.toFixed(1)}</b>. ` : `Test patch (${TESTNAME[M.test]}): Euclidean distance ${e.toFixed(2)}, Mahalanobis distance <b>${mh.toFixed(1)}</b>. `;
    t += `Good caps here reach at most ${nEu[q].toFixed(2)} and ${nMax[q].toFixed(1)}.`;
    if (!S.probe) t += ` Position-free nearest neighbor (SPADE-style): ${knn(p).toFixed(2)}.`;
    ctx.hud(badge('toy', t));
  }
  let cancel = () => {};
  function setPos(q) { cancel(); const same = q === S.q; S.q = q; S.tp = null; pv.redraw(); say(); if (same || PE3.instant()) { S.h = 1; build(); return; } cancel = s3.tween(650, u => { S.h = u; build(); }); }
  function moveTest(fn) { cancel(); const from = clampR(toWorld(S.q, testFeat())); fn(); const to = clampR(toWorld(S.q, testFeat())); pv.redraw(); say();
    cancel = s3.tween(550, u => { S.tp = [from[0] + (to[0] - from[0]) * u, from[1] + (to[1] - from[1]) * u, to[2]]; build(); }, () => { S.tp = null; build(); }); }
  const probeAt = () => { const g = gs[POS.rim]; return [g.m[0] + g.s * g.e2[0], g.m[1] + g.s * g.e2[1]]; };
  const pressProbe = () => press([ctx.q('[data-probe]')], () => !!S.probe);
  // picker: click or arrow keys choose the position; drag the yellow point on the floor (captured before pe3 orbits)
  pick.addEventListener('pointerdown', e => { e.stopPropagation(); const r = pick.getBoundingClientRect(), sz = r.width - 12, s = sz / TOY.N, x = (e.clientX - r.left - 6) / s, y = (e.clientY - r.top - 18) / s;
    if (x < 0 || y < 0 || x > TOY.N || y > TOY.N) return; S.probe = null; M.probe = false; pressProbe();
    setPos(PE.clamp(Math.round((y - TOY.P / 2) / TOY.S), 0, G - 1) * G + PE.clamp(Math.round((x - TOY.P / 2) / TOY.S), 0, G - 1)); });
  pick.addEventListener('keydown', e => { const d = { ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowUp: [0, -1], ArrowDown: [0, 1] }[e.key]; if (!d) return;
    e.preventDefault(); e.stopPropagation(); S.probe = null; M.probe = false; pressProbe(); setPos(PE.clamp(Math.floor(S.q / G) + d[1], 0, G - 1) * G + PE.clamp(S.q % G + d[0], 0, G - 1)); });
  const unproject = (sx, sy, w0) => { let w = [...w0];
    for (let it = 0; it < 6; it++) { const p = s3.project([w[0], 0, w[1]]), px = s3.project([w[0] + .01, 0, w[1]]), pz = s3.project([w[0], 0, w[1] + .01]);
      const J = [[(px.x - p.x) / .01, (pz.x - p.x) / .01], [(px.y - p.y) / .01, (pz.y - p.y) / .01]], det = J[0][0] * J[1][1] - J[0][1] * J[1][0]; if (Math.abs(det) < 1e-6) break;
      const rx = sx - p.x, ry = sy - p.y; w = [w[0] + (J[1][1] * rx - J[0][1] * ry) / det, w[1] + (-J[1][0] * rx + J[0][0] * ry) / det]; }
    return clampR(w); };
  let drag = false;
  const onDown = e => { if (e.target !== el) return; const r = el.getBoundingClientRect(), w = clampR(toWorld(S.q, testFeat())), p = s3.project([w[0], .02, w[1]]);
    if (Math.hypot(e.clientX - r.left - p.x, e.clientY - r.top - p.y) > 18) return; e.stopPropagation(); e.preventDefault(); drag = true; };
  const onMove = e => { if (!drag) return; const r = el.getBoundingClientRect(), w0 = clampR(toWorld(S.q, testFeat())), w = unproject(e.clientX - r.left, e.clientY - r.top, w0);
    S.probe = toFeat(S.q, w); S.tp = null; build(); say(); };
  const onUp = () => { drag = false; };
  ctx.stage.addEventListener('pointerdown', onDown, { capture: true }); window.addEventListener('pointermove', onMove); window.addEventListener('pointerup', onUp);
  const offTheme = PE.onTheme(build);
  build(); say();
  return {
    update(what) {
      if (what === 'test') { if (M.test === 'scratch') { S.probe = null; setPos(POS.scratch); } else if (M.test === 'hole' && S.q !== POS.hole) { S.probe = null; setPos(POS.hole); } else moveTest(() => { S.probe = null; }); }
      if (what === 'dist') { cancel(); cancel = s3.tween(450, u => { S.ring = u; build(); }); }
      if (what === 'probe') { if (S.q === POS.rim) moveTest(() => { S.probe = probeAt(); }); else { S.probe = probeAt(); setPos(POS.rim); } }
      if (what === 'reset') { S.probe = null; S.ring = 1; setPos(POS.hole); s3.setView({ yaw: .75, pitch: .62, dist: narrowP() ? 12.5 : 10.5 }, 500); build(); }
    },
    replay() { cancel(); S.h = 0; build(); cancel = s3.tween(1200, u => { S.h = u; build(); }); },
    destroy() { cancel(); offTheme(); ctx.stage.removeEventListener('pointerdown', onDown, { capture: true }); window.removeEventListener('pointermove', onMove); window.removeEventListener('pointerup', onUp); pv.destroy(); s3.destroy(); } };
}

const PCORE = lazy(() => {
  const B = TOY.bank('cap', 20, 1), n = B.length / 3;
  const d2 = (A, i, C, j) => (A[i * 3] - C[j * 3]) ** 2 + (A[i * 3 + 1] - C[j * 3 + 1]) ** 2 + (A[i * 3 + 2] - C[j * 3 + 2]) ** 2;
  const order = new Int32Array(n), rank = new Int32Array(n), rad = new Float32Array(n + 1);
  { const r = TOY.rng(7), anchors = Array.from({ length: 10 }, () => Math.floor(r() * n)), md = new Float64Array(n);
    for (let i = 0; i < n; i++) { let s = 0; for (const a of anchors) s += Math.sqrt(d2(B, i, B, a)); md[i] = s / 10; }
    for (let k = 0; k < n; k++) { let bi = 0; for (let i = 1; i < n; i++) if (md[i] > md[bi]) bi = i;
      rad[k] = md[bi]; order[k] = bi; rank[bi] = k; for (let i = 0; i < n; i++) { const d = Math.sqrt(d2(B, i, B, bi)); if (d < md[i]) md[i] = d; } }
    rad[n] = 0; }
  const tests = { good: TOY.make('cap', 104), scratch: TOY.defect(TOY.make('cap', 200), 'scratch', 200), hole: TOY.defect(TOY.make('cap', 207), 'hole', 207) };
  const tf = Object.fromEntries(Object.keys(tests).map(k => [k, TOY.features(tests[k])]));
  const cache = {};
  function scored(test, c) {
    const key = test + c; if (cache[key]) return cache[key];
    const f = tf[test], s = new Float32Array(TOY.G * TOY.G), nnI = new Int32Array(TOY.G * TOY.G);
    for (let q = 0; q < s.length; q++) { let b = Infinity, bi = 0; for (let k = 0; k < c; k++) { const d = d2(f, q, B, order[k]); if (d < b) { b = d; bi = order[k]; } } s[q] = Math.sqrt(b); nnI[q] = bi; }
    let w = 0; for (let q = 1; q < s.length; q++) if (s[q] > s[w]) w = q;
    let fb = Infinity, fi = 0; for (let i = 0; i < n; i++) { const d = d2(f, w, B, i); if (d < fb) { fb = d; fi = i; } }
    let onDefect = false; const m = tests[test].mask;
    if (m) { const gi = w % TOY.G, gj = Math.floor(w / TOY.G); for (let y = gj * TOY.S; y < gj * TOY.S + TOY.P && !onDefect; y++) for (let x = gi * TOY.S; x < gi * TOY.S + TOY.P; x++) if (m[y * TOY.N + x]) { onDefect = true; break; } }
    return cache[key] = { s, worst: w, score: s[w], coreNN: nnI[w], fullNN: fi, fullD: Math.sqrt(fb), onDefect, map: TOY.toPixels(s) };
  }
  const SC = 1.3, Wp = (A, i) => [A[i * 3] / SC, A[i * 3 + 2] / SC, A[i * 3 + 1] / SC];
  return { B, n, d2, order, rank, rad, tests, tf, scored, Wp, WB: Array.from({ length: n }, (_, i) => Wp(B, i)), scratchRef: scored('scratch', n).score };
});

function coreScene(ctx, M) {
  const D = PCORE(), { B, n, d2, order, rank, rad, tests, tf, scored, Wp, WB, scratchRef } = D;
  const el = mkCanvas(ctx.stage, 'pe3'), side = mkCanvas(ctx.stage), narrow = () => ctx.stage.clientWidth < 620;
  side.setAttribute('aria-hidden', 'true');
  const sideW = () => narrow() ? 120 : Math.min(400, ctx.stage.clientWidth * .45);
  side.style.cssText = `position:absolute;left:12px;top:12px;width:${sideW()}px;border-radius:8px;border:1px solid var(--line);background:var(--surface);pointer-events:none`;
  const lOf = p => Math.max(1, Math.floor(n * p / 100));
  const S = { c: lOf(M.keep), anim: false };
  let stop = () => {}, heat = null, heatKey = '';
  const s3 = PE3.scene(el, { aspect: S3ASPECT(ctx.stage), yaw: .75, pitch: .42, dist: narrow() ? 9 : 7.4, items: [] });
  function items() {
    const kept = [], rest = []; for (let i = 0; i < n; i++) (rank[i] < S.c ? kept : rest).push(WB[i]);
    const it = [{ type: 'axes', len: 2.3, color: '--muted', labels: ['h vs v edges', 'brightness', 'diagonal edges'] },
      { type: 'points', pts: rest, color: '--blue', size: 1.4, alpha: .16 }, { type: 'points', pts: kept, color: '--blue', size: 3.1, alpha: .95 }];
    if (S.c >= 1 && S.c < n) { const w = order[S.c]; let b = Infinity, bi = 0; for (let k = 0; k < S.c; k++) { const d = d2(B, w, B, order[k]); if (d < b) { b = d; bi = order[k]; } }
      it.push({ type: 'line', pts: [WB[w], WB[bi]], color: '--red', width: 2.6 }, { type: 'points', pts: [WB[w]], color: '--fg', size: 5 },
        { type: 'label', at: WB[w], text: `coverage radius ${rad[S.c].toFixed(2)}`, color: '--red', dx: 8, dy: 18 }); }
    if (!S.anim) { const R = scored(M.test, S.c), t = Wp(tf[M.test], R.worst);
      it.push({ type: 'line', pts: [t, WB[R.fullNN]], color: '--blue', width: 1.6, dash: [4, 3] }, { type: 'line', pts: [t, WB[R.coreNN]], color: '--red', width: 2.6 },
        { type: 'points', pts: [t], color: '--yellow', size: 6.5 }, { type: 'label', at: t, text: 'worst test patch', color: '--yellow', dx: 10, dy: -12 }); }
    return it;
  }
  const render3 = () => { s3.items = items(); s3.redraw(); };
  const sv = PE.canvas(side, () => narrow() ? 1.25 : .4, (c, W, H) => {
    if (!W) return; const C = COL(), nr = narrow(); c.clearRect(0, 0, W, H);
    const R = S.anim ? null : scored(M.test, S.c), isz = nr ? W - 16 : H - 44, ix = 8, iy = 20;
    if (R && M.test + S.c + C.red !== heatKey) { heat = TOY.heatBitmap(R.map, .3, 1.1, C.red); heatKey = M.test + S.c + C.red; }
    TOY.draw(c, tests[M.test], ix, iy, isz, { heat: R ? heat : null, vf: false });
    PE.label(c, 'camera view', ix, 13, C.muted, 10);
    if (R) { const sz = isz / TOY.N, w = R.worst; c.save(); c.strokeStyle = C.yellow; c.lineWidth = 2; c.strokeRect(ix + (w % TOY.G) * TOY.S * sz, iy + Math.floor(w / TOY.G) * TOY.S * sz, TOY.P * sz, TOY.P * sz); c.restore();
      PE.label(c, `image score ${R.score.toFixed(2)}`, ix, iy + isz + 16, C.fg, 11); }
    if (nr) return;
    const bx = { x: ix + isz + 50, y: 14, w: W - (ix + isz + 50) - 12, h: H - 50 }, ymax = 1.6;
    const A = PE.axes(c, bx, [.8, 100], [0, ymax], { logx: true, xt: [1, 10, 25, 100], yt: [0, .5, 1, 1.5], xf: v => v + '%', yf: v => v.toFixed(1), xl: 'share kept (log)', yl: 'radius', ylo: 30 });
    const pts = []; for (let k = 0; k <= 120; k++) { const pct = .8 * Math.pow(125, k / 120), l = Math.min(n - 1, Math.max(1, Math.floor(n * pct / 100))); pts.push([A.X(pct), A.Y(Math.min(ymax, rad[l]))]); }
    PE.polyline(c, pts, C.blue, 2);
    PE.polyline(c, [[bx.x, A.Y(scratchRef)], [bx.x + bx.w, A.Y(scratchRef)]], C.yellow, 1.4, [5, 4]);
    PE.label(c, 'scratch distance, full bank', bx.x + bx.w - 2, A.Y(scratchRef) - 5, C.yellow, 10, 'right');
    const pc = Math.max(.8, 100 * S.c / n), rv = S.c >= n ? 0 : rad[S.c]; PE.dot(c, A.X(pc), A.Y(Math.min(ymax, rv)), 5, C.red);
  });
  function say() {
    const R = scored(M.test, S.c), g = scored('good', S.c), rv = S.c >= n ? 0 : rad[S.c], pct = Math.round(100 * S.c / n);
    let t = `PatchCore keeping ${pct}% of the toy bank: ${S.c.toLocaleString('en-US')} of ${n.toLocaleString('en-US')} patches, coverage radius ${rv.toFixed(2)}. `;
    t += `The ${TESTNAME[M.test]}'s worst patch is ${R.score.toFixed(2)} from its nearest kept patch (${R.fullD.toFixed(2)} from the full bank)`;
    t += M.test === 'good' ? '.' : (R.onDefect ? ', on the defect.' : ', not on the defect: a coverage gap or a bolt hole scores higher.');
    if (M.test !== 'good') t += ` Good cap ${g.score.toFixed(2)} against ${R.score.toFixed(2)}: <b>${R.score > g.score + .05 ? 'separated' : 'not separated'}</b>.`;
    ctx.hud(badge('toy', t));
  }
  function settle() { S.anim = false; render3(); sv.redraw(); say(); }
  function setKeep() { stop(); const from = S.c, to = lOf(M.keep); if (from === to) { settle(); return; } S.anim = true;
    stop = PE.animate(600, u => { S.c = Math.round(from + (to - from) * u); render3(); sv.redraw(); }, settle); }
  let lastW = ctx.stage.clientWidth;
  settle();
  return {
    resize() { const w = ctx.stage.clientWidth; side.style.width = sideW() + 'px'; sv.resize(); if (w !== lastW) { lastW = w; render3(); } else el.style.height = ctx.stage.clientHeight + 'px'; },
    update(what) { if (what === 'keep' || what === 'reset') setKeep(); if (what === 'test') settle(); if (what === 'reset') s3.setView({ yaw: .75, pitch: .42, dist: narrow() ? 9 : 7.4 }, 600); },
    replay() { stop(); const to = lOf(M.keep); S.anim = true; S.c = 0; stop = PE.animate(2600, u => { S.c = Math.max(1, Math.round(to * u)); render3(); sv.redraw(); }, () => { S.c = to; settle(); }); },
    destroy() { stop(); sv.destroy(); s3.destroy(); } };
}

SCENES.memory = ctx => {
  const tb = ctx.qa('[data-test]'), sb = ctx.qa('[data-spade]'), db = ctx.qa('[data-dist]'), kb = ctx.qa('[data-keep]'), pb = ctx.q('[data-probe]');
  const DEF = { test: 'scratch', spade: 'pixel', dist: 'maha', probe: false, keep: 10 };
  const M = { test: (onKey(tb) || tb[0]).dataset.test, spade: (onKey(sb) || sb[1]).dataset.spade, dist: (onKey(db) || db[0]).dataset.dist, probe: pb.getAttribute('aria-pressed') === 'true', keep: +((onKey(kb) || kb[1]).dataset.keep) };
  const makers = [spadeScene, padimScene, coreScene];
  let cur = -1, sub = null;
  function show(n) { if (n === cur && sub) return; if (sub) sub.destroy(); ctx.stage.replaceChildren(); cur = n; document.body.dataset.flat = n === 0 ? '1' : '0'; sub = makers[n](ctx, M); }
  const sync = () => { press(tb, b => b.dataset.test === M.test); press(sb, b => b.dataset.spade === M.spade); press(db, b => b.dataset.dist === M.dist); press([pb], () => M.probe); press(kb, b => +b.dataset.keep === M.keep); };
  // A control that belongs to another tab switches to that tab first; the new sub-scene reads M when it mounts.
  const act = (n, what) => { sync(); if (cur !== n) { const t = ctx.q(`[role="tab"][data-step="${n}"]`); if (t) t.click(); } else sub.update(what); };
  const offs = [...tb.map(b => listen(b, 'click', () => { M.test = b.dataset.test; M.probe = false; sync(); sub && sub.update('test'); })),
    ...sb.map(b => listen(b, 'click', () => { M.spade = b.dataset.spade; act(0, 'spade'); })),
    ...db.map(b => listen(b, 'click', () => { M.dist = b.dataset.dist; act(1, 'dist'); })),
    listen(pb, 'click', () => { M.probe = true; act(1, 'probe'); }),
    ...kb.map(b => listen(b, 'click', () => { M.keep = +b.dataset.keep; act(2, 'keep'); })),
    listen(ctx.q('[data-act="reset"]'), 'click', () => { Object.assign(M, DEF); sync(); sub && sub.update('reset'); })];
  return { step: show, replay: () => sub && sub.replay(), resize: () => sub && sub.resize ? sub.resize() : null,
    destroy() { offs.forEach(f => f()); if (sub) sub.destroy(); } };
};

/* 4. Flows: the exact toy 2D flow (FASTFLOW_KIT) as a density surface (port of parts/viz-fastflow.js). Checked there
   (build/fastflow-sim.mjs): log pX(x) of each test point is identical at every stage (anomaly -7.5936, normal -0.9841); the
   anomaly ends at z = (0.74, 3.50), with 0.25% of training samples scoring lower. */
const FFLOW = lazy(() => {
  const F = FASTFLOW_KIT, train = F.sample(260, 7), ref = F.sample(2000, 11).map(x => F.logDensity(x, 0));
  const peakIdx = train.reduce((b, x, i) => F.logDensity(x, 0) > F.logDensity(train[b], 0) ? i : b, 0);
  return { train, ref, peakIdx };
});
SCENES.flows = ctx => {
  const F = FASTFLOW_KIT, { train, ref, peakIdx } = FFLOW(), el = mkCanvas(ctx.stage, 'pe3'), narrow = ctx.stage.clientWidth < 620;
  const lb = ctx.qa('[data-layer]'), fb = ctx.qa('[data-ff]');
  const EXT = 3.6, SC = 3 / EXT, HS = 5.5;
  const S = { stage: +((onKey(lb) || lb[0]).dataset.layer), test: (onKey(fb) || fb[0]).dataset.ff };
  S.target = S.stage;
  const circle = c => Array.from({ length: 25 }, (_, i) => [c[0] + .2 * Math.cos(i * Math.PI / 12), c[1] + .2 * Math.sin(i * Math.PI / 12)]);
  const W3 = h => [h[0] * SC, 0, -h[1] * SC];
  let stopT = () => {};
  const s3 = PE3.scene(el, { aspect: S3ASPECT(ctx.stage), fov: 36, dist: narrow ? 13 : 10.5, yaw: .55, pitch: .62, items: [] });
  function items() {
    const blue = PE.css('--blue'), panel = PE.css('--panel'), st = S.stage, ht = h => HS * Math.exp(F.logDensity(h, st));
    let ymax = 0; for (let i = 0; i <= 34; i++) for (let j = 0; j <= 34; j++) ymax = Math.max(ymax, ht([(-3 + 6 * i / 34) / SC, -(-3 + 6 * j / 34) / SC]));
    const surf = PE3.surface((x, z) => ht([x / SC, -z / SC]), { x0: -3, x1: 3, z0: -3, z1: 3, nx: 34, nz: 34, fill: y => PE3.mix(panel, blue, Math.min(1, .1 + .9 * y / ymax)) });
    const pts = train.map(x => { const h = F.push(x, st).h; return [h[0] * SC, ht(h) + .03, -h[1] * SC]; });
    const x0 = F.TEST[S.test], h = F.push(x0, st).h, top = [h[0] * SC, ht(h) + .03, -h[1] * SC], base = W3(h);
    return [{ type: 'grid', y: 0, size: 6, step: 1, color: '--rule' }, ...surf,
      { type: 'points', pts, color: '--fg', size: 2, alpha: .7 },
      { type: 'line', pts: circle(x0).map(c => W3(F.push(c, st).h)), color: '--yellow', width: 1.6 },
      { type: 'line', pts: [base, top], color: '--yellow', width: 1.4, dash: [3, 3] },
      { type: 'points', pts: [top], color: '--yellow', size: 7 },
      { type: 'label', at: top, text: S.test === 'anomaly' ? 'feature from a scratch' : 'feature from a good part', color: '--yellow', dy: -12, align: 'center' },
      { type: 'label', at: W3(F.push(train[peakIdx], st).h), text: 'normal features', color: '--blue', dy: 22, align: 'center' },
      { type: 'label', at: [0, 0, 3.2], text: `${Math.round(S.stage * 10) / 10} of ${F.K} coupling layers applied`, color: '--muted', dy: 18, align: 'center' }];
  }
  const redraw = () => { s3.items = items(); s3.redraw(); };
  function say() {
    const x0 = F.TEST[S.test], k = Math.round(S.stage), p = F.push(x0, k), here = F.logDensity(p.h, k), tot = here + p.logdet;
    const lower = ref.filter(v => v < tot).length / ref.length * 100, who = S.test === 'anomaly' ? 'Scratch feature' : 'Good-part feature', f = v => (v >= 0 ? '+ ' : '- ') + Math.abs(v).toFixed(2);
    ctx.hud(badge('toy', k === F.K
      ? `All ${F.K} layers applied: normal is now a standard Gaussian. ${who}: log N(z) ${here.toFixed(2)} ${f(p.logdet)} log-det = log-likelihood <b>${tot.toFixed(2)}</b>. ${lower.toFixed(lower < 1 ? 2 : 0)}% of toy normal features score lower.`
      : `${k} of ${F.K} layers applied. ${who}: log density here ${here.toFixed(2)} ${f(p.logdet)} log-det so far = <b>${tot.toFixed(2)}</b>. The total is the same at every stage; only the split changes.`));
  }
  const pressAll = () => { press(lb, b => +b.dataset.layer === S.target); press(fb, b => b.dataset.ff === S.test); };
  function goTo(k, from) {
    stopT(); const a = from ?? S.stage; S.target = k; pressAll();
    if (a === k) { S.stage = k; redraw(); say(); return; }
    stopT = s3.tween(Math.min(2600, 650 * Math.abs(k - a)), u => { S.stage = a + (k - a) * u; redraw(); }, () => { S.stage = k; redraw(); say(); });
  }
  const offs = [...lb.map(b => listen(b, 'click', () => goTo(+b.dataset.layer))), ...fb.map(b => listen(b, 'click', () => { S.test = b.dataset.ff; pressAll(); redraw(); say(); })),
    listen(ctx.q('[data-act="reset"]'), 'click', () => { stopT(); S.stage = 0; S.target = 0; S.test = 'anomaly'; pressAll(); s3.setView({ yaw: .55, pitch: .62, dist: narrow ? 13 : 10.5 }, 500); redraw(); say(); }),
    PE.onTheme(redraw)];
  redraw(); say(); pressAll();
  return { replay: () => { S.stage = 0; goTo(F.K, 0); }, destroy() { stopT(); offs.forEach(f => f()); s3.destroy(); } };
};

/* 5. Synthetic defects: CutPaste and DRAEM training samples from TOY images (port of parts/viz-synth.js). Checked there
   (build/synth-sim.mjs): Perlin threshold 0.70 keeps 9-21% of the image (seeds 1-6), 0.60 keeps 22-41%. */
const SYNTH_BASE = lazy(() => ({ mesh: TOY.make('mesh', 41), cap: TOY.make('cap', 41) }));
SCENES.synthetic = ctx => {
  const K = SYNTH_KIT, N = K.N, base = SYNTH_BASE(), el = mkCanvas(ctx.stage);
  el.tabIndex = 0; el.setAttribute('aria-label', 'Fake defects made from a good toy image. In CutPaste, drag the dashed source or the yellow target rectangle; arrow keys move the target.');
  const pb = ctx.qa('[data-product]'), cb = ctx.qa('[data-cp]'), area = ctx.q('#cp-area'), aspect = ctx.q('#cp-aspect'), thr = ctx.q('#dr-thr'), beta = ctx.q('#dr-beta');
  const out = id => ctx.q('#' + id + '-value');
  const S = { mode: 'cutpaste', product: (onKey(pb) || pb[0]).dataset.product, scar: (onKey(cb) || cb[0]).dataset.cp === 'scar', src: [24, 30], dst: [70, 66],
    area: +area.value / 100, aspect: +aspect.value, seed: 3, thr: +thr.value, beta: +beta.value, fade: 1, rw: 0, rh: 0, ang: 0 };
  const rect = () => S.scar ? { w: 4, h: 12, ang: .5 } : { w: Math.min(N - 4, Math.sqrt(S.area * N * N * S.aspect)), h: Math.min(N - 4, Math.sqrt(S.area * N * N / S.aspect)), ang: 0 };
  const cl = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
  let cache = {}, L = null, cancel = () => {};
  function data() {
    const key = [S.mode, S.product, S.src, S.dst, S.rw.toFixed(2), S.rh.toFixed(2), S.ang.toFixed(2), S.seed, S.thr, S.beta].join('|');
    if (cache.key === key) return cache; const I = base[S.product].px;
    if (S.mode === 'cutpaste') { const r = K.cutpaste(I, S.src, S.dst, S.rw, S.rh, S.ang, .08); cache = { key, I: { px: I }, Ia: { px: r.img }, mask: r.mask }; }
    else { const m = K.perlinMask(S.seed, S.thr), A = K.texture(S.seed); cache = { key, I: { px: I }, A: { px: A }, mask: m, Ia: { px: K.draem(I, m, A, S.beta) } }; }
    return cache;
  }
  const cv = PE.canvas(el, S3ASPECT(ctx.stage), (c, W, H) => draw(c, W, H));
  function layout(W, H) {
    const narrow = W < 620, n = S.mode === 'draem' ? 4 : 3, gap = narrow ? 16 : 28, top = 34, bot = narrow ? 16 : HUD_H;
    const s = Math.min((W - gap * 3) / 2, (H - bot - 2 * top - 18) / 2), x0 = (W - 2 * s - gap) / 2;
    const P = []; for (let i = 0; i < n; i++) P.push({ x: x0 + (i % 2) * (s + gap), y: top + Math.floor(i / 2) * (s + top + 6), s });
    return { P, s, narrow };
  }
  function rectPath(c, p, o, w, h, ang) {
    const k = p.s / N, cx = p.x + (o[0] + w / 2) * k, cy = p.y + (o[1] + h / 2) * k, co = Math.cos(ang), si = Math.sin(ang);
    c.beginPath(); [[-1, -1], [1, -1], [1, 1], [-1, 1]].forEach(([a, b], i) => { const u = a * w / 2 * k, v = b * h / 2 * k, X = cx + co * u - si * v, Y = cy + si * u + co * v; i ? c.lineTo(X, Y) : c.moveTo(X, Y); }); c.closePath();
  }
  function draw(c, W, H) {
    if (!W) return; L = layout(W, H); const D = data(), C = COL();
    c.clearRect(0, 0, W, H); c.globalAlpha = S.fade;
    const P = L.P, lab = (i, t, col) => PE.label(c, t, P[i].x, P[i].y - 9, col || C.muted, 12);
    const maskImg = m => { const f = new Float32Array(N * N); for (let k = 0; k < N * N; k++) f[k] = m[k]; return TOY.heatBitmap(f, 0, 1, C.red); };
    if (S.mode === 'cutpaste') {
      TOY.draw(c, D.I, P[0].x, P[0].y, P[0].s); lab(0, 'good image x', C.blue);
      c.save(); c.setLineDash([5, 4]); c.strokeStyle = C.blue; c.lineWidth = 2; rectPath(c, P[0], S.src, S.rw, S.rh, S.ang); c.stroke(); c.restore();
      TOY.draw(c, D.Ia, P[1].x, P[1].y, P[1].s); lab(1, 'CutPaste(x): the fake', C.yellow);
      c.save(); c.strokeStyle = C.yellow; c.lineWidth = 2; rectPath(c, P[1], S.dst, S.rw, S.rh, S.ang); c.stroke(); c.restore();
      const p = P[2]; c.save(); c.strokeStyle = C.rule; c.strokeRect(p.x, p.y, p.s, p.s); c.restore(); lab(2, 'what the classifier learns');
      const fs = Math.max(11, Math.min(16, p.s / 12)), lx = p.x + 14; let ly = p.y + fs * 2.2;
      PE.label(c, 'x', lx, ly, C.blue, fs); PE.label(c, 'class 0: normal', lx + fs * 2.6, ly, C.fg, fs); ly += fs * 2;
      PE.label(c, 'CutPaste(x)', lx, ly, C.yellow, fs); ly += fs * 1.4; PE.label(c, 'class 1: pasted', lx + fs * 2.6, ly, C.fg, fs); ly += fs * 2.2;
      c.save(); c.font = `${fs - 2}px system-ui, sans-serif`; c.fillStyle = C.muted;
      ['No mask: the classifier only', 'learns which image was edited.', 'Its features are then scored', 'with one Gaussian.'].forEach((t, i) => c.fillText(t, lx, ly + i * (fs + 3))); c.restore();
    } else {
      TOY.draw(c, D.I, P[0].x, P[0].y, P[0].s); lab(0, 'good image I', C.blue);
      TOY.draw(c, D.A, P[1].x, P[1].y, P[1].s, { vf: false }); lab(1, 'foreign texture A', C.brown);
      c.save(); c.fillStyle = C.bg; c.fillRect(P[2].x, P[2].y, P[2].s, P[2].s); c.drawImage(maskImg(D.mask), P[2].x, P[2].y, P[2].s, P[2].s); c.strokeStyle = C.rule; c.strokeRect(P[2].x, P[2].y, P[2].s, P[2].s); c.restore(); lab(2, 'mask M from Perlin noise', C.red);
      TOY.draw(c, D.Ia, P[3].x, P[3].y, P[3].s); lab(3, 'fake anomaly I_a', C.yellow);
    }
    c.globalAlpha = 1;
  }
  function say() {
    const D = data(), pct = (K.frac(D.mask) * 100).toFixed(0);
    ctx.hud(badge('toy', S.mode === 'cutpaste'
      ? `CutPaste${S.scar ? ' (scar)' : ''}: a ${Math.round(S.rw)} by ${Math.round(S.rh)} pixel ${S.scar ? 'strip, rotated,' : 'patch'} (${pct}% of the toy image) copied from one place in the same image to another, slightly brightened. The classifier learns class 0 for the original and class 1 for the edited copy.`
      : `DRAEM: the threshold keeps ${pct}% of the Perlin noise as the mask; inside it the toy image is ${Math.round(S.beta * 100)}% foreign texture and ${Math.round((1 - S.beta) * 100)}% original (beta ${S.beta.toFixed(2)}). Targets: the clean image I for the repair, the mask M for the segmentation.`));
  }
  function sync() {
    press(pb, b => b.dataset.product === S.product); press(cb, b => (b.dataset.cp === 'scar') === S.scar);
    out('cp-area').textContent = Math.round(S.area * 100) + '%'; out('cp-aspect').textContent = S.aspect.toFixed(1); out('dr-thr').textContent = S.thr.toFixed(2); out('dr-beta').textContent = S.beta.toFixed(2);
  }
  function clampRects() { const m = S.ang ? 6 : 0; S.src = [cl(S.src[0], m, N - S.rw - m), cl(S.src[1], m, N - S.rh - m)]; S.dst = [cl(S.dst[0], m, N - S.rw - m), cl(S.dst[1], m, N - S.rh - m)]; }
  function morphRect() { cancel(); const a = { w: S.rw, h: S.rh, ang: S.ang }, b = rect();
    cancel = PE.animate(450, u => { S.rw = PE.lerp(a.w, b.w, u); S.rh = PE.lerp(a.h, b.h, u); S.ang = PE.lerp(a.ang, b.ang, u); clampRects(); cv.redraw(); }, () => { Object.assign(S, { rw: b.w, rh: b.h, ang: b.ang }); clampRects(); cv.redraw(); say(); }); }
  function fadeIn() { cancel(); cancel = PE.animate(400, u => { S.fade = u; cv.redraw(); }, () => { S.fade = 1; cv.redraw(); }); }
  const toStep = n => { const t = ctx.q(`[role="tab"][data-step="${n}"]`); if (t) t.click(); };
  const need = (m, fn) => { if (S.mode !== m) { fn(); toStep(m === 'cutpaste' ? 0 : 1); } else fn(); };
  const setRect = () => { S.scar = false; const r = rect(); Object.assign(S, { rw: r.w, rh: r.h, ang: 0 }); clampRects(); sync(); cv.redraw(); say(); };
  const offs = [...pb.map(b => listen(b, 'click', () => { S.product = b.dataset.product; sync(); fadeIn(); say(); })),
    ...cb.map(b => listen(b, 'click', () => need('cutpaste', () => { S.scar = b.dataset.cp === 'scar'; sync(); morphRect(); }))),
    listen(ctx.q('[data-noise]'), 'click', () => need('draem', () => { S.seed++; fadeIn(); say(); })),
    listen(area, 'input', () => { S.area = area.value / 100; setRect(); }), listen(aspect, 'input', () => { S.aspect = +aspect.value; setRect(); }),
    listen(thr, 'input', () => { S.thr = +thr.value; sync(); cv.redraw(); say(); }), listen(beta, 'input', () => { S.beta = +beta.value; sync(); cv.redraw(); say(); }),
    listen(ctx.q('[data-act="reset"]'), 'click', () => { Object.assign(S, { product: 'mesh', scar: false, src: [24, 30], dst: [70, 66], area: .06, aspect: 1.6, seed: 3, thr: .72, beta: .6 });
      area.value = 6; aspect.value = 1.6; thr.value = .72; beta.value = .6; const r = rect(); Object.assign(S, { rw: r.w, rh: r.h, ang: 0 }); sync(); fadeIn(); say(); })];
  let drag = null;
  const hit = e => { if (!L || S.mode !== 'cutpaste') return null; const r = el.getBoundingClientRect(), x = e.clientX - r.left, y = e.clientY - r.top;
    for (const i of [0, 1]) { const p = L.P[i]; if (x >= p.x && x <= p.x + p.s && y >= p.y && y <= p.y + p.s) return { which: i ? 'dst' : 'src', u: (x - p.x) / p.s * N, v: (y - p.y) / p.s * N }; } return null; };
  el.addEventListener('pointerdown', e => { const h = hit(e); if (!h) return; drag = h.which; el.setPointerCapture(e.pointerId); S[drag] = [h.u - S.rw / 2, h.v - S.rh / 2]; clampRects(); cv.redraw(); });
  el.addEventListener('pointermove', e => { if (!drag) return; const h = hit(e); if (!h || h.which !== drag) return; S[drag] = [h.u - S.rw / 2, h.v - S.rh / 2]; clampRects(); cv.redraw(); });
  el.addEventListener('pointerup', () => { if (drag) { drag = null; say(); } });
  el.addEventListener('keydown', e => { const d = { ArrowLeft: [-3, 0], ArrowRight: [3, 0], ArrowUp: [0, -3], ArrowDown: [0, 3] }[e.key]; if (!d || S.mode !== 'cutpaste') return;
    e.preventDefault(); const k = e.shiftKey ? 'src' : 'dst'; S[k] = [S[k][0] + d[0], S[k][1] + d[1]]; clampRects(); cv.redraw(); say(); });
  { const r = rect(); Object.assign(S, { rw: r.w, rh: r.h, ang: r.ang }); clampRects(); }
  sync();
  return { flat: true, resize: cv.resize, step(n) { S.mode = n === 1 ? 'draem' : 'cutpaste'; cache = {}; fadeIn(); say(); }, destroy() { cancel(); offs.forEach(f => f()); cv.destroy(); } };
};

/* 6. Student-teacher: toy EfficientAD on the toy cap (port of parts/viz-efficientad.js). Checked there
   (build/toy-efficientad.mjs, 40 training caps, 3 neighbours, 8 components): displayed good cap 500 (image score 0.02),
   scratch 200 (local 0.43, global 0.09), extra hole on cap 201 (local 0.08, global 0.40). */
const EAD = lazy(() => {
  EAD_KIT.fit();
  const CASES = { good: TOY.make('cap', 500), scratch: TOY.defect(TOY.make('cap', 200), 'scratch', 200), extra: EAD_KIT.extraHole(TOY.make('cap', 201), 1) };
  const R = {}; for (const k in CASES) R[k] = EAD_KIT.score(CASES[k]);
  return { IMG: CASES, R };
});
SCENES.studentteacher = ctx => {
  const { IMG, R } = EAD(), el = mkCanvas(ctx.stage), cb = ctx.qa('[data-case]'), slider = ctx.q('#ead-ph'), sOut = ctx.q('#ead-ph-value');
  const PH = [0, .9, .99, .999, .9999, .99999], PHS = ['0', '0.9', '0.99', '0.999', '0.9999', '0.99999'], HI = .3, THR = .1;
  const S = { c: (onKey(cb) || cb[0]).dataset.case, ph: +slider.value, fade: 1 };
  const HB = {}, heat = (k, which, red) => HB[k + which + red] || (HB[k + which + red] = TOY.heatBitmap(TOY.toPixels(Float32Array.from(R[k][which])), 0, HI, red));
  let cancel = () => {};
  const cv = PE.canvas(el, S3ASPECT(ctx.stage), (c, W, H) => draw(c, W, H));
  function layout(W, H) {
    const narrow = W < 620, g = narrow ? 14 : 28, bot = narrow ? 30 : HUD_H, s = Math.min((W - 3 * g) / 2, (H - bot - 40 - 62 - 40) / 2), x0 = (W - 2 * s - g) / 2;
    return { s, at: i => ({ x: x0 + (i % 2) * (s + g), y: 40 + Math.floor(i / 2) * (s + 62) }) };
  }
  function draw(c, W, H) {
    if (!W) return; const C = COL(), L = layout(W, H), r = R[S.c], img = IMG[S.c]; c.clearRect(0, 0, W, H);
    const nr = W < 620, panels = [{ title: 'camera view' }, { title: nr ? 'local map' : 'student vs teacher (local)', map: 'nst', peak: r.maxSt }, { title: nr ? 'global map' : 'autoencoder vs student (global)', map: 'nae', peak: r.maxAe }, { title: nr ? 'combined' : 'combined map', map: 'comb', peak: r.image }];
    panels.forEach((p, i) => { const a = L.at(i);
      PE.label(c, p.title, a.x, a.y - 9, C.muted, 12);
      if (!p.map) TOY.draw(c, img, a.x, a.y, L.s, {});
      else { c.save(); c.globalAlpha = .35; c.imageSmoothingEnabled = false; c.drawImage(TOY.bitmap(img), a.x, a.y, L.s, L.s); c.restore();
        c.save(); c.globalAlpha = S.fade; c.drawImage(heat(S.c, p.map, C.red), a.x, a.y, L.s, L.s); c.restore();
        c.save(); c.strokeStyle = C.rule; c.strokeRect(a.x - .5, a.y - .5, L.s + 1, L.s + 1); c.restore();
        PE.label(c, `peak ${p.peak.toFixed(2)}`, a.x, a.y + L.s + 18, p.peak > THR ? C.red : C.muted, 12, 'left', 700); } });
    const m = EAD_KIT.hardMask(r, PH[S.ph]), a0 = L.at(0), ps = L.s / TOY.N;
    c.save(); c.strokeStyle = C.yellow; c.lineWidth = 1.2; c.globalAlpha = .9;
    for (let q = 0; q < 225; q++) if (m.keep[q]) c.strokeRect(a0.x + (q % 15) * 8 * ps, a0.y + Math.floor(q / 15) * 8 * ps, 16 * ps, 16 * ps); c.restore();
    PE.label(c, nr ? `${m.n} of 675 train` : `${m.n} of 675 toy elements train (p_hard ${PHS[S.ph]})`, a0.x, a0.y + L.s + 18, C.yellow, 12);
    const a3 = L.at(3), rej = r.image > THR; pill(c, a3.x + (nr ? 70 : 80), a3.y + L.s + 4, rej ? 'reject' : 'pass', rej ? C.red : C.blue);
  }
  function say() {
    const r = R[S.c], m = EAD_KIT.hardMask(r, PH[S.ph]);
    const lead = S.c === 'good' ? 'Good cap: both maps stay quiet' : S.c === 'scratch' ? 'Scratch: the student-versus-teacher map finds it; the autoencoder map barely moves' : 'Extra hole: the autoencoder map finds it; the student-versus-teacher map sees an ordinary hole';
    ctx.hud(badge('toy', `${lead}. Peaks ${r.maxSt.toFixed(2)} (local) and ${r.maxAe.toFixed(2)} (global); image score ${r.image.toFixed(2)}, ${r.image > THR ? 'over' : 'under'} the 0.1 level: <b>${r.image > THR ? 'reject' : 'pass'}</b>. At p_hard ${PHS[S.ph]} the hard feature loss would train on ${m.n} of the 675 toy feature elements (yellow).`));
  }
  const markCol = () => ctx.qa('#ead-table td[data-ph]').forEach(td => { const on = +td.dataset.ph === S.ph; td.style.color = on ? 'var(--yellow)' : ''; td.style.fontWeight = on ? '700' : ''; });
  function setCase(k) { const changed = k !== S.c; S.c = k; press(cb, b => b.dataset.case === k); cancel(); if (changed) cancel = PE.animate(500, u => { S.fade = u; cv.redraw(); }, () => { S.fade = 1; cv.redraw(); }); else cv.redraw(); say(); }
  function setPh(i) { S.ph = i; slider.value = i; sOut.textContent = PHS[i]; markCol(); cv.redraw(); say(); }
  const offs = [...cb.map(b => listen(b, 'click', () => setCase(b.dataset.case))), listen(slider, 'input', () => setPh(+slider.value)),
    listen(ctx.q('[data-act="reset"]'), 'click', () => { setPh(3); setCase('good'); })];
  markCol(); say();
  return { flat: true, resize: cv.resize, destroy() { cancel(); offs.forEach(f => f()); ctx.qa('#ead-table td[data-ph]').forEach(td => { td.style.color = ''; td.style.fontWeight = ''; }); cv.destroy(); } };
};

/* 7. One model for all: one attention layer rebuilds the 15 x 15 tokens of a scratched toy mesh (port of parts/viz-uniad.js).
   Checked there (build/toy-uniad2.mjs, tau 0.08, 7 x 7 window): full attention leaves the scratch error at 0.12-0.33, like
   normal texture (up to 0.39); the neighbour mask gives 1.42-1.48 against at most 0.73 elsewhere; the learned query
   1.44-1.52 against 0.74. */
const UNIAD = lazy(() => {
  const G = UNIAD_KIT.G, NP = G * G, img = TOY.defect(TOY.make('mesh', 200), 'scratch', 200), x = UNIAD_KIT.feats(img), tau = UNIAD_KIT.TAU;
  const Q = new Float64Array(NP * 3); for (let s = 0; s < 20; s++) { const f = UNIAD_KIT.feats(TOY.make('mesh', 600 + s)); for (let i = 0; i < NP * 3; i++) Q[i] += f[i] / 20; }
  function attend(mode, i) {
    const q = mode === 'query' ? Q : x, w = new Float64Array(NP); let mx = -Infinity;
    for (let j = 0; j < NP; j++) { if (mode === 'mask' && UNIAD_KIT.masked(i, j)) { w[j] = -Infinity; continue; }
      let d = 0; for (let c = 0; c < 3; c++) d += (q[i * 3 + c] - x[j * 3 + c]) ** 2; w[j] = -d / tau; if (w[j] > mx) mx = w[j]; }
    let z = 0; for (let j = 0; j < NP; j++) { w[j] = w[j] === -Infinity ? 0 : Math.exp(w[j] - mx); z += w[j]; } for (let j = 0; j < NP; j++) w[j] /= z; return w;
  }
  const R = {};
  ['full', 'mask', 'query'].forEach(m => { const err = new Float64Array(NP); for (let i = 0; i < NP; i++) { const w = attend(m, i); let e = 0;
    for (let c = 0; c < 3; c++) { let y = 0; for (let j = 0; j < NP; j++) y += w[j] * x[j * 3 + c]; e += (x[i * 3 + c] - y) ** 2; } err[i] = Math.sqrt(e); } R[m] = err; });
  const inScratch = new Uint8Array(NP); for (let q = 0; q < NP; q++) { const i = q % G, j = Math.floor(q / G); let n = 0; for (let y = j * 8; y < j * 8 + 16; y++) for (let xx = i * 8; xx < i * 8 + 16; xx++) n += img.mask[y * 128 + xx]; inScratch[q] = n > 20 ? 1 : 0; }
  let seed = 0; for (let q = 0; q < NP; q++) if (inScratch[q] && R.mask[q] > (inScratch[seed] ? R.mask[seed] : -1)) seed = q;
  const stats = m => { let a = 0, b = 0; for (let q = 0; q < NP; q++) inScratch[q] ? (a = Math.max(a, R[m][q])) : (b = Math.max(b, R[m][q])); return [a, b]; };
  return { G, NP, img, attend, R, inScratch, seed, stats };
});
SCENES.multiclass = ctx => {
  const D = UNIAD(), { G, NP, img, attend, R, inScratch, stats } = D, el = mkCanvas(ctx.stage), mb = ctx.qa('[data-mode]');
  el.tabIndex = 0; el.setAttribute('aria-label', 'Left: the toy mesh with the attention of the selected token. Right: the reconstruction error map. Click a token, or use the arrow keys.');
  const NAMES = { full: 'Full attention', mask: 'Neighbor mask', query: 'Learned query' }, Rw = UNIAD_KIT.R;
  const S = { mode: (onKey(mb) || mb[0]).dataset.mode, sel: D.seed, fade: 1 }; S.prev = S.mode;
  const HB = {}, heat = (m, red) => HB[m + red] || (HB[m + red] = TOY.heatBitmap(TOY.toPixels(Float32Array.from(R[m])), .1, 1.5, red));
  let L = null, cancel = () => {};
  const cv = PE.canvas(el, S3ASPECT(ctx.stage), (c, W, H) => draw(c, W, H));
  function layout(W, H) { const narrow = W < 620, g = narrow ? 14 : 30, bot = narrow ? 40 : HUD_H, s = Math.min((W - 3 * g) / 2, H - bot - 90), x0 = (W - 2 * s - g) / 2, y0 = narrow ? 40 : Math.max(50, (H - bot - s) / 2);
    return { narrow, s, a: { x: x0, y: y0 }, b: { x: x0 + s + g, y: y0 } }; }
  function draw(c, W, H) {
    if (!W) return; L = layout(W, H); const C = COL(), { a, b, s } = L, ps = s / TOY.N; c.clearRect(0, 0, W, H);
    const tokXY = q => [a.x + (q % G) * TOY.S * ps, a.y + Math.floor(q / G) * TOY.S * ps];
    PE.label(c, L.narrow ? 'attention of the token' : `where the selected token looks (${NAMES[S.mode].toLowerCase()})`, a.x, a.y - 10, C.muted, 12);
    TOY.draw(c, img, a.x, a.y, s, {});
    if (S.mode === 'mask') { const i = S.sel % G, j = Math.floor(S.sel / G), x0 = Math.max(0, i - Rw), y0 = Math.max(0, j - Rw), x1 = Math.min(G - 1, i + Rw), y1 = Math.min(G - 1, j + Rw);
      const X0 = a.x + x0 * TOY.S * ps, Y0 = a.y + y0 * TOY.S * ps, X1 = a.x + (x1 * TOY.S + TOY.P) * ps, Y1 = a.y + (y1 * TOY.S + TOY.P) * ps;
      c.save(); c.fillStyle = C.bg; c.globalAlpha = .62 * S.fade; c.fillRect(X0, Y0, X1 - X0, Y1 - Y0); c.globalAlpha = 1; c.strokeStyle = C.brown; c.lineWidth = 1.5; c.setLineDash([5, 3]); c.strokeRect(X0, Y0, X1 - X0, Y1 - Y0); c.restore(); }
    const w = attend(S.mode, S.sel), wm = Math.max(...w) || 1;
    c.save(); for (let q = 0; q < NP; q++) { const v = w[q] / wm; if (v < .04) continue; const [tx, ty] = tokXY(q);
      c.globalAlpha = (.25 + .7 * v) * S.fade; c.fillStyle = C.blue; const side = TOY.S * ps * .8, cx = tx + TOY.P * ps / 2, cy = ty + TOY.P * ps / 2; c.fillRect(cx - side / 2, cy - side / 2, side, side); } c.restore();
    const [sx, sy] = tokXY(S.sel); c.save(); c.strokeStyle = C.yellow; c.lineWidth = 2.5; c.strokeRect(sx, sy, TOY.P * ps, TOY.P * ps); c.restore();
    PE.label(c, 'reconstruction error', b.x, b.y - 10, C.muted, 12);
    c.save(); c.globalAlpha = .35; c.imageSmoothingEnabled = false; c.drawImage(TOY.bitmap(img), b.x, b.y, s, s); c.restore();
    if (S.fade < 1 && S.prev !== S.mode) { c.save(); c.globalAlpha = 1 - S.fade; c.drawImage(heat(S.prev, C.red), b.x, b.y, s, s); c.restore(); }
    c.save(); c.globalAlpha = S.fade; c.drawImage(heat(S.mode, C.red), b.x, b.y, s, s); c.restore();
    c.save(); c.strokeStyle = C.rule; c.strokeRect(b.x - .5, b.y - .5, s + 1, s + 1); c.restore();
    const [ins, outs] = stats(S.mode);
    PE.label(c, L.narrow ? `scratch ${ins.toFixed(2)}, elsewhere ${outs.toFixed(2)}` : `max error on the scratch ${ins.toFixed(2)}, elsewhere ${outs.toFixed(2)}`, b.x, b.y + s + 20, ins > 1.5 * outs ? C.red : C.muted, 12, 'left', 700);
    PE.label(c, `self weight ${(w[S.sel] * 100).toFixed(0)}%`, a.x, a.y + s + 20, w[S.sel] > .5 ? C.yellow : C.muted, 12, 'left', 700);
  }
  function say() {
    const w = attend(S.mode, S.sel), [ins, outs] = stats(S.mode), i = S.sel % G, j = Math.floor(S.sel / G);
    const why = S.mode === 'full' ? 'The rest goes to overlapping tokens that look almost the same, so the scratch is copied and the error map stays nearly blank: the identical shortcut.'
      : S.mode === 'mask' ? 'The token cannot see its own 7 by 7 window; normal texture finds look-alikes elsewhere, the scratch does not, so it lights up.'
      : 'Each position asks with the average good-mesh feature there, so the output looks like normal mesh; the scratch cannot be copied and lights up.';
    ctx.hud(badge('toy', `${NAMES[S.mode]}, token at column ${i + 1}, row ${j + 1}${inScratch[S.sel] ? ' (on the scratch)' : ''}: ${(w[S.sel] * 100).toFixed(0)}% of its weight on itself. ${why} Largest error on the scratch ${ins.toFixed(2)}, elsewhere ${outs.toFixed(2)}.`));
  }
  function setMode(m) { const prev = S.mode; press(mb, b => b.dataset.mode === m); if (prev === m) { cv.redraw(); say(); return; }
    cancel(); S.prev = prev; S.mode = m; cancel = PE.animate(600, u => { S.fade = u; cv.redraw(); }, () => { S.fade = 1; S.prev = m; cv.redraw(); }); say(); }
  const select = q => { S.sel = q; cv.redraw(); say(); };
  el.addEventListener('pointerdown', e => { if (!L) return; const r = el.getBoundingClientRect(), ps = L.s / TOY.N, i = Math.floor((e.clientX - r.left - L.a.x) / (TOY.S * ps)), j = Math.floor((e.clientY - r.top - L.a.y) / (TOY.S * ps)); if (i >= 0 && j >= 0 && i < G && j < G) select(j * G + i); });
  el.addEventListener('keydown', e => { const d = { ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowUp: [0, -1], ArrowDown: [0, 1] }[e.key]; if (!d) return; e.preventDefault(); select(PE.clamp(Math.floor(S.sel / G) + d[1], 0, G - 1) * G + PE.clamp(S.sel % G + d[0], 0, G - 1)); });
  const offs = [...mb.map(b => listen(b, 'click', () => setMode(b.dataset.mode))), listen(ctx.q('[data-act="reset"]'), 'click', () => { S.sel = D.seed; setMode('full'); cv.redraw(); say(); })];
  say();
  return { flat: true, resize: cv.resize, destroy() { cancel(); offs.forEach(f => f()); cv.destroy(); } };
};

/* Sourced data for chapters 8 and 9, generated from parts/viz-dinov2.js (AnomalyDINO v3 Tables 2-4: [shots, value]) and the
   speed-data block of parts/ch-speed.html (every value is a claim in notes/claims; src names the paper and table). */
const FEWSHOT = {"mvtec":{"img":{"dino448":[[1,"96.5"],[2,"96.7"],[4,"97.6"],[8,"98.0"],[16,"98.3"]],"dino672":[[1,"96.6"],[2,"96.9"],[4,"97.7"],[8,"98.2"],[16,"98.4"]],"winclip":[[1,"93.1"],[2,"94.4"],[4,"95.2"],[8,"94.6"],[16,"94.8"]],"aprilgan":[[1,"92.0"],[2,"92.4"],[4,"92.8"],[8,"93.1"],[16,"93.2"]],"patchcore":[[1,"83.4"],[2,"86.3"],[4,"88.8"]],"zs":[[0,"91.5","AnomalyCLIP"],[0,"91.8","WinCLIP"]]},"pix":{"dino448":[[1,"96.3"],[2,"96.5"],[4,"96.7"],[8,"97.0"],[16,"97.1"]],"dino672":[[1,"96.8"],[2,"97.0"],[4,"97.2"],[8,"97.4"],[16,"97.5"]],"winclip":[[1,"95.2"],[2,"96.0"],[4,"96.2"]],"aprilgan":[[1,"95.1"],[2,"95.5"],[4,"95.9"],[8,"96.2"],[16,"96.4"]],"patchcore":[[1,"92.0"],[2,"93.3"],[4,"94.3"]]},"pro":{"dino448":[[1,"91.7"],[2,"92.0"],[4,"92.4"],[8,"92.7"],[16,"92.9"]],"dino672":[[1,"92.7"],[2,"93.1"],[4,"93.4"],[8,"93.8"],[16,"94.0"]],"winclip":[[1,"87.1"],[2,"88.4"],[4,"89.0"]],"aprilgan":[[1,"90.6"],[2,"91.3"],[4,"91.8"],[8,"92.4"],[16,"92.6"]],"patchcore":[[1,"79.7"],[2,"82.3"],[4,"84.3"]]}},"visa":{"img":{"dino448":[[1,"85.6"],[2,"88.3"],[4,"91.3"],[8,"92.6"],[16,"93.8"]],"dino672":[[1,"87.4"],[2,"89.7"],[4,"92.6"],[8,"93.8"],[16,"94.8"]],"winclip":[[1,"83.8"],[2,"84.6"],[4,"87.3"],[8,"85.0"],[16,"85.0"]],"aprilgan":[[1,"91.2"],[2,"92.2"],[4,"92.6"],[8,"93.0"],[16,"93.2"]],"patchcore":[[1,"79.9"],[2,"81.6"],[4,"85.3"]],"zs":[[0,"82.1","AnomalyCLIP"],[0,"78.1","WinCLIP"]]},"pix":{"dino448":[[1,"97.5"],[2,"97.8"],[4,"98.0"],[8,"98.2"],[16,"98.3"]],"dino672":[[1,"97.8"],[2,"98.0"],[4,"98.2"],[8,"98.4"],[16,"98.5"]],"winclip":[[1,"96.4"],[2,"96.8"],[4,"97.2"]],"aprilgan":[[1,"96.0"],[2,"96.2"],[4,"96.2"],[8,"96.3"],[16,"96.3"]],"patchcore":[[1,"95.4"],[2,"96.1"],[4,"96.8"]]},"pro":{"dino448":[[1,"90.7"],[2,"91.7"],[4,"92.5"],[8,"93.3"],[16,"93.8"]],"dino672":[[1,"92.5"],[2,"93.4"],[4,"94.1"],[8,"94.8"],[16,"95.3"]],"winclip":[[1,"85.1"],[2,"86.2"],[4,"87.6"]],"aprilgan":[[1,"90.0"],[2,"90.1"],[4,"90.2"],[8,"90.2"],[16,"90.2"]],"patchcore":[[1,"80.5"],[2,"82.6"],[4,"84.9"]]}}};
const SPEED = [["desktop","s","EfficientAD-S","RTX A6000",2.2,"2.2 ms",98.8,"98.8","one-class","EfficientAD 2303.14535v3, Tables 1, 2 and 17 (float16, one image at a time; baselines re-run by the EfficientAD authors)",""],["desktop","m","EfficientAD-M","RTX A6000",4.5,"4.5 ms",99.1,"99.1","one-class","EfficientAD 2303.14535v3, Tables 1, 2 and 17 (float16, one image at a time; baselines re-run by the EfficientAD authors)",""],["desktop","gcad","GCAD","RTX A6000",10.7,"10.7 ms",89.1,"89.1","one-class","EfficientAD 2303.14535v3, Tables 1, 2 and 17 (float16, one image at a time; baselines re-run by the EfficientAD authors)",""],["desktop","simplenet","SimpleNet","RTX A6000",12.0,"12.0 ms",98.2,"98.2","one-class","EfficientAD 2303.14535v3, Tables 1, 2 and 17 (float16, one image at a time; baselines re-run by the EfficientAD authors)",""],["desktop","fastflow","FastFlow","RTX A6000",16.5,"16.5 ms",96.9,"96.9","one-class","EfficientAD 2303.14535v3, Tables 1, 2 and 17 (float16, one image at a time; baselines re-run by the EfficientAD authors)",""],["desktop","dsr","DSR","RTX A6000",17.2,"17.2 ms",98.1,"98.1","one-class","EfficientAD 2303.14535v3, Tables 1, 2 and 17 (float16, one image at a time; baselines re-run by the EfficientAD authors)",""],["desktop","patchcore","PatchCore","RTX A6000",32.0,"32.0 ms",98.7,"98.7","one-class","EfficientAD 2303.14535v3, Tables 1, 2 and 17 (float16, one image at a time; baselines re-run by the EfficientAD authors)",""],["desktop","ast","AST","RTX A6000",53.1,"53.1 ms",98.9,"98.9","one-class","EfficientAD 2303.14535v3, Tables 1, 2 and 17 (float16, one image at a time; baselines re-run by the EfficientAD authors)",""],["desktop","st","S-T","RTX A6000",74.7,"74.7 ms",93.2,"93.2","one-class","EfficientAD 2303.14535v3, Tables 1, 2 and 17 (float16, one image at a time; baselines re-run by the EfficientAD authors)",""],["desktop","patchcore-ens","PatchCore-Ens","RTX A6000",147.6,"147.6 ms",99.3,"99.3","one-class","EfficientAD 2303.14535v3, Tables 1, 2 and 17 (float16, one image at a time; baselines re-run by the EfficientAD authors)",""],["jetson","patchcore","PatchCore* (ResNet-18)","Jetson Orin Nano",119.0,"119.0 ms",92.5,"0.925","continual","2605.24251v2 Table 4 (latency) and Table 1 (accuracy, continual protocol)",""],["jetson","efficientad","EfficientAD*","Jetson Orin Nano",34.7,"34.7 ms",52.900000000000006,"0.529","continual","2605.24251v2 Table 4 (latency) and Table 1 (accuracy, continual protocol)",""],["jetson","dinosaur","DINOSaur (DINOv3 ViT-S/16)","Jetson Orin Nano",91.6,"91.6 ms",96.8,"0.968","continual","2605.24251v2 Table 4 (latency) and Table 1 (accuracy, continual protocol)",""],["jetson","dne","DNE","Jetson Orin Nano",60.7,"60.7 ms",50.0,"0.500","continual","2605.24251v2 Table 4 (latency) and Table 1 (accuracy, continual protocol)",""],["jetson","ucad","UCAD","Jetson Orin Nano",956.5,"956.5 ms",79.80000000000001,"0.798","continual","2605.24251v2 Table 4 (latency) and Table 1 (accuracy, continual protocol)",""],["jetson","edgezsad","EdgeZSAD (TensorRT FP16)","Jetson Orin Nano Super",32.83,"32.83 ms",91.6,"91.6","zero-shot","EdgeZSAD 2606.16119v1 Tables 4 and 5 (device-rescored)",""],["jetson","stfpm-fp32","STFPM FP32","Jetson Xavier NX",52.14,"52.14 ms",94.0,"0.94","multi-class","2407.02968v1 Table 5 (Jetson Xavier NX, 15-class model; \"mean AUROC\", image or pixel level not stated)",""],["jetson","stfpm-fp16","STFPM, student FP16","Jetson Xavier NX",24.98,"24.98 ms",94.0,"0.94","multi-class","2407.02968v1 Table 5 (Jetson Xavier NX, 15-class model; \"mean AUROC\", image or pixel level not stated)",""],["jetson","stfpm-int8","STFPM, student INT8","Jetson Xavier NX",24.87,"24.87 ms",92.0,"0.92","multi-class","2407.02968v1 Table 5 (Jetson Xavier NX, 15-class model; \"mean AUROC\", image or pixel level not stated)",""],["jetson","rd-fp32","RD FP32","Jetson Xavier NX",166.6,"166.60 ms",96.0,"0.96","multi-class","2407.02968v1 Table 5 (Jetson Xavier NX, 15-class model; \"mean AUROC\", image or pixel level not stated)",""],["jetson","rd-fp16","RD, student FP16","Jetson Xavier NX",19.68,"19.68 ms",82.0,"0.82","multi-class","2407.02968v1 Table 5 (Jetson Xavier NX, 15-class model; \"mean AUROC\", image or pixel level not stated)",""],["jetson","rd-int8","RD, student INT8","Jetson Xavier NX",18.02,"18.02 ms",82.0,"0.82","multi-class","2407.02968v1 Table 5 (Jetson Xavier NX, 15-class model; \"mean AUROC\", image or pixel level not stated)",""],["jetson","us-fp32","Uninformed Students FP32","Jetson Xavier NX",1392.97,"1392.97 ms",79.0,"0.79","multi-class","2407.02968v1 Table 5 (Jetson Xavier NX, 15-class model; \"mean AUROC\", image or pixel level not stated)",""],["jetson","us-fp16","Uninformed Students, student FP16","Jetson Xavier NX",699.24,"699.24 ms",79.0,"0.79","multi-class","2407.02968v1 Table 5 (Jetson Xavier NX, 15-class model; \"mean AUROC\", image or pixel level not stated)",""],["cpu","pc","PatchCore","Core i5-9600K",345.0,"345 ms",98.0,"0.98","one-class","2603.20288v1 Tables I and II (Intel Core i5-9600K)","PatchCore"],["cpu","pc-edge","PatchCore-Edge","Core i5-9600K",30.0,"30 ms",95.0,"0.95","one-class","2603.20288v1 Tables I and II (Intel Core i5-9600K)","PatchCore-Edge"],["cpu","pc-lite","PatchCore-Lite","Core i5-9600K",130.0,"130 ms",86.0,"0.86","one-class","2603.20288v1 Tables I and II (Intel Core i5-9600K)","PatchCore-Lite"],["cpu","padim","PaDiM","Core i5-9600K",22553.0,"22553 ms",96.0,"0.96","one-class","2603.20288v1 Tables I and II (Intel Core i5-9600K)","PaDiM"],["cpu","padim-edge","PaDiM-Edge","Core i5-9600K",35.0,"35 ms",94.0,"0.94","one-class","2603.20288v1 Tables I and II (Intel Core i5-9600K)","PaDiM-Edge"],["cpu","padim-lite","PaDiM-Lite","Core i5-9600K",24.0,"24 ms",85.0,"0.85","one-class","2603.20288v1 Tables I and II (Intel Core i5-9600K)","PaDiM-Lite"],["cpu","stfpm","STFPM","Core i5-9600K",123.0,"123 ms",95.0,"0.95","one-class","2603.20288v1 Tables I and II (Intel Core i5-9600K)","STFPM"],["cpu","stfpm-edge","STFPM-Edge","Core i5-9600K",13.0,"13 ms",90.0,"0.90","one-class","2603.20288v1 Tables I and II (Intel Core i5-9600K)","STFPM-Edge"],["cpu","paste","PaSTe","Core i5-9600K",9.0,"9 ms",92.0,"0.92","one-class","2603.20288v1 Tables I and II (Intel Core i5-9600K)","PaSTe"],["cpu","padim-paper","PaDiM-WR50-Rd550","Core i7-4710HQ",950.0,"0.95 s",95.3,"95.3","one-class","PaDiM 2011.08785v1 Table VII (time) and Table V (accuracy)","PaDiM-WR50-Rd550"],["cpu","spade-paper","SPADE","Core i7-4710HQ",7100.0,"7.10 s",85.5,"85.5","one-class","PaDiM 2011.08785v1 Table VII (time, PaDiM re-run) and Table V (accuracy as printed in PaDiM)","SPADE"],["cpu","pi-pc","PatchCore* (ResNet-18)","Raspberry Pi 5",1074.5,"1074.5 ms",92.5,"0.925","continual","2605.24251v2 Table 4 (latency) and Table 1 (accuracy, continual protocol)","PatchCore* Pi"],["cpu","pi-ead","EfficientAD*","Raspberry Pi 5",1318.2,"1318.2 ms",52.900000000000006,"0.529","continual","2605.24251v2 Table 4 (latency) and Table 1 (accuracy, continual protocol)","EfficientAD* Pi"],["cpu","pi-dino","DINOSaur","Raspberry Pi 5",630.6,"630.6 ms",96.8,"0.968","continual","2605.24251v2 Table 4 (latency) and Table 1 (accuracy, continual protocol)","DINOSaur Pi"]];
const SPEED_RUG = [["jetson","KairosAD (no on-device accuracy; supervised)","Jetson NX",211.0,"211","KairosAD 2505.24334v1 Sec. 4.4"],["jetson","KairosAD (no on-device accuracy; supervised)","Jetson AGX",218.0,"218","KairosAD 2505.24334v1 Sec. 4.4"],["jetson","CLIP ViT-L/14 backbone alone","Jetson Orin Nano Super",130.26,"130.26","EdgeZSAD 2606.16119v1 Table 4 (backbone-only probe)"],["jetson","DINOv2 ViT-L/14 backbone alone","Jetson Orin Nano Super",301.33,"301.33","EdgeZSAD 2606.16119v1 Table 4 (backbone-only probe)"]];

/* 8. Zero and few shot. WinCLIP tab: toy window scoring (port of parts/viz-winclip.js; checked in build/toy-winclip.mjs, M0 0.6,
   SC 0.12: mesh scratch -> max small-window p 0.998, mid 0.976, whole image 0.036; pixel AUROC small 0.990, whole image 0.500).
   AnomalyCLIP tab: AnomalyCLIP v12 Table 1 (WinCLIP rows copied from WinCLIP). AnomalyDINO tab: FEWSHOT above. */
const WINCLIP = lazy(() => {
  const { G, N, S: ST, P } = TOY, M0 = .6, SC = .12, SCALES = { small: 2, mid: 3, img: G }, sig = z => 1 / (1 + Math.exp(-z));
  const parts = { mesh: TOY.defect(TOY.make('mesh', 200), 'scratch', 200), cap: TOY.defect(TOY.make('cap', 200), 'scratch', 200) };
  const model = {};
  function build(part) {
    if (model[part]) return model[part];
    const f = TOY.features(parts[part]), d = []; for (let k = 0; k < G * G; k++) d.push(Math.abs(f[k * 3 + 1]));
    const W = {};
    for (const [name, k] of Object.entries(SCALES)) { W[name] = [];
      for (let j = 0; j + k <= G; j++) for (let i = 0; i + k <= G; i++) { let m = 0; for (let b = 0; b < k; b++) for (let a = 0; a < k; a++) m += d[(j + b) * G + i + a];
        m /= k * k; W[name].push({ i, j, k, m, p: sig((m - M0) / SC) }); } }
    return model[part] = { W, maps: {} };
  }
  const box = w => [w.i * ST, w.j * ST, (w.i + w.k - 1) * ST + P, (w.j + w.k - 1) * ST + P];
  function mapOf(ws, harm) {
    const acc = new Float64Array(N * N), cnt = new Float64Array(N * N);
    for (const w of ws) { const [x0, y0, x1, y1] = box(w); for (let y = y0; y < y1; y++) for (let x = x0; x < x1; x++) { acc[y * N + x] += harm ? 1 / Math.max(w.p, 1e-6) : w.p; cnt[y * N + x]++; } }
    const o = new Float32Array(N * N); for (let q = 0; q < N * N; q++) o[q] = cnt[q] ? (harm ? cnt[q] / acc[q] : acc[q] / cnt[q]) : 0; return o;
  }
  const order = scale => scale === 'all' ? ['small', 'mid', 'img'] : [scale];
  function fullMap(part, scale, harm) {
    const M = build(part), key = scale + harm; if (M.maps[key]) return M.maps[key];
    const ms = order(scale).map(s => mapOf(M.W[s], harm)); if (ms.length === 1) return M.maps[key] = ms[0];
    const o = new Float32Array(N * N); for (let q = 0; q < N * N; q++) o[q] = harm ? 3 / ms.reduce((s, m) => s + 1 / Math.max(m[q], 1e-6), 0) : (ms[0][q] + ms[1][q] + ms[2][q]) / 3;
    return M.maps[key] = o;
  }
  function auroc(map, mask) {
    const pos = [], neg = []; for (let q = 0; q < N * N; q++) (mask[q] ? pos : neg).push(map[q]); neg.sort((a, b) => a - b);
    const lb = (v, le) => { let a = 0, b = neg.length; while (a < b) { const m = (a + b) >> 1; if (le ? neg[m] <= v : neg[m] < v) a = m + 1; else b = m; } return a; };
    let s = 0; for (const v of pos) { const l = lb(v, false), h = lb(v, true); s += l + .5 * (h - l); } return s / (pos.length * neg.length);
  }
  return { G, N, SCALES, parts, build, box, mapOf, order, fullMap, auroc };
});
const ACLIP = [   // target, prompts trained on, AnomalyCLIP [img, pix, pro], WinCLIP [img, pix, pro] (null: not in Table 1)
  ['MVTec AD', 'VisA', ['91.5', '91.1', '81.4'], ['91.8', '85.1', '64.6']], ['VisA', 'MVTec AD', ['82.1', '95.5', '87.0'], ['78.1', '79.6', '56.8']],
  ['MPDD', 'MVTec AD', ['77.0', '96.5', null], [null, null, null]], ['BTAD', 'MVTec AD', ['88.3', '94.2', null], [null, null, null]],
  ['SDD', 'MVTec AD', ['84.7', '90.6', null], [null, null, null]], ['DAGM', 'MVTec AD', ['97.5', '95.6', null], [null, null, null]],
  ['DTD-Synthetic', 'MVTec AD', ['93.5', '97.9', null], [null, null, null]]];
SCENES.zeroshot = ctx => {
  const WC = WINCLIP(), el = mkCanvas(ctx.stage);
  el.tabIndex = 0; el.setAttribute('aria-label', 'WinCLIP: click the camera view or a grid cell to select a window. Other tabs: charts of the papers\' numbers.');
  const sb = ctx.qa('[data-scale]'), hb = ctx.qa('[data-harm]'), pb = ctx.qa('[data-part]'), db = ctx.qa('[data-ds]'), mb = ctx.qa('[data-metric]'), ks = ctx.q('#dino-k'), kOut = ctx.q('#dino-k-value');
  const KS = [1, 2, 4, 8, 16], MI = { img: 0, pix: 1, pro: 2 }, DSN = { mvtec: 'MVTec AD', visa: 'VisA' }, MN = { img: 'image AUROC', pix: 'pixel AUROC', pro: 'AU-PRO' };
  const Z = { view: 0, scale: (onKey(sb) || sb[0]).dataset.scale, harm: (onKey(hb) || hb[0]).dataset.harm === '1', part: (onKey(pb) || pb[0]).dataset.part,
    ds: (onKey(db) || db[0]).dataset.ds, metric: (onKey(mb) || mb[0]).dataset.metric, k: +ks.value, t: 1, u: 1, sel: null };
  let L = null, stop = () => {};
  const cv = PE.canvas(el, S3ASPECT(ctx.stage), (c, W, H) => draw(c, W, H));
  const seq = () => WC.order(Z.scale).flatMap(s => WC.build(Z.part).W[s]);
  const shown = () => { const all = seq(); return all.slice(0, Math.max(0, Math.round(Z.t * all.length))); };
  const best = () => { const all = seq().filter(w => w.k !== WC.G); return (all.length ? all : seq()).reduce((a, b) => b.p > a.p ? b : a); };
  const selected = () => { if (Z.t < 1) { const s = shown(); return s[s.length - 1] || seq()[0]; } return Z.sel || best(); };
  function drawWin(c, W, H, C) {
    const narrow = W < 620, { G, N, SCALES } = WC;
    if (narrow) { const s = Math.min(W * .5, H * .4); L = { narrow, img: { x: (W - s) / 2, y: 26, s }, grid: { x: 16, y: 26 + s + 40, w: W * .44, h: H - s - 96 }, bars: { x: W * .52, y: 26 + s + 40, w: W * .44 } }; }
    else { const s = Math.min(W * .36, H - HUD_H - 90); L = { narrow, img: { x: 24, y: 44, s }, grid: { x: 24 + s + 46, y: 44, w: W * .24, h: s }, bars: { x: 24 + s + 46 + W * .24 + 40, y: 44, w: W - (24 + s + 46 + W * .24 + 40) - 24 } }; }
    const img = WC.parts[Z.part], ws = shown();
    const map = Z.t >= 1 ? WC.fullMap(Z.part, Z.scale, Z.harm) : (() => { const per = WC.order(Z.scale).map(s => WC.mapOf(ws.filter(w => w.k === SCALES[s]), Z.harm)).filter((m, i) => ws.some(w => w.k === SCALES[WC.order(Z.scale)[i]]));
      if (!per.length) return new Float32Array(N * N); if (per.length === 1) return per[0]; const o = new Float32Array(N * N);
      for (let q = 0; q < N * N; q++) o[q] = Z.harm ? per.length / per.reduce((s, m) => s + 1 / Math.max(m[q], 1e-6), 0) : per.reduce((s, m) => s + m[q], 0) / per.length; return o; })();
    TOY.draw(c, img, L.img.x, L.img.y, L.img.s, { heat: TOY.heatBitmap(map, .02, .55, C.red), heatAlpha: .9 });
    PE.label(c, narrow ? 'camera view' : 'camera view, window scores laid over it', L.img.x, L.img.y - 12, C.muted, 12);
    const w = selected(), sc = L.img.s / N, [x0, y0, x1, y1] = WC.box(w);
    c.save(); c.strokeStyle = C.yellow; c.lineWidth = 2.5; c.strokeRect(L.img.x + x0 * sc, L.img.y + y0 * sc, (x1 - x0) * sc, (y1 - y0) * sc); c.restore();
    const grids = WC.order(Z.scale), gap = 14, gwid = (L.grid.w - gap * (grids.length - 1)) / grids.length;
    PE.label(c, 'one cell per window', L.grid.x, L.grid.y - 12, C.muted, 12);
    grids.forEach((s, gi) => {
      const n = G - SCALES[s] + 1, side = Math.min(gwid, L.grid.h - 22), cell = side / n, gx = L.grid.x + gi * (gwid + gap), gy = L.grid.y;
      c.save(); c.fillStyle = C.panel; c.fillRect(gx, gy, side, side); c.restore();
      const done = new Set(ws.filter(v => v.k === SCALES[s]).map(v => v.j * n + v.i));
      WC.build(Z.part).W[s].forEach(v => { if (!done.has(v.j * n + v.i)) return; c.save(); c.globalAlpha = .1 + .9 * v.p; c.fillStyle = C.red; c.fillRect(gx + v.i * cell + .5, gy + v.j * cell + .5, Math.max(1, cell - 1), Math.max(1, cell - 1)); c.restore(); });
      if (w.k === SCALES[s]) { c.save(); c.strokeStyle = C.yellow; c.lineWidth = 2; c.strokeRect(gx + w.i * cell, gy + w.j * cell, cell, cell); c.restore(); }
      PE.label(c, s === 'small' ? '2 x 2' : s === 'mid' ? '3 x 3' : 'whole', gx, gy + side + 15, C.muted, 11);
      if (gi === 0) L.cells = { gx, gy, cell, n };
    });
    const barSet = (y, label, p) => { PE.label(c, label, L.bars.x, y, C.fg, 12);
      [['flawless', 1 - p, C.blue], ['damaged', p, C.red]].forEach(([nm, v, col], r) => { const by = y + 10 + r * 24, bw = Math.max(30, L.bars.w - 104);
        c.save(); c.fillStyle = C.panel; c.fillRect(L.bars.x + 64, by, bw, 14); c.globalAlpha = .85; c.fillStyle = col; c.fillRect(L.bars.x + 64, by, bw * v, 14); c.restore();
        PE.label(c, nm, L.bars.x, by + 11, col, 11); PE.label(c, v.toFixed(2), L.bars.x + 64 + bw + 8, by + 11, C.muted, 11); }); };
    barSet(L.bars.y + 4, w.k === G ? 'whole image' : `selected window (${w.k} x ${w.k})`, w.p);
    barSet(L.bars.y + 92, 'whole image', WC.build(Z.part).W.img[0].p);
    PE.label(c, `toy cue: edge energy ${w.m.toFixed(2)}`, L.bars.x, L.bars.y + 176, C.muted, 11);
  }
  function drawAclip(c, W, H, C) {
    const narrow = W < 620, mi = MI[Z.metric], sq = narrow ? 7 : 12, x0 = narrow ? 112 : 150;
    const prow = (y, lbl, col, words) => { PE.label(c, lbl, 14, y + sq - 1, col, 12, 'left', 600);
      for (let k = 0; k < 12; k++) { c.save(); c.fillStyle = C.green; c.globalAlpha = .85; c.beginPath(); c.roundRect(x0 + k * (sq + 3), y, sq, sq, 2); c.fill(); c.restore(); }
      PE.label(c, words, x0 + 12 * (sq + 3) + 6, y + sq - 1, C.fg, 12); };
    prow(24, narrow ? 'normal' : 'normal prompt', C.blue, 'object'); prow(30 + sq, narrow ? 'abnormal' : 'abnormal prompt', C.red, 'damaged object');
    PE.label(c, narrow ? 'green: 12 learned vectors each' : 'green squares: learned vectors; CLIP itself stays frozen', 14, 52 + 2 * sq, C.muted, 11, 'left', 400);
    const rows = ACLIP.filter(r => r[2][mi] !== null), labW = narrow ? 92 : 170, bx = labW + 12, bw = W - bx - (narrow ? 50 : 90), top = 104 + 2 * sq, bot = narrow ? 16 : HUD_H;
    const rh = Math.min(64, (H - top - bot - 30) / rows.length), X = v => bx + (v - 50) / 50 * bw;
    PE.label(c, `zero-shot ${MN[Z.metric]} (bars start at 50, chance)`, bx, top - 8, C.muted, 11, 'left', 400);
    [50, 75, 100].forEach(t => { PE.polyline(c, [[X(t), top], [X(t), top + rows.length * rh]], C.rule, 1); PE.label(c, t, X(t), top + rows.length * rh + 14, C.muted, 10, 'center', 400); });
    rows.forEach((r, i) => { const y = top + i * rh + 4, hh = Math.min(16, (rh - 12) / 2);
      PE.label(c, r[0], labW, y + hh, C.fg, 12, 'right', 700); if (!narrow) PE.label(c, `prompts from ${r[1]}`, labW, y + hh + 15, C.muted, 10, 'right', 400);
      [[r[2][mi], C.green, 'AnomalyCLIP'], [r[3][mi], C.brown, 'WinCLIP']].forEach(([v, col, nm], k) => { if (v === null) return; const yy = y + k * (hh + 3), ww = (X(+v) - bx) * Z.u;
        c.save(); c.fillStyle = col; c.globalAlpha = .88; c.fillRect(bx, yy, Math.max(1, ww), hh); c.restore();
        PE.label(c, narrow ? v : `${nm} ${v}`, bx + ww + 6, yy + hh - 3, col, 11, 'left', 700); }); });
  }
  function drawDino(c, W, H, C) {
    const narrow = W < 620, data = FEWSHOT[Z.ds][Z.metric], cols = [0, ...KS];
    const SER = { dino448: ['AnomalyDINO, 448 px', C.fg, 2.6, null], dino672: ['AnomalyDINO, 672 px', C.fg, 1.4, [5, 3]], aprilgan: ['APRIL-GAN', C.muted, 1.6, [2, 3]], winclip: ['WinCLIP+', C.brown, 1.8, null], patchcore: ['PatchCore', C.muted, 1.4, [1, 4]] };
    const b = narrow ? { x: 40, y: 24, w: W - 56, h: H - 130 } : { x: 64, y: 50, w: W - 64 - 200, h: H - HUD_H - 110 };
    const vals = Object.values(data).flat().map(p => +p[1]), lo = Math.floor((Math.min(...vals) - 1) / 5) * 5, hi = 100;
    const X = k => b.x + cols.indexOf(k) / (cols.length - 1) * b.w, Y = v => b.y + b.h - (v - lo) / (hi - lo) * b.h, sk = KS[Z.k];
    c.save(); c.fillStyle = C.panel; c.beginPath(); c.roundRect(X(sk) - 16, b.y - 6, 32, b.h + 12, 6); c.fill(); c.restore();
    for (let v = lo; v <= hi; v += 5) { PE.polyline(c, [[b.x, Y(v)], [b.x + b.w, Y(v)]], C.rule, 1); PE.label(c, v, b.x - 6, Y(v) + 4, C.muted, 11, 'right', 400); }
    cols.forEach(k => PE.label(c, k === 0 ? 'zero' : k, X(k), b.y + b.h + 18, C.muted, 11, 'center', 400));
    PE.label(c, 'good images per product (k)', b.x + b.w / 2, b.y + b.h + 34, C.muted, 11, 'center', 400);
    PE.label(c, `${DSN[Z.ds]}, ${MN[Z.metric]}`, b.x, b.y - 14, C.fg, 13);
    const ends = [];
    Object.entries(SER).forEach(([key, st]) => { const pts = (data[key] || []).map(p => [X(p[0]), Y(lo + (+p[1] - lo) * Z.u), p[0]]); if (!pts.length) return;
      const solid = key === 'winclip' ? pts.filter(p => p[2] <= 4) : pts, star = key === 'winclip' ? pts.filter(p => p[2] >= 4) : [];
      if (solid.length > 1) PE.polyline(c, solid.map(p => [p[0], p[1]]), st[1], st[2], st[3]);
      if (star.length > 1) PE.polyline(c, star.map(p => [p[0], p[1]]), st[1], 1.2, [3, 3]);
      pts.forEach(p => { if (key === 'winclip' && p[2] >= 8) { PE.dot(c, p[0], p[1], 3.5, C.bg); PE.ring(c, p[0], p[1], 3.5, st[1], 1.5); } else PE.dot(c, p[0], p[1], key === 'dino448' ? 4.5 : 3.2, st[1]); });
      ends.push({ key, y: pts[pts.length - 1][1], x: pts[pts.length - 1][0], full: pts[pts.length - 1][2] === 16 }); });
    [...(data.zs || [])].sort((a, b2) => +b2[1] - +a[1]).forEach((z, i) => { const x = X(0), y = Y(lo + (+z[1] - lo) * Z.u), col = z[2] === 'AnomalyCLIP' ? C.green : C.brown;
      c.save(); c.fillStyle = col; c.beginPath(); c.moveTo(x, y - 6); c.lineTo(x + 6, y); c.lineTo(x, y + 6); c.lineTo(x - 6, y); c.closePath(); c.fill(); c.restore();
      PE.label(c, `${z[2]} ${z[1]}`, x + 9, i ? y + 14 : y - 6, col, 10); });
    if (!narrow) { const full = ends.filter(e => e.full).sort((a, b2) => a.y - b2.y); let last = -1e9; full.forEach(e => { const y = Math.max(e.y + 4, last + 14); last = y; PE.label(c, SER[e.key][0], b.x + b.w + 10, y, SER[e.key][1], 11); });
      ends.filter(e => !e.full).forEach(e => PE.label(c, SER[e.key][0], e.x + 8, e.y + 16, SER[e.key][1], 11)); }
    else { const keys = Object.keys(SER).filter(k => data[k]); keys.forEach((k, i) => { const lx = 12 + (i % 3) * (W - 24) / 3, ly = b.y + b.h + 56 + Math.floor(i / 3) * 16; PE.polyline(c, [[lx, ly - 4], [lx + 14, ly - 4]], SER[k][1], SER[k][2], SER[k][3]); PE.label(c, SER[k][0].replace('AnomalyDINO, ', 'DINO '), lx + 18, ly, SER[k][1], 10); }); }
  }
  function draw(c, W, H) { if (!W) return; const C = COL(); c.clearRect(0, 0, W, H); if (Z.view === 0) drawWin(c, W, H, C); else if (Z.view === 1) drawAclip(c, W, H, C); else drawDino(c, W, H, C); }
  function say() {
    if (Z.view === 0) {
      const M = WC.build(Z.part), map = WC.fullMap(Z.part, Z.scale, Z.harm), au = WC.auroc(map, WC.parts[Z.part].mask);
      const nWin = WC.order(Z.scale).reduce((s, k) => s + M.W[k].length, 0), top = Math.max(...WC.order(Z.scale).flatMap(k => M.W[k]).map(v => v.p)), whole = M.W.img[0].p;
      let fl = 0, inM = 0; for (let q = 0; q < WC.N * WC.N; q++) if (map[q] > .5) { fl++; inM += WC.parts[Z.part].mask[q]; }
      const sc = { small: 'Small windows (2 x 2 patches)', mid: 'Mid windows (3 x 3 patches)', img: 'The whole image as one window', all: 'All three scales' }[Z.scale];
      let t = `${sc}: ${nWin} ${nWin === 1 ? 'window' : 'windows'}, highest p(damaged) ${top.toFixed(3)}; whole image ${whole.toFixed(3)}. ${Z.harm ? 'Harmonic' : 'Arithmetic'} map: ${fl} pixels above 0.5 (${inM} inside the scratch), toy pixel AUROC ${au.toFixed(3)}.`;
      if (Z.scale === 'img') t += ' One score for every pixel: no map at all.'; if (Z.part === 'cap') t += ' On the cap the cue also fires on the designed rim.';
      ctx.hud(badge('toy', t)); return;
    }
    if (Z.view === 1) { const mi = MI[Z.metric], rows = ACLIP.filter(r => r[2][mi] !== null);
      ctx.hud(badge('sourced', `AnomalyCLIP (2310.18961v12, Table 1), zero-shot ${MN[Z.metric]}: ${rows.map(r => `${r[0]} ${r[2][mi]}${r[3][mi] ? ` (WinCLIP ${r[3][mi]})` : ''}`).join(', ')}. The prompts for MVTec AD are trained on VisA; for every other dataset, on MVTec AD.`)); return; }
    const data = FEWSHOT[Z.ds][Z.metric], k = KS[Z.k], v = key => (data[key] || []).find(p => p[0] === k);
    const names = { dino448: 'AnomalyDINO 448 px', dino672: 'AnomalyDINO 672 px', aprilgan: 'APRIL-GAN', winclip: 'WinCLIP+', patchcore: 'PatchCore' };
    const part = Object.keys(names).map(key => v(key) ? `${names[key]} ${v(key)[1]}` : null).filter(Boolean).join('; ');
    ctx.hud(badge('sourced', `AnomalyDINO (2405.14529v3, Tables 2 to 4), ${DSN[Z.ds]}, ${MN[Z.metric]}, ${k} good ${k === 1 ? 'image' : 'images'} per product: ${part}.` + (data.zs ? ` Zero-shot: ${data.zs.map(z => `${z[2]} ${z[1]}`).join(', ')}.` : '')));
  }
  const toStep = n => { const t = ctx.q(`[role="tab"][data-step="${n}"]`); if (t) t.click(); };
  const sync = () => { press(sb, b => b.dataset.scale === Z.scale); press(hb, b => (b.dataset.harm === '1') === Z.harm); press(pb, b => b.dataset.part === Z.part); press(db, b => b.dataset.ds === Z.ds); press(mb, b => b.dataset.metric === Z.metric); ks.value = Z.k; kOut.textContent = KS[Z.k]; };
  const grow = () => { stop(); Z.u = 0; stop = PE.animate(700, u => { Z.u = u; cv.redraw(); }, () => { Z.u = 1; cv.redraw(); }); };
  const win = fn => { fn(); Z.sel = null; sync(); if (Z.view !== 0) toStep(0); else { cv.redraw(); say(); } };
  const slide = () => { stop(); Z.sel = null; Z.t = 0; stop = PE.animate(2400, u => { Z.t = u; cv.redraw(); }, () => { Z.t = 1; cv.redraw(); say(); }); };
  const offs = [...sb.map(b => listen(b, 'click', () => win(() => { Z.scale = b.dataset.scale; }))), ...hb.map(b => listen(b, 'click', () => win(() => { Z.harm = b.dataset.harm === '1'; }))),
    ...pb.map(b => listen(b, 'click', () => win(() => { Z.part = b.dataset.part; }))),
    ...db.map(b => listen(b, 'click', () => { Z.ds = b.dataset.ds; sync(); if (Z.view !== 2) toStep(2); else { grow(); say(); } })),
    ...mb.map(b => listen(b, 'click', () => { Z.metric = b.dataset.metric; sync(); if (Z.view === 0) toStep(1); else { grow(); say(); } })),
    listen(ks, 'input', () => { Z.k = +ks.value; sync(); cv.redraw(); say(); }),
    listen(ctx.q('[data-act="reset"]'), 'click', () => { stop(); Object.assign(Z, { scale: 'small', harm: true, part: 'mesh', ds: 'mvtec', metric: 'img', k: 0, t: 1, u: 1, sel: null }); sync(); cv.redraw(); say(); })];
  const pick = (i, j) => { const k = WC.SCALES[WC.order(Z.scale)[0]], n = WC.G - k + 1; i = PE.clamp(i, 0, n - 1); j = PE.clamp(j, 0, n - 1); Z.sel = WC.build(Z.part).W[WC.order(Z.scale)[0]][j * n + i]; Z.t = 1; cv.redraw(); };
  el.addEventListener('click', e => { if (Z.view !== 0 || !L) return; const r = el.getBoundingClientRect(), x = e.clientX - r.left, y = e.clientY - r.top, k = WC.SCALES[WC.order(Z.scale)[0]], ST = TOY.S;
    if (x >= L.img.x && x <= L.img.x + L.img.s && y >= L.img.y && y <= L.img.y + L.img.s) { const px = (x - L.img.x) / L.img.s * WC.N, py = (y - L.img.y) / L.img.s * WC.N; pick(Math.round((px - (k * ST + ST) / 2) / ST), Math.round((py - (k * ST + ST) / 2) / ST)); }
    else if (L.cells && x >= L.cells.gx && y >= L.cells.gy && x < L.cells.gx + L.cells.cell * L.cells.n && y < L.cells.gy + L.cells.cell * L.cells.n) pick(Math.floor((x - L.cells.gx) / L.cells.cell), Math.floor((y - L.cells.gy) / L.cells.cell)); });
  el.addEventListener('keydown', e => { if (Z.view !== 0) return; const k = { ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowUp: [0, -1], ArrowDown: [0, 1] }[e.key]; if (!k) return; e.preventDefault(); const w = selected(); if (w.k === WC.G) return; pick(w.i + k[0], w.j + k[1]); });
  sync();
  return { flat: true, resize: cv.resize,
    step(n) { stop(); Z.view = n; Z.t = 1; Z.u = 1; cv.redraw(); say(); if (n > 0) grow(); },
    replay() { if (Z.view === 0) slide(); else grow(); },
    destroy() { stop(); offs.forEach(f => f()); cv.destroy(); } };
};

/* 9. Speed and scoreboard. Speed: latency (log x) against MVTec AD image AUROC, one point per (method, device, source), from
   SPEED above (port of parts/viz-speed.js, drawn on canvas). Scoreboard: rows of parts/ch-closing.html's table, each value a
   claim; an 'r' suffix marks a re-run by a later paper (drawn hollow). */
const BOARD = [
  ['SPADE', 'one-class', ['85.5', '96.0', '91.7'], [null, null, null]], ['PaDiM', 'one-class', ['95.3', '97.5', '92.1'], ['87.8r', '93.1r', null]],
  ['PatchCore', 'one-class', ['99.1', '98.1', '93.4'], ['92.4r', '98.4r', '79.7r']], ['FastFlow', 'one-class', ['99.4', '98.5', null], ['93.9r', null, '86.8r']],
  ['EfficientAD-S', 'one-class', ['98.8', null, '93.1'], ['97.5', null, '93.1']], ['EfficientAD-M', 'one-class', ['99.1', null, '93.5'], ['98.1', null, '94.0']],
  ['CutPaste', 'synthetic defects', ['95.2', '96.0', null], [null, null, null]], ['DRAEM', 'synthetic defects', ['98.0', '97.3', null], [null, null, null]],
  ['UniAD', 'multi-class', ['96.5', '96.8', null], [null, null, null]], ['Dinomaly', 'multi-class', ['99.6', '98.4', '94.8'], ['98.7', '98.7', '94.5']],
  ['WinCLIP+', '1-shot', ['93.1', '95.2', '87.1'], ['83.8', '96.4', '85.1']], ['AnomalyDINO', '1-shot', ['96.5', '96.3', '91.7'], ['85.6', '97.5', '90.7']],
  ['WinCLIP', 'zero-shot', ['91.8', '85.1', '64.6'], ['78.1', '79.6', '56.8']], ['AnomalyCLIP', 'zero-shot', ['91.5', '91.1', '81.4'], ['82.1', '95.5', '87.0']],
  ['EdgeZSAD', 'zero-shot', ['91.6', '79.3', '67.8'], ['88.2', '92.9', '74.2']]];
SCENES.speed = ctx => {
  const el = mkCanvas(ctx.stage), vb = ctx.qa('[data-view]'), bb = ctx.qa('[data-sb]'), rng = ctx.q('#speed-v'), rOut = ctx.q('#speed-v-value');
  el.setAttribute('aria-label', 'Speed against accuracy, or the scoreboard. Hover a point for its source.');
  const XR = { desktop: [1, 300], jetson: [10, 2000], cpu: [5, 30000] }, VIEW = { desktop: 'Desktop GPU (RTX A6000)', jetson: 'Jetson-class boards', cpu: 'CPUs and a Raspberry Pi 5' };
  const S = { tab: 0, view: (onKey(vb) || vb[0]).dataset.view, sb: (onKey(bb) || bb[0]).dataset.sb, v: +rng.value, focus: null };
  const pps = v => Math.pow(10, -1 + 4 * v / 100), fmtPps = p => p >= 10 ? Math.round(p).toString() : p >= 1 ? p.toFixed(1).replace(/\.0$/, '') : p.toFixed(2).replace(/0$/, '');
  const fmtMs = ms => ms >= 100 ? Math.round(ms).toLocaleString('en-US') : ms >= 10 ? ms.toFixed(0) : ms.toFixed(1);
  const pts = () => SPEED.filter(p => p[0] === S.view).map(p => ({ key: p[1], name: p[2], device: p[3], latMs: p[4], latV: p[5], acc: p[6], accV: p[7], proto: p[8], src: p[9], label: p[10] }));
  const rug = () => SPEED_RUG.filter(r => r[0] === S.view).map(r => ({ name: r[1], device: r[2], latMs: r[3], latV: r[4], src: r[5] }));
  let G = null, hits = [];
  const cv = PE.canvas(el, S3ASPECT(ctx.stage), (c, W, H) => draw(c, W, H));
  function marker(c, proto, x, y, r, fill, stroke) {
    c.save(); c.beginPath();
    if (proto === 'multi-class') c.rect(x - r, y - r, 2 * r, 2 * r);
    else if (proto === 'continual') { c.moveTo(x, y - r * 1.3); c.lineTo(x + r * 1.3, y); c.lineTo(x, y + r * 1.3); c.lineTo(x - r * 1.3, y); c.closePath(); }
    else if (proto === 'zero-shot') { c.moveTo(x, y - r * 1.3); c.lineTo(x + r * 1.2, y + r); c.lineTo(x - r * 1.2, y + r); c.closePath(); }
    else c.arc(x, y, r, 0, 7);
    if (fill) { c.fillStyle = fill; c.fill(); } c.strokeStyle = stroke; c.lineWidth = 1.6; c.stroke(); c.restore();
  }
  function drawSpeed(c, W, H, C) {
    const narrow = W < 620, P = pts(), xr = XR[S.view], lo = Math.min(...P.map(p => p.acc)), yr = [Math.max(40, Math.floor((lo - 2) / 5) * 5), 100];
    const b = narrow ? { x: 42, y: 20, w: W - 56, h: H - 74 } : { x: 64, y: 40, w: W - 100, h: H - HUD_H - 90 };
    const lx = Math.log10, X = v => b.x + (lx(v) - lx(xr[0])) / (lx(xr[1]) - lx(xr[0])) * b.w, Y = v => b.y + b.h - (v - yr[0]) / (yr[1] - yr[0]) * b.h;
    const budget = 1000 / pps(S.v), reserved = [];
    for (let e = -1; e <= 5; e++) for (const m of narrow ? [1] : [1, 2, 5]) { const v = m * 10 ** e; if (v < xr[0] || v > xr[1]) continue;
      PE.polyline(c, [[X(v), b.y], [X(v), b.y + b.h]], C.rule, 1); PE.label(c, v >= 1000 ? (v / 1000) + ' s' : v + (v === 1 ? ' ms' : ''), X(v), b.y + b.h + 15, C.muted, narrow ? 10 : 11, 'center', 400); }
    for (let v = yr[0]; v <= 100; v += yr[1] - yr[0] > 30 ? 10 : 5) { PE.polyline(c, [[b.x, Y(v)], [b.x + b.w, Y(v)]], C.rule, 1); PE.label(c, v, b.x - 6, Y(v) + 4, C.muted, narrow ? 10 : 11, 'right', 400); }
    PE.label(c, 'latency per image (log scale)', b.x + b.w / 2, b.y + b.h + (narrow ? 30 : 34), C.muted, 11, 'center', 400);
    c.save(); c.translate(narrow ? 11 : 16, b.y + b.h / 2); c.rotate(-Math.PI / 2); PE.label(c, 'MVTec AD image AUROC (%)', 0, 0, C.muted, 11, 'center', 400); c.restore();
    if (budget > xr[0]) { const bx = Math.min(X(Math.min(budget, xr[1])), b.x + b.w);
      if (budget < xr[1]) { c.save(); c.fillStyle = C.red; c.globalAlpha = .09; c.fillRect(bx, b.y, b.x + b.w - bx, b.h); c.restore(); }
      PE.polyline(c, [[bx, b.y], [bx, b.y + b.h]], C.red, 2);
      const lab = `budget ${fmtMs(budget)} ms per part`, tx = bx < b.x + 22 ? bx + 14 : bx - 6; c.save(); c.translate(tx, b.y + b.h - 6); c.rotate(-Math.PI / 2); PE.label(c, lab, 0, 0, C.red, 11); c.restore();
      reserved.push({ x: tx - 12, y: b.y + b.h - 6 - lab.length * 6.3, w: 15, h: lab.length * 6.3 }); }
    else { c.save(); c.fillStyle = C.red; c.globalAlpha = .09; c.fillRect(b.x, b.y, b.w, b.h); c.restore(); }
    hits = [];
    const R = rug(); if (R.length) { R.forEach(r => { const x = X(r.latMs); PE.polyline(c, [[x, b.y + b.h - 14], [x, b.y + b.h]], C.muted, 2); hits.push({ x, y: b.y + b.h - 7, r, rug: true }); });
      const x1 = Math.max(...R.map(r => X(r.latMs))), x0 = Math.min(...R.map(r => X(r.latMs))); PE.label(c, 'latency only', x1 + 6, b.y + b.h - 4, C.muted, 10, 'left', 400); reserved.push({ x: x0 - 8, y: b.y + b.h - 18, w: x1 - x0 + 80, h: 18 }); }
    const rr = narrow ? 4.5 : 5.5, fs = narrow ? 10 : 11, geo = P.map(p => ({ x: X(p.latMs), y: Y(p.acc) })), placed = reserved.slice();
    P.forEach((p, i) => placed.push({ x: geo[i].x - rr, y: geo[i].y - rr, w: 2 * rr, h: 2 * rr }));
    const labels = new Array(P.length).fill(null);
    P.map((p, i) => i).sort((i, j) => P[i].acc - P[j].acc).forEach(i => {
      const t = (P[i].label || P[i].name.replace(/ \(.*\)$/, '')).replace('TensorRT ', '').replace('Uninformed Students', 'US'), w = t.length * fs * .56, h = fs + 2, { x, y } = geo[i];
      for (const [dx, dy, al] of [[rr + 3, 4, 'left'], [-rr - 3, 4, 'right'], [rr + 2, -rr - 3, 'left'], [rr + 2, rr + 11, 'left'], [-rr - 2, -rr - 3, 'right'], [-rr - 2, rr + 11, 'right']]) {
        const bx = al === 'left' ? x + dx : x + dx - w, by = y + dy - fs;
        if (bx < b.x || bx + w > b.x + b.w || by < b.y || by + h > b.y + b.h) continue;
        if (placed.some(q => bx < q.x + q.w && bx + w > q.x && by < q.y + q.h && by + h > q.y)) continue;
        placed.push({ x: bx, y: by, w, h }); labels[i] = { x: x + dx, y: y + dy, al, t }; break; } });
    P.forEach((p, i) => { const ok = p.latMs <= budget, { x, y } = geo[i], on = S.focus === p.key;
      marker(c, p.proto, x, y, rr, ok ? C.fg : null, on ? C.yellow : ok ? C.fg : C.muted);
      if (labels[i]) PE.label(c, labels[i].t, labels[i].x, labels[i].y, on ? C.yellow : ok ? C.fg : C.muted, fs, labels[i].al, 600);
      hits.push({ x, y, p }); });
    if (!narrow) { const items = [['one-class', 'one model per product'], ['multi-class', 'one model for all'], ['continual', 'continual protocol'], ['zero-shot', 'zero-shot']].filter(([k]) => P.some(p => p.proto === k));
      items.forEach(([k, t], i) => { const x = b.x + b.w - 150, y = b.y + b.h - 16 - (items.length - 1 - i) * 18 - (R.length ? 22 : 0); marker(c, k, x, y - 4, 4.5, C.fg, C.fg); PE.label(c, t, x + 12, y, C.muted, 11, 'left', 400); }); }
  }
  function drawBoard(c, W, H, C) {
    const narrow = W < 620, di = S.sb === 'mvtec' ? 2 : 3, groups = [...new Set(BOARD.map(r => r[1]))], lines = BOARD.length + groups.length;
    const labW = narrow ? 92 : 150, b = narrow ? { x: labW + 8, y: 40, w: W - labW - 24, h: H - 70 } : { x: labW + 30, y: 70, w: Math.min(W - labW - 100, 560), h: H - HUD_H - 125 };
    const rh = b.h / lines, X = v => b.x + (v - 50) / 50 * b.w;
    PE.label(c, `${S.sb === 'mvtec' ? 'MVTec AD' : 'VisA'}: each method's own paper`, narrow ? 12 : 24, narrow ? 16 : 32, C.fg, 13);
    const lg = narrow ? [12, 30] : [b.x, 54]; [['image AUROC', 'dot'], ['pixel AUROC', 'ring'], ['AU-PRO', 'square']].forEach(([t, kind], i) => { const x = lg[0] + i * (narrow ? 112 : 130), y = lg[1];
      if (kind === 'dot') PE.dot(c, x + 4, y - 4, 4.5, C.fg); else if (kind === 'ring') PE.ring(c, x + 4, y - 4, 4.5, C.fg, 1.6); else { c.save(); c.strokeStyle = C.fg; c.lineWidth = 1.6; c.strokeRect(x, y - 8, 8, 8); c.restore(); }
      PE.label(c, t, x + 14, y, C.muted, 11, 'left', 400); });
    [50, 60, 70, 80, 90, 100].forEach(t => { PE.polyline(c, [[X(t), b.y], [X(t), b.y + b.h]], C.rule, 1); PE.label(c, t, X(t), b.y + b.h + 14, C.muted, 10, 'center', 400); });
    let line = 0; hits = [];
    groups.forEach(g => { PE.label(c, g, narrow ? 6 : 24, b.y + (line + .7) * rh, C.muted, 11, 'left', 700); line++;
      BOARD.filter(r => r[1] === g).forEach(r => { const y = b.y + (line + .5) * rh, vals = r[di];
        PE.label(c, r[0], b.x - 10, y + 4, C.fg, narrow ? 10 : 12, 'right', 600);
        const nums = vals.filter(Boolean).map(v => parseFloat(v)); if (nums.length > 1) PE.polyline(c, [[X(Math.min(...nums)), y], [X(Math.max(...nums)), y]], C.rule, 2);
        vals.forEach((v, k) => { if (!v) return; const x = X(parseFloat(v)), re = v.endsWith('r');
          if (k === 0) { if (re) { PE.dot(c, x, y, 4.5, C.bg); PE.ring(c, x, y, 4.5, C.fg, 1.2); } else PE.dot(c, x, y, 4.5, C.fg); }
          if (k === 1) PE.ring(c, x, y, 4.5, re ? C.muted : C.fg, 1.6);
          if (k === 2) { c.save(); c.strokeStyle = re ? C.muted : C.fg; c.globalAlpha = 1; c.lineWidth = 1.6; c.strokeRect(x - 4, y - 4, 8, 8); c.restore(); } });
        hits.push({ x: X(Math.max(...nums, 50)), y, row: r }); line++; }); });
  }
  function draw(c, W, H) { if (!W) return; G = { W, H }; const C = COL(); c.clearRect(0, 0, W, H); if (S.tab === 0) drawSpeed(c, W, H, C); else drawBoard(c, W, H, C); }
  function say(extra) {
    if (S.tab === 1) { const di = S.sb === 'mvtec' ? 2 : 3, ds = S.sb === 'mvtec' ? 'MVTec AD' : 'VisA', top = BOARD.filter(r => r[di][0]).sort((a, b) => parseFloat(b[di][0]) - parseFloat(a[di][0]))[0];
      ctx.hud(badge('sourced', extra || `${ds} scores from each method's own paper${S.sb === 'visa' ? '; hollow or grey marks are re-runs by a later paper (PaDiM and PatchCore by the VisA authors; FastFlow, and the PatchCore AU-PRO, by the EfficientAD authors)' : ''}. Highest image AUROC here: ${top[0]} ${top[di][0].replace('r', '')} (${top[1]}). Hover a row for its numbers.`)); return; }
    const P = pts(), budget = 1000 / pps(S.v), n = P.filter(p => p.latMs <= budget).length;
    ctx.hud(badge('sourced', `${VIEW[S.view]}. At ${fmtPps(pps(S.v))} parts per second each part gets ${fmtMs(budget)} ms: ${n} of ${P.length} plotted methods keep up.` + (extra ? ' ' + extra : ' Hover a point for its source.')));
  }
  const toStep = n => { const t = ctx.q(`[role="tab"][data-step="${n}"]`); if (t) t.click(); };
  function setV(v) { S.v = v; rng.value = v; rOut.textContent = `${fmtPps(pps(v))} parts per second`; cv.redraw(); say(); }
  el.addEventListener('pointermove', e => { const r = el.getBoundingClientRect(), x = e.clientX - r.left, y = e.clientY - r.top; let best = null, bd = 14;
    hits.forEach(h => { const d = S.tab === 1 ? Math.abs(h.y - y) : Math.hypot(h.x - x, h.y - y); if (d < bd) { bd = d; best = h; } });
    if (!best) return;
    if (best.row) { const r0 = best.row, di = S.sb === 'mvtec' ? 2 : 3, f = v => v ? v.replace('r', ' (re-run)') : 'not reported'; say(`${r0[0]} (${r0[1]}): image AUROC ${f(r0[di][0])}, pixel AUROC ${f(r0[di][1])}, AU-PRO ${f(r0[di][2])}.`); return; }
    if (best.rug) { say(`${best.r.name}, ${best.r.device}: ${best.r.latV} ms. Source: ${best.r.src}.`); return; }
    const p = best.p; if (S.focus !== p.key) { S.focus = p.key; cv.redraw(); } say(`${p.name} on ${p.device}: ${p.latV}, image AUROC ${p.accV} (${p.proto}). Source: ${p.src}.`); });
  const offs = [...vb.map(b => listen(b, 'click', () => { S.view = b.dataset.view; S.focus = null; press(vb, x => x === b); if (S.tab !== 0) toStep(0); else { cv.redraw(); say(); } })),
    ...bb.map(b => listen(b, 'click', () => { S.sb = b.dataset.sb; press(bb, x => x === b); if (S.tab !== 1) toStep(1); else { cv.redraw(); say(); } })),
    listen(rng, 'input', () => setV(+rng.value)),
    listen(ctx.q('[data-act="reset"]'), 'click', () => { S.view = 'desktop'; S.sb = 'mvtec'; S.focus = null; press(vb, x => x.dataset.view === 'desktop'); press(bb, x => x.dataset.sb === 'mvtec'); setV(50); })];
  rOut.textContent = `${fmtPps(pps(S.v))} parts per second`;
  return { flat: true, resize: cv.resize, step(n) { S.tab = n; S.focus = null; cv.redraw(); say(); }, destroy() { offs.forEach(f => f()); cv.destroy(); } };
};
