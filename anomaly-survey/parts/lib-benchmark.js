<script>
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
if (typeof module !== 'undefined') module.exports = BENCH_KIT;
</script>
