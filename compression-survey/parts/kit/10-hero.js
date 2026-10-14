/* Hero: one 16x16 weight tile goes through the four levers. Toy weights; the 2:4 pruning and the rounding are exact. */
KIT.hero = (el, opts = {}) => {
  const N = 16, r = PE.rng(7), g = PE.gauss(r);
  const W = Array.from({ length: N * N }, () => g() * .5);
  const maxA = Math.max(...W.map(Math.abs));
  const LV = 7;                               // levels per sign shown (a 4-bit grid, so the snapping is visible)
  const S = maxA / LV;
  const Q = W.map(w => Math.round(w / S));
  const keep = W.map(() => true);             // exact 2:4: in each row, each group of 4 keeps its 2 largest magnitudes
  for (let i = 0; i < N; i++) for (let j = 0; j < N; j += 4) {
    const idx = [0, 1, 2, 3].map(k => i * N + j + k).sort((a, b) => Math.abs(W[b]) - Math.abs(W[a]));
    keep[idx[2]] = keep[idx[3]] = false;
  }
  const M = 8, r2 = PE.rng(11), g2 = PE.gauss(r2), SW = Array.from({ length: M * M }, () => Math.round(g2() * 2.2));
  const T = { t: 0 }, END = 6.2;
  let raf = 0;
  const seg = (t, a, b) => PE.clamp((t - a) / (b - a), 0, 1);
  const ease = u => u < .5 ? 2 * u * u : 1 - Math.pow(-2 * u + 2, 2) / 2;

  function layout(w, h) {
    const narrow = w < 620;
    if (!narrow) {
      const cw = w / 5, tile = Math.min(cw * .62, h * .5);
      return { narrow, tile, cw, st: [0, 1, 2, 3, 4].map(k => ({ x: cw * (k + .5), y: h * .46 })) };
    }
    const top = 30, ch = (h - top) / 3, cw = w / 2, tile = Math.min(cw * .56, ch - 78);
    const st = [0, 1, 2, 3].map(k => ({ x: cw * (k % 2 + .5), y: top + ch * Math.floor(k / 2) + 22 + tile / 2 }));
    st.push({ x: w / 2, y: top + ch * 2 + 22 + tile / 2 });
    return { narrow, tile, cw, st };
  }
  function cellGrid(ctx, cx, cy, size, n, fill) {
    const c = size / n, x0 = cx - size / 2, y0 = cy - size / 2;
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) fill(i, j, x0 + j * c + .5, y0 + i * c + .5, c - 1);
  }
  function draw(ctx, w, h) {
    ctx.clearRect(0, 0, w, h);
    const t = T.t, L = layout(w, h), C = n => PE.css(n);
    const blue = C('--blue'), yellow = C('--yellow'), green = C('--green'), brown = C('--brown'), red = C('--red'), mut = C('--muted'), fg = C('--fg'), rule = C('--rule');
    const names = ['trained, FP32', 'quantize', 'prune 2:4', 'distill', 'compile'];
    const notes = [`256 ${KIT.times} 32 bits = 1024 B`, `256 ${KIT.times} 8 bits = 256 B`, `128 kept ${KIT.times} 8 bits + 2-bit indices = 160 B`, 'a smaller student: 64 weights', 'one fused kernel, one trip to memory'];
    const fs = L.narrow ? 11 : 12;
    const shown = [seg(t, 0, .6), seg(t, 1, 2), seg(t, 2.2, 3.2), seg(t, 3.4, 4.4), seg(t, 4.6, 5.6)];
    // arrows between stations
    for (let k = 0; k < 4; k++) {
      const a = L.st[k], b = L.st[k + 1]; if (shown[k + 1] <= 0) continue;
      if (Math.abs(a.y - b.y) > 1) continue;
      PE.arrow(ctx, a.x + L.tile / 2 + 8, a.y, b.x - L.tile / 2 - 8, a.y, rule, 1.5, 7, Math.min(1, shown[k + 1] * 2));
    }
    // station 0: FP32 tile
    const s0 = shown[0];
    cellGrid(ctx, L.st[0].x, L.st[0].y, L.tile, N, (i, j, x, y, c) => {
      ctx.globalAlpha = s0 * (.18 + .82 * Math.abs(W[i * N + j]) / maxA); ctx.fillStyle = blue; ctx.fillRect(x, y, c, c); });
    // station 1: quantized tile, cells fly from station 0 and snap color to discrete levels
    const s1 = ease(shown[1]);
    if (s1 > 0) cellGrid(ctx, PE.lerp(L.st[0].x, L.st[1].x, s1), PE.lerp(L.st[0].y, L.st[1].y, s1), L.tile, N, (i, j, x, y, c) => {
      const k = i * N + j, cont = Math.abs(W[k]) / maxA, disc = Math.abs(Q[k]) / LV;
      ctx.globalAlpha = .18 + .82 * PE.lerp(cont, disc, s1); ctx.fillStyle = s1 < .5 ? blue : yellow; ctx.fillRect(x, y, c, c); });
    // station 2: 2:4 pruned copy
    const s2 = ease(shown[2]);
    if (s2 > 0) cellGrid(ctx, PE.lerp(L.st[1].x, L.st[2].x, s2), PE.lerp(L.st[1].y, L.st[2].y, s2), L.tile, N, (i, j, x, y, c) => {
      const k = i * N + j, a = .18 + .82 * Math.abs(Q[k]) / LV;
      if (keep[k]) { ctx.globalAlpha = a; ctx.fillStyle = yellow; ctx.fillRect(x, y, c, c); }
      else { ctx.globalAlpha = a * (1 - s2); ctx.fillStyle = yellow; ctx.fillRect(x, y, c, c);
        ctx.globalAlpha = s2 * .9; ctx.strokeStyle = brown; ctx.lineWidth = 1; ctx.strokeRect(x + .5, y + .5, c - 1, c - 1); } });
    ctx.globalAlpha = 1;
    // station 3: a smaller student tile grows in
    const s3 = ease(shown[3]);
    if (s3 > 0) { const size = L.tile * (1 - .5 * s3);
      cellGrid(ctx, PE.lerp(L.st[2].x, L.st[3].x, s3), PE.lerp(L.st[2].y, L.st[3].y, s3), size, s3 < .5 ? N : M, (i, j, x, y, c) => {
        const v = s3 < .5 ? Math.abs(Q[i * N + j]) / LV : Math.abs(SW[i * M + j]) / 6;
        ctx.globalAlpha = .2 + .8 * Math.min(1, v); ctx.fillStyle = yellow; ctx.fillRect(x, y, c, c); });
      ctx.globalAlpha = 1; }
    // station 4: three kernels with memory trips merge into one fused kernel
    const s4 = ease(shown[4]);
    if (s4 > 0) {
      const { x, y } = L.st[4], bw = L.tile * .8, bh = L.tile * .2, mem = y + L.tile * .46;
      ctx.globalAlpha = s4; PE.polyline(ctx, [[x - bw * .6, mem], [x + bw * .6, mem]], mut, 3); ctx.globalAlpha = 1;
      PE.label(ctx, 'memory', x, mem + 14, mut, 10, 'center', 400);
      const labs = ['conv', '+ bias', 'ReLU'];
      for (let k = 0; k < 3; k++) {
        const y0 = y - L.tile * .42 + k * bh * 1.35, yk = PE.lerp(y0, y - bh / 2, s4 > .55 ? (s4 - .55) / .45 : 0);
        const merged = s4 > .55 ? (s4 - .55) / .45 : 0;
        PE.box(ctx, x - bw / 2, yk, bw, bh, merged > .5 ? green : blue, merged > .7 ? '' : labs[k], s4 * (k === 1 ? 1 : 1 - merged * .99));
        if (merged < .5) { ctx.globalAlpha = s4 * (1 - merged * 2) * .7; PE.polyline(ctx, [[x + bw / 2 + 6, yk + bh / 2], [x + bw / 2 + 6, mem]], red, 1.2, [3, 3]); ctx.globalAlpha = 1; }
      }
      if (s4 > .8) { PE.label(ctx, 'fused kernel', x, y + 4, green, 12, 'center'); ctx.globalAlpha = (s4 - .8) / .2; PE.polyline(ctx, [[x + bw / 2 + 6, y], [x + bw / 2 + 6, mem]], red, 1.2, [3, 3]); ctx.globalAlpha = 1; }
    }
    // labels
    for (let k = 0; k < 5; k++) {
      const a = k === 0 ? s0 : seg(t, [0, 2, 3.2, 4.4, 5.6][k] - .2, [0, 2, 3.2, 4.4, 5.6][k] + .4);
      if (a <= 0) continue;
      const { x, y } = L.st[k], top = y - L.tile / 2 - 10, bot = y + L.tile / 2 + (k === 4 ? 26 : 18);
      ctx.globalAlpha = a;
      PE.label(ctx, names[k], x, top, k === 0 ? blue : k === 4 ? green : yellow, fs + 1, 'center');
      const words = notes[k].split(' '), lines = [], maxW = L.cw * .92;
      ctx.font = `400 ${fs}px system-ui, sans-serif`;
      words.forEach(wd => { const cur = lines[lines.length - 1]; if (cur && ctx.measureText(cur + ' ' + wd).width < maxW) lines[lines.length - 1] = cur + ' ' + wd; else lines.push(wd); });
      lines.forEach((ln, i) => PE.label(ctx, ln, x, bot + i * (fs + 3), k === 0 ? mut : fg, fs, 'center', 400));
      ctx.globalAlpha = 1;
    }
  }
  const cv = PE.canvas(el, opts.aspect || (w => w < 620 ? 1.62 : .36), draw);
  function replay() {
    cancelAnimationFrame(raf);
    if (PE.instant()) { T.t = END; cv.redraw(); return; }
    const t0 = performance.now();
    const f = now => { T.t = Math.min(END, (now - t0) / 1000); cv.redraw(); if (T.t < END) raf = requestAnimationFrame(f); };
    raf = requestAnimationFrame(f);
  }
  replay();
  return { replay, redraw: cv.redraw, resize: cv.resize, destroy: () => { cancelAnimationFrame(raf); cv.destroy(); } };
};
