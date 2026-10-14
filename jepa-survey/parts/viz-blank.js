<script>
(() => {
  // Toy: one synthetic scene on a G x G patch grid, one hidden block. Three ways to learn from it:
  // reconstruct its pixels, make two views agree, or predict its embedding. The "embedding" is four
  // hand-picked patch statistics (brightness, greenness, texture, edge direction), so the toy's point
  // is where the loss is measured, not what a real encoder learns.
  const G = 16, P = 8, HB = { r: 7, c: 9, n: 4 };   // hidden block: rows 7-10, cols 9-12
  const r0 = PE.rng(7), gs = PE.gauss(r0);
  // scene(x, y) -> [r, g, b] in 0..1 for pixel coordinates in [0, G*P): sky, a hill, a textured tree.
  const base = [];
  for (let y = 0; y < G * P; y++) { base.push([]); for (let x = 0; x < G * P; x++) {
    const u = x / (G * P), v = y / (G * P);
    let c = [.42 + .3 * (1 - v), .55 + .25 * (1 - v), .78];                       // sky
    const hill = .62 + .08 * Math.sin(u * 6.5) + .05 * Math.sin(u * 17);
    if (v > hill) c = [.36 + .1 * (v - hill), .5 + .12 * (v - hill), .25];          // ground
    const tx = u - .68, ty = v - .5; const inTree = tx * tx * 1.3 + ty * ty < .05;
    if (inTree) { const n = gs(); c = [.18 + .09 * n, .42 + .16 * n, .16 + .06 * n]; } // leaves: textured
    if (Math.abs(u - .68) < .012 && v > .62 && v < .78) c = [.35, .24, .14];        // trunk
    base[y].push(c.map(t => PE.clamp(t, 0, 1)));
  } }
  const pix = (y, x, noise, rr) => base[y][x].map(t => PE.clamp(t + noise * rr(), 0, 1));
  // Patch embedding: brightness, greenness, texture (std of brightness), edge direction (h vs v energy).
  function feat(py, px, noise, rr) {
    let s = 0, s2 = 0, g = 0, eh = 0, ev = 0, n = 0; const L = [];
    for (let y = 0; y < P; y++) { L.push([]); for (let x = 0; x < P; x++) {
      const c = pix(py * P + y, px * P + x, noise, rr), l = .3 * c[0] + .59 * c[1] + .11 * c[2];
      L[y].push(l); s += l; s2 += l * l; g += c[1] - .5 * (c[0] + c[2]); n++; } }
    for (let y = 1; y < P; y++) for (let x = 1; x < P; x++) { eh += Math.abs(L[y][x] - L[y][x - 1]); ev += Math.abs(L[y][x] - L[y - 1][x]); }
    const m = s / n, sd = Math.sqrt(Math.max(0, s2 / n - m * m));
    return [m, g / n * 2 + .5, Math.min(1, sd * 4), (eh + ev) > 1e-6 ? eh / (eh + ev) : .5];
  }
  const hidden = []; for (let i = 0; i < HB.n; i++) for (let j = 0; j < HB.n; j++) hidden.push([HB.r + i, HB.c + j]);
  const isHidden = (i, j) => i >= HB.r && i < HB.r + HB.n && j >= HB.c && j < HB.c + HB.n;
  // Neighbours of the hidden block: a one-patch ring (the context the toy predictor is allowed to use).
  const ring = []; for (let i = HB.r - 1; i <= HB.r + HB.n; i++) for (let j = HB.c - 1; j <= HB.c + HB.n; j++) if (!isHidden(i, j) && i >= 0 && j >= 0 && i < G && j < G) ring.push([i, j]);

  const fig = document.getElementById('viz-blank'), el = document.getElementById('blank-canvas'), out = document.getElementById('blank-readout');
  const S = { mode: 'pixels', noise: 0, u: 1 };
  let stop = () => {};
  function losses(noise) {
    const rr = PE.rng(99), rp = PE.rng(99);
    // Pixel route: predict each hidden pixel as the mean of the ring's pixels at the same offset (a crude
    // in-painter); the loss is the mean squared error against the (possibly noisy) hidden pixels.
    let se = 0, n = 0;
    const ringMean = []; for (let y = 0; y < P; y++) { ringMean.push([]); for (let x = 0; x < P; x++) {
      const m = [0, 0, 0]; ring.forEach(([i, j]) => { const c = base[i * P + y][j * P + x]; m[0] += c[0]; m[1] += c[1]; m[2] += c[2]; }); ringMean[y].push(m.map(t => t / ring.length)); } }
    hidden.forEach(([i, j]) => { for (let y = 0; y < P; y++) for (let x = 0; x < P; x++) {
      const t = pix(i * P + y, j * P + x, noise, rr), p = ringMean[y][x]; for (let k = 0; k < 3; k++) { se += (t[k] - p[k]) ** 2; n++; } } });
    // Embedding route: predict the hidden block's four statistics as the ring's mean statistics.
    const fr = [0, 0, 0, 0]; ring.forEach(([i, j]) => feat(i, j, 0, rp).forEach((v, k) => fr[k] += v / ring.length));
    let fe = 0; hidden.forEach(([i, j]) => { const f = feat(i, j, noise, rr); fe += f.reduce((s, v, k) => s + (v - fr[k]) ** 2, 0) / 4; });
    return { pixel: se / n, embed: fe / hidden.length, fr };
  }
  const L0 = losses(0);
  const cv = PE.canvas(el, w => w < 560 ? 1.05 : .56, draw);
  function draw(ctx, W, H) {
    const C = { bg: PE.css('--bg'), fg: PE.css('--fg'), mu: PE.css('--muted'), blue: PE.css('--blue'), yel: PE.css('--yellow'), grn: PE.css('--green'), red: PE.css('--red'), brn: PE.css('--brown'), rule: PE.css('--rule') };
    ctx.fillStyle = C.bg; ctx.fillRect(0, 0, W, H);
    const narrow = W < 560, pad = 14;
    const side = narrow ? Math.min(W - 2 * pad, H * .46) : Math.min(H - 2 * pad - 22, W * .42);
    const ox = pad, oy = narrow ? pad + 18 : (H - side) / 2 + 8, cs = side / G;
    const rr = PE.rng(99), rt = PE.rng(123);
    // The scene, patch by patch; hidden patches are drawn as the model sees them (blank).
    for (let i = 0; i < G; i++) for (let j = 0; j < G; j++) {
      if (isHidden(i, j)) { ctx.fillStyle = C.yel; ctx.globalAlpha = .18; ctx.fillRect(ox + j * cs, oy + i * cs, cs, cs); ctx.globalAlpha = 1; continue; }
      for (let y = 0; y < P; y++) for (let x = 0; x < P; x++) { const c = base[i * P + y][j * P + x]; ctx.fillStyle = `rgb(${c.map(t => Math.round(t * 255)).join(',')})`; ctx.fillRect(ox + j * cs + x * cs / P, oy + i * cs + y * cs / P, cs / P + .5, cs / P + .5); }
    }
    ctx.strokeStyle = C.blue; ctx.lineWidth = 2; ctx.strokeRect(ox + .5, oy + .5, side - 1, side - 1);
    ctx.strokeStyle = C.yel; ctx.lineWidth = 2.5; ctx.strokeRect(ox + HB.c * cs, oy + HB.r * cs, HB.n * cs, HB.n * cs);
    PE.label(ctx, 'context (seen)', ox, oy - 6, C.blue, 12); PE.label(ctx, 'hidden block', ox + HB.c * cs + HB.n * cs / 2, oy + HB.r * cs - 5, C.yel, 11, 'center');
    // Right side: what is predicted and what the loss measures.
    const rx = narrow ? pad : ox + side + 28, ry = narrow ? oy + side + 34 : oy, rw = narrow ? W - 2 * pad : W - rx - pad;
    const noise = S.noise, Ln = losses(noise), u = S.u;
    const barY = (narrow ? ry + 120 : ry + side - 46), bw = Math.min(150, rw * .42);
    const bar = (x, v, vmax, color, name) => { const h = 50 * PE.clamp(v / vmax, 0, 1); ctx.fillStyle = C.rule; ctx.fillRect(x, barY - 50, bw, 50); ctx.fillStyle = color; ctx.globalAlpha = .9; ctx.fillRect(x, barY - h, bw, h); ctx.globalAlpha = 1; PE.label(ctx, name, x, barY + 14, C.mu, 11); PE.label(ctx, v.toFixed(3), x + bw, barY - h - 4, C.fg, 11, 'right'); };
    if (S.mode === 'pixels') {
      PE.label(ctx, 'Reconstruct the pixels', rx, ry + 2, C.fg, 14);
      // The crude in-painter's output: blurry average of the ring, next to the truth.
      const bs = Math.min(72, rw * .22), by = ry + 16;
      const ringMean = (y, x) => { const m = [0, 0, 0]; ring.forEach(([i, j]) => { const c = base[i * P + y][j * P + x]; m[0] += c[0]; m[1] += c[1]; m[2] += c[2]; }); return m.map(t => t / ring.length); };
      for (let y = 0; y < P; y++) for (let x = 0; x < P; x++) { const c = ringMean(y, x); ctx.fillStyle = `rgb(${c.map(t => Math.round(t * 255)).join(',')})`; ctx.fillRect(rx + x * bs / P, by + y * bs / P, bs / P + .5, bs / P + .5); }
      ctx.strokeStyle = C.grn; ctx.lineWidth = 2; ctx.strokeRect(rx + .5, by + .5, bs - 1, bs - 1); PE.label(ctx, 'guess', rx, by + bs + 13, C.grn, 11);
      const tx = rx + bs + 26; const [i0, j0] = hidden[5];
      for (let y = 0; y < P; y++) for (let x = 0; x < P; x++) { const c = pix(i0 * P + y, j0 * P + x, noise, rt); ctx.fillStyle = `rgb(${c.map(t => Math.round(t * 255)).join(',')})`; ctx.fillRect(tx + x * bs / P, by + y * bs / P, bs / P + .5, bs / P + .5); }
      ctx.strokeStyle = C.yel; ctx.lineWidth = 2; ctx.strokeRect(tx + .5, by + .5, bs - 1, bs - 1); PE.label(ctx, 'truth (one patch)', tx, by + bs + 13, C.yel, 11);
      PE.arrow(ctx, rx + bs + 4, by + bs / 2, tx - 4, by + bs / 2, C.red, 2, 7);
      PE.label(ctx, 'every pixel is graded', rx, by + bs + 32, C.mu, 11);
      bar(rx, Ln.pixel * u + L0.pixel * (1 - u), .08, C.red, 'pixel loss (mean squared error)');
    } else if (S.mode === 'views') {
      PE.label(ctx, 'Make two views agree', rx, ry + 2, C.fg, 14);
      const bs = Math.min(64, rw * .2), by = ry + 16;
      const crop = (x0, ci, cj, col, name) => { for (let i = 0; i < 6; i++) for (let j = 0; j < 6; j++) { const ii = ci + i, jj = cj + j; const c = base[ii * P + 3][jj * P + 3]; ctx.fillStyle = `rgb(${c.map(t => Math.round(t * 255)).join(',')})`; ctx.fillRect(x0 + j * bs / 6, by + i * bs / 6, bs / 6 + .5, bs / 6 + .5); }
        ctx.strokeStyle = col; ctx.lineWidth = 2; ctx.strokeRect(x0 + .5, by + .5, bs - 1, bs - 1); PE.label(ctx, name, x0, by + bs + 13, col, 11);
        ctx.strokeStyle = col; ctx.setLineDash([3, 3]); ctx.strokeRect(ox + cj * cs, oy + ci * cs, 6 * cs, 6 * cs); ctx.setLineDash([]); };
      crop(rx, 2, 1, C.blue, 'view A'); crop(rx + bs + 22, 6, 7, C.blue, 'view B');
      const ex = rx + 2 * bs + 60; PE.dot(ctx, ex + 10, by + bs * .35, 7, C.grn); PE.dot(ctx, ex + 16, by + bs * .5, 7, C.grn);
      PE.arrow(ctx, rx + bs, by + bs * .4, ex + 2, by + bs * .38, C.grn, 1.5, 6); PE.arrow(ctx, rx + 2 * bs + 22, by + bs * .5, ex + 8, by + bs * .5, C.grn, 1.5, 6);
      PE.label(ctx, 'same embedding', ex - 6, by + bs + 13, C.grn, 11);
      PE.label(ctx, 'nothing is predicted; the two embeddings are pulled together', rx, by + bs + 32, C.mu, 11);
      PE.label(ctx, 'needs: a trick so both do not become constant', rx, by + bs + 48, C.brn, 11);
    } else {
      PE.label(ctx, 'Predict the embedding', rx, ry + 2, C.fg, 14);
      const by = ry + 16, names = ['bright', 'green', 'texture', 'edges'], fr = Ln.fr;
      const [i0, j0] = hidden[5]; const ft = feat(i0, j0, noise, rt);
      const col = Math.min(140, rw * .4);
      names.forEach((nm, k) => { const y = by + k * 18;
        PE.label(ctx, nm, rx, y + 10, C.mu, 11);
        ctx.fillStyle = C.rule; ctx.fillRect(rx + 52, y + 2, col, 10); ctx.fillStyle = C.grn; ctx.fillRect(rx + 52, y + 2, col * fr[k], 5); ctx.fillStyle = C.yel; ctx.fillRect(rx + 52, y + 7, col * ft[k], 5); });
      PE.label(ctx, 'guess (green) vs truth (yellow)', rx, by + 86, C.mu, 11);
      PE.label(ctx, 'four numbers are graded, not 192 pixels', rx, by + 102, C.mu, 11);
      bar(rx, Ln.embed * u + L0.embed * (1 - u), .08, C.red, 'embedding loss (mean squared error)');
    }
    if (S.mode !== 'views') { const xs = rx + bw + 22; bar(xs, S.mode === 'pixels' ? Ln.embed : Ln.pixel, .08, C.brn, S.mode === 'pixels' ? 'the other route, for scale' : 'the other route, for scale'); }
    if (noise > 0) PE.label(ctx, `target noise: ${noise.toFixed(2)}`, ox + side, oy + side + 15, C.yel, 11, 'right');
  }
  function say() {
    const Ln = losses(S.noise), L00 = L0;
    const m = { pixels: `Pixel route: loss ${Ln.pixel.toFixed(3)}${S.noise ? ` (clean target: ${L00.pixel.toFixed(3)})` : ''}. The guess is a blur, and every leaf it got wrong counts.`,
      views: 'Two views of one scene are pushed to the same embedding. No prediction, so no blur, but nothing stops both embeddings from becoming the same constant.',
      embed: `Embedding route: loss ${Ln.embed.toFixed(3)}${S.noise ? ` (clean target: ${L00.embed.toFixed(3)})` : ''}. Four summary numbers are predicted; pixel-level detail is never graded.` }[S.mode];
    out.textContent = m;
  }
  function setMode(m) { S.mode = m; fig.querySelectorAll('[data-mode]').forEach(b => b.setAttribute('aria-pressed', String(b.dataset.mode === m))); stop(); stop = PE.animate(500, u => { S.u = u; cv.redraw(); }); say(); }
  function setNoise(v) { S.noise = v; fig.querySelectorAll('[data-noise]').forEach(b => b.setAttribute('aria-pressed', String(+b.dataset.noise === v))); stop(); stop = PE.animate(500, u => { S.u = u; cv.redraw(); }); say(); }
  fig.querySelectorAll('[data-mode]').forEach(b => b.addEventListener('click', () => setMode(b.dataset.mode)));
  fig.querySelectorAll('[data-noise]').forEach(b => b.addEventListener('click', () => setNoise(+b.dataset.noise)));
  document.getElementById('btn-blank-reset').addEventListener('click', () => { S.noise = 0; fig.querySelectorAll('[data-noise]').forEach(b => b.setAttribute('aria-pressed', String(b.dataset.noise === '0'))); setMode('pixels'); });
  PE.linkPulses(fig, () => cv.redraw());
  setMode('pixels');
})();
</script>
