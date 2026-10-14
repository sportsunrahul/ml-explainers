/* KIT: helpers shared by every figure on both pages (long page and atlas). Needs PE (both pages define it). */
const KIT = (() => {
  const NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'];
  // Reviewer marks: uneven pen strokes so they read as a person's marks, not chart glyphs. `u` in 0..1 draws them partly.
  function jitter(seed) { const r = PE.rng(seed * 7919 + 13); return () => r() - .5; }
  function ring(c, x, y, rad, color, o = {}) {
    const j = jitter(o.seed || 1), u = o.u ?? 1, lw = o.lw || 2.2, n = 34, a0 = -2.1 + j() * .6, turn = 2 * Math.PI * 1.08 * u;
    if (u <= 0) return;
    c.save(); c.strokeStyle = color; c.lineWidth = lw; c.lineCap = 'round'; c.globalAlpha = o.alpha ?? 1; c.beginPath();
    const wob = [j(), j(), j()];
    for (let i = 0; i <= n; i++) {
      const t = a0 + turn * i / n, rr = rad * (1 + .07 * Math.sin(2 * t + wob[0] * 6) + .04 * wob[1] * Math.sin(3 * t));
      const px = x + rr * 1.06 * Math.cos(t), py = y + rr * .94 * Math.sin(t);
      i ? c.lineTo(px, py) : c.moveTo(px, py);
    }
    c.stroke(); c.restore();
  }
  function strike(c, x, y, w, color, o = {}) {
    const j = jitter(o.seed || 2), u = o.u ?? 1; if (u <= 0) return;
    c.save(); c.strokeStyle = color; c.lineWidth = o.lw || 2.6; c.lineCap = 'round'; c.globalAlpha = o.alpha ?? 1;
    const x0 = x - w / 2, y0 = y + w * .32 + j() * 2, x1 = x + w / 2, y1 = y - w * .32 + j() * 2;
    c.beginPath(); c.moveTo(x0, y0); c.quadraticCurveTo((x0 + x1) / 2 + j() * 4, (y0 + y1) / 2 + j() * 4, x0 + (x1 - x0) * u, y0 + (y1 - y0) * u); c.stroke(); c.restore();
  }
  function tick(c, x, y, s, color, o = {}) {
    const u = o.u ?? 1; if (u <= 0) return;
    c.save(); c.strokeStyle = color; c.lineWidth = o.lw || 2.6; c.lineCap = 'round'; c.lineJoin = 'round'; c.globalAlpha = o.alpha ?? 1;
    const p = [[x - s * .45, y], [x - s * .1, y + s * .35], [x + s * .5, y - s * .4]];
    c.beginPath(); c.moveTo(...p[0]);
    if (u < .4) c.lineTo(p[0][0] + (p[1][0] - p[0][0]) * u / .4, p[0][1] + (p[1][1] - p[0][1]) * u / .4);
    else { c.lineTo(...p[1]); const v = (u - .4) / .6; c.lineTo(p[1][0] + (p[2][0] - p[1][0]) * v, p[1][1] + (p[2][1] - p[1][1]) * v); }
    c.stroke(); c.restore();
  }
  // Thumbnail sprites (CIFAR-10 train images, 32 x 32, packed 16 per row). SPRITES holds data URIs.
  const imgs = {}, waiters = {};
  function sprite(name, onload) {
    if (imgs[name] && imgs[name].complete && imgs[name].naturalWidth) { onload && onload(imgs[name]); return imgs[name]; }
    (waiters[name] = waiters[name] || []).push(onload);
    if (!imgs[name]) { const im = new Image(); im.onload = () => (waiters[name] || []).forEach(f => f && f(im)); im.src = SPRITES[name].src; imgs[name] = im; }
    return null;
  }
  function tile(c, im, k, x, y, size, cols = 16) {
    if (!im) { c.fillStyle = PE.css('--panel2'); c.fillRect(x, y, size, size); return; }
    const sx = (k % cols) * 32, sy = Math.floor(k / cols) * 32;
    c.save(); c.imageSmoothingEnabled = false; c.drawImage(im, sx, sy, 32, 32, x, y, size, size); c.restore();
  }
  const entropy = p => -p.reduce((a, v) => a + (v > 0 ? v * Math.log(v) : 0), 0);
  // Two-component 1D Gaussian mixture by EM, as DivideMix fits per-sample losses (min-max normalized first).
  function gmm2(xs, iters = 60) {
    const n = xs.length, lo = Math.min(...xs), hi = Math.max(...xs), x = xs.map(v => (v - lo) / Math.max(hi - lo, 1e-9));
    let mu = [.1, .9], va = [.02, .02], w = [.5, .5]; const reg = 5e-4;
    const N = (v, m, s2) => Math.exp(-(v - m) * (v - m) / (2 * s2)) / Math.sqrt(2 * Math.PI * s2);
    let r = new Array(n).fill(.5);
    for (let it = 0; it < iters; it++) {
      r = x.map(v => { const a = w[0] * N(v, mu[0], va[0]), b = w[1] * N(v, mu[1], va[1]); return a / Math.max(a + b, 1e-300); });
      const s0 = r.reduce((a, b) => a + b, 0), s1 = n - s0;
      mu = [r.reduce((a, q, i) => a + q * x[i], 0) / s0, r.reduce((a, q, i) => a + (1 - q) * x[i], 0) / s1];
      va = [r.reduce((a, q, i) => a + q * (x[i] - mu[0]) ** 2, 0) / s0 + reg, r.reduce((a, q, i) => a + (1 - q) * (x[i] - mu[1]) ** 2, 0) / s1 + reg];
      w = [s0 / n, s1 / n];
    }
    const lowIs0 = mu[0] <= mu[1];
    const clean = r.map(q => lowIs0 ? q : 1 - q);   // posterior of the low-mean component
    const dens = v => { const z = (v - lo) / Math.max(hi - lo, 1e-9); return [w[0] * N(z, mu[0], va[0]) / (hi - lo), w[1] * N(z, mu[1], va[1]) / (hi - lo)]; };
    return { clean, mu: mu.map(m => lo + m * (hi - lo)), sd: va.map(s => Math.sqrt(s) * (hi - lo)), w, lowIs0, dens };
  }
  // A smooth heat layer: value(i, j) in 0..1 on a G x G grid, drawn once offscreen and scaled with smoothing (no seams).
  const heatCanvas = document.createElement('canvas');
  function heat(c, x, y, w, h, G, value, color, amax = .6, mask) {
    heatCanvas.width = G; heatCanvas.height = G; const hc = heatCanvas.getContext('2d'), img = hc.createImageData(G, G);
    const m = /#([0-9a-f]{2})([0-9a-f]{2})([0-9a-f]{2})/i.exec(color) || [0, '58', 'c4', 'dd'], rgb = [1, 2, 3].map(k => parseInt(m[k], 16));
    for (let i = 0; i < G; i++) for (let j = 0; j < G; j++) { const k = (i * G + j) * 4, v = mask && !mask(i, j) ? 0 : value(i, j);
      img.data[k] = rgb[0]; img.data[k + 1] = rgb[1]; img.data[k + 2] = rgb[2]; img.data[k + 3] = Math.round(255 * amax * PE.clamp(v, 0, 1)); }
    hc.putImageData(img, 0, 0); c.save(); c.imageSmoothingEnabled = true; c.drawImage(heatCanvas, x, y, w, h); c.restore();
  }
  // A mark legend drawn into small canvases with data-mark="ring|strike|tick".
  function legends(root = document) {
    root.querySelectorAll('canvas[data-mark]').forEach(el => {
      const draw = () => { const r = devicePixelRatio || 1; el.width = 22 * r; el.height = 22 * r; const c = el.getContext('2d'); c.setTransform(r, 0, 0, r, 0, 0); c.clearRect(0, 0, 22, 22);
        const m = el.dataset.mark; if (m === 'ring') ring(c, 11, 11, 7.5, PE.css('--yellow'), { seed: 3 }); else if (m === 'strike') strike(c, 11, 11, 16, PE.css('--red'), { seed: 4 }); else tick(c, 11, 12, 15, PE.css('--green')); };
      draw(); PE.onTheme(draw);
    });
  }
  // Pressed state for a group of buttons.
  const press = (btns, on) => btns.forEach(b => b.setAttribute('aria-pressed', String(b === on)));
  return { NAMES, ring, strike, tick, sprite, tile, entropy, gmm2, legends, press, heat };
})();

/* Figure cores shared by the long page and the atlas: FIG.<name>(canvas, opts) -> handle. */
const FIG = {};
