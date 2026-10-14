/* ch-before: the same desk scene as four models would label it. Tabs: FCN (coarse score map, stride buttons),
   U-Net (full-resolution map through skip connections), Mask R-CNN (box, then a small per-box mask),
   Mask2Former (N queries, each a mask or "no object"; stuff and things). Toy throughout. */
SCENES.before = ctx => {
  const el = mkCanvas(ctx.stage);
  const CLASSES = [['cup', 'mugset'], ['plant', 'plant'], ['lamp', 'lamp'], ['book', 'book']];
  const S = { step: 0, stride: 32, u: 1 };
  const GX = 160, GY = 100;
  const rast = CLASSES.map(([, id]) => TOY.raster(TOY.byId[id], GX, GY));
  // class index per raster cell (-1 = background or not in the label list)
  const lab = new Int8Array(GX * GY).fill(-1); rast.forEach((m, k) => m.forEach((v, i) => { if (v) lab[i] = k; }));
  const strideCells = { 32: 16, 16: 8, 8: 4 };
  let cancel = () => {};
  const glow = {}; // equation terms (long page): mask, query, pixel
  const onLink = e => { const k = e.detail; if (S.step !== 3) { const t = ctx.q('[role="tab"][data-step="3"]'); if (t) t.click(); }
    PE.animate(1100, u => { glow[k] = Math.sin(Math.PI * u); cv.redraw(); }, () => { glow[k] = 0; cv.redraw(); }); };
  ctx.article.addEventListener('pe-link', onLink);
  const cv = PE.canvas(el, S3ASPECT(ctx.stage), draw);
  const strideBtns = ctx.qa('[data-stride]');
  const offs = strideBtns.map(b => listen(b, 'click', () => { S.stride = +b.dataset.stride; press(strideBtns, x => x === b); if (S.step !== 0) { const t = ctx.q('[role="tab"][data-step="0"]'); if (t) t.click(); } else animateIn(); }));
  const pressed = strideBtns.find(b => b.getAttribute('aria-pressed') === 'true'); if (pressed) S.stride = +pressed.dataset.stride; // survives a theme remount
  press(strideBtns, b => +b.dataset.stride === S.stride);
  function coarse(cell) { // majority vote per cell
    const nx = Math.ceil(GX / cell), ny = Math.ceil(GY / cell), out = new Int8Array(nx * ny).fill(-1);
    for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) { const cnt = new Array(CLASSES.length + 1).fill(0);
      for (let y = j * cell; y < Math.min(GY, (j + 1) * cell); y++) for (let x = i * cell; x < Math.min(GX, (i + 1) * cell); x++) cnt[lab[y * GX + x] + 1]++;
      let best = 0; cnt.forEach((v, k) => { if (v > cnt[best]) best = k; }); out[j * nx + i] = best - 1; }
    return { nx, ny, out };
  }
  function draw(c, W, H) {
    c.clearRect(0, 0, W, H);
    const narrow = W < 620, pad = 12, top = 34, bottom = ctx.reserve || 0;
    const imgW = narrow ? W - 2 * pad : W * .58, imgH = narrow ? (H - bottom) * .55 : Math.min(H - bottom - top - pad, imgW / TOY.WW);
    const top2 = narrow ? top : Math.max(top, (H - bottom - imgH) / 2);
    const V = TOY.fit(pad, top2, imgW, imgH), G = PE.css('--green'), u = S.u;
    TOY.drawWorld(c, V, { dim: .25 });
    const cellW = V.w / GX, cellH = V.h / GY;
    if (S.step === 0) {
      const { nx, ny, out } = coarse(strideCells[S.stride]), cs = strideCells[S.stride];
      c.save(); c.globalAlpha = .42 * u; c.fillStyle = G;
      for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) if (out[j * nx + i] >= 0) c.fillRect(V.ox + i * cs * cellW, V.oy + j * cs * cellH, Math.min(cs, GX - i * cs) * cellW, Math.min(cs, GY - j * cs) * cellH);
      c.globalAlpha = .5 * u; c.strokeStyle = PE.css('--rule'); c.lineWidth = 1;
      for (let i = 0; i <= nx; i++) { const x = V.ox + Math.min(GX, i * cs) * cellW; c.beginPath(); c.moveTo(x, V.oy); c.lineTo(x, V.oy + V.h); c.stroke(); }
      for (let j = 0; j <= ny; j++) { const y = V.oy + Math.min(GY, j * cs) * cellH; c.beginPath(); c.moveTo(V.ox, y); c.lineTo(V.ox + V.w, y); c.stroke(); }
      c.restore();
      classLabels(c, V, u);
    } else if (S.step === 1) {
      CLASSES.forEach(([, id]) => TOY.drawMask(c, V, TOY.byId[id], { alpha: u, lift: 0, fill: .42 }));
      classLabels(c, V, u);
    } else if (S.step === 2) {
      CLASSES.forEach(([name, id], k) => { const n = TOY.byId[id], [x0, y0, x1, y1] = TOY.bbox(n), t = PE.clamp(u * 1.6 - k * .15, 0, 1);
        const X0 = V.X(x0), Y0 = V.Y(y0), X1 = V.X(x1), Y1 = V.Y(y1);
        c.save(); c.globalAlpha = Math.min(1, t * 2); c.strokeStyle = G; c.lineWidth = 1.5; c.setLineDash([5, 3]); c.strokeRect(X0, Y0, X1 - X0, Y1 - Y0); c.restore();
        const m = 14, mt = PE.clamp(t * 2 - 1, 0, 1); // a small fixed-size mask grid per box (Mask R-CNN predicts m x m; drawn coarse here)
        c.save(); c.globalAlpha = .45 * mt; c.fillStyle = G;
        for (let j = 0; j < m; j++) for (let i = 0; i < m; i++) { const wx = x0 + (i + .5) / m * (x1 - x0), wy = y0 + (j + .5) / m * (y1 - y0); if (TOY.inside(n, wx, wy)) c.fillRect(X0 + i * (X1 - X0) / m, Y0 + j * (Y1 - Y0) / m, (X1 - X0) / m + .5, (Y1 - Y0) / m + .5); }
        c.restore();
        if (t > 0) { c.save(); c.globalAlpha = Math.min(1, t * 2); c.font = '600 11px system-ui, sans-serif'; c.fillStyle = PE.css('--panel'); const tw = c.measureText(name).width; c.fillRect(X0, Y0 - 16, tw + 8, 16); PE.label(c, name, X0 + 4, Y0 - 4, G, 11); c.restore(); } });
    } else {
      // panoptic: stuff (wall, table) then things; every pixel gets exactly one label
      c.save(); c.globalAlpha = .16 * u; c.fillStyle = G; c.fillRect(V.ox, V.oy, V.w, V.Y(.62) - V.oy); c.globalAlpha = .1 * u; c.fillRect(V.ox, V.Y(.62), V.w, V.oy + V.h - V.Y(.62));
      c.globalAlpha = .7 * u; c.strokeStyle = G; c.setLineDash([3, 4]); c.lineWidth = 1.5; c.strokeRect(V.ox + 2, V.oy + 2, V.w - 4, V.Y(.62) - V.oy - 3); c.strokeRect(V.ox + 2, V.Y(.62) + 1, V.w - 4, V.oy + V.h - V.Y(.62) - 3); c.restore();
      CLASSES.forEach(([, id], k) => TOY.drawMask(c, V, TOY.byId[id], { alpha: PE.clamp(u * 1.5 - k * .12, 0, 1), lift: 0, fill: .42, lw: 2 + 3 * (glow.mask || 0) }));
      PE.label(c, 'wall (stuff)', V.ox + 8, V.oy + 16, G, 11); PE.label(c, 'table (stuff)', V.ox + 8, V.oy + V.h - 8, G, 11);
      classLabels(c, V, u);
    }
    diagram(c, narrow ? { x: pad, y: top + imgH + 16, w: W - 2 * pad, h: H - bottom - top - imgH - 22 } : { x: pad + imgW + 18, y: top2, w: W - imgW - 2 * pad - 18, h: imgH }, narrow);
  }
  function classLabels(c, V, u) {
    const spots = { cup: [1.12, .52], plant: [1.455, .43], lamp: [.2, .2], book: [.73, .985] };
    c.save(); c.globalAlpha = u; Object.entries(spots).forEach(([k, [x, y]]) => { PE.label(c, k, V.X(x), V.Y(y), PE.css('--green'), 12, 'center', 700); }); c.restore();
    c.save(); c.globalAlpha = .9 * u; PE.label(c, 'window: not a class', V.X(.75), V.Y(.07), PE.css('--brown'), 11, 'center'); PE.label(c, 'pencil: not a class', V.X(.86), V.Y(.77), PE.css('--brown'), 11, 'center'); c.restore();
  }
  function diagram(c, b, narrow) {
    const B = PE.css('--blue'), G = PE.css('--green'), M = PE.css('--muted'), F = PE.css('--fg');
    c.save(); c.strokeStyle = PE.css('--rule'); c.lineWidth = 1; c.beginPath(); c.roundRect(b.x, b.y, b.w, b.h, 10); c.stroke(); c.restore();
    const x = b.x + 12, y = b.y + 12, w = b.w - 24, h = b.h - 24;
    if (S.step === 0 || S.step === 1) {
      const lv = 4, enc = [], dec = [];
      for (let i = 0; i < lv; i++) { const bh = h * (1 - i * .2) * .7, bw = Math.max(10, w * .07); enc.push({ x: x + i * w * .11, y: y + (h - bh) / 2 - (narrow ? 0 : 0), w: bw, h: bh }); }
      const bot = enc[lv - 1];
      for (let i = 0; i < lv; i++) { const e = enc[lv - 1 - i]; dec.push({ x: x + w * .45 + i * w * .11, y: e.y, w: e.w, h: e.h }); }
      enc.forEach((r, i) => { c.fillStyle = B; c.globalAlpha = .55; c.fillRect(r.x, r.y, r.w, r.h); c.globalAlpha = 1; if (i) PE.arrow(c, enc[i - 1].x + enc[i - 1].w + 2, y + h / 2, r.x - 2, y + h / 2, M, 1.2, 6); });
      PE.label(c, 'downsample', enc[0].x, b.y + b.h - 8, M, 11);
      if (S.step === 1) {
        dec.forEach((r, i) => { c.fillStyle = G; c.globalAlpha = .55; c.fillRect(r.x, r.y, r.w, r.h); c.globalAlpha = 1; if (i) PE.arrow(c, dec[i - 1].x + dec[i - 1].w + 2, y + h / 2, r.x - 2, y + h / 2, M, 1.2, 6); });
        for (let i = 0; i < lv - 1; i++) { const e = enc[i], d = dec[lv - 1 - i]; c.save(); c.strokeStyle = B; c.setLineDash([4, 3]); c.lineWidth = 1.3; c.beginPath(); c.moveTo(e.x + e.w, e.y + 6 + i * 3); c.lineTo(d.x, d.y + 6 + i * 3); c.stroke(); c.restore(); }
        PE.label(c, 'skip connections carry detail across', x + w * .02, y + 10, B, 11);
        PE.label(c, 'upsample', dec[0].x, b.y + b.h - 8, M, 11);
      } else {
        const out = { x: x + w * .58, y: bot.y, w: w * .3, h: bot.h };
        PE.arrow(c, bot.x + bot.w + 4, y + h / 2, out.x - 4, y + h / 2, M, 1.2, 6);
        c.fillStyle = G; c.globalAlpha = .55; c.fillRect(out.x, out.y, out.w, out.h); c.globalAlpha = 1;
        PE.label(c, 'class scores per cell', out.x, out.y - 6, G, 11);
        PE.label(c, `stride ${S.stride}: one cell = ${S.stride} x ${S.stride} pixels`, x, y + 10, F, 11);
        PE.label(c, 'upsample to image size', out.x, b.y + b.h - 8, M, 11);
      }
    } else if (S.step === 2) {
      const cols = ['image', 'backbone', 'boxes', 'features\nper box', 'class, box,\nmask m x m'];
      const vert = !narrow, bw = vert ? Math.min(w * .7, 180) : (w - (cols.length - 1) * 10) / cols.length, bh = vert ? Math.min(46, (h - 30 - 4 * 14) / 5) : Math.min(70, h * .5);
      cols.forEach((t, i) => { const bx = vert ? x + (w - bw) / 2 : x + i * (bw + 10), by = vert ? y + 26 + i * (bh + 14) : y + (h - bh) / 2; PE.box(c, bx, by, bw, bh, i < 2 ? B : G, null); t.split('\n').forEach((s, k) => PE.label(c, s, bx + bw / 2, by + bh / 2 + 4 + (k - (t.split('\n').length - 1) / 2) * 13, i < 2 ? B : G, narrow ? 9 : 11, 'center'));
        if (i) { if (vert) PE.arrow(c, bx + bw / 2, by - 13, bx + bw / 2, by - 1, M, 1.2, 5); else PE.arrow(c, bx - 9, by + bh / 2, bx - 1, by + bh / 2, M, 1.2, 5); } });
      PE.label(c, 'find each object, then mask inside its box', x, y + 10, F, 11);
    } else {
      const Nq = 8, qx = x + 6, used = [true, true, true, true, true, true, false, false], names = ['cup', 'plant', 'lamp', 'book', 'wall', 'table', 'no object', 'no object'];
      const pix = { x: x + w * .5, y: y + 44, w: w * .45, h: h - 52 };
      c.fillStyle = B; c.globalAlpha = .4 + .4 * (glow.pixel || 0); c.fillRect(pix.x, pix.y, pix.w, pix.h); c.globalAlpha = 1; if (glow.pixel) { c.save(); c.strokeStyle = B; c.lineWidth = 2 + 3 * glow.pixel; c.strokeRect(pix.x, pix.y, pix.w, pix.h); c.restore(); }
      PE.label(c, 'per-pixel embeddings', pix.x, pix.y - 6, B, 11);
      for (let i = 0; i < Nq; i++) { const qy = y + 44 + i * (h - 52) / Nq + 8, col = used[i] ? G : PE.css('--brown');
        PE.dot(c, qx + 6, qy, 5 + 3 * (used[i] ? glow.query || 0 : 0), col); PE.label(c, names[i], qx + 16, qy + 4, col, narrow ? 9 : 11);
        if (used[i]) PE.arrow(c, qx + w * .3, qy, pix.x - 4, pix.y + pix.h * (i + .5) / Nq, PE.css('--muted'), 1, 5, .6); }
      PE.label(c, 'N queries; each dot product with the pixel', x, y + 6, F, 11);
      PE.label(c, 'embeddings gives one mask', x, y + 19, F, 11);
    }
  }
  const text = [
    'FCN labels every pixel with one class from a fixed list. Its scores come out on a coarse grid; coarser strides blur edges.',
    'U-Net adds skip connections, so fine detail from early layers reaches the output. The map is sharp, but the class list is still fixed.',
    'Mask R-CNN finds each object first, then predicts a small mask inside its box. Two cups would be two instances, not one blob.',
    'Mask2Former predicts a set of masks, each with a class or "no object". One design covers semantic, instance and panoptic segmentation.'];
  function readout() { ctx.hud(badge('toy', text[S.step] + ' The window and the pencil are not in the label list, so no model here can mask them.')); }
  function animateIn() { cancel(); cancel = PE.animate(700, t => { S.u = t; cv.redraw(); }); }
  readout();
  return { flat: true, step(n) { S.step = n; readout(); animateIn(); }, resize: () => cv.resize(), destroy: () => { cancel(); offs.forEach(f => f()); ctx.article.removeEventListener('pe-link', onLink); cv.destroy(); } };
};
