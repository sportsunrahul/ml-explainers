/* KIT.photo: the page's running-example "photo" (a mug on a table, a window, a ball, a plant, a book), drawn
   procedurally so every augmentation figure can crop, jitter, grey, blur, rotate, cut out, add noise or take
   Sobel edges of the same image. The pixel colors are image content, not page roles, so they are fixed hex values
   and do not follow the theme. Pure pixel loops (no ctx.filter), so every browser renders the same views.
   API (all deterministic from the seed):
     KIT.photoBase(seed)                 -> cached offscreen canvas (KIT.PHOTO.BASE px square) of scene `seed`
     KIT.photoView(seed, view)           -> offscreen canvas (KIT.PHOTO.RES px) of one view; view = {
                                            crop: {x, y, w, h} (fractions of the base), flip: bool,
                                            seq: [op, ...] applied in order after the crop }
       ops: {k:'crop', x, y, w, h} | {k:'cutout', x, y, s} | {k:'color', b, c, s, h, grey} | {k:'sobel'}
            {k:'noise', sigma, seed} | {k:'blur', sigma} | {k:'rotate', deg}   (sigma in px at RES; h in turns)
     KIT.photoOp(kind, rng)              -> one random op of a kind, with SimCLR-like ranges (App. A)
     KIT.photoAug(rng)                   -> a full SimCLR-style view: crop + flip + color (p 0.8) + grey (p 0.2)
                                            + blur (p 0.5)
     KIT.photo(ctx, x, y, w, h, seed, view, alpha) draws a view scaled into a rectangle
     KIT.photoHist(canvas, bins) / KIT.histOverlap(a, b): per-channel color histogram and its overlap in [0, 1] */
KIT.PHOTO = { BASE: 192, RES: 128 };
KIT.photoScenes = {};
KIT.photoBase = seed => {
  if (KIT.photoScenes[seed]) return KIT.photoScenes[seed];
  const S = KIT.PHOTO.BASE, c = document.createElement('canvas'); c.width = c.height = S;
  const g = c.getContext('2d'), r = PE.rng(1000 + seed * 7919), pick = a => a[Math.floor(r() * a.length)];
  const hsl = (h, s, l) => `hsl(${h.toFixed(0)},${s.toFixed(0)}%,${l.toFixed(0)}%)`;
  // seed 0 is the canonical photo (beige wall, wooden table, red mug); other seeds vary wall, table and object colors widely
  const wallH = seed === 0 ? 40 : 360 * r(), wallS = seed === 0 ? 30 : 15 + 35 * r(), wallL = seed === 0 ? 78 : 45 + 40 * r();
  const TAB = [[28, 45, 46], [20, 40, 28], [40, 10, 86], [210, 18, 48], [150, 25, 36], [0, 0, 22]];
  const [tableH, tableS, tableL] = seed === 0 ? TAB[0] : TAB[Math.floor(r() * TAB.length)], mugH = seed === 0 ? 8 : 360 * r();
  const ty = S * (.56 + .08 * r());
  // wall with a soft vertical gradient
  const wg = g.createLinearGradient(0, 0, 0, ty);
  wg.addColorStop(0, hsl(wallH, wallS, wallL)); wg.addColorStop(1, hsl(wallH, wallS * .85, wallL - 14)); g.fillStyle = wg; g.fillRect(0, 0, S, ty);
  // window with sky and cross bars
  const wx = S * (r() < .5 ? .06 : .58), wy = S * .08, ww = S * .34, wh = S * .3;
  const sk = g.createLinearGradient(0, wy, 0, wy + wh); sk.addColorStop(0, hsl(205, 70, 70)); sk.addColorStop(1, hsl(195, 60, 86));
  g.fillStyle = hsl(wallH, 15, 92); g.fillRect(wx - 4, wy - 4, ww + 8, wh + 8); g.fillStyle = sk; g.fillRect(wx, wy, ww, wh);
  g.fillStyle = hsl(wallH, 15, 92); g.fillRect(wx + ww / 2 - 2, wy, 4, wh); g.fillRect(wx, wy + wh / 2 - 2, ww, 4);
  // table top and front edge
  const tg = g.createLinearGradient(0, ty, 0, S); tg.addColorStop(0, hsl(tableH, tableS, tableL)); tg.addColorStop(1, hsl(tableH, tableS * 1.1, tableL * .66));
  g.fillStyle = tg; g.fillRect(0, ty, S, S - ty);
  g.strokeStyle = hsl(tableH, tableS, tableL * .5); g.globalAlpha = .35; g.lineWidth = 1;
  for (let i = 0; i < 6; i++) { const y = ty + 6 + i * (S - ty) / 6 + 3 * r(); g.beginPath(); g.moveTo(0, y); g.bezierCurveTo(S * .3, y + 3, S * .7, y - 3, S, y + 1); g.stroke(); }
  g.globalAlpha = 1; g.fillStyle = hsl(tableH, tableS, Math.min(92, tableL + 22)); g.fillRect(0, ty, S, 3);
  // plant: pot and leaves, on the opposite side from the mug
  const mx = S * (.42 + .16 * r()), side = mx > S / 2 ? -1 : 1, px = mx + side * S * (.26 + .06 * r());
  g.fillStyle = hsl(14, 55, 45); g.beginPath(); g.moveTo(px - 13, ty + 2); g.lineTo(px + 13, ty + 2); g.lineTo(px + 9, ty + 24); g.lineTo(px - 9, ty + 24); g.closePath(); g.fill();
  const leaf = 110 + 30 * r();
  for (let k = 0; k < 6; k++) { const a = -Math.PI / 2 + (k - 2.5) * .38, L = 22 + 12 * r();
    g.fillStyle = hsl(leaf, 50, 30 + 14 * (k % 2)); g.beginPath(); g.ellipse(px + Math.cos(a) * L / 2, ty + 2 + Math.sin(a) * L / 2, L / 2, 5, a, 0, 7); g.fill(); }
  // book lying on the table
  const bx = mx - side * S * (.2 + .08 * r()) - 22, by = ty + 26 + 10 * r();
  g.fillStyle = hsl(pick([215, 280, 340, 45]), 45, 40); g.fillRect(bx, by, 46, 11); g.fillStyle = hsl(50, 30, 88); g.fillRect(bx + 2, by + 11, 44, 3);
  // ball
  const balx = mx + side * S * .02 + (r() - .5) * 20, baly = S * .9, ballH = pick([48, 52, 190, 300]);
  const bgd = g.createRadialGradient(balx - 4, baly - 5, 2, balx, baly, 12); bgd.addColorStop(0, hsl(ballH, 90, 75)); bgd.addColorStop(1, hsl(ballH, 85, 45));
  g.fillStyle = bgd; g.beginPath(); g.arc(balx, baly, 11, 0, 7); g.fill();
  // mug: shadow, body, handle, rim, coffee, highlight
  const mw = S * .22, mh = S * .27, mtop = ty + 10 - mh;
  g.fillStyle = 'rgba(0,0,0,.22)'; g.beginPath(); g.ellipse(mx + 4, ty + 12, mw * .62, 6, 0, 0, 7); g.fill();
  g.strokeStyle = hsl(mugH, 65, 40); g.lineWidth = 6; g.beginPath(); g.ellipse(mx + mw / 2 + 2, mtop + mh * .45, 10, 14, 0, -Math.PI / 2, Math.PI / 2); g.stroke();
  const mg = g.createLinearGradient(mx - mw / 2, 0, mx + mw / 2, 0);
  mg.addColorStop(0, hsl(mugH, 65, 38)); mg.addColorStop(.35, hsl(mugH, 70, 56)); mg.addColorStop(1, hsl(mugH, 65, 34));
  g.fillStyle = mg; g.beginPath(); g.moveTo(mx - mw / 2, mtop); g.lineTo(mx - mw / 2, ty + 6);
  g.ellipse(mx, ty + 6, mw / 2, 5, 0, Math.PI, 0, true); g.lineTo(mx + mw / 2, mtop); g.closePath(); g.fill();
  g.fillStyle = hsl(mugH, 50, 74); g.beginPath(); g.ellipse(mx, mtop, mw / 2, 6, 0, 0, 7); g.fill();
  g.fillStyle = hsl(25, 55, 22); g.beginPath(); g.ellipse(mx, mtop + 1, mw / 2 - 3, 4, 0, 0, 7); g.fill();
  g.fillStyle = 'rgba(255,255,255,.35)'; g.fillRect(mx - mw / 2 + 6, mtop + 8, 4, mh - 18);
  KIT.photoScenes[seed] = c;
  return c;
};
KIT.photoPix = (c, fn) => {
  const g = c.getContext('2d'), im = g.getImageData(0, 0, c.width, c.height); fn(im.data, c.width, c.height); g.putImageData(im, 0, 0);
};
KIT.photoApply = (c, op) => {
  const R = c.width, g = c.getContext('2d'), cl = v => v < 0 ? 0 : v > 255 ? 255 : v;
  if (op.k === 'crop' || op.k === 'rotate') {
    const t = document.createElement('canvas'); t.width = t.height = R; const tg = t.getContext('2d'); tg.imageSmoothingQuality = 'high';
    if (op.k === 'crop') tg.drawImage(c, op.x * R, op.y * R, op.w * R, op.h * R, 0, 0, R, R);
    else { tg.translate(R / 2, R / 2); tg.rotate(op.deg * Math.PI / 180); tg.drawImage(c, -R / 2, -R / 2); }
    g.clearRect(0, 0, R, R); g.drawImage(t, 0, 0); return;
  }
  if (op.k === 'cutout') { g.fillStyle = '#000'; g.fillRect(op.x * R, op.y * R, op.s * R, op.s * R); return; }
  if (op.k === 'color') KIT.photoPix(c, d => {
    // brightness, contrast around the mean grey, saturation around the pixel's grey, hue as a YIQ rotation
    let m = 0; for (let i = 0; i < d.length; i += 4) m += .299 * d[i] + .587 * d[i + 1] + .114 * d[i + 2]; m /= d.length / 4;
    const ca = Math.cos(op.h * 2 * Math.PI), sa = Math.sin(op.h * 2 * Math.PI);
    for (let i = 0; i < d.length; i += 4) {
      let r = d[i] * op.b, gg = d[i + 1] * op.b, b = d[i + 2] * op.b;
      r = (r - m) * op.c + m; gg = (gg - m) * op.c + m; b = (b - m) * op.c + m;
      const y = .299 * r + .587 * gg + .114 * b;
      r = y + (r - y) * op.s; gg = y + (gg - y) * op.s; b = y + (b - y) * op.s;
      let I = .596 * r - .274 * gg - .322 * b, Q = .211 * r - .523 * gg + .312 * b; const I2 = I * ca - Q * sa; Q = I * sa + Q * ca; I = I2;
      r = y + .956 * I + .621 * Q; gg = y - .272 * I - .647 * Q; b = y - 1.106 * I + 1.703 * Q;
      if (op.grey) { const v = .299 * r + .587 * gg + .114 * b; r = gg = b = v; }
      d[i] = cl(r); d[i + 1] = cl(gg); d[i + 2] = cl(b);
    }
  });
  else if (op.k === 'noise') { const n = PE.gauss(PE.rng(op.seed || 1)); KIT.photoPix(c, d => { for (let i = 0; i < d.length; i += 4) for (let j = 0; j < 3; j++) d[i + j] = cl(d[i + j] + op.sigma * n()); }); }
  else if (op.k === 'sobel') KIT.photoPix(c, (d, W, H) => {
    const y = new Float32Array(W * H); for (let i = 0; i < W * H; i++) y[i] = .299 * d[4 * i] + .587 * d[4 * i + 1] + .114 * d[4 * i + 2];
    const at = (x, yy) => y[Math.min(H - 1, Math.max(0, yy)) * W + Math.min(W - 1, Math.max(0, x))];
    for (let yy = 0; yy < H; yy++) for (let x = 0; x < W; x++) {
      const gx = at(x + 1, yy - 1) + 2 * at(x + 1, yy) + at(x + 1, yy + 1) - at(x - 1, yy - 1) - 2 * at(x - 1, yy) - at(x - 1, yy + 1);
      const gy = at(x - 1, yy + 1) + 2 * at(x, yy + 1) + at(x + 1, yy + 1) - at(x - 1, yy - 1) - 2 * at(x, yy - 1) - at(x + 1, yy - 1);
      const v = cl(Math.hypot(gx, gy) * .7), i = 4 * (yy * W + x); d[i] = d[i + 1] = d[i + 2] = v;
    }
  });
  else if (op.k === 'blur') KIT.photoPix(c, (d, W, H) => {
    // three box passes per axis approximate a Gaussian of this sigma
    const r = Math.max(1, Math.round(Math.sqrt(4 * op.sigma * op.sigma + 1) / 2)), tmp = new Float32Array(d.length);
    const pass = (src, dst, dx, dy) => { for (let yy = 0; yy < H; yy++) for (let x = 0; x < W; x++) for (let j = 0; j < 3; j++) {
      let s = 0; for (let k = -r; k <= r; k++) { const xx = Math.min(W - 1, Math.max(0, x + k * dx)), y2 = Math.min(H - 1, Math.max(0, yy + k * dy)); s += src[4 * (y2 * W + xx) + j]; }
      dst[4 * (yy * W + x) + j] = s / (2 * r + 1); } };
    for (let p = 0; p < 3; p++) { pass(d, tmp, 1, 0); pass(tmp, d, 0, 1); }
  });
};
KIT.photoView = (seed, view = {}) => {
  const R = KIT.PHOTO.RES, B = KIT.PHOTO.BASE, c = document.createElement('canvas'); c.width = c.height = R;
  const g = c.getContext('2d'), cr = view.crop || { x: 0, y: 0, w: 1, h: 1 }; g.imageSmoothingQuality = 'high';
  if (view.flip) { g.translate(R, 0); g.scale(-1, 1); }
  g.drawImage(KIT.photoBase(seed), cr.x * B, cr.y * B, cr.w * B, cr.h * B, 0, 0, R, R);
  g.setTransform(1, 0, 0, 1, 0, 0);
  (view.seq || []).forEach(op => KIT.photoApply(c, op));
  return c;
};
// A random op of one kind. Ranges follow SimCLR App. A where it gives them (crop area 0.08-1, aspect 3/4-4/3, color strength 1,
// blur sigma 0.1-2.0 at 224 px); cutout size, noise level and rotation angles (90, 180, 270) are our choices for the drawing.
KIT.photoOp = (kind, r) => {
  const R = KIT.PHOTO.RES, U = (a, b) => a + (b - a) * r();
  if (kind === 'crop') { const A = U(.25, .6), ar = Math.exp(U(Math.log(3 / 4), Math.log(4 / 3))), w = Math.min(1, Math.sqrt(A * ar)), h = Math.min(1, Math.sqrt(A / ar));
    return { k: 'crop', x: U(0, 1 - w), y: U(0, 1 - h), w, h }; }
  if (kind === 'cutout') { const s = U(.3, .45); return { k: 'cutout', x: U(.05, .95 - s), y: U(.05, .95 - s), s }; }
  if (kind === 'color') return { k: 'color', b: U(.2, 1.8), c: U(.2, 1.8), s: U(.2, 1.8), h: U(-.2, .2), grey: r() < .2 };
  if (kind === 'sobel') return { k: 'sobel' };
  if (kind === 'noise') return { k: 'noise', sigma: U(30, 50), seed: Math.floor(r() * 1e6) };
  if (kind === 'blur') return { k: 'blur', sigma: U(1, 2) * R / 224 * 2.2 };
  if (kind === 'rotate') return { k: 'rotate', deg: [90, 180, 270][Math.floor(r() * 3)] };
  throw new Error('KIT.photoOp: unknown kind ' + kind);
};
// One SimCLR-style training view: random crop (area 0.08-1) and flip, then color (p 0.8) with grey (p 0.2), then blur (p 0.5).
KIT.photoAug = r => {
  const U = (a, b) => a + (b - a) * r(), A = U(.08, 1), ar = Math.exp(U(Math.log(3 / 4), Math.log(4 / 3)));
  const w = Math.min(1, Math.sqrt(A * ar)), h = Math.min(1, Math.sqrt(A / ar)), seq = [];
  if (r() < .8) { const c = KIT.photoOp('color', r); c.grey = false; seq.push(c); }
  if (r() < .2) seq.push({ k: 'color', b: 1, c: 1, s: 0, h: 0, grey: true });
  if (r() < .5) seq.push({ k: 'blur', sigma: U(.1, 2) * KIT.PHOTO.RES / 224 * 2.2 });
  return { crop: { x: U(0, 1 - w), y: U(0, 1 - h), w, h }, flip: r() < .5, seq };
};
KIT.photo = (ctx, x, y, w, h, seed, view, alpha = 1) => {
  const c = view instanceof HTMLCanvasElement ? view : KIT.photoView(seed, view);
  ctx.save(); ctx.globalAlpha = alpha; ctx.imageSmoothingQuality = 'high'; ctx.drawImage(c, x, y, w, h); ctx.restore();
  return c;
};
KIT.photoHist = (c, bins = 8) => {
  const d = c.getContext('2d').getImageData(0, 0, c.width, c.height).data, h = new Array(3 * bins).fill(0), n = d.length / 4;
  for (let i = 0; i < d.length; i += 4) for (let j = 0; j < 3; j++) h[j * bins + Math.min(bins - 1, Math.floor(d[i + j] / 256 * bins))] += 1 / n;
  return h;
};
KIT.histOverlap = (a, b) => a.reduce((s, v, i) => s + Math.min(v, b[i]), 0) / 3;
