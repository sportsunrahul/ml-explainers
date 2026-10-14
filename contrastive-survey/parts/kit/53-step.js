/* viz-step (ch-step interlude): one SimCLR training step at N = 4096, ten stages. The drawing shows 4 of the 4096 images
   (KIT.photo scenes, SimCLR-style random views from KIT.photoAug); the embeddings, bar heights and losses are toy values;
   the shapes in KIT.simclrStepText are computed from the paper's config (N = 4096, 224 x 224 crops, h 2048, z 128). */
KIT.STEP_SHOWN = 4;
KIT.stepBatch = seed => {
  const r = PE.rng(seed * 7919 + 13), g = PE.gauss(r), imgs = [];
  for (let i = 0; i < KIT.STEP_SHOWN; i++) {
    const s = (seed * KIT.STEP_SHOWN + i) % 23, base = 2 * Math.PI * i / KIT.STEP_SHOWN + .5 * r();
    const hb = Array.from({ length: 14 }, () => r());
    imgs.push({ seed: s, views: [KIT.photoAug(r), KIT.photoAug(r)],
      ang: [base - .22 + .1 * g(), base + .22 + .1 * g()], len: [.55 + .8 * r(), .55 + .8 * r()],
      h: [hb.map(v => PE.clamp(v + .25 * g(), .05, 1)), hb.map(v => PE.clamp(v + .25 * g(), .05, 1))],
      z: [0, 1].map(() => Array.from({ length: 6 }, () => .15 + .85 * r())) });
  }
  imgs.forEach(im => { im.canv = im.views.map(v => KIT.photoView(im.seed, v)); });
  const th = imgs.flatMap(im => im.ang), m = KIT.ntxentRows(th, .5);
  return { imgs, th, m };
};
KIT.simclrStep = (canvas, opts = {}) => {
  const state = { stage: 0, seed: 1, t: 1 };
  let batch = KIT.stepBatch(1), cancel = () => {};
  const glow = opts.fig ? PE.linkPulses(opts.fig, () => cv.redraw()) : () => 0;
  const n = 2 * KIT.STEP_SHOWN;
  const views = () => batch.imgs.flatMap((im, i) => im.canv.map((c, j) => ({ c, i, j })));
  const C = () => ({ B: PE.css('--blue'), Y: PE.css('--yellow'), G: PE.css('--green'), R: PE.css('--red'), Br: PE.css('--brown'), M: PE.css('--muted'), F: PE.css('--fg'), P2: PE.css('--panel2'), RU: PE.css('--rule') });
  const frame = (ctx, x, y, s, col, lw = 2.5) => { ctx.save(); ctx.strokeStyle = col; ctx.lineWidth = lw; ctx.strokeRect(x - lw / 2, y - lw / 2, s + lw, s + lw); ctx.restore(); };
  const bars = (ctx, x, y, w, h, vals, col, a = 1) => { const bw = w / vals.length; ctx.save(); ctx.fillStyle = col; ctx.globalAlpha = a;
    vals.forEach((v, k) => ctx.fillRect(x + k * bw + .5, y + h - v * h, bw - 1.5, v * h)); ctx.restore(); };
  const boxT = (ctx, x, y, w, h, col, lines, c) => { ctx.save(); ctx.fillStyle = c.P2; ctx.strokeStyle = col; ctx.lineWidth = 1.5; ctx.beginPath(); ctx.roundRect(x, y, w, h, 8); ctx.fill(); ctx.stroke(); ctx.restore();
    lines.forEach((t, k) => PE.label(ctx, t, x + w / 2, y + h / 2 + 4 + (k - (lines.length - 1) / 2) * 14, k ? c.M : col, k ? 10.5 : 12, 'center', k ? 400 : 600)); };
  // rows of the 8 drawn views: thumbnail on the left, then whatever the stage draws to the right
  const rowsLayout = (W, H) => { const top = 22, rh = (H - top - 6) / n; return { top, rh, s: Math.min(rh - 4, 40) }; };
  const matrix = (ctx, x0, y0, cs, t, c, masks) => {
    const S = batch.m.S, P = batch.m.P;
    for (let i = 0; i < n; i++) { const a = PE.clamp(t * 1.6 * n - i, 0, 1); if (!a) continue;
      for (let k = 0; k < n; k++) { const x = x0 + k * cs, y = y0 + i * cs; ctx.save(); ctx.globalAlpha = a;
        if (masks && k === i) { ctx.fillStyle = c.P2; ctx.fillRect(x + .5, y + .5, cs - 1, cs - 1); ctx.strokeStyle = c.M; ctx.lineWidth = 1;
          ctx.beginPath(); ctx.moveTo(x + 3, y + 3); ctx.lineTo(x + cs - 3, y + cs - 3); ctx.moveTo(x + cs - 3, y + 3); ctx.lineTo(x + 3, y + cs - 3); ctx.stroke(); }
        else if (masks) { const pos = k === (i ^ 1); ctx.fillStyle = pos ? c.G : c.R; ctx.globalAlpha = a * (.1 + .85 * P[i][k]); ctx.fillRect(x + .5, y + .5, cs - 1, cs - 1);
          if (pos) { ctx.globalAlpha = a; ctx.strokeStyle = c.Y; ctx.lineWidth = 2; ctx.strokeRect(x + 1.5, y + 1.5, cs - 3, cs - 3); } }
        else { ctx.fillStyle = c.F; ctx.globalAlpha = a * (.06 + .6 * (S[i][k] + 1) / 2); ctx.fillRect(x + .5, y + .5, cs - 1, cs - 1); }
        ctx.restore(); } }
    for (let k = 0; k < n; k++) { const lab = (Math.floor(k / 2) + 1) + 'ab'[k % 2], col = k % 2 ? c.Y : c.B;
      PE.label(ctx, lab, x0 + (k + .5) * cs, y0 - 5, col, 10, 'center'); PE.label(ctx, lab, x0 - 5, y0 + (k + .65) * cs, col, 10, 'right'); }
  };
  const DRAW = [
    (ctx, W, H, t, c) => { // sample
      const nar = KIT.narrow(W), cols = nar ? 2 : 4, gap = 14, s = Math.min((W - gap * (cols - 1) - 10) / cols, (H - 40) / (4 / cols) - 26);
      batch.imgs.forEach((im, i) => { const x = 5 + (i % cols) * (s + gap), y = 8 + Math.floor(i / cols) * (s + 26);
        KIT.photo(ctx, x, y, s, s, im.seed, {}, PE.clamp(t * 4 - i, 0, 1)); PE.label(ctx, 'image ' + (i + 1), x, y + s + 15, c.F, 11); });
      PE.label(ctx, '+ 4092 more images in the batch', W - 6, H - 6, c.M, 11, 'right', 400);
    },
    (ctx, W, H, t, c) => { // augment
      const nar = KIT.narrow(W), cols = nar ? 2 : 4, cw = (W - 10) / cols, rows = 4 / cols, rh = (H - 6) / rows;
      const s0 = Math.min(cw * .5, rh * .42), s1 = Math.min((cw - 18) / 2, rh - s0 - 40), oy = Math.max(0, (rh - s0 - s1 - 30) / 2);
      batch.imgs.forEach((im, i) => { const x = 5 + (i % cols) * cw, y = 6 + oy + Math.floor(i / cols) * rh;
        KIT.photo(ctx, x + (cw - s0) / 2 - 6, y, s0, s0, im.seed, {});
        im.canv.forEach((cv2, j) => { const vx = x + j * (s1 + 8), vy = y + s0 + 22, a = PE.clamp(t * 2 - j * .4, 0, 1), sz = s1 * (.6 + .4 * a);
          KIT.photo(ctx, vx + (s1 - sz) / 2, vy + (s1 - sz) / 2, sz, sz, 0, cv2, a); if (a > .05) frame(ctx, vx + (s1 - sz) / 2, vy + (s1 - sz) / 2, sz, j ? c.Y : c.B, 2);
          PE.label(ctx, (i + 1) + 'ab'[j], vx, vy - 5, j ? c.Y : c.B, 10.5); });
      });
    },
    (ctx, W, H, t, c) => { // encoder
      const L = rowsLayout(W, H), bx = W * .3, bw = Math.min(150, W * .26), hx = bx + bw + 30, hw = W - hx - 8;
      PE.label(ctx, 'views', 0, 14, c.M, 11, 'left', 400); PE.label(ctx, KIT.narrow(W) ? 'h (2048 each)' : 'h, 2048 numbers each (14 drawn)', hx, 14, c.M, 11, 'left', 400);
      views().forEach((v, k) => { const y = L.top + k * L.rh; KIT.photo(ctx, 4, y, L.s, L.s, 0, v.c); frame(ctx, 4, y, L.s, v.j ? c.Y : c.B, 1.5);
        ctx.save(); ctx.strokeStyle = c.RU; ctx.beginPath(); ctx.moveTo(10 + L.s, y + L.s / 2); ctx.lineTo(bx, L.top + (H - L.top) / 2); ctx.stroke(); ctx.restore();
        bars(ctx, hx, y + 2, hw * PE.clamp(t * 1.4 - k * .05, 0, 1), L.s - 2, batch.imgs[v.i].h[v.j], v.j ? c.Y : c.B, .85); });
      boxT(ctx, bx, L.top + (H - L.top) / 2 - 34, bw, 68, c.F, KIT.narrow(W) ? ['ResNet-50 f', 'one network'] : ['ResNet-50 f', 'one network,', 'same weights for all'], c);
    },
    (ctx, W, H, t, c) => { // head
      const L = rowsLayout(W, H), hw = W * .3, bx = hw + 34, bw = Math.min(140, W * .24), zx = bx + bw + 30, zw = Math.min(W - zx - 8, hw * .5);
      PE.label(ctx, 'h (2048)', 0, 14, c.M, 11, 'left', 400); PE.label(ctx, 'z (128)', zx, 14, c.M, 11, 'left', 400);
      views().forEach((v, k) => { const y = L.top + k * L.rh, col = v.j ? c.Y : c.B; bars(ctx, 0, y + 2, hw, L.s - 2, batch.imgs[v.i].h[v.j], col, .45);
        bars(ctx, zx, y + 2, zw * PE.clamp(t * 1.4 - k * .05, 0, 1), L.s - 2, batch.imgs[v.i].z[v.j], col, .9); });
      boxT(ctx, bx, L.top + (H - L.top) / 2 - 34, bw, 68, c.F, KIT.narrow(W) ? ['head g', 'MLP'] : ['head g', 'Linear, ReLU, Linear', 'used only for the loss'], c);
    },
    (ctx, W, H, t, c) => { // normalize
      const R = Math.min(H * .36, W * .22), cx = W * (KIT.narrow(W) ? .5 : .3), cy = H / 2 + 6;
      PE.ring(ctx, cx, cy, R, c.RU, 1.5); PE.label(ctx, 'length 1', cx + R * .72, cy - R * .78, c.M, 11, 'left', 400);
      batch.imgs.forEach((im, i) => im.ang.forEach((a, j) => { const L = PE.lerp(im.len[j], 1, t) * R, x = cx + L * Math.cos(a), y = cy - L * Math.sin(a), col = j ? c.Y : c.B;
        PE.polyline(ctx, [[cx, cy], [x, y]], col, 1.5, null, .55); PE.dot(ctx, x, y, 5.5, col);
        PE.label(ctx, (i + 1) + 'ab'[j], cx + (L + 13) * Math.cos(a), cy - (L + 13) * Math.sin(a) + 4, col, 10.5, 'center'); }));
      if (!KIT.narrow(W)) { const x = W * .6; PE.label(ctx, 'z / |z|: every row scaled to length 1,', x, H * .42, c.F, 12, 'left', 400);
        PE.label(ctx, 'so a dot product is a cosine', x, H * .42 + 18, c.F, 12, 'left', 400);
        PE.label(ctx, '(a 2D sketch of 128-number vectors)', x, H * .42 + 40, c.M, 11, 'left', 400); }
    },
    (ctx, W, H, t, c) => { // similarity
      const cs = Math.min((H - 40) / n, (W * (KIT.narrow(W) ? .8 : .45)) / n), x0 = 34, y0 = 26;
      matrix(ctx, x0, y0, cs, t, c, false);
      const tx = KIT.narrow(W) ? x0 : x0 + n * cs + 30, ty = KIT.narrow(W) ? y0 + n * cs + 22 : H * .35;
      PE.label(ctx, '8 drawn views: an 8 x 8 table', tx, ty, c.F, 12, 'left', 400);
      PE.label(ctx, 'the real one is 8192 x 8192', tx, ty + 18, c.F, 12, 'left', 400);
      PE.label(ctx, 'brighter = more similar', tx, ty + 38, c.M, 11, 'left', 400);
    },
    (ctx, W, H, t, c) => { // masks
      const cs = Math.min((H - 40) / n, (W * (KIT.narrow(W) ? .8 : .45)) / n), x0 = 34, y0 = 26;
      matrix(ctx, x0, y0, cs, 1, c, true);
      const tx = KIT.narrow(W) ? x0 : x0 + n * cs + 30, ty = KIT.narrow(W) ? y0 + n * cs + 22 : H * .3;
      PE.label(ctx, 'x  diagonal: a view vs itself, removed', tx, ty, c.M, 12);
      PE.label(ctx, 'yellow box: the partner view (positive)', tx, ty + 20, c.Y, 12);
      PE.label(ctx, 'red: every other view (negatives)', tx, ty + 40, c.R, 12);
      PE.label(ctx, 'shading: the row softmax (toy, \u03c4 = 0.5)', tx, ty + 62, c.M, 11, 'left', 400);
    },
    (ctx, W, H, t, c) => { // loss
      const L = batch.m.loss, top = 24, rh = (H - top - 40) / n, lx = 40, mx = Math.max(...L), bw = W - lx - 70;
      PE.label(ctx, 'each row: cross-entropy of "which view is my partner?"', 0, 14, c.M, 11, 'left', 400);
      L.forEach((v, i) => { const y = top + i * rh, col = i % 2 ? c.Y : c.B; PE.label(ctx, 'row ' + (Math.floor(i / 2) + 1) + 'ab'[i % 2], 0, y + rh * .65, col, 10.5);
        ctx.save(); ctx.fillStyle = c.R; ctx.globalAlpha = .85; ctx.fillRect(lx, y + 2, v / mx * bw * t, rh - 5); ctx.restore();
        PE.label(ctx, v.toFixed(2), lx + v / mx * bw * t + 5, y + rh * .65, c.F, 10.5, 'left', 400); });
      PE.label(ctx, 'mean of the rows = the loss: ' + batch.m.mean.toFixed(3) + ' (toy)', lx, H - 10, c.R, 12);
    },
    (ctx, W, H, t, c) => { // gradient: the weights on one row (anchor 1a), then the backward path
      const nar = KIT.narrow(W), P = batch.m.P[0], tau = .5, w = P.map((p, k) => k === 0 ? 0 : k === 1 ? (1 - p) / tau : p / tau);
      const top = 26, bh = nar ? H * .62 : H * .66, lx = 44, cw = (W - lx - 10) / (n - 1), mx = Math.max(...w), base = top + bh - 18;
      PE.label(ctx, 'row 1a: how hard each view acts on the anchor (toy, \u03c4 = 0.5)', 0, 14, c.M, 11, 'left', 400);
      for (let k = 1; k < n; k++) {
        const x = lx + (k - 1) * cw, h = w[k] / mx * (bh - 44) * PE.clamp(t * 1.5, 0, 1), pos = k === 1, col = pos ? c.Y : c.R;
        ctx.save(); ctx.fillStyle = col; ctx.globalAlpha = .85; ctx.fillRect(x + cw * .18, base - h, cw * .64, h); ctx.restore();
        PE.label(ctx, w[k].toFixed(2), x + cw / 2, base - h - 5, c.F, 10.5, 'center', 400);
        PE.label(ctx, (Math.floor(k / 2) + 1) + 'ab'[k % 2], x + cw / 2, base + 14, k % 2 ? c.Y : c.B, 10.5, 'center');
      }
      PE.label(ctx, 'pull', lx + cw / 2, top + 2, c.Y, 11, 'center');
      PE.label(ctx, nar ? 'push = p / \u03c4' : 'push = softmax probability / \u03c4', lx + 1.5 * cw, top + 2, c.R, 11, 'left');
      const names = [['loss', ''], ['cosines', ''], ['normalize', ''], ['head g', 'W1, W2'], [nar ? 'ResNet' : 'ResNet-50 f', 'all weights']], k5 = names.length;
      const gp = nar ? 12 : 22, bw = (W - 8 - (k5 - 1) * gp) / k5, y = top + bh + 14, hh = nar ? 40 : 38, a2 = PE.clamp(t * 2 - 1, 0, 1);
      names.forEach(([a, b], j) => { const x = 4 + j * (bw + gp), tr = j >= 3;
        ctx.save(); ctx.globalAlpha = .3 + .7 * a2; ctx.fillStyle = c.P2; ctx.strokeStyle = tr ? c.R : c.RU; ctx.lineWidth = tr ? 2.2 : 1.2;
        ctx.beginPath(); ctx.roundRect(x, y, bw, hh, 7); ctx.fill(); ctx.stroke(); ctx.restore();
        PE.label(ctx, a, x + bw / 2, y + (b ? 16 : 23), j ? c.F : c.R, nar ? 10 : 11.5, 'center'); if (b) PE.label(ctx, b, x + bw / 2, y + 31, c.R, 10, 'center', 400);
        if (j && a2 > 0) PE.arrow(ctx, x - gp + 2, y + hh / 2, x - 2, y + hh / 2, c.R, 2, 7, a2); });
      PE.label(ctx, nar ? 'back through both views; all weights trained' : 'then back through both views of every photo; every weight is trained', 4, y + hh + 18, c.F, nar ? 10.5 : 11.5, 'left', 400);
    },
    (ctx, W, H, t, c) => { // LARS step: the paper's learning-rate schedule
      const b = { x: 46, y: 26, w: W - 70, h: H - 70 }, lr = e => e < 10 ? 4.8 * e / 10 : 4.8 * .5 * (1 + Math.cos(Math.PI * (e - 10) / 90));
      const ax = PE.axes(ctx, b, [0, 100], [0, 5], { xt: [0, 10, 25, 50, 75, 100], yt: [0, 1.2, 2.4, 3.6, 4.8], xl: 'epoch (100-epoch ablation run)', yl: 'learning rate', ylo: 36 });
      const pts = []; for (let e = 0; e <= 100 * t; e += .5) pts.push([ax.X(e), ax.Y(lr(e))]); PE.polyline(ctx, pts, c.G, 2.5);
      const e0 = 30; if (t > .3) { PE.dot(ctx, ax.X(e0), ax.Y(lr(e0)), 6, c.R); PE.label(ctx, 'this step (say epoch ' + e0 + ')', ax.X(e0) + 9, ax.Y(lr(e0)) - 6, c.R, 11); }
      PE.label(ctx, 'warmup: 10 epochs', ax.X(10) + 6, ax.Y(.5), c.M, 10.5, 'left', 400);
    },
  ];
  const draw = (ctx, W, H) => { if (!W) return; const c = C(); ctx.fillStyle = PE.css('--bg'); ctx.fillRect(0, 0, W, H); DRAW[state.stage](ctx, W, H, state.t, c); };
  const cv = PE.canvas(canvas, w => KIT.narrow(w) ? 1.15 : .5, draw);
  const set = (stage, seed) => {
    cancel(); if (seed !== state.seed) { state.seed = seed; batch = KIT.stepBatch(seed); }
    state.stage = stage;
    if (PE.instant()) { state.t = 1; cv.redraw(); return; }
    state.t = 0; cancel = PE.animate(900, u => { state.t = u; cv.redraw(); });
  };
  return { set, state, redraw: cv.redraw, resize: cv.resize, destroy: () => { cancel(); cv.destroy(); }, stages: DRAW.length };
};
// Shapes at batch N, from the paper's config: 224 x 224 crops, h 2048 (ResNet-50), z 128. All computed.
KIT.simclrStepText = (stage, N = 4096) => {
  const V = 2 * N, f = x => x.toLocaleString('en-US'), x = ' x ';
  return [
    'Sampled N = ' + f(N) + ' images (4 drawn). Their sizes differ until they are cropped.',
    '2N = ' + f(V) + ' views, each 3' + x + '224' + x + '224: one tensor of ' + f(V) + x + '3' + x + '224' + x + '224 = ' + f(V * 3 * 224 * 224) + ' numbers.',
    'h = f(view): ' + f(V) + x + '2048, one 2048-number vector per view (after average pooling).',
    'z = g(h): ' + f(V) + x + '128. The hidden width of g is not given in the paper.',
    'z / |z|: still ' + f(V) + x + '128, now every row has length 1.',
    'S = Z Z^T: ' + f(V) + x + f(V) + ' = ' + f(V * V) + ' cosines in one matrix product.',
    'Removed: ' + f(V) + ' diagonal entries. Kept per row: 1 positive and 2(N - 1) = ' + f(V - 2) + ' negatives (our arithmetic).',
    f(V) + ' row losses, each a softmax over ' + f(V - 1) + ' entries, averaged into one number.',
    'Row 1a of the toy batch: the partner pulls with weight (1 - p)/\u03c4, each negative pushes with weight p/\u03c4. At full size each row has 1 pull and ' + f(V - 2) + ' pushes.',
    'One LARS update of every weight. The learning rate follows 0.3' + x + 'N / 256 = ' + (0.3 * N / 256) + ' at its peak, with 10 warmup epochs, then cosine decay.',
  ][stage];
};
