/* viz-augment (ch-simclr): SimCLR Fig. 5's asymmetric protocol on the running-example photo. Both branches get the same
   crop; branch 2 then gets the 1st and the 2nd transformation (the same one twice means a single transformation, as in
   the figure's diagonal). Under the views: their per-channel color histograms, and the overlap between them, which is the
   shortcut a crop-only network can use. The parameters of each transformation are fixed per kind so the figure is stable. */
KIT.AUG_KINDS = ['crop', 'cutout', 'color', 'sobel', 'noise', 'blur', 'rotate'];
KIT.AUG_NAMES = { crop: 'Crop', cutout: 'Cutout', color: 'Color', sobel: 'Sobel', noise: 'Noise', blur: 'Blur', rotate: 'Rotate' };
KIT.AUG_CROP0 = { x: .14, y: .2, w: .74, h: .74 };
KIT.augOp = (kind, slot) => {
  const op = KIT.photoOp(kind, PE.rng(31 + 17 * KIT.AUG_KINDS.indexOf(kind) + 101 * slot));
  if (kind === 'color') Object.assign(op, slot ? { b: .85, c: 1.2, s: 1.3, h: -.3, grey: false } : { b: 1.1, c: 1.1, s: 1.5, h: .25, grey: false });
  return op;
};
KIT.augViews = (a, b, seed = 0) => {
  const seq = a === b ? [KIT.augOp(a, 0)] : [KIT.augOp(a, 0), KIT.augOp(b, 1)];
  const A = KIT.photoView(seed, { crop: KIT.AUG_CROP0 }), B = KIT.photoView(seed, { crop: KIT.AUG_CROP0, seq });
  const hA = KIT.photoHist(A), hB = KIT.photoHist(B);
  // the same crop of 8 other photos: how much color mass a non-partner typically shares with branch 1
  let other = 0; for (let s = 1; s <= 8; s++) other += KIT.histOverlap(hA, KIT.photoHist(KIT.photoView(seed + s, { crop: KIT.AUG_CROP0 }))) / 8;
  return { A, B, hA, hB, partner: KIT.histOverlap(hA, hB), other };
};
KIT.augment = (canvas, opts = {}) => {
  const state = { a: 'crop', b: 'crop', u: 1, seed: opts.seed || 0 };
  let cur = KIT.augViews('crop', 'crop', state.seed), prev = cur, cancel = () => {};
  const glow = opts.fig ? PE.linkPulses(opts.fig, () => cv.redraw()) : () => 0;
  const draw = (ctx, W, H) => {
    if (!W) return;
    const Bc = PE.css('--blue'), Yc = PE.css('--yellow'), Gc = PE.css('--green'), M = PE.css('--muted'), RU = PE.css('--rule'), F = PE.css('--fg');
    ctx.clearRect(0, 0, W, H);
    const nar = KIT.narrow(W), fs = nar ? 10.5 : 12, gap = nar ? 10 : 26, top = nar ? 34 : 26, histH = nar ? H * .3 : H * .3;
    const s = Math.min((W - 2 * gap - 8) / 3, H - histH - top - 30), x0 = (W - 3 * s - 2 * gap) / 2;
    const xs = [x0, x0 + s + gap, x0 + 2 * (s + gap)], u = state.u;
    // the photo, with the shared crop dashed
    KIT.photo(ctx, xs[0], top, s, s, state.seed, {});
    const c0 = KIT.AUG_CROP0; ctx.save(); ctx.setLineDash([5, 4]); ctx.strokeStyle = F; ctx.lineWidth = 1.5;
    ctx.strokeRect(xs[0] + c0.x * s, top + c0.y * s, c0.w * s, c0.h * s); ctx.restore();
    // two branches: A = crop only, B = crop then the chosen transformation(s); cross-fade B while it changes
    KIT.photo(ctx, xs[1], top, s, s, 0, cur.A);
    if (u < 1) KIT.photo(ctx, xs[2], top, s, s, 0, prev.B, 1 - u);
    KIT.photo(ctx, xs[2], top, s, s, 0, cur.B, u);
    ctx.save(); ctx.lineWidth = 3; ctx.strokeStyle = Bc; ctx.strokeRect(xs[1] - 1.5, top - 1.5, s + 3, s + 3); ctx.strokeStyle = Yc; ctx.strokeRect(xs[2] - 1.5, top - 1.5, s + 3, s + 3); ctx.restore();
    const name = k => KIT.AUG_NAMES[k], bl = state.a === state.b ? name(state.a) : name(state.a) + ', then ' + name(state.b);
    PE.label(ctx, 'the photo', xs[0], top - 8, M, fs);
    PE.label(ctx, nar ? 'crop only' : 'branch 1: crop only', xs[1], top - 8, Bc, fs);
    PE.label(ctx, nar ? bl : 'branch 2: crop, then ' + bl, xs[2], top - 8, Yc, fs);
    if (nar) { PE.label(ctx, 'branch 1', xs[1], top - 21, Bc, fs); PE.label(ctx, 'branch 2', xs[2], top - 21, Yc, fs); }
    // histograms: blue = branch 1, yellow = branch 2, green = the mass they share
    const hy = top + s + (nar ? 34 : 40), hh = H - hy - 18, hx = xs[0], hw = xs[2] + s - xs[0], n = cur.hA.length, bw = hw / (n + 2);
    const hB = cur.hB.map((v, i) => PE.lerp(prev.hB[i], v, u)), mx = Math.max(...cur.hA, ...hB, ...prev.hB) || 1;
    PE.label(ctx, nar ? 'color histograms' : 'color histograms (pixel counts per channel and brightness bin)', hx, hy - 8, M, fs, 'left', 400);
    ctx.save(); ctx.strokeStyle = RU; ctx.beginPath(); ctx.moveTo(hx, hy + hh); ctx.lineTo(hx + hw, hy + hh); ctx.stroke(); ctx.restore();
    for (let i = 0; i < n; i++) {
      const ch = Math.floor(i / 8), x = hx + (i + ch) * bw, a = cur.hA[i] / mx * hh, b = hB[i] / mx * hh, m = Math.min(a, b);
      ctx.globalAlpha = .4; ctx.fillStyle = Bc; ctx.fillRect(x + 1, hy + hh - a, bw * .5 - 1, a);
      ctx.fillStyle = Yc; ctx.fillRect(x + bw * .5, hy + hh - b, bw * .5 - 1, b);
      ctx.globalAlpha = .9; ctx.fillStyle = Gc; ctx.fillRect(x + 1, hy + hh - m, bw - 2, m); ctx.globalAlpha = 1;
    }
    ['red channel', 'green channel', 'blue channel'].forEach((t, ch) => PE.label(ctx, t, hx + (ch * 9 + 4) * bw, hy + hh + 14, M, nar ? 10 : 11, 'center', 400));
    const ov = PE.lerp(prev.partner, cur.partner, u);
    PE.label(ctx, 'other photos: ' + Math.round(cur.other * 100) + '%', hx + hw, hy - 8, M, fs, 'right');
    PE.label(ctx, 'partner shares: ' + Math.round(ov * 100) + '%', hx + hw, hy - 8 - (nar ? 14 : 15), Gc, fs, 'right');
  };
  const cv = PE.canvas(canvas, w => KIT.narrow(w) ? .8 : .5, draw);
  const set = (a, b) => {
    cancel(); prev = cur; state.a = a; state.b = b; cur = KIT.augViews(a, b, state.seed);
    if (PE.instant()) { state.u = 1; cv.redraw(); return; }
    state.u = 0; cancel = PE.animate(700, u => { state.u = u; cv.redraw(); });
  };
  return { set, state, redraw: cv.redraw, resize: cv.resize, destroy: () => { cancel(); cv.destroy(); }, views: () => cur };
};
// Readout HTML; `cell` is the Fig. 5 value as printed in the static table, `id` its claim id.
KIT.augmentText = (a, b, cell, id, v) => {
  const N = KIT.AUG_NAMES, pair = a === b ? 'single transformation ' + N[a] : '1st ' + N[a] + ', 2nd ' + N[b];
  const pct = x => Math.round(x * 100) + '%';
  const cheat = v.partner > v.other + .05 ? 'The histograms alone still point to the partner, so the network can match colors instead of objects.'
    : 'The histograms no longer single out the partner, so matching colors does not solve the task.';
  return 'Fig. 5, ' + pair + ': linear probe top-1 <span data-claim="' + id + '">' + cell + '</span>% on ImageNet. ' +
    'Toy histograms: the two views share ' + pct(v.partner) + ' of their color mass; branch 1 and the same crop of 8 other photos share ' + pct(v.other) + ' on average. ' + cheat;
};
