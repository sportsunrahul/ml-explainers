/* Nagel 2021 Fig. 8, redrawn: the standard PTQ pipeline, with the data / no-data branch at step 4. Step 0 = overview. */
KIT.PTQ_STEPS = [
  { key: 'fp', t: ['pre-trained', 'FP32 model'], c: '--blue' },
  { key: 'cle', t: ['cross-layer', 'equalization'], c: '--green', n: 1 },
  { key: 'add', t: ['add', 'quantizers'], c: '--yellow', n: 2 },
  { key: 'wr', t: ['weight range', '(MSE)'], c: '--yellow', n: 3 },
  { key: 'ada', t: ['AdaRound', '(with data)'], c: '--green', n: 4, br: -1 },
  { key: 'bc', t: ['bias correction', '(no data)'], c: '--green', n: 4, br: 1 },
  { key: 'ar', t: ['activation', 'range'], c: '--yellow', n: 5 },
];
KIT.ptqpipe = (el, opts = {}) => {
  const S = { step: 0, u: 1 };
  function layout(w, h) {
    const narrow = w < 560, pos = {};
    if (!narrow) { const cw = (w - 20) / 6, bw = Math.min(cw - 18, 118), bh = 46, cy = h / 2;
      KIT.PTQ_STEPS.forEach(s => { const col = s.key === 'fp' ? 0 : s.key === 'cle' ? 1 : s.key === 'add' ? 2 : s.key === 'wr' ? 3 : s.br ? 4 : 5;
        pos[s.key] = { x: 10 + cw * (col + .5) - bw / 2, y: cy - bh / 2 + (s.br || 0) * 40, w: bw, h: bh }; }); }
    else { const rh = (h - 20) / 6, bw = Math.min(w / 2 - 16, 150), bh = 40;
      KIT.PTQ_STEPS.forEach(s => { const row = s.key === 'fp' ? 0 : s.key === 'cle' ? 1 : s.key === 'add' ? 2 : s.key === 'wr' ? 3 : s.br ? 4 : 5;
        pos[s.key] = { x: w / 2 - bw / 2 + (s.br || 0) * (bw / 2 + 8), y: 10 + rh * (row + .5) - bh / 2, w: s.br ? bw * .95 : bw, h: bh }; }); }
    return { narrow, pos };
  }
  function draw(ctx, w, h) {
    ctx.clearRect(0, 0, w, h);
    const L = layout(w, h), P = L.pos, mut = PE.css('--muted'), fg = PE.css('--fg');
    const mid = b => [b.x + b.w / 2, b.y + b.h / 2];
    const edge = (a, b, lab) => { const A = P[a], B = P[b];
      const [x1, y1] = L.narrow ? [mid(A)[0], A.y + A.h] : [A.x + A.w, mid(A)[1]], [x2, y2] = L.narrow ? [mid(B)[0], B.y] : [B.x, mid(B)[1]];
      PE.arrow(ctx, x1 + (L.narrow ? 0 : 3), y1 + (L.narrow ? 3 : 0), x2 - (L.narrow ? 0 : 3), y2 - (L.narrow ? 3 : 0), mut, 1.4, 7, .8);
      if (lab) PE.label(ctx, lab, (x1 + x2) / 2 - (L.narrow ? 0 : 8), (y1 + y2) / 2 + (L.narrow ? -2 : lab === 'data' ? -8 : 16), mut, 10, L.narrow ? 'center' : 'right', 400); };
    edge('fp', 'cle'); edge('cle', 'add'); edge('add', 'wr'); edge('wr', 'ada', 'data'); edge('wr', 'bc', 'no data'); edge('ada', 'ar'); edge('bc', 'ar');
    KIT.PTQ_STEPS.forEach(s => {
      const b = P[s.key], col = PE.css(s.c), on = S.step === 0 || s.n === S.step || (s.key === 'fp' && S.step === 0);
      const cur = s.n === S.step, al = S.step === 0 ? 1 : cur ? .35 + .65 * S.u : (s.n || 0) < S.step ? .75 : .35;
      ctx.save(); ctx.globalAlpha = al; ctx.fillStyle = PE.css('--panel'); ctx.strokeStyle = col; ctx.lineWidth = cur ? 2.6 : 1.4;
      ctx.beginPath(); ctx.roundRect(b.x, b.y, b.w, b.h, 8); ctx.fill(); ctx.stroke(); ctx.restore();
      s.t.forEach((t, k) => PE.label(ctx, t, b.x + b.w / 2, b.y + b.h / 2 + 4 + (k - .5) * 13, on || cur ? fg : mut, 11, 'center', cur ? 700 : 500));
      if (s.n && !L.narrow) PE.label(ctx, String(s.n), b.x + 8, b.y - 5, mut, 10.5, 'left', 600);
      if (cur) PE.dot(ctx, b.x + b.w - 8, b.y + 8, 3.5, col, S.u);
    });
  }
  const cv = PE.canvas(el, opts.aspect || (w => w < 560 ? 1.15 : .26), draw);
  let stop = () => {};
  function set(next) { if (next.step != null && next.step !== S.step) { stop(); S.step = next.step; S.u = 0; stop = KIT.tween(S, { u: 1 }, 500, cv.redraw); } cv.redraw(); }
  return { set, state: S, redraw: cv.redraw, resize: cv.resize, destroy: () => { stop(); cv.destroy(); } };
};
