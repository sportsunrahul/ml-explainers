<script>
/* viz-owldrop (long page only): OWLv2's token dropping. Left, a toy 2 x 2 mosaic cut into 16 x 16 patches; the
   paper's rule (App. A.5: add uniform noise in [0, 0.01] to each patch's mean pixel variance, drop the lowest share)
   runs on the toy's own pixels, rendered once off-screen in fixed grays so the ranking is the same in both themes.
   Right, the paper's measured cost of each drop rate (OWLv2 Table A1). "At inference" shows that nothing is dropped. */
(() => {
  // OWLv2 Table A1 (PDF p14): standard OWL-ViT L/14 trained on O365+VG, LVIS val. [drop rate, AP_all, AP_rare].
  const T = [[0, 33.3, 31.8], [.25, 33.1, 31.0], [.33, 33.6, 32.6], [.5, 32.9, 30.8], [.7, 30.4, 28.2]];
  const SD = [.33, 1.16]; // Table A1, drop 0.00: std over three runs (AP_all, AP_rare)
  const NP = 16, NPATCH = NP * NP;
  const TILES = [
    { objs: DK.WAREHOUSE, ar: 16 / 9 },
    { objs: [{ kind: 'pallet', box: [.15, .45, .7, .42] }], ar: 1 },
    { objs: [{ kind: 'person', box: [.3, .1, .4, .78] }], ar: 3 / 4 },
    { objs: [{ kind: 'crate', box: [.12, .42, .36, .46] }, { kind: 'cone', box: [.6, .52, .2, .36] }], ar: 4 / 3 },
  ];
  // Each tile is fitted into its cell and padded with gray at the bottom or right, as in the paper's preprocessing.
  function drawMosaic(ctx, X, Y, S, k, pad) {
    const t = S / 2;
    TILES.forEach((tl, m) => {
      const x = X + (m % 2) * t, y = Y + Math.floor(m / 2) * t, iw = tl.ar >= 1 ? t : t * tl.ar, ih = tl.ar >= 1 ? t / tl.ar : t;
      ctx.fillStyle = pad; ctx.fillRect(x, y, t, t); ctx.fillStyle = k.p; ctx.fillRect(x, y, iw, ih);
      DK.drawScene(ctx, tl.objs, x + iw * .03, y + ih * .03, iw * .94, ih * .94, k);
    });
  }
  // Rank patches by variance once, from an off-screen render in fixed grays.
  const order = (() => {
    const S = 256, ps = S / NP, c = document.createElement('canvas'); c.width = c.height = S; const x = c.getContext('2d');
    const K = { bg: '#f4f4f4', fg: '#202020', mu: '#777777', rule: '#cccccc', p: '#ececec', p2: '#e0e0e0', B: '#3070a0', Y: '#a08000', G: '#408020', R: '#c03020', Br: '#8b5a2b' };
    x.fillStyle = K.bg; x.fillRect(0, 0, S, S); drawMosaic(x, 0, 0, S, K, '#808080');
    const px = x.getImageData(0, 0, S, S).data, r = PE.rng(41), v = [];
    for (let pj = 0; pj < NP; pj++) for (let pi = 0; pi < NP; pi++) {
      let tot = 0;
      for (let ch = 0; ch < 3; ch++) { let s = 0, s2 = 0;
        for (let yy = 0; yy < ps; yy++) for (let xx = 0; xx < ps; xx++) { const q = px[((pj * ps + yy) * S + pi * ps + xx) * 4 + ch] / 255; s += q; s2 += q * q; }
        const n = ps * ps; tot += s2 / n - (s / n) ** 2; }
      v.push({ i: pj * NP + pi, v: tot / 3 + r() * .01 });
    }
    return v.sort((a, b) => a.v - b.v).map(e => e.i);
  })();
  const rankOf = new Array(NPATCH); order.forEach((p, n) => { rankOf[p] = n; });

  const el = document.getElementById('owldrop-canvas'), out = document.getElementById('owldrop-readout');
  const KEYS = ['0', '25', '33', '50', '70', 'infer'], RATE = { 0: 0, 25: .25, 33: .33, 50: .5, 70: .7, infer: 0 };
  let S = { key: '0', from: 0, to: 0, u: 1 }, stop = () => {};
  const nDrop = key => Math.round(RATE[key] * NPATCH);
  function draw(ctx, w, h) {
    const k = DK.C(); ctx.clearRect(0, 0, w, h); if (w < 160) return;
    const narrow = w < 620, fs = narrow ? 11 : 12;
    const MS = narrow ? Math.min(w - 24, 330) : Math.min(h - 62, w * .44), mx = narrow ? (w - MS) / 2 : 14, my = narrow ? 50 : 34;
    ctx.save(); ctx.font = `600 ${fs + .5}px system-ui, sans-serif`; ctx.fillStyle = k.fg;
    ctx.fillText(S.key === 'infer' ? 'At inference: every patch is kept' : `Training: drop the ${Math.round(RATE[S.key] * 100)}% flattest patches`, mx, my - 12); ctx.restore();
    drawMosaic(ctx, mx, my, MS, k, DK.mixh(k.bg, k.mu, .45));
    const ps = MS / NP, cut = PE.lerp(S.from, S.to, S.u);
    for (let p = 0; p < NPATCH; p++) {
      const a = PE.clamp(cut - rankOf[p], 0, 1); if (a <= 0) continue;
      const x = mx + (p % NP) * ps, y = my + Math.floor(p / NP) * ps;
      ctx.globalAlpha = .9 * a; ctx.fillStyle = DK.mixh(k.bg, k.Br, .28); ctx.fillRect(x, y, ps, ps); ctx.globalAlpha = 1;
    }
    ctx.strokeStyle = k.rule; ctx.lineWidth = 1; ctx.globalAlpha = .7; ctx.beginPath();
    for (let i = 0; i <= NP; i++) { ctx.moveTo(mx + i * ps, my); ctx.lineTo(mx + i * ps, my + MS); ctx.moveTo(mx, my + i * ps); ctx.lineTo(mx + MS, my + i * ps); }
    ctx.stroke(); ctx.globalAlpha = 1;
    ctx.save(); ctx.font = `400 ${fs - .5}px system-ui, sans-serif`; ctx.fillStyle = k.mu;
    ctx.fillText(`${NPATCH} patches; brown = dropped (lowest pixel variance)`, mx, my + MS + 16); ctx.restore();
    // chart
    const b = narrow ? { x: 46, y: my + MS + 70, w: w - 46 - 24, h: h - (my + MS + 70) - 44 } : { x: mx + MS + 70, y: 64, w: w - (mx + MS + 70) - 24, h: h - 64 - 50 };
    ctx.save(); ctx.font = `600 ${fs + .5}px system-ui, sans-serif`; ctx.fillStyle = k.fg; ctx.fillText('LVIS val AP after training with dropping', b.x - 36, b.y - 36);
    ctx.font = `400 ${fs - .5}px system-ui, sans-serif`; ctx.fillStyle = k.mu; ctx.fillText('OWL-ViT L/14 on O365+VG, OWLv2 Table A1', b.x - 36, b.y - 19); ctx.restore();
    const A = PE.axes(ctx, b, [0, .7], [27, 35], { xt: [0, .25, .5, .7], xf: v => Math.round(v * 100) + '%', yt: [28, 30, 32, 34], xl: 'share of patches dropped in training' });
    const sel = S.key === 'infer' ? null : RATE[S.key];
    if (sel != null) { ctx.fillStyle = DK.mixh(k.bg, k.fg, .1); ctx.fillRect(A.X(sel) - 9, b.y, 18, b.h); }
    [[1, k.mu, [5, 4], 'all classes'], [2, k.fg, null, 'rare classes']].forEach(([j, col, dash, name]) => {
      PE.polyline(ctx, T.map(r => [A.X(r[0]), A.Y(r[j])]), col, j === 2 ? 2.4 : 1.8, dash);
      T.forEach(r => PE.dot(ctx, A.X(r[0]), A.Y(r[j]), r[0] === sel ? 5.5 : 3.5, col));
      const sd = SD[j - 1]; ctx.strokeStyle = col; ctx.lineWidth = 1.3; ctx.beginPath(); ctx.moveTo(A.X(0) + (j - 1.5) * 8, A.Y(T[0][j] - sd)); ctx.lineTo(A.X(0) + (j - 1.5) * 8, A.Y(T[0][j] + sd)); ctx.stroke();
      ctx.save(); ctx.font = `600 ${fs - .5}px system-ui, sans-serif`; ctx.fillStyle = col; ctx.textAlign = 'right';
      ctx.fillText(name, b.x + b.w, A.Y(T[4][j]) + (j === 1 ? -9 : 15)); ctx.restore();
      if (sel != null) { const r = T.find(t => t[0] === sel); ctx.save(); ctx.font = `600 ${fs}px system-ui, sans-serif`; ctx.fillStyle = col; ctx.textAlign = 'left';
        ctx.fillText(r[j].toFixed(1), A.X(sel) + 12, A.Y(r[j]) + (j === 1 ? -6 : 12)); ctx.restore(); }
    });
    if (sel == null) { ctx.save(); ctx.font = `600 ${fs}px system-ui, sans-serif`; ctx.fillStyle = k.fg; ctx.textAlign = 'center'; ctx.fillText('no rate: nothing is dropped at test time', b.x + b.w / 2, b.y + 14); ctx.restore(); }
  }
  const cv = PE.canvas(el, w => (w < 620 ? (50 + Math.min(w - 24, 330) + 70 + 230) / w : .46), draw);
  function describe() {
    if (S.key === 'infer') return 'At inference OWLv2 keeps every patch, so the deployed model is the same as OWL-ViT. Dropping only makes training cheaper.';
    const r = T.find(t => t[0] === RATE[S.key]), n = nDrop(S.key);
    return S.key === '0' ? `No dropping. Table A1: AP ${r[1]} (std ${SD[0]}) and rare-class AP ${r[2]} (std ${SD[1]}) over three runs.`
      : `Training with ${Math.round(RATE[S.key] * 100)}% of patches dropped: the toy loses ${n} of ${NPATCH} patches, mostly padding and blank wall. Table A1: AP ${r[1]}, rare-class AP ${r[2]}.`;
  }
  const B = key => document.getElementById('btn-owldrop-' + key);
  const press = () => KEYS.forEach(key => B(key).setAttribute('aria-pressed', String(key === S.key)));
  function set(key) { stop(); S.from = PE.lerp(S.from, S.to, S.u); S.to = nDrop(key); S.key = key; S.u = 0;
    stop = PE.animate(700, u => { S.u = u; cv.redraw(); }); press(); out.textContent = describe(); }
  KEYS.forEach(key => B(key).addEventListener('click', () => set(key)));
  B('reset').addEventListener('click', () => { stop(); S = { key: '0', from: 0, to: 0, u: 1 }; cv.redraw(); press(); out.textContent = describe(); });
  press(); out.textContent = describe();
})();
</script>
