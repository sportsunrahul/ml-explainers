<script>
(() => {
  // YOLO-World Table 2: zero-shot LVIS minival, Fixed AP; FPS on one V100 without TensorRT (the authors' measurements).
  // fps = the only FPS for the baselines and the original (bracketed) FPS for YOLO-World; rep = re-parameterized FPS.
  const ROWS = [
    { m: 'GLIP-T', fps: .12, ap: 24.9, apr: 17.7 },
    { m: 'GLIP-T', fps: .12, ap: 26.0, apr: 20.8, extra: 'Cap4M' },
    { m: 'Grounding DINO-T', fps: 1.5, ap: 25.6, apr: 14.4 },
    { m: 'Grounding DINO-T', fps: 1.5, ap: 27.4, apr: 18.1, extra: 'Cap4M' },
    { m: 'DetCLIP-T', fps: 2.3, ap: 34.4, apr: 26.9, above: true },
    { m: 'YOLO-World-S', fps: 19.9, rep: 74.1, ap: 26.2, apr: 19.1 },
    { m: 'YOLO-World-M', fps: 18.5, rep: 58.1, ap: 31.0, apr: 23.8 },
    { m: 'YOLO-World-L', fps: 17.6, rep: 52.0, ap: 35.0, apr: 27.1 },
    { m: 'YOLO-World-L', fps: 17.6, rep: 52.0, ap: 35.4, apr: 27.6, extra: 'CC3M' },
  ];
  const fig = document.getElementById('viz-speed'), out = document.getElementById('speed-readout');
  const B = id => document.getElementById('btn-speed-' + id);
  const S = { metric: 'ap', data: 'og', uM: 0, uX: 0 };
  let stop = () => {};
  function marker(ctx, x, y, r, col, kind) {
    ctx.save(); ctx.strokeStyle = col; ctx.fillStyle = col; ctx.lineWidth = 1.6; ctx.beginPath();
    if (kind === 'diamond') { ctx.moveTo(x, y - r - 1); ctx.lineTo(x + r + 1, y); ctx.lineTo(x, y + r + 1); ctx.lineTo(x - r - 1, y); ctx.closePath(); ctx.stroke(); }
    else if (kind === 'ring') { ctx.arc(x, y, r, 0, 7); ctx.stroke(); }
    else { ctx.arc(x, y, r, 0, 7); ctx.fill(); }
    ctx.restore();
  }
  function draw(ctx, w, h) {
    const k = DK.C(); ctx.clearRect(0, 0, w, h); if (w < 80) return;
    const narrow = w < 520, fs = narrow ? 10.5 : 12;
    const b = { x: narrow ? 42 : 56, y: narrow ? 64 : 34, w: w - (narrow ? 52 : 76), h: h - (narrow ? 64 : 34) - (narrow ? 42 : 46) };
    const ax = PE.axes(ctx, b, [.08, 130], [10, 38], { logx: true, xt: [.1, 1, 10, 100], xf: v => String(v), yt: [10, 15, 20, 25, 30, 35],
      xl: narrow ? 'frames per second, one V100 (log)' : 'frames per second on one V100, no TensorRT (log scale)',
      yl: S.uM < .5 ? 'LVIS minival Fixed AP' : 'LVIS minival Fixed APr', ylo: narrow ? 30 : 38 });
    const yv = r => PE.lerp(r.ap, r.apr, S.uM), base = DK.mixh(k.bg, k.fg, .6);
    ctx.save(); ctx.textBaseline = 'middle';
    ROWS.forEach(r => {
      const a = r.extra ? S.uX : 1; if (a < .02) return;
      ctx.globalAlpha = a;
      const yw = !!r.rep, col = yw ? k.B : base, X = ax.X(r.fps), Y = ax.Y(yv(r));
      if (yw) {
        const X2 = ax.X(r.rep);
        // the CC3M row has the same speeds as YOLO-World-L, so it gets only its diamond
        if (!r.extra) { marker(ctx, X, Y, 4.5, col, 'ring'); PE.arrow(ctx, X + 6, Y, X2 - 6, Y, col, 1.4, 6, a); }
        marker(ctx, X2, Y, 4.5, col, r.extra ? 'diamond' : 'dot');
        ctx.font = `600 ${fs}px system-ui, sans-serif`; ctx.fillStyle = k.fg;
        if (r.extra) { if (!narrow) { ctx.textAlign = 'left'; ctx.fillStyle = k.mu; ctx.fillText('+' + r.extra, X2 + 9, Y - 4); } }
        else { ctx.textAlign = 'center'; ctx.fillText(r.m, (X + X2) / 2, Y - (narrow ? 10 : 11)); }
      } else {
        marker(ctx, X, Y, 4.5, col, r.extra ? 'diamond' : 'dot');
        ctx.font = `600 ${fs}px system-ui, sans-serif`;
        if (r.extra) { ctx.textAlign = 'left'; ctx.fillStyle = k.mu; ctx.fillText('+' + r.extra, X + 9, Y); }
        else { ctx.textAlign = 'center'; ctx.fillStyle = k.fg; ctx.fillText(r.m, PE.clamp(X, b.x + 40, b.x + b.w - 40), Y + (r.above ? -13 : 14)); }
      }
    });
    ctx.globalAlpha = 1;
    // legend: top-left of the plot on desktop, above it on phones
    const lx = narrow ? 8 : b.x + 10, ly = narrow ? 14 : b.y + 12;
    ctx.font = `400 ${fs - .5}px system-ui, sans-serif`; ctx.textAlign = 'left'; ctx.fillStyle = k.mu;
    marker(ctx, lx + 4, ly, 4, k.B, 'ring'); PE.arrow(ctx, lx + 11, ly, lx + 27, ly, k.B, 1.3, 5); marker(ctx, lx + 33, ly, 4, k.B, 'dot');
    ctx.fillText('YOLO-World: original, then re-parameterized', lx + 42, ly);
    marker(ctx, lx + 4, ly + 17, 4, k.mu, 'diamond'); ctx.fillText('with caption data added (Cap4M or CC3M)', lx + 42, ly + 17);
    if (narrow) { ctx.fillText('all FPS measured by the YOLO-World authors', lx, ly + 34); }
    ctx.restore();
  }
  const cv = PE.canvas(document.getElementById('speed-canvas'), w => (w < 520 ? 1.08 : .5), draw);
  function say() {
    const key = S.metric, name = key === 'ap' ? 'Fixed AP' : 'Fixed APr';
    const L = ROWS[7], D = ROWS[4], G = ROWS[2], f = v => v.toFixed(1);
    let t = `${name}, ${S.data === 'all' ? 'all Table 2 rows' : 'only models pretrained on O365 + GoldG'}: YOLO-World-L ${f(L[key])} at ${f(L.rep)} FPS re-parameterized (${f(L.fps)} original); DetCLIP-T ${f(D[key])} at ${f(D.fps)} FPS; Grounding DINO-T ${f(G[key])} at ${f(G.fps)} FPS.`;
    if (S.data === 'all') { const d = (i, j) => (ROWS[j][key] - ROWS[i][key]).toFixed(1);
      t += ` Extra caption data adds ${d(0, 1)} for GLIP-T, ${d(2, 3)} for Grounding DINO-T and ${d(7, 8)} for YOLO-World-L.`; }
    if (key === 'apr') t += ' Minival has only 70 rare classes.';
    out.textContent = t;
  }
  function tween() {
    const m0 = S.uM, x0 = S.uX, m1 = S.metric === 'apr' ? 1 : 0, x1 = S.data === 'all' ? 1 : 0; stop();
    stop = PE.animate(700, u => { S.uM = PE.lerp(m0, m1, u); S.uX = PE.lerp(x0, x1, u); cv.redraw(); });
  }
  const press = () => { B('ap').setAttribute('aria-pressed', String(S.metric === 'ap')); B('apr').setAttribute('aria-pressed', String(S.metric === 'apr'));
    B('og').setAttribute('aria-pressed', String(S.data === 'og')); B('all').setAttribute('aria-pressed', String(S.data === 'all')); };
  const set = (key, val) => { S[key] = val; press(); say(); tween(); };
  B('ap').addEventListener('click', () => set('metric', 'ap'));
  B('apr').addEventListener('click', () => set('metric', 'apr'));
  B('og').addEventListener('click', () => set('data', 'og'));
  B('all').addEventListener('click', () => set('data', 'all'));
  B('reset').addEventListener('click', () => { stop(); S.metric = 'ap'; S.data = 'og'; S.uM = 0; S.uX = 0; press(); say(); cv.redraw(); });
  press(); say();
})();
</script>
