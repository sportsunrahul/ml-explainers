<script>
(() => {
  const fig = document.getElementById('viz-ijsched'), el = document.getElementById('ijsched-canvas'), out = document.getElementById('ijsched-readout');
  const E = 300, SRC = { paper: { lr0: 1e-4, warm: 15 }, config: { lr0: 2e-4, warm: 40 } };
  const S = { src: 'paper' };
  const lr = (e, c) => e < c.warm ? c.lr0 + (1e-3 - c.lr0) * e / c.warm : 1e-6 + (1e-3 - 1e-6) * .5 * (1 + Math.cos(Math.PI * (e - c.warm) / (E - c.warm)));
  const cv = PE.canvas(el, w => w < 560 ? .9 : .42, (ctx, W, H) => {
    const C = { bg: PE.css('--bg'), blue: PE.css('--blue'), red: PE.css('--red'), brn: PE.css('--brown'), mu: PE.css('--muted') };
    ctx.fillStyle = C.bg; ctx.fillRect(0, 0, W, H);
    const narrow = W < 560, pw = narrow ? W - 50 : (W - 100) / 3, ph = narrow ? (H - 70) / 3 : H - 52;
    const panels = [['learning rate', e => lr(e, SRC[S.src]), [1e-6, 1e-3], true, C.red], ['weight decay', e => .04 + .36 * e / E, [0, .4], false, C.brn], ['EMA momentum', e => .996 + .004 * e / E, [.996, 1], false, C.blue]];
    panels.forEach(([name, f, yr, logy, col], k) => {
      const b = narrow ? { x: 44, y: 16 + k * (ph + 20), w: pw, h: ph - 16 } : { x: 44 + k * (pw + 28), y: 14, w: pw, h: ph - 20 };
      const a = PE.axes(ctx, b, [0, E], yr, { xt: [0, 100, 200, 300], yt: logy ? [1e-6, 1e-5, 1e-4, 1e-3] : [yr[0], (yr[0] + yr[1]) / 2, yr[1]], logy, xl: 'epoch', yf: v => logy ? v.toExponential(0) : String(+v.toFixed(3)) });
      const pts = []; for (let e = 0; e <= E; e++) pts.push([a.X(e), a.Y(Math.max(f(e), yr[0]))]);
      PE.polyline(ctx, pts, col, 2); PE.label(ctx, name, b.x, b.y - 4, col, 12);
    });
  });
  function set(src) { S.src = src; fig.querySelectorAll('[data-src]').forEach(b => b.setAttribute('aria-pressed', String(b.dataset.src === src))); cv.redraw();
    const c = SRC[src]; out.textContent = `${src === 'paper' ? 'Paper' : 'Released config'}: learning rate from ${c.lr0} to 1e-3 over ${c.warm} epochs, cosine to 1e-6 by epoch 300; weight decay 0.04 to 0.4; momentum 0.996 to 1.0.`; }
  fig.querySelectorAll('[data-src]').forEach(b => b.addEventListener('click', () => set(b.dataset.src)));
  document.getElementById('btn-ijsched-reset').addEventListener('click', () => set('paper'));
  set('paper');
})();
</script>
