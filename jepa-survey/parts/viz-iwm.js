<script>
(() => {
  const fig = document.getElementById('viz-iwm'), el = document.getElementById('iwm-canvas'), out = document.getElementById('iwm-readout');
  const S = { pred: 'cond', act: 'jitter', u: 1 };
  const SRC = [-1.3, 0], TGT = { jitter: [.9, .8], blur: [1.1, -.1], gray: [.6, -.9] };
  const cv = PE.canvas(el, w => w < 560 ? .8 : .46, draw);
  function draw(ctx, W, H) {
    const C = { bg: PE.css('--bg'), blue: PE.css('--blue'), yel: PE.css('--yellow'), grn: PE.css('--green'), red: PE.css('--red'), mu: PE.css('--muted'), rule: PE.css('--rule') };
    ctx.fillStyle = C.bg; ctx.fillRect(0, 0, W, H);
    const v = PE.view(W, H, 3.2);
    ctx.strokeStyle = C.rule; for (let g = -2; g <= 2; g++) { ctx.beginPath(); ctx.moveTo(v.X(g), 0); ctx.lineTo(v.X(g), H); ctx.moveTo(0, v.Y(g)); ctx.lineTo(W, v.Y(g)); ctx.stroke(); }
    const blind = S.pred === 'blind', u = S.u;
    // Under a blind predictor the encoder must merge the three targets; ease them toward their centroid.
    const keys = Object.keys(TGT), cen = [keys.reduce((s, k) => s + TGT[k][0], 0) / 3, keys.reduce((s, k) => s + TGT[k][1], 0) / 3];
    const pos = {}; keys.forEach(k => { const t = TGT[k], m = blind ? u : 1 - u; pos[k] = [t[0] + (cen[0] - t[0]) * m, t[1] + (cen[1] - t[1]) * m]; });
    PE.dot(ctx, v.X(SRC[0]), v.Y(SRC[1]), 8, C.blue); PE.label(ctx, 'context embedding', v.X(SRC[0]), v.Y(SRC[1]) + 24, C.blue, 12, 'center');
    keys.forEach(k => { const p = pos[k]; PE.dot(ctx, v.X(p[0]), v.Y(p[1]), 7, C.yel, k === S.act ? 1 : .45); PE.label(ctx, k, v.X(p[0]) + 11, v.Y(p[1]) + 4, C.yel, 12); });
    const dst = blind ? cen : TGT[S.act];
    PE.arrow(ctx, v.X(SRC[0]) + 8, v.Y(SRC[1]), v.X(dst[0]) - 8, v.Y(dst[1]), C.grn, 2.5, 10);
    PE.label(ctx, blind ? 'predictor output (same for every action)' : `predictor output given "${S.act}"`, v.X(-1.2), v.Y(1.35), C.grn, 12);
    const tp = pos[S.act], err = Math.hypot(dst[0] - tp[0], dst[1] - tp[1]);
    if (err > .02) { ctx.save(); ctx.strokeStyle = C.red; ctx.setLineDash([4, 4]); ctx.lineWidth = 2; ctx.beginPath(); ctx.moveTo(v.X(dst[0]), v.Y(dst[1])); ctx.lineTo(v.X(tp[0]), v.Y(tp[1])); ctx.stroke(); ctx.restore(); }
    fig._err = err;
  }
  function say() { out.textContent = S.pred === 'cond' ? `Conditioned: the predictor reads the action "${S.act}" and lands on that action's target. The encoder can keep color and sharpness in the embedding (equivariant).` :
    'Blind: one output for every action, so the loss is only small if the encoder places all three targets together. Color, blur and grayscale are discarded (invariant).'; }
  function set(k, val, btn) { S[k] = val; fig.querySelectorAll(`[data-${k === 'pred' ? 'pred' : 'act'}]`).forEach(x => x.setAttribute('aria-pressed', String(x === btn))); S.u = 0; PE.animate(600, u => { S.u = u; cv.redraw(); }); say(); }
  fig.querySelectorAll('[data-pred]').forEach(b => b.addEventListener('click', () => set('pred', b.dataset.pred, b)));
  fig.querySelectorAll('[data-act]').forEach(b => b.addEventListener('click', () => set('act', b.dataset.act, b)));
  document.getElementById('btn-iwm-reset').addEventListener('click', () => { S.act = 'jitter'; fig.querySelectorAll('[data-act]').forEach(x => x.setAttribute('aria-pressed', String(x.dataset.act === 'jitter'))); set('pred', 'cond', document.getElementById('btn-iwm-cond')); });
  PE.linkPulses(fig, () => cv.redraw());
  say(); cv.redraw();
})();
</script>
