SCENES.limits = ctx => {
  const el = mkCanvas(ctx.stage);
  const sN = ctx.q('#limits-atlas-N'), sK = ctx.q('#limits-atlas-K'), sn = ctx.q('#limits-atlas-n');
  const kb = ctx.qa('[data-keys]'), nb = ctx.qa('[data-n]'), cb = ctx.qa('[data-ctx]');
  // State lives in the article's controls, so a theme remount keeps it.
  const ctxL = () => { const b = cb.find(x => x.getAttribute('aria-pressed') === 'true'); return b ? +b.dataset.ctx : 262144; };
  const R = { N: Math.pow(2, +sN.value), K: CoreRecall.fromSlider(+sK.value), ortho: kb.some(b => b.dataset.keys === 'ortho' && b.getAttribute('aria-pressed') === 'true'), fade: 1 };
  const Hy = { attn: CoreHybrid.target(+sn.value), L: ctxL() };
  let step = 0, cancel = () => {};
  const usedH = (W, H) => W < 620 ? H : H - 180;
  const cv = PE.canvas(el, S3ASPECT(ctx.stage), (c, W, H) => {
    c.fillStyle = PE.css('--bg'); c.fillRect(0, 0, W, H);
    if (step === 0) CoreRecall.draw(c, W, usedH(W, H), R); else CoreHybrid.draw(c, W, usedH(W, H), Hy);
  });
  const hud = () => ctx.hud(step === 0 ? badge('toy', CoreRecall.readout(R)) : badge('computed', CoreHybrid.readout(Hy)));
  function setKeys(ortho) { press(kb, b => (b.dataset.keys === 'ortho') === ortho); if (R.ortho === ortho) return; cancel(); R.ortho = ortho; cancel = PE.animate(700, u => { R.fade = u; cv.redraw(); }, hud); }
  function goHy(n, L) {
    cancel(); sn.value = n; press(cb, b => +b.dataset.ctx === L);
    const a0 = Hy.attn.slice(), a1 = CoreHybrid.target(n), l0 = Math.log(Hy.L), l1 = Math.log(L);
    cancel = PE.animate(800, u => { Hy.attn = a0.map((v, i) => PE.lerp(v, a1[i], u)); Hy.L = u >= 1 ? L : Math.exp(PE.lerp(l0, l1, u)); cv.redraw(); }, hud);
  }
  const offs = [
    listen(sN, 'input', () => { cancel(); R.fade = 1; R.N = Math.pow(2, +sN.value); cv.redraw(); hud(); }),
    listen(sK, 'input', () => { cancel(); R.fade = 1; R.K = CoreRecall.fromSlider(+sK.value); cv.redraw(); hud(); }),
    ...kb.map(b => listen(b, 'click', () => setKeys(b.dataset.keys === 'ortho'))),
    listen(sn, 'input', () => { cancel(); Hy.attn = CoreHybrid.target(+sn.value); cv.redraw(); hud(); }),
    ...nb.map(b => listen(b, 'click', () => goHy(+b.dataset.n, Hy.L))),
    ...cb.map(b => listen(b, 'click', () => goHy(Math.round(CoreHybrid.nAttn(Hy)), +b.dataset.ctx))),
  ];
  hud();
  return {
    flat: true,
    step(n) { if (n === step) return; cancel(); step = n; cv.redraw(); hud(); },
    resize() { cv.resize(); },
    destroy() { cancel(); offs.forEach(f => f()); cv.destroy(); el.remove(); },
  };
};
