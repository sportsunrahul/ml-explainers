SCENES.s4 = ctx => {
  const slider = ctx.q('#s4-atlas-dt'), vb = ctx.qa('[data-view]');
  const pv = vb.find(b => b.getAttribute('aria-pressed') === 'true');
  const S = { view: pv ? pv.dataset.view : 'all', dt: CoreS4.fromSlider(+slider.value), k: 44, p: 1 };
  let cancel = () => {}, cancelD = () => {};
  const el = mkCanvas(ctx.stage);
  // Desktop keeps the bottom of the stage free for the readout box.
  const cv = PE.canvas(el, S3ASPECT(ctx.stage), (c, W, H) => { c.clearRect(0, 0, W, H); CoreS4.draw(c, W, W < 620 ? H : H - 190, S); });
  const hud = () => ctx.hud(badge('computed', CoreS4.readout(S).split(' Building')[0]));
  function setView(v) {
    cancel(); S.view = v; press(vb, b => b.dataset.view === v); S.p = 0;
    cancel = PE.animate(v === 'rec' ? 2600 : v === 'cont' ? 1600 : 800, u => { S.p = u; cv.redraw(); }, hud);
  }
  const offs = [
    ...vb.map(b => listen(b, 'click', () => setView(b.dataset.view))),
    listen(slider, 'input', () => { cancelD(); S.dt = CoreS4.fromSlider(+slider.value); cv.redraw(); hud(); }),
  ];
  hud();
  return {
    flat: true,
    step(n) { setView(n === 1 ? 'conv' : 'all'); },
    replay() { setView(S.view); },
    resize() { cv.resize(); },
    destroy() { cancel(); cancelD(); offs.forEach(f => f()); cv.destroy(); el.remove(); },
  };
};
