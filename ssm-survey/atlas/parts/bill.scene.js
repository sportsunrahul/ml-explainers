SCENES.bill = ctx => {
  const slider = ctx.q('#bill-atlas-L');
  const S = { step: 0, L: 8192, u: 1 };
  let view = null, cancel = () => {};
  slider.value = CoreBill.toSlider(S.L);
  const hud = () => ctx.hud(S.step === 0
    ? badge('toy', `Twenty tokens arrive. The KV cache gains a column of keys and values for each one; the state keeps its size and only changes shade.`)
    : badge('computed', CoreBill.readout(S.L)));
  function mount3d() {
    const el = mkCanvas(ctx.stage);
    const s3 = PE3.scene(el, { aspect: S3ASPECT(ctx.stage), fov: 34, dist: 6.4, yaw: .38, pitch: .34 });
    const V = () => CoreHero.view(ctx.stage.clientWidth, ctx.stage.clientHeight);
    const frame = v => { S.u = v; s3.items = CoreHero.items(v, V().narrow); s3.redraw(); };
    const fit = () => s3.setView(V(), 0);
    const play = () => { cancel(); if (PE3.instant()) { frame(1); return; } cancel = s3.tween(6200, frame); };
    const off = PE.onTheme(() => frame(S.u));
    fit(); frame(PE3.instant() ? 1 : 0); play();
    return { replay: play, resize: () => { el.style.height = ctx.stage.clientHeight + 'px'; fit(); frame(S.u); }, destroy: () => { cancel(); off(); s3.destroy(); el.remove(); } };
  }
  function mount2d() {
    const el = mkCanvas(ctx.stage);
    const cv = PE.canvas(el, S3ASPECT(ctx.stage), (c, W, H) => { c.clearRect(0, 0, W, H); CoreBill.draw(c, W, W < 620 ? H : H - 200, S.L); });
    return { replay: () => setL(8192), resize: () => cv.resize(), destroy: () => { cancel(); cv.destroy(); el.remove(); }, redraw: () => cv.redraw() };
  }
  function show(n) {
    if (view) view.destroy();
    // The shell reads handle.flat once at mount; this scene switches between 3D and 2D, so it sets the flag itself.
    S.step = n; view = n === 0 ? mount3d() : mount2d(); document.body.dataset.flat = n === 0 ? '0' : '1'; hud();
  }
  // Switching to the chart goes through the tab, so the shell updates the tabs and the step text too.
  const toChart = () => { if (S.step !== 1) ctx.q('[role="tab"][data-step="1"]').click(); };
  function setL(L) {
    cancel(); const a = CoreBill.toSlider(S.L), b = CoreBill.toSlider(L); slider.value = b;
    toChart();
    cancel = PE.animate(700, u => { S.L = u >= 1 ? L : CoreBill.fromSlider(a + (b - a) * u); view.redraw && view.redraw(); }, hud);
  }
  const offs = [
    listen(slider, 'input', () => { cancel(); S.L = CoreBill.fromSlider(+slider.value); toChart(); view.redraw(); hud(); }),
    ...ctx.qa('[data-l]').map(b => listen(b, 'click', () => setL(+b.dataset.l))),
  ];
  show(0);
  return {
    step(n) { if (n !== S.step) show(n); },
    replay() { view && view.replay(); },
    resize() { view && view.resize(); },
    destroy() { offs.forEach(f => f()); if (view) view.destroy(); },
  };
};
