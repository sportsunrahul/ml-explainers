SCENES.hippo = ctx => {
  const slider = ctx.q('#hippo-atlas-N'), sb = ctx.qa('[data-sig]');
  const pressed = sb.find(b => b.getAttribute('aria-pressed') === 'true');
  // A theme toggle remounts the scene, so N and the signal are read back from the article's controls.
  const S = { step: 0, sig: pressed ? pressed.dataset.sig : 'paper', method: 'bilinear', N: +slider.value, k: 1000, u: 1, narrow: false };
  let view = null, cancel = () => {};
  const hud = () => ctx.hud(badge('computed', CoreHippo.readout(S)));
  function mount3d() {
    const el = mkCanvas(ctx.stage);
    const s3 = PE3.scene(el, { aspect: S3ASPECT(ctx.stage), fov: 34, dist: 6.2, yaw: -.42, pitch: .62 });
    const fit = () => { const w = ctx.stage.clientWidth, h = ctx.stage.clientHeight, n = w < 620 || w / h < 1.05; S.narrow = n; s3.setView({ dist: n ? Math.max(7.2, 6.6 * h / Math.max(w, 1)) : 6.2, yaw: n ? -.62 : -.42 }, 0); };
    const build = () => { s3.items = CoreHippo.items(S); s3.redraw(); };
    const off = PE.onTheme(build);
    fit(); build();
    return { redraw: build, resize: () => { el.style.height = ctx.stage.clientHeight + 'px'; fit(); build(); }, destroy: () => { off(); s3.destroy(); el.remove(); } };
  }
  function mount2d() {
    const el = mkCanvas(ctx.stage);
    const cv = PE.canvas(el, S3ASPECT(ctx.stage), (c, W, H) => { c.clearRect(0, 0, W, H); CoreHippo.draw(c, W, W < 620 ? H : H - 150, S); });
    return { redraw: () => cv.redraw(), resize: () => cv.resize(), destroy: () => { cv.destroy(); el.remove(); } };
  }
  function show(n) {
    if (view) view.destroy();
    S.step = n; view = n === 0 ? mount3d() : mount2d(); document.body.dataset.flat = n === 0 ? '0' : '1'; hud();
  }
  function setN(N) { cancel(); const a = S.N; cancel = PE.animate(600, u => { S.N = Math.round(a + (N - a) * u); view.redraw(); }, hud); }
  const offs = [
    listen(slider, 'input', () => { cancel(); S.N = +slider.value; view.redraw(); hud(); }),
    ...sb.map(b => listen(b, 'click', () => { cancel(); S.sig = b.dataset.sig; press(sb, x => x === b); view.redraw(); hud(); })),
  ];
  show(0);
  return {
    step(n) { if (n !== S.step) show(n); },
    replay() { slider.value = 64; setN(64); },
    resize() { view && view.resize(); },
    destroy() { cancel(); offs.forEach(f => f()); if (view) view.destroy(); },
  };
};
