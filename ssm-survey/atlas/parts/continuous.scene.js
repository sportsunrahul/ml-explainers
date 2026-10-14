/* 3. Continuous time: the N = 3 state as a 3D curve; the slider re-samples it, the mode buttons keep the tokens instead. */
SCENES.continuous = ctx => {
  const C = CoreState3d, el = mkCanvas(ctx.stage), slider = ctx.q('#continuous-atlas-dt');
  const S = { mode: 'signal', dt: 1 / 4 };
  const s3 = PE3.scene(el, { aspect: S3ASPECT(ctx.stage), fov: 34, dist: 9, yaw: .24, pitch: .18 });
  let frame = C.frameLTI(S.mode, S.dt), cur = { to: frame }, cancel = () => {}, narrow = false;
  const draw = st => { cur = st; s3.items = C.items({ ...st, narrow, lift: narrow ? 0 : .6 }); s3.redraw(); };
  const fit = () => { const v = C.view(ctx.stage.clientWidth, ctx.stage.clientHeight, 'lti'); narrow = v.narrow; s3.setView({ ...v, dist: v.dist * (narrow ? 1 : 1.3) }, 0); };
  const ui = () => {
    slider.value = C.DTS.indexOf(S.dt);
    press(ctx.qa('[data-mode]'), b => b.dataset.mode === S.mode);
    press(ctx.qa('[data-dt]'), b => Math.abs(+b.dataset.dt - S.dt) < 1e-9);
    ctx.hud(badge('toy', C.readLTI(S.mode, S.dt)));
  };
  function set(mode, dt, ms = 800) {
    cancel(); const from = frame; S.mode = mode; S.dt = dt; frame = C.frameLTI(mode, dt); ui();
    cancel = s3.tween(ms, u => draw({ from, to: frame, u }), () => draw({ to: frame }));
  }
  const offs = [
    ...ctx.qa('[data-mode]').map(b => listen(b, 'click', () => set(b.dataset.mode, S.dt))),
    ...ctx.qa('[data-dt]').map(b => listen(b, 'click', () => set(S.mode, C.DTS.find(d => Math.abs(d - +b.dataset.dt) < 1e-9)))),
    listen(slider, 'input', () => set(S.mode, C.DTS[+slider.value], 350)),
  ];
  const offTheme = PE.onTheme(() => draw(cur));
  fit(); ui(); draw(cur);
  return {
    resize() { el.style.height = ctx.stage.clientHeight + 'px'; fit(); draw(cur); },
    destroy() { cancel(); offs.forEach(f => f()); offTheme(); s3.destroy(); el.remove(); },
  };
};
