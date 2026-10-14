/* 6. Selection: per-token Delta bends the state's path; important tokens are written, filler is skipped. */
SCENES.select = ctx => {
  const C = CoreState3d, el = mkCanvas(ctx.stage);
  const presets = { keys: C.keyMarks(), allfill: C.VAL.map(() => false), allimp: C.VAL.map(() => true) };
  const S = { marks: presets.keys.slice() };
  const s3 = PE3.scene(el, { aspect: S3ASPECT(ctx.stage), fov: 34, dist: 9, yaw: .3, pitch: .22 });
  let frame = C.frameSel(S.marks), cur = { to: frame }, cancel = () => {}, narrow = false;
  const draw = st => { cur = st; s3.items = C.items({ ...st, narrow, lift: narrow ? 0 : .5 }); s3.redraw(); };
  const fit = () => { const v = C.view(ctx.stage.clientWidth, ctx.stage.clientHeight, 'select'); narrow = v.narrow; s3.setView({ ...v, dist: v.dist * (narrow ? 1 : 1.3) }, 0); };
  const same = (a, b) => a.every((v, i) => v === b[i]);
  const ui = () => {
    press(ctx.qa('[data-preset]'), b => same(presets[b.dataset.preset], S.marks));
    press(ctx.qa('[data-tok]'), b => S.marks[+b.dataset.tok]);
    ctx.hud(badge('toy', C.readSel(S.marks)));
  };
  function set(marks, ms = 800) {
    cancel(); const from = frame; S.marks = marks.slice(); frame = C.frameSel(S.marks); ui();
    cancel = s3.tween(ms, u => draw({ from, to: frame, u }), () => draw({ to: frame }));
  }
  const offs = [
    ...ctx.qa('[data-preset]').map(b => listen(b, 'click', () => set(presets[b.dataset.preset]))),
    ...ctx.qa('[data-tok]').map(b => listen(b, 'click', () => { const m = S.marks.slice(), t = +b.dataset.tok; m[t] = !m[t]; set(m, 600); })),
  ];
  const offTheme = PE.onTheme(() => draw(cur));
  fit(); ui(); draw(cur);
  return {
    resize() { el.style.height = ctx.stage.clientHeight + 'px'; fit(); draw(cur); },
    destroy() { cancel(); offs.forEach(f => f()); offTheme(); s3.destroy(); el.remove(); },
  };
};
