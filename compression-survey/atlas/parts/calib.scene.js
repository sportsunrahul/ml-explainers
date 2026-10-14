SCENES.calib = ctx => {
  const el = mkCanvas(ctx.stage);
  const kit = KIT.calib(el, { aspect: w => Math.max(.3, ctx.stage.clientHeight / Math.max(w, 1)), bottomPad: innerWidth > 860 ? 130 : 0 });
  const mb = ctx.qa('[data-method]');
  const sync = () => { press(mb, b => b.dataset.method === kit.state.method); ctx.hud(badge('computed', KIT.calibText(kit.info()))); };
  const offs = mb.map(b => listen(b, 'click', () => { kit.set({ method: b.dataset.method }); sync(); }));
  sync();
  return { flat: true, resize: kit.resize, destroy() { offs.forEach(f => f()); kit.destroy(); } };
};
