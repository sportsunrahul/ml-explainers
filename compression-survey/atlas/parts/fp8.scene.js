SCENES.fp8 = ctx => {
  const el = mkCanvas(ctx.stage);
  const kit = KIT.fp8(el, { aspect: w => Math.max(.3, ctx.stage.clientHeight / Math.max(w, 1)), bottomPad: innerWidth > 860 ? 130 : 0 });
  const db = ctx.qa('[data-dist]');
  const sync = () => { press(db, b => b.dataset.dist === kit.state.dist); ctx.hud(badge('computed', KIT.fp8Text(kit.info()))); };
  const offs = db.map(b => listen(b, 'click', () => { kit.set({ dist: b.dataset.dist }); sync(); }));
  sync();
  return { flat: true, resize: kit.resize, destroy() { offs.forEach(f => f()); kit.destroy(); } };
};
