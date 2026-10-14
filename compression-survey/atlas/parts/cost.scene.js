SCENES.cost = ctx => {
  const el = mkCanvas(ctx.stage);
  const kit = KIT.roofline(el, { aspect: w => Math.max(.3, ctx.stage.clientHeight / Math.max(w, 1)), bottomPad: innerWidth > 860 ? 130 : 0 });
  const pb = ctx.qa('[data-prec]'), lb = ctx.qa('[data-layer]');
  const sync = () => { press(pb, b => b.dataset.prec === kit.state.prec); press(lb, b => b.dataset.layer === kit.state.sel);
    ctx.hud(badge('toy', KIT.roofText(kit.info()))); };
  const offs = [...pb.map(b => listen(b, 'click', () => { kit.set({ prec: b.dataset.prec }); sync(); })),
    ...lb.map(b => listen(b, 'click', () => { kit.set({ sel: b.dataset.layer }); sync(); }))];
  sync();
  return { flat: true, resize: kit.resize, destroy() { offs.forEach(f => f()); kit.destroy(); } };
};
