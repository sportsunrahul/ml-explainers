SCENES.granularity = ctx => {
  const el = mkCanvas(ctx.stage);
  const kit = KIT.chanland(el, { aspect: S3ASPECT(ctx.stage) });
  const mb = ctx.qa('[data-mode]'), zb = ctx.qa('[data-zoom]');
  let zoom = 0;
  const sync = () => { press(mb, b => b.dataset.mode === kit.state.mode); press(zb, b => +b.dataset.zoom === zoom); ctx.hud(badge('toy', KIT.chanlandText(kit.info()))); };
  const offs = [...mb.map(b => listen(b, 'click', () => { kit.set({ mode: b.dataset.mode }); sync(); })),
    ...zb.map(b => listen(b, 'click', () => { zoom = +b.dataset.zoom; kit.set({ zoom }); sync(); }))];
  sync();
  return { resize: kit.resize, destroy() { offs.forEach(f => f()); kit.destroy(); } };
};
