SCENES.grid = ctx => {
  const el = mkCanvas(ctx.stage);
  let preset = 'minmax', tm = 0;
  const show = () => ctx.hud(badge('computed', KIT.rulerText(kit.info())));
  const kit = KIT.ruler(el, { aspect: w => Math.max(.3, ctx.stage.clientHeight / Math.max(w, 1)), bottomPad: innerWidth > 860 ? 130 : 0, onInput: () => { preset = ''; sync(); } });
  const bb = ctx.qa('[data-bits]'), sb = ctx.qa('[data-sym]'), rb = ctx.qa('[data-range]');
  const sync = () => {
    press(bb, b => +b.dataset.bits === kit.state.bits); press(sb, b => (b.dataset.sym === '1') === kit.state.sym); press(rb, b => b.dataset.range === preset);
    clearTimeout(tm); tm = setTimeout(show, PE.instant() ? 0 : 700);
  };
  const offs = [...bb.map(b => listen(b, 'click', () => { kit.set({ bits: +b.dataset.bits }); sync(); })),
    ...sb.map(b => listen(b, 'click', () => { kit.set({ sym: b.dataset.sym === '1' }); sync(); })),
    ...rb.map(b => listen(b, 'click', () => { const [lo, hi] = KIT.RULER_PRESETS[b.dataset.range](); preset = b.dataset.range; kit.set({ lo, hi }); sync(); }))];
  sync(); show();
  return { flat: true, resize: kit.resize, destroy() { clearTimeout(tm); offs.forEach(f => f()); kit.destroy(); } };
};
