SCENES.qat = ctx => {
  const el = mkCanvas(ctx.stage);
  const kit = KIT.ste(el, { aspect: w => Math.max(.3, ctx.stage.clientHeight / Math.max(w, 1)), bottomPad: innerWidth > 860 ? 130 : 0 });
  const mb = ctx.qa('[data-mode]'), ab = ctx.qa('[data-act]');
  const sync = () => { press(mb, b => b.dataset.mode === kit.state.mode); ctx.hud(badge('computed', KIT.steText(kit.info()))); };
  const act = { step1: { step: 1 }, step5: { step: 5 }, reset: { mode: 'ste', reset: true } };
  const offs = [...mb.map(b => listen(b, 'click', () => { kit.set({ mode: b.dataset.mode }); sync(); })),
    ...ab.map(b => listen(b, 'click', () => { kit.set(act[b.dataset.act]); sync(); }))];
  sync();
  return { flat: true, resize: kit.resize, destroy() { offs.forEach(f => f()); kit.destroy(); } };
};
