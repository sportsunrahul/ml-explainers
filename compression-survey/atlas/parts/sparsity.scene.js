SCENES.sparsity = ctx => {
  const el = mkCanvas(ctx.stage);
  const kit = KIT.sparse(el, { aspect: w => Math.max(.3, ctx.stage.clientHeight / Math.max(w, 1)), bottomPad: innerWidth > 860 ? 130 : 0 });
  const pb = ctx.qa('[data-pattern]'), sl = ctx.q('#atlas-sparse-s'), sv = ctx.q('#atlas-sparse-s-value');
  const sync = () => {
    press(pb, b => b.dataset.pattern === kit.state.pattern);
    sl.value = String(kit.state.s * 100); sl.disabled = kit.state.pattern === '24'; sv.textContent = kit.state.s * 100 + '%';
    ctx.hud(badge('computed', KIT.sparseText(kit.info())));
  };
  const on = pb.find(b => b.getAttribute('aria-pressed') === 'true');
  kit.set({ pattern: on ? on.dataset.pattern : 'unstructured', s: +sl.value / 100 });
  const offs = [...pb.map(b => listen(b, 'click', () => { kit.set({ pattern: b.dataset.pattern }); sync(); })),
    listen(sl, 'input', () => { kit.set({ s: +sl.value / 100 }); sv.textContent = sl.value + '%'; }),
    listen(sl, 'change', sync)];
  sync();
  return { flat: true, resize: kit.resize, destroy() { offs.forEach(f => f()); kit.destroy(); } };
};
