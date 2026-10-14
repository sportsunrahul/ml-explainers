SCENES.outliers = ctx => {
  const el = mkCanvas(ctx.stage);
  const wide = innerWidth > 860;
  const kit = KIT.actsurf(el, { aspect: S3ASPECT(ctx.stage), dist: w => wide ? 8.6 : 8.4, lift: wide ? .45 : .1 });
  const mb = ctx.qa('[data-act]'), sl = ctx.q('[data-alpha]'), val = ctx.q('[data-alpha-value]');
  const sync = () => { press(mb, b => b.dataset.act === kit.state.mode); val.textContent = (+sl.value).toFixed(2);
    ctx.hud(badge('toy', KIT.actsurfText(kit.info()))); };
  // a theme toggle remounts the scene; restore the state the article's controls still show
  const was = (mb.find(b => b.getAttribute('aria-pressed') === 'true') || {}).dataset;
  if (was && was.act === 'int8') kit.set({ mode: 'int8' }, 0);
  else if ((was && was.act === 'smooth') || +sl.value !== .5) kit.set({ mode: 'smooth', alpha: +sl.value }, 0);
  const offs = [...mb.map(b => listen(b, 'click', () => {
      if (b.dataset.act === 'smooth') { sl.value = '0.5'; kit.set({ mode: 'smooth', alpha: .5 }); } else kit.set({ mode: b.dataset.act });
      sync(); })),
    listen(sl, 'input', () => { val.textContent = (+sl.value).toFixed(2); kit.set({ mode: 'smooth', alpha: +sl.value }, 250); }),
    listen(sl, 'change', sync)];
  sync();
  return { resize: kit.resize, destroy() { offs.forEach(f => f()); kit.destroy(); } };
};
