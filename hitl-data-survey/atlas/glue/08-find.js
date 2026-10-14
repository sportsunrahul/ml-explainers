/* 8. Find: confident learning on run C. */
SCENES.find = ctx => {
  const el = mkCanvas(ctx.stage, 'tiles'), narrow = () => ctx.stage.clientWidth < 620;
  const f = FIG.cl(el, { aspect: S3ASPECT(ctx.stage), padBottom: narrow() ? 0 : 130, say: t => ctx.hud(ours(t)) });
  const cs = ctx.qa('[data-cell]'), vb = ctx.q('[data-view]');
  const offs = [...cs.map(b => listen(b, 'click', () => { press(cs, x => x === b); f.cell(+b.dataset.cell); })),
    listen(vb, 'click', () => { const on = vb.getAttribute('aria-pressed') !== 'true'; vb.setAttribute('aria-pressed', String(on)); f.view(on ? 'true' : 'joint'); })];
  return { flat: true, resize: f.redraw, destroy() { offs.forEach(g => g()); f.destroy(); } };
};
