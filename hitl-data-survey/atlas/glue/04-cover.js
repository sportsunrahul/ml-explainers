/* 4. Cover: filling one batch. */
SCENES.cover = ctx => {
  const el = mkCanvas(ctx.stage), narrow = () => ctx.stage.clientWidth < 620;
  const f = FIG.cover(el, { aspect: S3ASPECT(ctx.stage), padBottom: narrow() ? 0 : 130, say: t => ctx.hud(badge('toy', t)) });
  const bs = ctx.qa('[data-strategy]');
  const offs = bs.map(b => listen(b, 'click', () => { press(bs, x => x === b); f.strategy(b.dataset.strategy); }));
  press(bs, x => x === bs[0]); f.reset();
  return { flat: true, resize: f.redraw, destroy() { offs.forEach(g => g()); f.destroy(); } };
};
