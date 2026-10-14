/* 6. Memorize: run A's loss ridge. */
SCENES.memorize = ctx => {
  const el = mkCanvas(ctx.stage, 'pe3');
  const f = FIG.ridge(el, null, { aspect: S3ASPECT(ctx.stage), say: t => ctx.hud(ours(t)) });
  const bs = ctx.qa('[data-epoch]');
  const offs = bs.map(b => listen(b, 'click', () => { press(bs, x => x === b); f.epoch(+b.dataset.epoch); }));
  const on = bs.find(b => b.getAttribute('aria-pressed') === 'true'); if (on) f.epoch(+on.dataset.epoch);
  return { resize: f.redraw, destroy() { offs.forEach(g => g()); f.destroy(); } };
};
