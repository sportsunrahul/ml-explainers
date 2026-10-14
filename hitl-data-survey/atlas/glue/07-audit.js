/* 7. Audit: test-set errors and the re-ranking. */
SCENES.audit = ctx => {
  const el = mkCanvas(ctx.stage), narrow = () => ctx.stage.clientWidth < 620;
  const f = FIG.audit(el, { aspect: S3ASPECT(ctx.stage), padBottom: narrow() ? 0 : 130, say: t => ctx.hud(badge('sourced', t)) });
  const bs = ctx.qa('[data-corr]');
  const offs = bs.map(b => listen(b, 'click', () => { press(bs, x => x === b); f.corrected(b.dataset.corr === '1'); }));
  f.reset(); press(bs, x => x === bs[0]);
  return { flat: true, resize: f.redraw, destroy() { offs.forEach(g => g()); f.destroy(); } };
};
