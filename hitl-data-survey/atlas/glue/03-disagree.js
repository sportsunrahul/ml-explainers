/* 3. Disagree: the entropy dome and BALD. */
SCENES.disagree = ctx => {
  const el = mkCanvas(ctx.stage, 'pe3');
  const f = FIG.dome(el, { aspect: S3ASPECT(ctx.stage), say: t => ctx.hud(badge('toy', t)) });
  const ps = ctx.qa('[data-preset]');
  const offs = ps.map(b => listen(b, 'click', () => { press(ps, x => x === b); f.preset(b.dataset.preset); }));
  return { resize: f.redraw, destroy() { offs.forEach(g => g()); f.destroy(); } };
};
