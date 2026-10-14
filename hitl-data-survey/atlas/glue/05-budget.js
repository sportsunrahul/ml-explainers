/* 5. Budget: TypiClust's two-region model and Munjal et al. Table 4. */
SCENES.budget = ctx => {
  const el = mkCanvas(ctx.stage), narrow = () => ctx.stage.clientWidth < 620;
  const f = FIG.budget(el, { aspect: S3ASPECT(ctx.stage), padBottom: narrow() ? 0 : 130, say: t => ctx.hud(badge('computed', t)) });
  const bs = ctx.qa('[data-sr]'), p = ctx.q('#atlas-budget-p'), pv = ctx.q('#budget-pv');
  const offs = [...bs.map(b => listen(b, 'click', () => { press(bs, x => x === b); f.sr(b.dataset.sr === '1'); })),
    listen(p, 'input', () => { pv.textContent = (+p.value).toFixed(2); f.p(+p.value); })];
  f.p(+p.value); f.sr(bs[1].getAttribute('aria-pressed') === 'true');
  return { flat: true, resize: f.redraw, destroy() { offs.forEach(g => g()); f.destroy(); } };
};
