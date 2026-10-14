/* 2. Ask: uncertainty sampling on the toy pool. */
SCENES.ask = ctx => {
  const el = mkCanvas(ctx.stage), narrow = () => ctx.stage.clientWidth < 620;
  const f = FIG.pool(el, { aspect: S3ASPECT(ctx.stage), padBottom: narrow() ? 0 : 130, say: t => ctx.hud(badge('toy', t)) });
  const ms = ctx.qa('[data-measure]'), rs = ctx.qa('[data-rounds]'), vb = ctx.q('[data-view]');
  const offs = [
    ...ms.map(b => listen(b, 'click', () => { press(ms, x => x === b); f.measure(b.dataset.measure); })),
    ...rs.map(b => listen(b, 'click', () => { press(rs, x => x === b); f.rounds(+b.dataset.rounds); })),
    listen(ctx.q('[data-step-round]'), 'click', () => { press(rs, () => false); f.step(); }),
    listen(vb, 'click', () => { const on = vb.getAttribute('aria-pressed') !== 'true'; vb.setAttribute('aria-pressed', String(on)); f.view(on ? 'rand' : 'unc'); }),
  ];
  return { flat: true, resize: f.redraw, destroy() { offs.forEach(g => g()); f.destroy(); } };
};
