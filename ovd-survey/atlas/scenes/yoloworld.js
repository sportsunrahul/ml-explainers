/* YOLO-World: the chapter-10 toy (SceneRepvl), driven by the step tabs: dense YOLO, text in the neck, fold into weights. */
SCENES.yoloworld = ctx => {
  const el = mkCanvas(ctx.stage); el.tabIndex = 0;
  const sc = SceneRepvl(el, { aspect: S3ASPECT(ctx.stage), bottom: ctx.stage.clientWidth < 620 ? 0 : 130,
    onSay: t => ctx.hud(badge('toy', t)) });
  const vb = ctx.qa('[data-vocab]');
  const sync = () => press(vb, b => +b.dataset.vocab === sc.state().vocab);
  const offs = vb.map(b => listen(b, 'click', () => { sc.setVocab(+b.dataset.vocab); sync(); }));
  sync();
  return { flat: true, step: n => sc.set(['dense', 'gate', 'reparam'][n]), resize: sc.resize,
    destroy: () => { offs.forEach(f => f()); sc.destroy(); } };
};
