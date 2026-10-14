/* 2. The class list becomes a text encoder: the chapter-1 classifier toy, driven by the step tabs. */
SCENES.reads = ctx => {
  const el = mkCanvas(ctx.stage); el.tabIndex = 0;
  const sc = SceneClassifier(el, { aspect: S3ASPECT(ctx.stage), bottom: ctx.stage.clientWidth < 620 ? 0 : 130,
    onSay: t => ctx.hud(badge('toy', t)) });
  return { flat: true, step: n => sc.set(['closed', 'text', 'add'][n]), resize: sc.redraw, destroy: sc.destroy };
};
