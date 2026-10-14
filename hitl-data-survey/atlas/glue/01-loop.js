/* 1. The loop: the page hero, on the stage. */
SCENES.loop = ctx => {
  const el = mkCanvas(ctx.stage, 'tiles'), narrow = () => ctx.stage.clientWidth < 620;
  const h = FIG.hero(el, { aspect: S3ASPECT(ctx.stage), padBottom: narrow() ? 0 : 130 });
  ctx.hud(badge('toy', 'Real CIFAR-10 images; the round itself is a toy: confidence bars and losses are drawn, not computed.'));
  return { flat: true, replay: h.replay, resize: h.redraw, destroy: h.destroy };
};
