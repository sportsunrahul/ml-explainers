/* 1. Overview: the hero motion, then the tap strip. */
SCENES.overview = ctx => {
  const el = mkCanvas(ctx.stage), narrow = () => ctx.stage.clientWidth < 620;
  const sc = SceneHero(el, { aspect: S3ASPECT(ctx.stage), bottom: narrow() ? 0 : 130 });
  ctx.hud(badge('toy', 'A drawn warehouse. The fixed class list misses the forklift; a typed name fixes it. The strip shows where text enters each model.'));
  sc.play();
  return { flat: true, replay: sc.play, resize: sc.redraw, destroy: sc.destroy };
};
