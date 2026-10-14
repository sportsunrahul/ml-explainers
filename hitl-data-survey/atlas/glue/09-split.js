/* 9. Split: Co-teaching's small-loss swap (step 0) and DivideMix's two-bump split (step 1). */
SCENES.split = ctx => {
  let el = null, f = null, cur = -1; const narrow = () => ctx.stage.clientWidth < 620;
  function mount(n) {
    if (n === cur) return; cur = n; if (f) f.destroy(); ctx.stage.replaceChildren();
    el = mkCanvas(ctx.stage, 'tiles'); const o = { aspect: S3ASPECT(ctx.stage), padBottom: narrow() ? 0 : 130, say: t => ctx.hud(ours(t)) };
    f = n === 0 ? FIG.coteach(el, o) : FIG.gmm(el, o); if (n === 0) f.reveal(true);
  }
  mount(0);
  return { flat: true, step: mount, resize: () => f && f.redraw(), destroy() { f && f.destroy(); } };
};
