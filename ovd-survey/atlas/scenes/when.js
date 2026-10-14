/* Prompt or train: the chapter-12 charts, one per step tab. */
SCENES.when = ctx => {
  const narrow = () => ctx.stage.clientWidth < 620;
  let el = null, sc = null;
  const MAKE = [el2 => SceneLadder(el2, opts()), el2 => { const s = SceneOdinw(el2, opts()); s.set('ft'); return s; }, el2 => { const s = SceneShots(el2, opts()); s.set('match'); return s; }];
  const BADGE = ['sourced', 'sourced', 'sourced'];
  let cur = 0;
  function opts() { return { aspect: S3ASPECT(ctx.stage), bottom: narrow() ? 0 : 130, onSay: t => ctx.hud(badge(BADGE[cur], t)) }; }
  function mount(n) { if (sc) sc.destroy(); if (el) el.remove(); cur = n; el = mkCanvas(ctx.stage); sc = MAKE[n](el); }
  mount(0);
  return { flat: true, step: n => { if (n !== cur || !sc) mount(n); }, resize: () => sc && sc.redraw(), destroy: () => { if (sc) sc.destroy(); } };
};
