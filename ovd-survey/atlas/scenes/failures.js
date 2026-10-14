/* Failure modes: jargon (the 3D sphere), tiny defects (computed), false alarms (D3 threshold sweep). */
SCENES.failures = ctx => {
  const narrow = () => ctx.stage.clientWidth < 620;
  let el = null, sc = null, cur = -1, t1 = 0;
  const BADGE = ['toy', 'computed', 'sourced'];
  function opts() { return { aspect: S3ASPECT(ctx.stage), bottom: narrow() ? 0 : 130, onSay: t => ctx.hud(badge(BADGE[cur], t)) }; }
  const MAKE = [
    el2 => { const s = SceneCloud(el2, { ...opts(), start: 'gap' }); s.set('jargon'); clearTimeout(t1); t1 = setTimeout(() => s.set('describe'), PE.instant() ? 0 : 1600); return s; },
    el2 => SceneTiny(el2, opts()),
    el2 => { const s = SceneFppc(el2, opts()); s.set(6); return s; }];
  function mount(n) { clearTimeout(t1); if (sc) sc.destroy(); if (el) el.remove(); cur = n; el = mkCanvas(ctx.stage); sc = MAKE[n](el); }
  mount(0);
  return { flat: false, step: n => { if (n !== cur) mount(n); }, resize: () => sc && sc.redraw(), destroy: () => { clearTimeout(t1); if (sc) sc.destroy(); } };
};
