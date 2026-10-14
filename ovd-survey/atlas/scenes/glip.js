/* GLIP: the grounding matrix (tabs 0 and 1) and the fusion toy (tab 2), swapped on one canvas. */
SCENES.glip = ctx => {
  const narrow = () => ctx.stage.clientWidth < 620;
  let el = null, sc = null, kind = '';
  const say = t => ctx.hud(badge('toy', t));
  function mount(k) {
    if (kind === k) return;
    if (sc) sc.destroy(); if (el) el.remove();
    el = mkCanvas(ctx.stage); el.tabIndex = 0; kind = k;
    const o = { aspect: S3ASPECT(ctx.stage), bottom: narrow() ? 0 : 130, onSay: say, onPick: () => press() };
    sc = k === 'ground' ? SceneGlipGround(el, o) : SceneGlipFusion(el, o);
  }
  function press() {
    ctx.qa('[data-phrase]').forEach(b => b.setAttribute('aria-pressed', String(kind === 'ground' && sc.state() === b.dataset.phrase)));
    ctx.qa('[data-fuse]').forEach(b => b.setAttribute('aria-pressed', String(kind === 'fusion' && sc.mode() === b.dataset.fuse)));
  }
  function step(n) {
    if (n === 2) { mount('fusion'); sc.setMode('deep'); } else { mount('ground'); sc.set(n === 0 ? 'phrases' : 'all'); }
    press();
  }
  // A control on another tab switches the tab first (the shell then calls step), then sets its own state.
  const tab = n => { const t = ctx.q(`[role="tab"][data-step="${n}"]`); if (t && +ctx.article.dataset.step !== n) t.click(); };
  const offs = [
    ...ctx.qa('[data-phrase]').map(b => listen(b, 'click', () => { tab(1); mount('ground'); sc.set(b.dataset.phrase); press(); })),
    ...ctx.qa('[data-fuse]').map(b => listen(b, 'click', () => { tab(2); mount('fusion'); sc.setMode(b.dataset.fuse); press(); })),
  ];
  return { flat: true, step, resize: () => sc && sc.resize(),
    destroy: () => { offs.forEach(f => f()); if (sc) sc.destroy(); if (el) el.remove(); } };
};
