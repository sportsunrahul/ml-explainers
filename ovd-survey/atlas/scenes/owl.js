/* OWL-ViT and OWLv2: the token-grid toy (steps 0-2), the self-training pipeline (step 3) and the sourced fine-tuning
   trade-off (step 4). One canvas at a time. The query row and the slider drive the toy; a query click on another step
   switches to the matching tab. The pressed query and the slider live in the article, so a theme remount keeps them. */
SCENES.owl = ctx => {
  const thr = ctx.q('#atlas-owl-thr'), thrV = ctx.q('#atlas-owl-thr-value'), tools = ctx.qa('[data-q]');
  let kind = '', sc = null, el = null, step = 0;
  const narrow = () => ctx.stage.clientWidth < 620;
  const pressedQ = () => { const b = tools.find(t => t.getAttribute('aria-pressed') === 'true'); return b ? b.dataset.q : null; };
  const press = q => tools.forEach(t => t.setAttribute('aria-pressed', String(t.dataset.q === q)));
  function mount(k) {
    if (kind === k && sc) return sc;
    if (sc) sc.destroy(); if (el) el.remove();
    el = mkCanvas(ctx.stage); kind = k;
    const base = { aspect: S3ASPECT(ctx.stage), bottom: narrow() ? 0 : 130 };
    if (k === 'owl') { el.tabIndex = 0; sc = SceneOwl(el, { ...base, onSay: t => ctx.hud(badge('toy', t)) }); sc.setThr(+thr.value); }
    else if (k === 'st') sc = SceneOwlST(el, { ...base, onSay: t => ctx.hud(badge('toy', t)) });
    else sc = SceneOwlTrade(el, { ...base, onSay: t => ctx.hud(badge('sourced', t)) });
    return sc;
  }
  function apply(n) {
    step = n;
    if (n === 0) { press(null); mount('owl').set('boxes'); }
    else if (n === 1) { const q = pressedQ(), t = q && q !== 'image' ? q : 'forklift'; press(t); mount('owl').set('text', t); }
    else if (n === 2) { press('image'); mount('owl').set('image'); }
    else if (n === 3) mount('st').play();
    else { const s = mount('trade'); s.reset(); s.set('ens'); }
  }
  const offs = tools.map(b => listen(b, 'click', () => {
    const target = b.dataset.q === 'image' ? 2 : 1; press(b.dataset.q);
    if (step !== target) ctx.q(`[role="tab"][data-step="${target}"]`).click(); else apply(target);
  }));
  offs.push(listen(thr, 'input', () => { thrV.textContent = (+thr.value).toFixed(2); if (kind === 'owl') sc.setThr(+thr.value); }));
  return { flat: true, step: apply, replay: () => apply(step), resize: () => sc && sc.resize(),
    destroy: () => { offs.forEach(f => f()); if (sc) sc.destroy(); } };
};
