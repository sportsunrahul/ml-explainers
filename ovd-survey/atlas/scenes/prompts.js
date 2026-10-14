/* Visual prompts: steps 0 and 1 drive the chapter-11 toy (name, then box); step 2 swaps the stage to the sourced bench.
   The prompt buttons sit outside the step texts; pressed on step 2 they go back to the toy's tab first. */
SCENES.prompts = ctx => {
  const narrow = () => ctx.stage.clientWidth < 620;
  let el = null, toy = null, bench = null, cur = 0;
  const drop = () => { if (toy) toy.destroy(); if (bench) bench.destroy(); toy = bench = null; if (el) el.remove(); el = null; };
  const mountToy = () => { if (toy) return; drop(); el = mkCanvas(ctx.stage); el.tabIndex = 0;
    toy = SceneVisualPrompts(el, { aspect: S3ASPECT(ctx.stage), bottom: narrow() ? 0 : 130, onSay: t => ctx.hud(badge('toy', t)) }); };
  const mountBench = () => { if (bench) return; drop(); el = mkCanvas(ctx.stage);
    bench = SceneVisualPromptsBench(el, { aspect: S3ASPECT(ctx.stage), bottom: narrow() ? 0 : 130, mixed: true,
      onSay: t => ctx.hud(badge('sourced', t + ' T-Rex2 Tables 1 and 8; RF100-VL Table 2; FSOD-VLM Table 3.')) }); };
  const modeBtns = ctx.qa('[data-vpmode]');
  const sync = () => press(modeBtns, b => !!toy && toy.mode() === b.dataset.vpmode);
  const offs = modeBtns.map(b => listen(b, 'click', () => {
    if (cur === 2) { const tab = ctx.q(`[role="tab"][data-step="${b.dataset.vpmode === 'text' ? 0 : 1}"]`); if (tab) tab.click(); }
    mountToy(); toy.set(b.dataset.vpmode); sync(); }));
  return { flat: true,
    step(n) {
      cur = n;
      if (n === 2) { mountBench(); sync(); return; }
      mountToy(); toy.set(n === 0 ? 'text' : 'box'); sync();
    },
    resize() { if (toy) toy.redraw(); if (bench) bench.redraw(); },
    destroy() { offs.forEach(f => f()); drop(); } };
};
