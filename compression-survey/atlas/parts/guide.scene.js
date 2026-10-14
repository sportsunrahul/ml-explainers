SCENES.guide = ctx => {
  const el = mkCanvas(ctx.stage), txt = ctx.q('#atlas-guide-text');
  const kit = KIT.guideFlow(el, { aspect: w => Math.max(.3, ctx.stage.clientHeight / Math.max(w, 1)), bottomPad: innerWidth > 860 ? 130 : 0 });
  const g = ctx.qa('[data-gpu]'), m = ctx.qa('[data-model]'), b = ctx.qa('[data-budget]'), t = ctx.qa('[data-train]');
  const sync = () => { const s = kit.state;
    press(g, x => x.dataset.gpu === s.gpu); press(m, x => x.dataset.model === s.model); press(b, x => +x.dataset.budget === s.budget); press(t, x => (x.dataset.train === '1') === s.train);
    txt.innerHTML = KIT.guideHTML(s).replace(/<span class="g-ev">[\s\S]*?<\/span>/g, '');
    ctx.hud(badge('sourced', 'Our reading of the sourced evidence: ' + KIT.GUIDE.models[s.model] + '.')); };
  const offs = [...g.map(x => listen(x, 'click', () => { kit.set({ gpu: x.dataset.gpu }); sync(); })),
    ...m.map(x => listen(x, 'click', () => { kit.set({ model: x.dataset.model }); sync(); })),
    ...b.map(x => listen(x, 'click', () => { kit.set({ budget: +x.dataset.budget }); sync(); })),
    ...t.map(x => listen(x, 'click', () => { kit.set({ train: x.dataset.train === '1' }); sync(); }))];
  sync();
  return { flat: true, resize: kit.resize, destroy() { offs.forEach(f => f()); kit.destroy(); } };
};
