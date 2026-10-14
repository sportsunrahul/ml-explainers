SCENES.distill = ctx => {
  const el = mkCanvas(ctx.stage);
  const kit = KIT.temp(el, { aspect: w => Math.max(.3, ctx.stage.clientHeight / Math.max(w, 1)), bottomPad: innerWidth > 860 ? 180 : 0 });
  const sl = ctx.q('[data-temp]'), val = ctx.q('[data-temp-val]'), tb = ctx.qa('[data-t2]');
  // A theme toggle remounts the scene, so start from the controls' current state.
  const on = tb.find(b => b.getAttribute('aria-pressed') === 'true');
  kit.set({ T: +sl.value, t2: !!on && on.dataset.t2 === 'on' });
  const sync = () => { val.textContent = kit.state.Tt.toFixed(1); press(tb, b => (b.dataset.t2 === 'on') === kit.state.t2);
    ctx.hud(badge('computed', KIT.tempText(kit.info()))); };
  const offs = [listen(sl, 'input', () => { kit.set({ T: +sl.value }); val.textContent = (+sl.value).toFixed(1); }),
    listen(sl, 'change', sync),
    ...tb.map(b => listen(b, 'click', () => { kit.set({ t2: b.dataset.t2 === 'on' }); sync(); }))];
  sync();
  return { flat: true, resize: kit.resize, destroy() { offs.forEach(f => f()); kit.destroy(); } };
};
