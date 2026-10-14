SCENES.recurrence = ctx => {
  const el = mkCanvas(ctx.stage);
  const sr = ctx.q('#recurrence-atlas-r'), st = ctx.q('#recurrence-atlas-theta'), vr = ctx.q('#recurrence-atlas-r-val'), vt = ctx.q('#recurrence-atlas-theta-val');
  const modeBtns = ctx.qa('[data-mode]'), P = CoreRecurrence.PRESETS;
  // State is read back from the controls, so the remount after a theme toggle keeps it.
  const tanhOn = () => modeBtns.some(b => b.dataset.mode === 'tanh' && b.getAttribute('aria-pressed') === 'true');
  const S = { r: +sr.value, th: +st.value, m: tanhOn() ? 1 : 0 };
  let cancel = () => {}, cancelPulse = () => {};
  const pulse = { k: '', v: 0 }, glow = k => pulse.k === k ? pulse.v : 0;
  const cv = PE.canvas(el, S3ASPECT(ctx.stage), (c, W, H) => { c.clearRect(0, 0, W, H); CoreRecurrence.draw(c, W, W < 620 ? H : H - 200, S, glow); });
  const hud = () => ctx.hud(badge('toy', CoreRecurrence.readout(S)));
  const sync = () => { sr.value = S.r; st.value = S.th; vr.textContent = S.r.toFixed(Math.abs(S.r * 100 - Math.round(S.r * 100)) > 1e-6 ? 3 : 2); vt.textContent = S.th.toFixed(2); };
  function go(t) {
    cancel();
    if ('m' in t) press(modeBtns, b => (b.dataset.mode === 'tanh') === (t.m === 1));
    const a = { ...S };
    cancel = PE.animate(750, u => { for (const k in t) S[k] = u >= 1 ? t[k] : PE.lerp(a[k], t[k], u); sync(); cv.redraw(); }, hud);
  }
  // Entering a step pulses the part of the stage its text is about.
  function flash(k) {
    cancelPulse(); pulse.k = k;
    cancelPulse = PE.animate(1100, u => { pulse.v = Math.sin(Math.PI * u); cv.redraw(); }, () => { pulse.v = 0; cv.redraw(); });
  }
  const drag = (input, key) => listen(input, 'input', () => { cancel(); S[key] = +input.value; sync(); cv.redraw(); hud(); });
  const offs = [
    drag(sr, 'r'), drag(st, 'th'),
    ...ctx.qa('[data-preset]').map(b => listen(b, 'click', () => go(P[b.dataset.preset]))),
    ...modeBtns.map(b => listen(b, 'click', () => go({ m: b.dataset.mode === 'tanh' ? 1 : 0 }))),
  ];
  sync(); hud();
  return {
    flat: true,
    // The recipe step is about the linear cell's gain, which the tanh cell does not have.
    step(n) { if (n === 2 && S.m > 0) go({ m: 0 }); flash(['magnitude', 'bars', 'gain'][n] || ''); },
    resize: () => cv.resize(),
    destroy() { cancel(); cancelPulse(); offs.forEach(f => f()); cv.destroy(); },
  };
};
