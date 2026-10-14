/* ViLD: the chapter-3 two-stage toy (step 0), then the chapter-4 embedding sphere under the text loss (1) and both losses (2).
   Both canvases are made at mount and shown by step, so the shell's one-time flat flag is set here per step. */
SCENES.vild = ctx => {
  const narrow = () => ctx.stage.clientWidth < 620;
  const el2 = mkCanvas(ctx.stage), el3 = mkCanvas(ctx.stage);
  el2.tabIndex = 0;
  let step = 0;
  const rpn = SceneRpn(el2, { aspect: S3ASPECT(ctx.stage), bottom: narrow() ? 0 : 130, onSay: t => { if (step === 0) ctx.hud(badge('toy', t)); } });
  const cloud = SceneCloud(el3, { aspect: S3ASPECT(ctx.stage), onSay: t => { if (step > 0) ctx.hud(badge('toy', t)); } });
  const slider = ctx.q('#vild-k'), kv = ctx.q('#vild-k-value');
  const offs = [listen(slider, 'input', () => { kv.textContent = slider.value; rpn.setK(+slider.value); })];
  const playRpn = () => { rpn.reset(); rpn.setK(+slider.value); rpn.play(); };
  // style.display, not the hidden attribute: the shell's #stage canvas { display: block } would override it
  function show(n) { step = n; el2.style.display = n === 0 ? '' : 'none'; el3.style.display = n === 0 ? 'none' : ''; document.body.dataset.flat = n === 0 ? '1' : '0'; }
  return {
    flat: true,
    step(n) { show(n); if (n === 0) playRpn(); else cloud.set(n === 1 ? 'text' : 'both'); },
    replay() { if (step === 0) playRpn(); else { cloud.reset(); cloud.set(step === 1 ? 'text' : 'both'); } },
    // pe3 sizes from width only, so a stage height change is pushed into its canvas by hand
    resize() { rpn.resize(); if (step > 0) el3.style.height = ctx.stage.clientHeight + 'px'; },
    destroy() { offs.forEach(f => f()); rpn.destroy(); cloud.destroy(); },
  };
};
