/* Grounding DINO: the parts/scene-gdino.js toy in its three stages (encoder fusion, query selection, decoder),
   driven by the step tabs and the prompt buttons. The pressed prompt button survives a remount. */
SCENES.gdino = ctx => {
  const el = mkCanvas(ctx.stage); el.tabIndex = 0;
  const btns = ctx.qa('[data-prompt]'), on = btns.find(b => b.getAttribute('aria-pressed') === 'true');
  const sc = SceneGdino(el, { aspect: S3ASPECT(ctx.stage), bottom: ctx.stage.clientWidth < 620 ? 0 : 130, k: 8,
    onSay: t => ctx.hud(badge('toy', t)) });
  const pressP = () => press(btns, b => b.dataset.prompt === sc.state().prompt);
  if (on) sc.set({ prompt: on.dataset.prompt });
  pressP();
  const offs = btns.map(b => listen(b, 'click', () => { sc.set({ prompt: b.dataset.prompt }); pressP(); }));
  return { flat: true, step: n => sc.set({ stage: ['fusion', 'qsel', 'decoder'][n] }), resize: () => sc.resize(),
    destroy: () => { offs.forEach(f => f()); sc.destroy(); } };
};
