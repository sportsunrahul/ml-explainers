SCENES.ssd = ctx => {
  const el = mkCanvas(ctx.stage);
  const mb = ctx.qa('[data-mode]'), qb = ctx.qa('[data-q]');
  // State lives in the pressed buttons, so the remount after a theme toggle keeps it.
  const on = (bs, k) => (bs.find(b => b.getAttribute('aria-pressed') === 'true') || bs[0]).dataset[k];
  const S = { T: 16, mode: on(mb, 'mode'), Q: +on(qb, 'q'), narrow: false };
  const s3 = PE3.scene(el, { aspect: S3ASPECT(ctx.stage), fov: 34, dist: 8, yaw: .42, pitch: .72 });
  let v = CoreSsd.target(S.T, S.mode, S.Q), cancel = () => {};
  const draw = () => { s3.items = CoreSsd.items(v, S, () => 0); s3.redraw(); };
  const fit = () => { const c = CoreSsd.view(ctx.stage.clientWidth, ctx.stage.clientHeight); S.narrow = c.narrow; s3.setView(c, 0); };
  const hud = () => ctx.hud(badge('computed', CoreSsd.readout(S)));
  function go(patch) {
    cancel(); const from = v; Object.assign(S, patch);
    press(mb, b => b.dataset.mode === S.mode); press(qb, b => +b.dataset.q === S.Q);
    const to = CoreSsd.target(S.T, S.mode, S.Q);
    cancel = s3.tween(850, u => { v = CoreSsd.blend(from, to, u); draw(); }); hud();
  }
  const offs = [...mb.map(b => listen(b, 'click', () => go({ mode: b.dataset.mode }))), ...qb.map(b => listen(b, 'click', () => go({ Q: +b.dataset.q }))), PE.onTheme(draw)];
  fit(); draw(); hud();
  return {
    replay() { go({ mode: 'softmax', Q: 0 }); },
    resize() { el.style.height = ctx.stage.clientHeight + 'px'; fit(); draw(); },
    destroy() { cancel(); offs.forEach(f => f()); s3.destroy(); el.remove(); },
  };
};
