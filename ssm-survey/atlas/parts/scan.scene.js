SCENES.scan = ctx => {
  const S = { step: 0, L: 8, seed: 3, round: 0, u: 1 }, M = { mode: 'fused', stage: 0, u: 1 };
  let cancel = () => {}, play = null;
  const el = mkCanvas(ctx.stage);
  const bottom = () => (ctx.stage.clientWidth < 620 ? 0 : 130);
  const cv = PE.canvas(el, S3ASPECT(ctx.stage), (c, W, H) => {
    c.clearRect(0, 0, W, H);
    const pad = W < 620 ? 0 : 40, w = Math.min(W - 2 * pad, 1000), x = (W - w) / 2;
    c.save(); c.translate(x, 0);
    if (S.step === 0) CoreScan.draw(c, w, Math.min(H, 640), { ...S, bottom: H > 640 ? 0 : bottom() });
    else CoreScan.drawMem(c, w, Math.min(H, 560), { ...M, bottom: H > 560 ? 0 : bottom() });
    c.restore();
  });
  const hud = () => ctx.hud(S.step === 0 ? badge('toy', CoreScan.readout(S)) : badge('computed', CoreScan.readoutMem(M)));
  const N = () => CoreScan.lastRound(S.L);
  function go(r) { cancel(); S.round = PE.clamp(r, 0, N()); S.u = 0; hud(); cancel = PE.animate(650, u => { S.u = u; cv.redraw(); }); }
  function stopPlay() { if (play) { clearTimeout(play); play = null; } }
  function replay() {
    stopPlay();
    if (S.step === 1) { setStage(1); return; }
    if (PE.instant()) { go(N()); return; }
    go(0); const tick = () => { if (S.round < N()) { go(S.round + 1); play = setTimeout(tick, 900); } else play = null; }; play = setTimeout(tick, 700);
  }
  function setStage(k) { cancel(); M.stage = k; M.u = 0; press(); hud(); cancel = PE.animate(800, u => { M.u = u; cv.redraw(); }); }
  const press = () => {
    ctx.qa('[data-n]').forEach(b => b.setAttribute('aria-pressed', String(+b.dataset.n === S.L)));
    ctx.qa('[data-mode]').forEach(b => b.setAttribute('aria-pressed', String(b.dataset.mode === M.mode)));
    ctx.qa('[data-stage]').forEach(b => b.setAttribute('aria-pressed', String(+b.dataset.stage === M.stage)));
  };
  const offs = [
    ...ctx.qa('[data-n]').map(b => listen(b, 'click', () => { stopPlay(); S.L = +b.dataset.n; press(); go(0); })),
    ...ctx.qa('[data-dir]').map(b => listen(b, 'click', () => { stopPlay(); const d = +b.dataset.dir; go(d === 0 ? 0 : S.round + d); })),
    ...ctx.qa('[data-mode]').map(b => listen(b, 'click', () => { M.mode = b.dataset.mode; setStage(M.stage); })),
    ...ctx.qa('[data-stage]').map(b => listen(b, 'click', () => setStage(+b.dataset.stage))),
  ];
  press(); hud();
  return {
    flat: true,
    step(n) { stopPlay(); cancel(); S.step = n; S.u = 1; M.u = 1; cv.redraw(); hud(); },
    replay,
    resize() { cv.resize(); },
    destroy() { stopPlay(); cancel(); offs.forEach(f => f()); cv.destroy(); el.remove(); },
  };
};
