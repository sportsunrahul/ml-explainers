SCENES.vision = ctx => {
  const V = CoreVision, el = mkCanvas(ctx.stage);
  el.tabIndex = 0; el.style.cursor = 'pointer'; el.style.touchAction = 'pan-y';
  el.setAttribute('role', 'img'); el.setAttribute('aria-label', 'A 14 by 14 grid of image patches; the blue heat shows how strongly each patch can reach the green one after one scan layer.');
  const dec = ctx.q('#vision-atlas-decay'), res = ctx.q('#vision-atlas-res');
  const sb = ctx.qa('[data-scan]'), pb = ctx.qa('[data-patch]'), db = ctx.qa('[data-d]');
  const pressed = (btns, k, d) => { const b = btns.find(x => x.getAttribute('aria-pressed') === 'true'); return b ? b.dataset[k] : d; };
  // Everything but the green patch lives in the article's controls, so a theme remount keeps it.
  const T = { kind: pressed(sb, 'scan', 'raster'), t: 6 * V.G + 6, a: +dec.value / 100, res: 224 + 32 * +res.value, patch: +pressed(pb, 'patch', 16), D: +pressed(db, 'd', 384) };
  const S = { kind: T.kind, a: T.a, res: T.res, patch: T.patch, D: T.D, tr: 6, tc: 6, heat: V.influence(T.kind, T.t, T.a), step: null };
  const pulse = {}, glow = k => pulse[k] || 0;
  let cancel = () => {}, cancelGlow = () => {};
  const usedH = (W, H) => W < 620 ? H : H - 150;   // desktop: keep the bottom of the stage clear for the readout box
  const cv = PE.canvas(el, S3ASPECT(ctx.stage), (c, W, H) => { c.fillStyle = PE.css('--bg'); c.fillRect(0, 0, W, H); V.draw(c, W, usedH(W, H), S, glow); });
  const hud = () => ctx.hud(badge('toy', V.readout(S)));
  const sync = () => { press(sb, b => b.dataset.scan === T.kind); press(pb, b => +b.dataset.patch === T.patch); press(db, b => +b.dataset.d === T.D); };
  function go(ms = 650) {
    cancel(); sync();
    const from = S.heat.slice(), to = V.influence(T.kind, T.t, T.a), o = { tr: S.tr, tc: S.tc, res: S.res, patch: S.patch, D: S.D };
    const n = { tr: Math.floor(T.t / V.G), tc: T.t % V.G, res: T.res, patch: T.patch, D: T.D };
    S.kind = T.kind; S.a = T.a;
    cancel = PE.animate(ms, u => { S.heat = V.blend(from, to, u); for (const k in n) S[k] = u >= 1 ? n[k] : PE.lerp(o[k], n[k], u); cv.redraw(); }, hud);
  }
  const flash = k => { cancelGlow(); cancelGlow = PE.animate(1400, u => { pulse[k] = Math.sin(Math.PI * u); cv.redraw(); }, () => { pulse[k] = 0; cv.redraw(); }); };
  const offs = [
    ...sb.map(b => listen(b, 'click', () => { T.kind = b.dataset.scan; go(); })),
    ...pb.map(b => listen(b, 'click', () => { T.patch = +b.dataset.patch; go(); })),
    ...db.map(b => listen(b, 'click', () => { T.D = +b.dataset.d; go(); })),
    listen(dec, 'input', () => { cancel(); T.a = S.a = +dec.value / 100; S.heat = V.influence(T.kind, T.t, T.a); cv.redraw(); hud(); }),
    listen(res, 'input', () => { cancel(); T.res = S.res = 224 + 32 * +res.value; cv.redraw(); hud(); }),
    listen(el, 'pointerdown', e => { const r = el.getBoundingClientRect(), W = cv.state.w, q = V.hit(W, usedH(W, cv.state.h), e.clientX - r.left, e.clientY - r.top); if (q >= 0) { T.t = q; go(450); } }),
    listen(el, 'keydown', e => { const q = V.move(T.t, e.key); if (q !== T.t) { e.preventDefault(); T.t = q; go(250); } }),
  ];
  sync(); hud();
  return {
    flat: true,
    // Tabs set the scan the step text talks about; the first call at mount keeps what the controls already show.
    step(n) {
      const first = S.step === null; S.step = n; if (first) return;
      if (n === 0 && T.kind !== 'raster') { T.kind = 'raster'; go(); }
      else if (n === 1 && T.kind === 'raster') { T.kind = 'vim'; go(); }
      else if (n === 2) flash('threshold');
    },
    resize: () => cv.resize(),
    destroy() { cancel(); cancelGlow(); offs.forEach(f => f()); cv.destroy(); el.remove(); },
  };
};
