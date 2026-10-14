<script>
(() => {
  const V = CoreVision, fig = document.getElementById('viz-vision'), el = document.getElementById('vision-canvas'), out = document.getElementById('vision-readout');
  const dec = document.getElementById('vision-decay'), res = document.getElementById('vision-res');
  const btn = k => document.getElementById('btn-vision-' + k);
  const D0 = { kind: 'raster', t: 6 * V.G + 6, a: .85, res: 224, patch: 16, D: 384 };
  const T = { ...D0 };   // target state set by the controls
  const S = { kind: T.kind, a: T.a, res: T.res, patch: T.patch, D: T.D, tr: 6, tc: 6, heat: V.influence(T.kind, T.t, T.a) };
  let glow = () => 0, cancel = () => {};
  const cv = PE.canvas(el, w => w < 520 ? 1.78 : .56, (ctx, W, H) => V.draw(ctx, W, H, S, glow));
  glow = PE.linkPulses(fig, cv.redraw);
  const write = () => { out.textContent = V.readout(S); };
  const press = () => {
    ['raster', 'vim', 'vmamba'].forEach(k => btn(k).setAttribute('aria-pressed', String(T.kind === k)));
    [8, 16, 32].forEach(p => btn('p' + p).setAttribute('aria-pressed', String(T.patch === p)));
    [192, 384, 768].forEach(d => btn('d' + d).setAttribute('aria-pressed', String(T.D === d)));
    [224, 1248].forEach(r => btn('r' + r).setAttribute('aria-pressed', String(T.res === r)));
    dec.value = Math.round(T.a * 100); res.value = (T.res - 224) / 32;
  };
  // Tween everything the controls changed: heat, target outline, resolution, patch size and width.
  function go(ms = 650) {
    cancel(); press();
    const from = S.heat.slice(), to = V.influence(T.kind, T.t, T.a), o = { tr: S.tr, tc: S.tc, res: S.res, patch: S.patch, D: S.D };
    const n = { tr: Math.floor(T.t / V.G), tc: T.t % V.G, res: T.res, patch: T.patch, D: T.D };
    S.kind = T.kind; S.a = T.a;
    cancel = PE.animate(ms, u => {
      S.heat = V.blend(from, to, u);
      for (const k in n) S[k] = u >= 1 ? n[k] : PE.lerp(o[k], n[k], u);
      cv.redraw();
    }, write);
  }
  const set = (k, v, ms) => () => { T[k] = v; go(ms); };
  ['raster', 'vim', 'vmamba'].forEach(k => btn(k).addEventListener('click', set('kind', k)));
  [8, 16, 32].forEach(p => btn('p' + p).addEventListener('click', set('patch', p)));
  [192, 384, 768].forEach(d => btn('d' + d).addEventListener('click', set('D', d)));
  [224, 1248].forEach(r => btn('r' + r).addEventListener('click', set('res', r, 800)));
  // Sliders follow the hand directly.
  dec.addEventListener('input', () => { cancel(); T.a = S.a = +dec.value / 100; S.heat = V.influence(T.kind, T.t, T.a); Object.assign(S, { tr: Math.floor(T.t / V.G), tc: T.t % V.G, res: T.res, patch: T.patch, D: T.D }); cv.redraw(); write(); });
  res.addEventListener('input', () => { cancel(); T.res = S.res = 224 + 32 * +res.value; press(); cv.redraw(); write(); });
  el.addEventListener('pointerdown', e => {
    const r = el.getBoundingClientRect(), q = V.hit(cv.state.w, cv.state.h, e.clientX - r.left, e.clientY - r.top);
    if (q >= 0) { T.t = q; go(450); }
  });
  el.addEventListener('keydown', e => {
    const q = V.move(T.t, e.key);
    if (q !== T.t) { e.preventDefault(); T.t = q; go(250); }
  });
  btn('reset').addEventListener('click', () => {
    cancel(); Object.assign(T, D0);
    Object.assign(S, { kind: T.kind, a: T.a, res: T.res, patch: T.patch, D: T.D, tr: 6, tc: 6, heat: V.influence(T.kind, T.t, T.a) });
    press(); cv.redraw(); write();
  });
  press(); write();
})();
</script>
