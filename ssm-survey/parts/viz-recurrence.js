<script>
(() => {
  const fig = document.getElementById('viz-recurrence'), el = document.getElementById('recurrence-canvas');
  const sr = document.getElementById('recurrence-r'), st = document.getElementById('recurrence-theta');
  const vr = document.getElementById('recurrence-r-val'), vt = document.getElementById('recurrence-theta-val');
  const out = document.getElementById('recurrence-readout');
  const P = CoreRecurrence.PRESETS, S = { ...P.reset };
  const bLin = document.getElementById('btn-recurrence-linear'), bTanh = document.getElementById('btn-recurrence-tanh');
  let glow = () => 0, cancel = () => {};
  const cv = PE.canvas(el, w => w < 520 ? 1.5 : .66, (ctx, W, H) => CoreRecurrence.draw(ctx, W, H, S, glow));
  glow = PE.linkPulses(fig, cv.redraw);
  const sync = () => { sr.value = S.r; st.value = S.th; vr.textContent = S.r.toFixed(Math.abs(S.r * 100 - Math.round(S.r * 100)) > 1e-6 ? 3 : 2); vt.textContent = S.th.toFixed(2); };
  const write = () => { out.textContent = CoreRecurrence.readout(S); };
  // Every button sets a target state and tweens to it; pressing twice lands on the same state.
  function go(target, ms = 750) {
    cancel();
    if ('m' in target) { bLin.setAttribute('aria-pressed', String(target.m === 0)); bTanh.setAttribute('aria-pressed', String(target.m === 1)); }
    const a = { ...S };
    cancel = PE.animate(ms, u => { for (const k in target) S[k] = u >= 1 ? target[k] : PE.lerp(a[k], target[k], u); sync(); cv.redraw(); }, write);
  }
  [['fast', P.fast], ['long', P.long], ['ring', P.ring], ['explode', P.explode], ['linear', { m: 0 }], ['tanh', { m: 1 }], ['reset', P.reset]].forEach(([k, t]) =>
    document.getElementById('btn-recurrence-' + k).addEventListener('click', () => go(t)));
  const drag = (input, key) => input.addEventListener('input', () => { cancel(); S[key] = +input.value; sync(); cv.redraw(); write(); });
  drag(sr, 'r'); drag(st, 'th');
  sync(); write();
})();
</script>
