<script>
(() => {
  // One toy SSM, three views (LSSL Fig. 1). Logic lives in CoreS4.
  const fig = document.getElementById('viz-s4-views'), el = document.getElementById('s4-views-canvas'), out = document.getElementById('s4-readout');
  const sD = document.getElementById('s4-dt'), sK = document.getElementById('s4-k');
  const D = { view: 'all', dt: .1, k: 44, p: 1 }, S = { ...D };
  let glow = () => 0, cancel = () => {}, cancelP = () => {};
  const cv = PE.canvas(el, w => w < 520 ? 1.15 : .56, (ctx, W, H) => CoreS4.draw(ctx, W, H, S, glow));
  glow = PE.linkPulses(fig, cv.redraw);
  const write = () => { out.textContent = CoreS4.readout(S); };
  const press = () => fig.querySelectorAll('[data-view]').forEach(b => b.setAttribute('aria-pressed', String(b.dataset.view === S.view)));
  // A view button sets the view and replays its drawing from the start; its end state is the same however often it is pressed.
  function setView(v) {
    cancelP(); S.view = v; press(); S.p = 0;
    cancelP = PE.animate(v === 'rec' ? 2600 : v === 'cont' ? 1600 : 800, u => { S.p = u; cv.redraw(); }, write);
  }
  function setDt(dt, ms = 700) {
    cancel(); const a = CoreS4.toSlider(S.dt), b = CoreS4.toSlider(dt); sD.value = b;
    cancel = PE.animate(ms, u => { S.dt = u >= 1 ? dt : CoreS4.fromSlider(a + (b - a) * u); cv.redraw(); }, write);
  }
  fig.querySelectorAll('[data-view]').forEach(b => b.addEventListener('click', () => setView(b.dataset.view)));
  document.getElementById('btn-s4-dt-small').addEventListener('click', () => setDt(.03));
  document.getElementById('btn-s4-dt-large').addEventListener('click', () => setDt(.3));
  sD.addEventListener('input', () => { cancel(); S.dt = CoreS4.fromSlider(+sD.value); cv.redraw(); write(); });
  sK.addEventListener('input', () => { cancelP(); S.k = +sK.value; if (S.view !== 'conv') { S.view = 'conv'; press(); } S.p = 1; cv.redraw(); write(); });
  document.getElementById('btn-s4-reset').addEventListener('click', () => {
    cancel(); cancelP(); Object.assign(S, D); sD.value = CoreS4.toSlider(D.dt); sK.value = D.k; press(); cv.redraw(); write();
  });
  sD.value = CoreS4.toSlider(D.dt); sK.value = D.k; press(); write();
})();
</script>
