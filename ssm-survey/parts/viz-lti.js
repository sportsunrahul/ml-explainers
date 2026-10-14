<script>
(() => {
  // Selective copying toy: a fixed decay against a gate set by each token. Logic lives in CoreLti.
  const fig = document.getElementById('viz-lti'), el = document.getElementById('lti-canvas'), out = document.getElementById('lti-readout'), slider = document.getElementById('lti-p');
  const S0 = { mode: 'random', seed: 2, p: .17, revealed: false, u: 0 }, S = { ...S0 };
  const btn = k => document.getElementById('btn-lti-' + k);
  let glow = () => 0, cancel = () => {};
  const cv = PE.canvas(el, w => (w < 520 ? 306 : 334) / w, (ctx, W, H) => CoreLti.draw(ctx, W, H, S, glow));
  glow = PE.linkPulses(fig, cv.redraw);
  const press = () => { btn('random').setAttribute('aria-pressed', String(S.mode === 'random')); btn('even').setAttribute('aria-pressed', String(S.mode === 'even')); btn('reveal').setAttribute('aria-pressed', String(S.revealed)); };
  const write = () => { out.textContent = CoreLti.readout(S); };
  // The reveal sweeps a cursor over the tokens; both states update as it passes.
  function sweep() { cancel(); S.u = 0; cancel = PE.animate(1800, u => { S.u = u; cv.redraw(); }, write); }
  function setMode(m) { S.mode = m; press(); if (S.revealed) sweep(); else { cv.redraw(); write(); } }
  btn('random').addEventListener('click', () => setMode('random'));
  btn('even').addEventListener('click', () => setMode('even'));
  // A new random draw: cue-exempt; the reset restores the seed.
  btn('new').addEventListener('click', () => { S.seed++; setMode('random'); });
  btn('reveal').addEventListener('click', () => { S.revealed = true; press(); sweep(); });
  slider.addEventListener('input', () => { cancel(); S.p = +slider.value / 100; S.u = 1; cv.redraw(); write(); });
  btn('reset').addEventListener('click', () => { cancel(); Object.assign(S, S0); slider.value = 17; press(); cv.redraw(); write(); });
  press(); write();
})();
</script>
