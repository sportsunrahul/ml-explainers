<script>
(() => {
  const fig = document.getElementById('viz-ground'), out = document.getElementById('ground-readout');
  const B = id => document.getElementById('btn-ground-' + id), IDS = ['phrases', 'tokens', 'person', 'cone', 'tooth', 'all'];
  let sc = null;
  const glow = PE.linkPulses(fig, () => sc && sc.redraw());
  const press = () => { const s = sc.state(); IDS.forEach(id => B(id).setAttribute('aria-pressed', String(id === s))); };
  sc = SceneGlipGround(document.getElementById('ground-canvas'), { glow, onSay: t => { out.textContent = t; }, onPick: press });
  IDS.forEach(id => B(id).addEventListener('click', () => { sc.set(id); press(); }));
  B('reset').addEventListener('click', () => { sc.reset(); press(); });
  press();
})();
</script>
