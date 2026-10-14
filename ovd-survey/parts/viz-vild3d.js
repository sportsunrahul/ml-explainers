<script>
(() => {
  const out = document.getElementById('vild3d-readout'), B = id => document.getElementById('btn-vild3d-' + id), STATES = ['text', 'image', 'both'];
  const sc = SceneCloud(document.getElementById('vild3d-canvas'), { onSay: t => { out.textContent = t; } });
  const press = () => STATES.forEach(s => B(s).setAttribute('aria-pressed', String(sc.state() === s)));
  STATES.forEach(s => B(s).addEventListener('click', () => { sc.set(s); press(); }));
  B('reset').addEventListener('click', () => { sc.reset(); press(); });
  press();
})();
</script>
