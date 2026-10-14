<script>
(() => {
  const fig = document.getElementById('viz-fusion'), out = document.getElementById('fusion-readout');
  const B = id => document.getElementById('btn-fusion-' + id);
  let sc = null;
  const glow = PE.linkPulses(fig, () => sc && sc.redraw());
  sc = SceneGlipFusion(document.getElementById('fusion-canvas'), { glow, onSay: t => { out.textContent = t; } });
  const press = () => {
    ['late', 'deep'].forEach(m => B(m).setAttribute('aria-pressed', String(sc.mode() === m)));
    ['ray', 'peng'].forEach(w => B(w).setAttribute('aria-pressed', String(sc.word() === w)));
  };
  ['late', 'deep'].forEach(m => B(m).addEventListener('click', () => { sc.setMode(m); press(); }));
  ['ray', 'peng'].forEach(w => B(w).addEventListener('click', () => { sc.setWord(w); press(); }));
  B('reset').addEventListener('click', () => { sc.reset(); press(); });
  press();
})();
</script>
