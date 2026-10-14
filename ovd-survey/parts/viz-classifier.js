<script>
(() => {
  const fig = document.getElementById('viz-classifier'), out = document.getElementById('classifier-readout');
  const B = id => document.getElementById('btn-classifier-' + id);
  let sc = null;
  const glow = PE.linkPulses(fig, () => sc && sc.redraw());
  sc = SceneClassifier(document.getElementById('classifier-canvas'), { glow, onSay: t => { out.textContent = t; } });
  const pressAll = () => { const m = sc.mode();
    B('closed').setAttribute('aria-pressed', String(m === 'closed')); B('text').setAttribute('aria-pressed', String(m !== 'closed')); B('add').setAttribute('aria-pressed', String(m === 'add')); };
  ['closed', 'text', 'add'].forEach(m => B(m).addEventListener('click', () => { sc.set(m); pressAll(); }));
  B('reset').addEventListener('click', () => { sc.reset(); pressAll(); });
  pressAll();
})();
</script>
