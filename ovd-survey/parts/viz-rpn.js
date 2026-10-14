<script>
(() => {
  const fig = document.getElementById('viz-rpn'), out = document.getElementById('rpn-readout'), slider = document.getElementById('rpn-k'), kv = document.getElementById('rpn-k-value');
  const STAGES = ['features', 'anchors', 'objectness', 'deltas', 'topk', 'roialign', 'head'];
  const B = id => document.getElementById('btn-rpn-' + id);
  let sc = null;
  const glow = PE.linkPulses(fig, () => sc && sc.redraw());
  sc = SceneRpn(document.getElementById('rpn-canvas'), { glow, onSay: t => { out.textContent = t; } });
  const press = () => STAGES.forEach(s => B(s).setAttribute('aria-pressed', String(sc.stage() === s)));
  STAGES.forEach(s => B(s).addEventListener('click', () => { sc.set(s); press(); }));
  slider.addEventListener('input', () => { kv.textContent = slider.value; sc.setK(+slider.value); });
  B('reset').addEventListener('click', () => { slider.value = 5; kv.textContent = '5'; sc.reset(); press(); });
  press();
})();
</script>
