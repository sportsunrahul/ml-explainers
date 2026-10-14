<script>
(() => {
  const fig = document.getElementById('viz-owl'), out = document.getElementById('owl-readout'), thr = document.getElementById('owl-thr');
  const thrV = document.getElementById('owl-thr-value'), B = id => document.getElementById('btn-owl-' + id);
  let sc = null;
  const glow = PE.linkPulses(fig, () => sc && sc.redraw());
  sc = SceneOwl(document.getElementById('owl-canvas'), { glow, onSay: t => { out.textContent = t; } });
  const press = () => { const s = sc.state();
    B('boxes').setAttribute('aria-pressed', String(s.mode === 'boxes'));
    ['forklift', 'pallet', 'rare'].forEach(q => B(q).setAttribute('aria-pressed', String(s.mode === 'text' && s.q === q)));
    B('image').setAttribute('aria-pressed', String(s.mode === 'image')); };
  B('boxes').addEventListener('click', () => { sc.set('boxes'); press(); });
  ['forklift', 'pallet', 'rare'].forEach(q => B(q).addEventListener('click', () => { sc.set('text', q); press(); }));
  B('image').addEventListener('click', () => { sc.set('image'); press(); });
  thr.addEventListener('input', () => { thrV.textContent = (+thr.value).toFixed(2); sc.setThr(thr.value); });
  B('reset').addEventListener('click', () => { sc.reset(); thr.value = .5; thrV.textContent = '0.50'; press(); });
  press();
})();
</script>
