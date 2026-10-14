<script>
(() => {
  const fig = document.getElementById('viz-repvl'), out = document.getElementById('repvl-readout');
  const B = id => document.getElementById('btn-repvl-' + id);
  let sc = null;
  const glow = PE.linkPulses(fig, () => sc && sc.redraw());
  sc = SceneRepvl(document.getElementById('repvl-canvas'), { glow, onSay: t => { out.textContent = t; } });
  const pressAll = () => { const s = sc.state();
    ['dense', 'gate', 'reparam'].forEach(m => B(m).setAttribute('aria-pressed', String(s.mode === m)));
    B('vocab4').setAttribute('aria-pressed', String(s.vocab === 4)); B('vocab3').setAttribute('aria-pressed', String(s.vocab === 3)); };
  ['dense', 'gate', 'reparam'].forEach(m => B(m).addEventListener('click', () => { sc.set(m); pressAll(); }));
  B('vocab4').addEventListener('click', () => { sc.setVocab(4); pressAll(); });
  B('vocab3').addEventListener('click', () => { sc.setVocab(3); pressAll(); });
  B('reset').addEventListener('click', () => { sc.reset(); pressAll(); });
  pressAll();
})();
</script>
