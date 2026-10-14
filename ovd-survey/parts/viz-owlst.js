<script>
(() => {
  const out = document.getElementById('owlst-readout'), texts = [...document.querySelectorAll('#viz-owlst .owlst-text > div')];
  const KEYS = ['web', 'ngrams', 'teacher', 'filter', 'student'], B = id => document.getElementById('btn-owlst-' + id);
  const sc = SceneOwlST(document.getElementById('owlst-canvas'), { onSay: t => { out.textContent = t; } });
  const show = () => { const s = sc.stage(); KEYS.forEach(k => B(k).setAttribute('aria-pressed', String(k === s))); texts.forEach(d => { d.hidden = d.dataset.stage !== s; }); };
  KEYS.forEach(k => B(k).addEventListener('click', () => { sc.set(k); show(); }));
  B('reset').addEventListener('click', () => { sc.reset(); show(); });
  show();
})();
(() => {
  const out = document.getElementById('owltrade-readout'), KEYS = ['st', 'ft', 'ens'], B = id => document.getElementById('btn-owltrade-' + id);
  const sc = SceneOwlTrade(document.getElementById('owltrade-canvas'), { onSay: t => { out.textContent = t; } });
  const press = () => KEYS.forEach(k => B(k).setAttribute('aria-pressed', String(k === sc.state())));
  KEYS.forEach(k => B(k).addEventListener('click', () => { sc.set(k); press(); }));
  B('reset').addEventListener('click', () => { sc.reset(); press(); });
  press();
})();
</script>
