<script>
(() => {
  const fig = document.getElementById('viz-vprompt'), out = document.getElementById('vprompt-readout');
  const B = id => document.getElementById('btn-vprompt-' + id), MODES = ['text', 'box', 'both'];
  let sc = null;
  const glow = PE.linkPulses(fig, () => sc && sc.redraw());
  sc = SceneVisualPrompts(document.getElementById('vprompt-canvas'), { glow, onSay: t => { out.textContent = t; } });
  const pressAll = () => MODES.forEach(m => B(m).setAttribute('aria-pressed', String(sc.mode() === m)));
  MODES.forEach(m => B(m).addEventListener('click', () => { sc.set(m); pressAll(); }));
  B('reset').addEventListener('click', () => { sc.reset(); pressAll(); });
  pressAll();

  const out2 = document.getElementById('vpbench-readout'), C = id => document.getElementById('btn-vpbench-' + id);
  const bench = SceneVisualPromptsBench(document.getElementById('vpbench-canvas'), { onSay: t => { out2.textContent = t; } });
  const press2 = () => ['pair', 'mixed'].forEach(m => C(m).setAttribute('aria-pressed', String(bench.mode() === m)));
  ['pair', 'mixed'].forEach(m => C(m).addEventListener('click', () => { bench.set(m); press2(); }));
  C('reset').addEventListener('click', () => { bench.reset(); press2(); });
  press2();
})();
</script>
