<script>
(() => {
  const fig = document.getElementById('viz-qsel'), out = document.getElementById('qsel-readout');
  const B = id => document.getElementById('btn-qsel-' + id), kIn = document.getElementById('qsel-k'), kVal = document.getElementById('qsel-k-val');
  const STAGES = { fusion: 'fusion', select: 'qsel', decoder: 'decoder' }, PROMPT = { forklift: 'forklift', pallet: 'pallet-person', cone: 'cone', ladder: 'ladder' };
  let sc = null;
  const glow = PE.linkPulses(fig, () => sc && sc.redraw());
  sc = SceneGdino(document.getElementById('qsel-canvas'), { glow, k: 8, onSay: t => { out.textContent = t; } });
  const press = () => { const st = sc.state();
    Object.entries(STAGES).forEach(([b, s]) => B(b).setAttribute('aria-pressed', String(st.stage === s)));
    Object.entries(PROMPT).forEach(([b, p]) => B(b).setAttribute('aria-pressed', String(st.prompt === p)));
    kIn.value = st.k; kVal.textContent = st.k; };
  Object.entries(STAGES).forEach(([b, s]) => B(b).addEventListener('click', () => { sc.set({ stage: s }); press(); }));
  Object.entries(PROMPT).forEach(([b, p]) => B(b).addEventListener('click', () => { sc.set({ prompt: p }); press(); }));
  kIn.addEventListener('input', () => { sc.set({ k: +kIn.value }); press(); });
  B('reset').addEventListener('click', () => { sc.reset(); press(); });
  press();
})();
</script>
