<script>
(() => {
  const $ = id => document.getElementById(id);
  const press = (btns, on) => btns.forEach(b => b.setAttribute('aria-pressed', String(b === on)));
  // Ladder
  { const fig = $('viz-ladder'), out = $('ladder-readout'); let sc = null;
    const glow = PE.linkPulses(fig, () => sc && sc.redraw());
    sc = SceneLadder($('ladder-canvas'), { glow, onSay: t => { out.textContent = t; } });
    const btns = [...fig.querySelectorAll('[data-cat]')];
    btns.forEach(b => b.addEventListener('click', () => { sc.set(b.dataset.cat); press(btns, b); }));
    $('btn-ladder-reset').addEventListener('click', () => { sc.reset(); press(btns, btns[0]); }); }
  // ODinW strip plot
  { const fig = $('viz-odinw'), out = $('odinw-readout'); let sc = null;
    const glow = PE.linkPulses(fig, () => sc && sc.redraw());
    sc = SceneOdinw($('odinw-canvas'), { glow, onSay: t => { out.textContent = t; } });
    const steps = ['13', '35', 'ft', 'gdino'], btns = steps.map(s => $('btn-odinw-' + s));
    steps.forEach((s, i) => btns[i].addEventListener('click', () => { sc.set(s); press(btns, btns[i]); }));
    $('btn-odinw-reset').addEventListener('click', () => { sc.reset(); press(btns, btns[0]); }); }
  // GLIP few-shot lines
  { const fig = $('viz-shots'), out = $('shots-readout'); let sc = null;
    const glow = PE.linkPulses(fig, () => sc && sc.redraw());
    sc = SceneShots($('shots-canvas'), { glow, onSay: t => { out.textContent = t; } });
    const steps = ['dyhead', 'glipt', 'glipl', 'match'], btns = steps.map(s => $('btn-shots-' + s));
    steps.forEach((s, i) => btns[i].addEventListener('click', () => { sc.set(s); press(btns, btns[i]); }));
    $('btn-shots-reset').addEventListener('click', () => { sc.reset(); press(btns, btns[0]); }); }
  // Decision guide (our reading). Each answer is a state; the advice is recomputed from all four.
  { const out = $('guide-readout'), Q = ['common', 'name', 'object', 'policy'], A = {};
    const advise = () => {
      const n = Q.filter(q => A[q] !== undefined).length;
      if (!n) return 'Answer the questions above.';
      const t = [];
      if (A.policy === false) t.push('Your label policy is not the web meaning of the name, and a prompt cannot read your annotation guide: fine-tune on a few boxes per class (chapter 13, item 2).');
      if (A.object === false) t.push('Region-like targets have no box shape for the detector to latch onto; expect weak zero-shot results and plan to fine-tune (our reading; chapter 13, item 3).');
      if (A.name === false) t.push('Describe the look in plain words instead of the jargon name; on the GLIP Mushrooms dataset, describing the look raised zero-shot AP from 13.6 to 73.8 (chapter 8). If it stays weak, fine-tune.');
      if (A.common === false) t.push('Rare concepts are where prompt-only detectors score lowest (RF100-VL); budget for about ten labelled boxes per class and a short fine-tune of an open-vocabulary detector.');
      if (!t.length && n === 4) t.push('A prompt may be enough. Measure it on a labelled test set; if it falls short, ten boxes per class and a short fine-tune is the next step (card 1).');
      if (A.common === false || A.name === false || A.object === false || A.policy === false) t.push('If you also need the model to keep detecting other things, keep the original weights or average them with the fine-tuned ones (card 4).');
      return (n < 4 ? '(Partial answers) ' : '') + t.join(' ');
    };
    Q.forEach(q => ['yes', 'no'].forEach(v => $(`btn-guide-${q}-${v}`).addEventListener('click', () => {
      A[q] = v === 'yes'; $(`btn-guide-${q}-yes`).setAttribute('aria-pressed', String(A[q])); $(`btn-guide-${q}-no`).setAttribute('aria-pressed', String(!A[q]));
      out.textContent = advise(); })));
    $('btn-guide-reset').addEventListener('click', () => { Q.forEach(q => { delete A[q]; $(`btn-guide-${q}-yes`).setAttribute('aria-pressed', 'false'); $(`btn-guide-${q}-no`).setAttribute('aria-pressed', 'false'); }); out.textContent = advise(); });
    out.textContent = advise(); }
})();
</script>
