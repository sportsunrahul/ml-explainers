<script>
(() => {
  const $ = id => document.getElementById(id);
  const press = (btns, on) => btns.forEach(b => b.setAttribute('aria-pressed', String(b === on)));
  const wire = (fig, canvasId, outId, make, steps, prefix, toArg = s => s) => {
    const out = $(outId); let sc = null;
    const glow = PE.linkPulses($(fig), () => sc && sc.redraw());
    sc = make($(canvasId), { glow, onSay: t => { out.textContent = t; } });
    const btns = steps.map(s => $(`btn-${prefix}-${s}`));
    steps.forEach((s, i) => btns[i].addEventListener('click', () => { sc.set(toArg(s)); press(btns, btns[i]); }));
    $(`btn-${prefix}-reset`).addEventListener('click', () => { sc.reset(); press(btns, btns[0]); });
    return sc;
  };
  wire('viz-gap3d', 'gap3d-canvas', 'gap3d-readout', (el, o) => SceneCloud(el, { ...o, start: 'gap' }), ['gap', 'jargon', 'describe'], 'gap3d');
  wire('viz-rename', 'rename-canvas', 'rename-readout', SceneRename, ['names', 'described'], 'rename');
  wire('viz-fppc', 'fppc-canvas', 'fppc-readout', SceneFppc, ['0', '2', '4', '6'], 'fppc', s => +s);
  const tiny = wire('viz-tiny', 'tiny-canvas', 'tiny-readout', SceneTiny, ['owlvit', 'owlv2', 'glip', 'yolow'], 'tiny');
  const W = $('tiny-width'), Dz = $('tiny-defect');
  const upd = () => { $('tiny-width-v').textContent = W.value; $('tiny-defect-v').textContent = Dz.value; tiny.sizes(+W.value, +Dz.value); };
  W.addEventListener('input', upd); Dz.addEventListener('input', upd);
  $('btn-tiny-reset').addEventListener('click', () => { W.value = 4096; Dz.value = 20; $('tiny-width-v').textContent = '4096'; $('tiny-defect-v').textContent = '20'; });
})();
</script>
