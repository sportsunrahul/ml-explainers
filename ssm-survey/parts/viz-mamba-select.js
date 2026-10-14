<script>
(() => {
  const C = CoreState3d, el = document.getElementById('mamba-select-canvas'), out = document.getElementById('mamba-select-readout');
  const row = document.querySelector('#viz-mamba-select .tok-row');
  const presets = { keys: C.keyMarks(), allfill: C.VAL.map(() => false), allimp: C.VAL.map(() => true) };
  const btns = Object.keys(presets).map(k => [k, document.getElementById('btn-mamba-' + k)]);
  const boxes = C.VAL.map((v, t) => {
    const lab = document.createElement('label'), cb = document.createElement('input');
    cb.type = 'checkbox'; cb.id = 'mamba-tok-' + t; lab.append(cb, ` ${t + 1}`); lab.title = `token ${t + 1}, x = ${v}`; row.append(lab); return cb;
  });
  const S = { marks: presets.keys.slice() };
  let frame = C.frameSel(S.marks), cur = { to: frame }, cancel = () => {}, narrow = false;
  const s3 = PE3.scene(el, { aspect: w => w < 520 ? 1.08 : .56, fov: 34, dist: 8, yaw: .4, pitch: .3 });
  const draw = st => { cur = st; s3.items = C.items({ ...st, narrow }); s3.redraw(); };
  new ResizeObserver(() => { const v = C.view(el.clientWidth, el.clientHeight, 'select'); narrow = v.narrow; s3.setView(v, 0); draw(cur); }).observe(el);
  PE.onTheme(() => draw(cur));
  const same = (a, b) => a.every((v, i) => v === b[i]);
  function ui() {
    boxes.forEach((b, t) => { b.checked = S.marks[t]; });
    btns.forEach(([k, b]) => b.setAttribute('aria-pressed', String(same(presets[k], S.marks))));
    out.textContent = C.readSel(S.marks);
  }
  function set(marks, ms = 800) {
    cancel(); const from = frame; S.marks = marks.slice(); frame = C.frameSel(S.marks); ui();
    cancel = s3.tween(ms, u => draw({ from, to: frame, u }), () => draw({ to: frame }));
  }
  btns.forEach(([k, b]) => b.addEventListener('click', () => set(presets[k])));
  boxes.forEach((b, t) => b.addEventListener('change', () => { const m = S.marks.slice(); m[t] = b.checked; set(m, 600); }));
  document.getElementById('btn-mamba-reset').addEventListener('click', () => { cancel(); S.marks = presets.keys.slice(); frame = C.frameSel(S.marks); ui(); draw({ to: frame }); });
  ui(); draw(cur);
})();
</script>
