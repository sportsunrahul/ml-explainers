<script>
(() => {
  // The SSD matrix surface: three matrices on the same toy inputs, a chunk overlay, and a work count. Logic lives in CoreSsd.
  const fig = document.getElementById('viz-ssd'), el = document.getElementById('ssd-canvas'), out = document.getElementById('ssd-readout');
  const D0 = { T: 16, mode: 'softmax', Q: 0 }, S = { ...D0, narrow: false };
  const s3 = PE3.scene(el, { aspect: w => w < 520 ? 1.05 : .62, fov: 34, dist: 7, yaw: .42, pitch: .74 });
  let v = CoreSsd.target(S.T, S.mode, S.Q), cancel = () => {}, glow = () => 0;
  const draw = () => { s3.items = CoreSsd.items(v, S, glow); s3.redraw(); };
  glow = PE.linkPulses(fig, draw);
  new ResizeObserver(() => { const c = CoreSsd.view(el.clientWidth, el.clientHeight); S.narrow = c.narrow; s3.setView(c, 0); draw(); }).observe(el);
  PE.onTheme(draw);
  const groups = { mode: ['softmax', 'linattn', 'ssd'], Q: [0, 1, 4, 8], T: [16, 32] };
  const id = (k, x) => 'btn-ssd-' + (k === 'mode' ? x : k === 'Q' ? (x ? 'q' + x : 'whole') : 't' + x);
  const press = () => Object.entries(groups).forEach(([k, xs]) => xs.forEach(x => document.getElementById(id(k, x)).setAttribute('aria-pressed', String(S[k] === x))));
  const write = () => { out.textContent = CoreSsd.readout(S); };
  function go(patch) {
    cancel();
    const from = v; Object.assign(S, patch);
    const to = CoreSsd.target(S.T, S.mode, S.Q);
    cancel = s3.tween(850, u => { v = CoreSsd.blend(from, to, u); draw(); });
    press(); write();
  }
  Object.entries(groups).forEach(([k, xs]) => xs.forEach(x => document.getElementById(id(k, x)).addEventListener('click', () => go({ [k]: x }))));
  document.getElementById('btn-ssd-reset').addEventListener('click', () => {
    cancel(); Object.assign(S, D0); v = CoreSsd.target(S.T, S.mode, S.Q); s3.setView(CoreSsd.view(el.clientWidth, el.clientHeight), 0); draw(); press(); write();
  });
  press(); write(); draw();
})();
</script>
