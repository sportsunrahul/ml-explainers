<script>
(() => {
  const C = CoreState3d, el = document.getElementById('continuous-canvas');
  const slider = document.getElementById('continuous-dt'), dtv = document.getElementById('continuous-dt-value'), out = document.getElementById('continuous-readout');
  const bSig = document.getElementById('btn-continuous-signal'), bTok = document.getElementById('btn-continuous-tokens');
  const b1 = document.getElementById('btn-continuous-d1'), b01 = document.getElementById('btn-continuous-d01');
  const DT0 = 1 / 4, S = { mode: 'signal', dt: DT0, frame: C.frameLTI('signal', DT0) };
  const s3 = PE3.scene(el, { aspect: w => w < 520 ? 1.08 : .56, fov: 34, dist: 7, yaw: .42, pitch: .3 });
  let narrow = false, cur = { to: S.frame }, cancel = () => {};
  const draw = st => { cur = st; s3.items = C.items({ ...st, narrow }); s3.redraw(); };
  new ResizeObserver(() => { const v = C.view(el.clientWidth, el.clientHeight, 'lti'); narrow = v.narrow; s3.setView(v, 0); draw(cur); }).observe(el);
  PE.onTheme(() => draw(cur));
  function ui() {
    slider.value = C.DTS.indexOf(S.dt); dtv.textContent = C.fmtDt(S.dt);
    bSig.setAttribute('aria-pressed', String(S.mode === 'signal')); bTok.setAttribute('aria-pressed', String(S.mode === 'tokens'));
    b1.setAttribute('aria-pressed', String(S.dt === 1)); b01.setAttribute('aria-pressed', String(S.dt === 1 / 10));
  }
  function set(mode, dt, ms = 800) {
    cancel();
    const from = S.frame; S.mode = mode; S.dt = dt; S.frame = C.frameLTI(mode, dt); ui();
    out.textContent = C.readLTI(mode, dt);
    if (from === S.frame) { draw({ to: S.frame }); return; }
    cancel = s3.tween(ms, u => draw({ from, to: S.frame, u }), () => draw({ to: S.frame }));
  }
  bSig.addEventListener('click', () => set('signal', S.dt));
  bTok.addEventListener('click', () => set('tokens', S.dt));
  b1.addEventListener('click', () => set(S.mode, 1));
  b01.addEventListener('click', () => set(S.mode, 1 / 10));
  slider.addEventListener('input', () => set(S.mode, C.DTS[+slider.value], 350));
  document.getElementById('btn-continuous-reset').addEventListener('click', () => { cancel(); S.mode = 'signal'; S.dt = DT0; S.frame = C.frameLTI('signal', DT0); ui(); out.textContent = C.readLTI('signal', DT0); draw({ to: S.frame }); });
  ui(); out.textContent = C.readLTI(S.mode, S.dt); draw(cur);
})();
</script>
