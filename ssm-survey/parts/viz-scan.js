<script>
(() => {
  // Scan tree stepper. Phase buttons set a round; Previous / Next step one round (plain controls, not cues).
  const fig = document.getElementById('viz-scan'), el = document.getElementById('scan-canvas'), out = document.getElementById('scan-readout');
  const S0 = { L: 8, seed: 3, round: 0, u: 1 }, S = { ...S0 };
  const btn = k => document.getElementById('btn-scan-' + k);
  let glow = () => 0, cancel = () => {};
  const cv = PE.canvas(el, w => (w < 520 ? 400 : 430) / w, (ctx, W, H) => CoreScan.draw(ctx, W, H, S, glow));
  glow = PE.linkPulses(fig, cv.redraw);
  const m = () => Math.round(Math.log2(S.L)), N = () => CoreScan.lastRound(S.L);
  const press = () => {
    btn('8').setAttribute('aria-pressed', String(S.L === 8)); btn('16').setAttribute('aria-pressed', String(S.L === 16));
    [['inputs', 0], ['up', m()], ['down', 2 * m()], ['states', N()]].forEach(([k, r]) => btn(k).setAttribute('aria-pressed', String(S.round === r)));
  };
  const write = () => { out.textContent = CoreScan.readout(S); };
  function go(r) { cancel(); S.round = PE.clamp(r, 0, N()); S.u = 0; press(); write(); cancel = PE.animate(700, u => { S.u = u; cv.redraw(); }); }
  btn('8').addEventListener('click', () => { S.L = 8; go(0); });
  btn('16').addEventListener('click', () => { S.L = 16; go(0); });
  btn('inputs').addEventListener('click', () => go(0));
  btn('up').addEventListener('click', () => go(m()));
  btn('down').addEventListener('click', () => go(2 * m()));
  btn('states').addEventListener('click', () => go(N()));
  btn('next').addEventListener('click', () => go(S.round + 1));
  btn('prev').addEventListener('click', () => go(S.round - 1));
  btn('reset').addEventListener('click', () => { cancel(); Object.assign(S, S0); press(); write(); cv.redraw(); });
  press(); write();
})();
(() => {
  // Where the state lives: HBM traffic per stage, fused against standard.
  const fig = document.getElementById('viz-scanmem'), el = document.getElementById('scanmem-canvas'), out = document.getElementById('scanmem-readout');
  const M0 = { mode: 'fused', stage: 0, u: 1 }, M = { ...M0 };
  const btn = k => document.getElementById('btn-scanmem-' + k), ST = CoreScan.STAGES;
  let glow = () => 0, cancel = () => {};
  const cv = PE.canvas(el, w => (w < 520 ? 420 : 340) / w, (ctx, W, H) => CoreScan.drawMem(ctx, W, H, M, glow));
  glow = PE.linkPulses(fig, cv.redraw);
  const press = () => { btn('fused').setAttribute('aria-pressed', String(M.mode === 'fused')); btn('standard').setAttribute('aria-pressed', String(M.mode === 'standard'));
    ST.slice(1).forEach((k, i) => btn(k).setAttribute('aria-pressed', String(M.stage === i + 1))); };
  const write = () => { out.textContent = CoreScan.readoutMem(M); };
  function set(mode, stage) { cancel(); M.mode = mode; M.stage = stage; M.u = 0; press(); write(); cancel = PE.animate(800, u => { M.u = u; cv.redraw(); }); }
  btn('fused').addEventListener('click', () => set('fused', M.stage));
  btn('standard').addEventListener('click', () => set('standard', M.stage));
  ST.slice(1).forEach((k, i) => btn(k).addEventListener('click', () => set(M.mode, i + 1)));
  btn('reset').addEventListener('click', () => { cancel(); Object.assign(M, M0); press(); write(); cv.redraw(); });
  press(); write();
})();
</script>
