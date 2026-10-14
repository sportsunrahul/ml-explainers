<script>
(() => {
  const fig = document.getElementById('viz-hybrid'), el = document.getElementById('hybrid-canvas');
  const slider = document.getElementById('hybrid-n'), nOut = document.getElementById('hybrid-n-val'), out = document.getElementById('hybrid-readout');
  const ctxBtns = [['8k', 8192], ['128k', 131072], ['256k', 262144]].map(([k, L]) => [document.getElementById('btn-hybrid-' + k), L]);
  const N0 = 4, L0 = 262144;
  const S = { attn: CoreHybrid.target(N0), L: L0 };
  let glow = () => 0, cancel = () => {};
  const cv = PE.canvas(el, w => w < 520 ? 1.32 : .6, (ctx, W, H) => CoreHybrid.draw(ctx, W, H, S, glow));
  glow = PE.linkPulses(fig, cv.redraw);
  const write = () => { out.textContent = CoreHybrid.readout(S); };
  const sync = () => { const n = Math.round(CoreHybrid.nAttn(S)); slider.value = n; nOut.textContent = n; ctxBtns.forEach(([b, L]) => b.setAttribute('aria-pressed', String(L === S.L))); };
  // Each layer fades between Mamba and attention; the context tweens on a log scale.
  function go(n, L = S.L) {
    cancel();
    const a0 = S.attn.slice(), a1 = CoreHybrid.target(n), l0 = Math.log(S.L), l1 = Math.log(L);
    cancel = PE.animate(800, u => { S.attn = a0.map((v, i) => PE.lerp(v, a1[i], u)); S.L = u >= 1 ? L : Math.exp(PE.lerp(l0, l1, u)); sync(); cv.redraw(); }, () => { sync(); write(); });
  }
  slider.addEventListener('input', () => { cancel(); S.attn = CoreHybrid.target(+slider.value); sync(); cv.redraw(); write(); });
  [['mamba', 0], ['jamba', 4], ['13', 8], ['attn', 32]].forEach(([k, n]) => document.getElementById('btn-hybrid-' + k).addEventListener('click', () => go(n)));
  ctxBtns.forEach(([b, L]) => b.addEventListener('click', () => go(Math.round(CoreHybrid.nAttn(S)), L)));
  document.getElementById('btn-hybrid-reset').addEventListener('click', () => { cancel(); S.attn = CoreHybrid.target(N0); S.L = L0; sync(); cv.redraw(); write(); });
  sync(); write();
})();
</script>
