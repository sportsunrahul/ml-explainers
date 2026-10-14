<script>
(() => {
  const fig = document.getElementById('viz-recall'), el = document.getElementById('recall-canvas');
  const sN = document.getElementById('recall-N'), sK = document.getElementById('recall-K'), out = document.getElementById('recall-readout');
  const nOut = document.getElementById('recall-N-val'), kOut = document.getElementById('recall-K-val');
  const bRand = document.getElementById('btn-recall-random'), bOrtho = document.getElementById('btn-recall-ortho');
  const D = { N: 16, K: 32, ortho: false, fade: 1 }, S = { ...D };
  let glow = () => 0, cancel = () => {};
  const cv = PE.canvas(el, w => w < 520 ? 1.36 : .52, (ctx, W, H) => CoreRecall.draw(ctx, W, H, S, glow));
  glow = PE.linkPulses(fig, cv.redraw);
  const write = () => { out.textContent = CoreRecall.readout(S); };
  function sync() {
    sN.value = Math.log2(S.N); sK.value = CoreRecall.toSlider(Math.round(S.K));
    nOut.textContent = S.N; kOut.textContent = Math.round(S.K);
    bRand.setAttribute('aria-pressed', String(!S.ortho)); bOrtho.setAttribute('aria-pressed', String(S.ortho));
  }
  // Tween K in slider space so the marker glides along the log axis.
  function setK(K) {
    cancel();
    const a = CoreRecall.toSlider(S.K), b = CoreRecall.toSlider(K);
    cancel = PE.animate(700, u => { S.K = u >= 1 ? K : CoreRecall.fromSlider(a + (b - a) * u); sync(); cv.redraw(); }, write);
  }
  function setKeys(ortho) {
    if (S.ortho === ortho) return;
    cancel(); S.ortho = ortho; sync();
    cancel = PE.animate(700, u => { S.fade = u; cv.redraw(); }, write);
  }
  sN.addEventListener('input', () => { cancel(); S.fade = 1; S.N = Math.pow(2, +sN.value); sync(); cv.redraw(); write(); });
  sK.addEventListener('input', () => { cancel(); S.fade = 1; S.K = CoreRecall.fromSlider(+sK.value); sync(); cv.redraw(); write(); });
  document.getElementById('btn-recall-half').addEventListener('click', () => setK(S.N / 2));
  document.getElementById('btn-recall-n').addEventListener('click', () => setK(S.N));
  document.getElementById('btn-recall-4n').addEventListener('click', () => setK(4 * S.N));
  bRand.addEventListener('click', () => setKeys(false));
  bOrtho.addEventListener('click', () => setKeys(true));
  document.getElementById('btn-recall-reset').addEventListener('click', () => { cancel(); Object.assign(S, D); sync(); cv.redraw(); write(); });
  sync(); write();
})();
</script>
