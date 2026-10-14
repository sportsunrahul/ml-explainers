<script>
(() => {
  // HiPPO-LegS, computed: the coefficient surface (pe3) and the reconstruction panel (2D). Logic lives in CoreHippo.
  const fig = document.getElementById('viz-hippo'), el3 = document.getElementById('hippo-canvas'), el2 = document.getElementById('hippo-recon-canvas');
  const sN = document.getElementById('hippo-N'), sT = document.getElementById('hippo-t'), out = document.getElementById('hippo-readout');
  const D = { sig: 'paper', method: 'bilinear', N: 16, k: 1000 };
  const S = { ...D, sig0: null, m0: null, u: 1, narrow: false };
  const s3 = PE3.scene(el3, { aspect: w => w < 560 ? .92 : .5, fov: 34, dist: 5.9, yaw: -.42, pitch: .62 });
  let glow = () => 0;
  const cv = PE.canvas(el2, w => w < 520 ? .95 : .4, (ctx, W, H) => CoreHippo.draw(ctx, W, H, S, glow));
  glow = PE.linkPulses(fig, cv.redraw);
  const build = () => { s3.items = CoreHippo.items(S); s3.redraw(); };
  const redraw = () => { build(); cv.redraw(); };
  new ResizeObserver(() => { const n = el3.clientWidth < 560; if (n !== S.narrow) { S.narrow = n; s3.setView({ dist: n ? 7.2 : 5.9, yaw: n ? -.62 : -.42 }, 0); build(); } }).observe(el3);
  PE.onTheme(build);
  const write = () => { out.textContent = CoreHippo.readout(S); };
  const press = () => {
    fig.querySelectorAll('[data-sig]').forEach(b => b.setAttribute('aria-pressed', String(b.dataset.sig === S.sig)));
    fig.querySelectorAll('[data-method]').forEach(b => b.setAttribute('aria-pressed', String(b.dataset.method === S.method)));
  };
  let cancel = () => {};
  // N steps through whole numbers: one more row of the surface, one more cell of the state.
  function setN(N, ms = 700) {
    cancel(); const a = S.N; sN.value = N;
    cancel = PE.animate(ms, u => { S.N = Math.round(a + (N - a) * u); redraw(); }, write);
  }
  function setK(k, ms = 700) {
    cancel(); const a = S.k; sT.value = k;
    cancel = PE.animate(ms, u => { S.k = Math.round(a + (k - a) * u); redraw(); }, write);
  }
  // A new signal or update rule morphs the coefficients from the old source to the new one.
  function setSource(sig, method, ms = 800) {
    cancel(); if (sig === S.sig && method === S.method) { S.u = 1; redraw(); write(); press(); return; }
    S.sig0 = S.sig; S.m0 = S.method; S.sig = sig; S.method = method; S.u = 0; press();
    cancel = PE.animate(ms, u => { S.u = u; redraw(); }, () => { S.sig0 = S.m0 = null; S.u = 1; redraw(); write(); });
  }
  sN.addEventListener('input', () => { cancel(); S.N = +sN.value; redraw(); write(); });
  sT.addEventListener('input', () => { cancel(); S.k = +sT.value; redraw(); write(); });
  [4, 16, 64].forEach(n => document.getElementById('btn-hippo-n' + n).addEventListener('click', () => setN(n)));
  fig.querySelectorAll('[data-sig]').forEach(b => b.addEventListener('click', () => setSource(b.dataset.sig, S.method)));
  fig.querySelectorAll('[data-method]').forEach(b => b.addEventListener('click', () => setSource(S.sig, b.dataset.method)));
  // Play restarts the clock from the start: the window [0, t] grows and the same N numbers must cover more of it.
  document.getElementById('btn-hippo-play').addEventListener('click', () => {
    cancel(); S.k = 20; sT.value = 20; redraw();
    cancel = PE.animate(3200, u => { S.k = Math.round(20 + 980 * u); sT.value = S.k; redraw(); }, write);
  });
  document.getElementById('btn-hippo-reset').addEventListener('click', () => {
    cancel(); Object.assign(S, D, { sig0: null, m0: null, u: 1 }); sN.value = D.N; sT.value = D.k; press(); redraw(); write();
  });
  sN.value = D.N; sT.value = D.k; press(); build(); write();
})();
</script>
