<script>
(() => {
  const fig = document.getElementById('viz-bill'), el = document.getElementById('bill-canvas');
  const slider = document.getElementById('bill-L'), out = document.getElementById('bill-readout');
  const L0 = 8192, S = { L: L0 };
  let glow = () => 0, cancel = () => {};
  const cv = PE.canvas(el, w => w < 520 ? .78 : .46, (ctx, W, H) => CoreBill.draw(ctx, W, H, S.L, glow));
  glow = PE.linkPulses(fig, cv.redraw);
  const write = () => { out.textContent = CoreBill.readout(S.L); };
  // Tween in slider space so the marker glides along the log axis.
  function setL(L, ms = 700) {
    cancel();
    const a = CoreBill.toSlider(S.L), b = CoreBill.toSlider(L);
    slider.value = b;
    cancel = PE.animate(ms, u => { S.L = u >= 1 ? L : CoreBill.fromSlider(a + (b - a) * u); cv.redraw(); }, write);
  }
  slider.addEventListener('input', () => { cancel(); S.L = CoreBill.fromSlider(+slider.value); cv.redraw(); write(); });
  [['1k', 1024], ['8k', 8192], ['128k', 131072], ['1m', 1048576]].forEach(([k, L]) =>
    document.getElementById('btn-bill-' + k).addEventListener('click', () => setL(L)));
  document.getElementById('btn-bill-reset').addEventListener('click', () => { cancel(); S.L = L0; slider.value = CoreBill.toSlider(L0); cv.redraw(); write(); });
  slider.value = CoreBill.toSlider(L0); write();
})();
</script>
