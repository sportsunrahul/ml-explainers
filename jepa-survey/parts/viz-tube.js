<script>
(() => {
  // V-JEPA mask sampler on the 8 x 14 x 14 token grid: blocks of a given fraction of the frame with aspect
  // ratio in (0.75, 1.5), placed uniformly, extended through every time slot (short: 8 x 15%, long: 2 x 70%).
  const fig = document.getElementById('viz-tube'), el = document.getElementById('tube-canvas'), out = document.getElementById('tube-readout');
  const T = 8, G = 14; const S = { mask: 'short', seed: 1 };
  function sample() {
    const r = PE.rng(S.seed * 131 + 7), m = []; for (let i = 0; i < G; i++) m.push(new Array(G).fill(0));
    const cfg = S.mask === 'long' ? [2, .7] : [8, .15];
    for (let k = 0; k < cfg[0]; k++) { const n = G * G * cfg[1], a = .75 + r() * .75;
      const h = Math.min(G, Math.max(1, Math.round(Math.sqrt(n * a)))), w = Math.min(G, Math.max(1, Math.round(Math.sqrt(n / a))));
      const top = Math.floor(r() * (G - h + 1)), left = Math.floor(r() * (G - w + 1));
      for (let i = top; i < top + h; i++) for (let j = left; j < left + w; j++) m[i][j] = 1; }
    S.m = m; S.frameOnly = S.mask === 'frame';
    if (S.frameOnly) { const r2 = PE.rng(S.seed * 131 + 9); const n = G * G * .7, a = 1, h = Math.round(Math.sqrt(n * a)), w = Math.round(Math.sqrt(n / a)); for (let i = 0; i < G; i++) for (let j = 0; j < G; j++) S.m[i][j] = 0;
      const top = Math.floor(r2() * (G - h + 1)), left = Math.floor(r2() * (G - w + 1)); for (let i = top; i < top + h; i++) for (let j = left; j < left + w; j++) S.m[i][j] = 1; }
  }
  const s3 = PE3.scene(el, { aspect: w => w < 560 ? .95 : .56, fov: 34, dist: 7.2, yaw: .75, pitch: .42 });
  function build() {
    const items = [], sp = .17, cell = .15; let masked = 0;
    for (let t = 0; t < T; t++) for (let i = 0; i < G; i++) for (let j = 0; j < G; j++) {
      const hidden = S.m[i][j] && (!S.frameOnly || t === 3); if (hidden) masked++;
      const x = (j - (G - 1) / 2) * sp, y = ((G - 1) / 2 - i) * sp, z = (t - (T - 1) / 2) * sp * 1.6;
      items.push({ type: 'quad', pts: [[x - cell / 2, y - cell / 2, z], [x + cell / 2, y - cell / 2, z], [x + cell / 2, y + cell / 2, z], [x - cell / 2, y + cell / 2, z]], fill: hidden ? '--yellow' : '--blue', alpha: hidden ? .9 : .22, link: hidden ? 'mask' : 'visible' }); }
    items.push({ type: 'label', at: [0, -(G / 2) * sp - .25, -(T / 2) * sp * 1.6], text: 'time ->', color: '--muted', dy: 14 });
    items.push({ type: 'label', at: [0, (G / 2) * sp + .2, (T / 2) * sp * 1.6], text: '14 x 14 tokens per time slot, 8 slots', color: '--muted', dy: -6 });
    s3.items = items; s3.redraw();
    const tot = T * G * G, f = masked / tot;
    out.textContent = S.frameOnly ? `Ablation (not the paper's setting): one frame hidden, ${masked} of ${tot} tokens (${(100 * f).toFixed(0)}%). Every hidden token has a visible copy one slot away; the predictor can copy instead of modelling motion.` :
      `${S.mask === 'short' ? 'Short-range' : 'Long-range'} mask: ${masked} of ${tot} tokens hidden (${(100 * f).toFixed(0)}% of the clip), the same footprint in all 8 time slots. The paper's average over both mask types is about 90%.`;
  }
  fig.querySelectorAll('[data-mask]').forEach(b => b.addEventListener('click', () => { S.mask = b.dataset.mask; fig.querySelectorAll('[data-mask]').forEach(x => x.setAttribute('aria-pressed', String(x === b))); sample(); build(); }));
  document.getElementById('btn-tube-sample').addEventListener('click', () => { S.seed++; sample(); build(); });
  document.getElementById('btn-tube-reset').addEventListener('click', () => { S.seed = 1; S.mask = 'short'; fig.querySelectorAll('[data-mask]').forEach(x => x.setAttribute('aria-pressed', String(x.dataset.mask === 'short'))); sample(); build(); });
  sample(); build();
})();
</script>
