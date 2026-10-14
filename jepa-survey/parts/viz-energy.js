<script>
(() => {
  // Toy energy landscape: F(x, y) over the unit square, drawn as a height field (up = energy).
  // Two valleys stand in for the data; the buttons rewrite the formula, not a trained network.
  const fig = document.getElementById('viz-energy'), el = document.getElementById('energy-canvas'), out = document.getElementById('energy-readout');
  const sx = document.getElementById('energy-x'), sy = document.getElementById('energy-y');
  const V = [[-.45, .5], [.5, -.4]];                 // valley centres in (x, y)
  const bowl = (x, y) => .35 * (x * x + y * y);
  const well = (x, y, c, w) => Math.exp(-((x - c[0]) ** 2 + (y - c[1]) ** 2) / w);
  const F = {
    trained: (x, y) => 1 - .95 * Math.max(well(x, y, V[0], .09), well(x, y, V[1], .09)) + bowl(x, y),
    flat: (x, y) => .25 + 0 * x,
    contrastive: (x, y) => 1 - .95 * Math.max(well(x, y, V[0], .09), well(x, y, V[1], .09)) + bowl(x, y)
      + .5 * (well(x, y, [.1, .1], .02) + well(x, y, [-.6, -.5], .02) + well(x, y, [.7, .6], .02)),
    reg: (x, y) => 1.25 - 1.15 * Math.max(well(x, y, V[0], .03), well(x, y, V[1], .03)) + bowl(x, y),
  };
  const S = { land: 'trained', cur: null };
  const s = PE3.scene(el, { aspect: w => w < 560 ? .9 : .58, fov: 38, dist: 5.2, yaw: -.7, pitch: .55, autoRotate: 0 });
  const N = 28;
  let hf = []; // current height field, eased between landscapes
  const sample = f => { const h = []; for (let i = 0; i <= N; i++) { h.push([]); for (let j = 0; j <= N; j++) h[i].push(f(-1 + 2 * i / N, -1 + 2 * j / N)); } return h; };
  hf = sample(F.trained);
  function build() {
    const items = [];
    const ramp = v => { const t = PE3.clamp ? PE3.clamp(v / 1.4, 0, 1) : Math.max(0, Math.min(1, v / 1.4)); return `color-mix(in srgb, var(--blue) ${Math.round((1 - t) * 100)}%, var(--yellow))`; };
    for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) {
      const p = (a, b) => [-1.6 + 3.2 * a / N, hf[a][b] * 1.1 - .7, -1.6 + 3.2 * b / N];
      const q = [p(i, j), p(i + 1, j), p(i + 1, j + 1), p(i, j + 1)];
      items.push({ type: 'quad', pts: q, fill: ramp((hf[i][j] + hf[i + 1][j + 1]) / 2), stroke: '--rule', alpha: .92, link: 'surface' });
    }
    V.forEach(c => items.push({ type: 'label', at: [c[0] * 1.6, -.6 + .25, c[1] * 1.6], text: 'data', color: '--yellow', dy: -8, link: 'valley' }));
    const x = +sx.value, y = +sy.value, fi = Math.round((x + 1) / 2 * N), fj = Math.round((y + 1) / 2 * N);
    const e = hf[fi][fj];
    items.push({ type: 'line', pts: [[x * 1.6, -.75, y * 1.6], [x * 1.6, e * 1.1 - .7, y * 1.6]], color: '--red', width: 1.5, dash: [3, 3] });
    items.push({ type: 'points', pts: [[x * 1.6, e * 1.1 - .7 + .03, y * 1.6]], color: '--red', size: 9, link: 'marker' });
    items.push({ type: 'label', at: [x * 1.6, e * 1.1 - .7, y * 1.6], text: `F = ${e.toFixed(2)}`, color: '--red', dx: 10, dy: -8, link: 'marker' });
    items.push({ type: 'label', at: [1.75, -.7, 0], text: 'x', color: '--blue', dx: 6 });
    items.push({ type: 'label', at: [0, -.7, 1.75], text: 'y', color: '--yellow', dy: 12 });
    items.push({ type: 'label', at: [-1.7, .5, -1.7], text: 'energy', color: '--muted', dx: -6, align: 'right' });
    s.items = items; s.redraw();
    out.textContent = { trained: `Trained landscape: two valleys where data pairs live. At (x, y) = (${x.toFixed(2)}, ${y.toFixed(2)}) the energy is ${e.toFixed(2)}; slide y to find the valley for this x.`,
      flat: 'Collapsed: every (x, y) has the same energy. Inference has nowhere to go; the model has learned nothing. This is what two encoders can do by outputting a constant.',
      contrastive: 'Contrastive: chosen wrong pairs have been pushed up into spikes. It works where you push; in high dimensions there are too many places to push.',
      reg: 'Regularised: the low-energy region has been shrunk by limiting its volume, so the valleys are narrow without naming any wrong answer.' }[S.land];
  }
  let stop = () => {};
  function setLand(k) {
    S.land = k; fig.querySelectorAll('[data-land]').forEach(b => b.setAttribute('aria-pressed', String(b.dataset.land === k)));
    const from = hf.map(r => r.slice()), to = sample(F[k]); stop();
    stop = s.tween(700, u => { hf = from.map((r, i) => r.map((v, j) => v + (to[i][j] - v) * u)); build(); });
  }
  fig.querySelectorAll('[data-land]').forEach(b => b.addEventListener('click', () => setLand(b.dataset.land)));
  sx.addEventListener('input', build); sy.addEventListener('input', build);
  document.getElementById('btn-energy-reset').addEventListener('click', () => { sx.value = -0.4; sy.value = 0.5; setLand('trained'); });
  build();
})();
</script>
