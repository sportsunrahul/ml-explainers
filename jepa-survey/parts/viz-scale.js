<script>
(() => {
  // V-JEPA 2 Fig. 3: average frozen-probe accuracy as each scaling ingredient is added.
  const fig = document.getElementById('viz-scale'), svg = document.getElementById('scale-chart'), out = document.getElementById('scale-readout');
  const steps = [['V-JEPA baseline', 84.2], ['+ data (22M videos)', 85.2], ['+ model (ViT-g)', 86.7], ['+ longer training', 87.5], ['+ resolution', 88.2]];
  const S = { n: 0, u: 1 };
  const NS = 'http://www.w3.org/2000/svg';
  function draw() {
    while (svg.firstChild) svg.removeChild(svg.firstChild);
    const narrow = svg.clientWidth < 520; svg.setAttribute('viewBox', narrow ? '0 0 360 230' : '0 0 640 230');
    const W = narrow ? 360 : 640, x0 = 44, y0 = 190, bw = (W - x0 - 20) / 5, y = v => y0 - (v - 83) * (150 / 6);
    const el = (t, a, txt) => { const e = document.createElementNS(NS, t); for (const k in a) e.setAttribute(k, a[k]); if (txt != null) e.textContent = txt; svg.appendChild(e); return e; };
    [83, 84, 85, 86, 87, 88, 89].forEach(v => { el('line', { x1: x0, x2: W - 10, y1: y(v), y2: y(v), stroke: 'var(--rule)' }); el('text', { x: x0 - 6, y: y(v) + 4, 'text-anchor': 'end', fill: 'var(--muted)', 'font-size': narrow ? 9 : 11, 'font-family': 'system-ui' }, v); });
    steps.forEach(([name, v], i) => { const on = i <= S.n, h = on ? (y0 - y(v)) * (i === S.n ? S.u : 1) + (i === S.n && i > 0 ? (y0 - y(steps[i - 1][1])) * (1 - S.u) : 0) : 0;
      el('rect', { x: x0 + i * bw + 8, y: y0 - h, width: bw - 16, height: h, fill: i === 0 ? 'var(--brown)' : 'var(--blue)', opacity: on ? .9 : .15, rx: 3 });
      if (on) el('text', { x: x0 + i * bw + bw / 2, y: y0 - h - 6, 'text-anchor': 'middle', fill: 'var(--fg)', 'font-size': narrow ? 10 : 12, 'font-family': 'system-ui' }, (83 + h / (150 / 6)).toFixed(1));
      el('text', { x: x0 + i * bw + bw / 2, y: y0 + 16, 'text-anchor': 'middle', fill: on ? 'var(--fg)' : 'var(--muted)', 'font-size': narrow ? 8 : 11, 'font-family': 'system-ui' }, narrow ? name.replace(/ \(.*\)/, '').slice(0, 14) : name); });
    el('text', { x: x0, y: 14, fill: 'var(--muted)', 'font-size': 11, 'font-family': 'system-ui' }, 'average over 6 frozen-probe tasks (%)');
  }
  function set(n) { S.n = n; fig.querySelectorAll('[data-step]').forEach(b => b.setAttribute('aria-pressed', String(+b.dataset.step === n))); S.u = 0; PE.animate(600, u => { S.u = u; draw(); });
    out.textContent = n === 0 ? 'V-JEPA (2024) ViT-L on 2 million videos: 84.2.' : `${steps[n][0]}: ${steps[n][1]} (+${(steps[n][1] - steps[n - 1][1]).toFixed(1)} over the previous step).`; }
  fig.querySelectorAll('[data-step]').forEach(b => b.addEventListener('click', () => set(+b.dataset.step)));
  document.getElementById('btn-scale-reset').addEventListener('click', () => set(0));
  new ResizeObserver(draw).observe(svg); PE.onTheme(draw); set(0);
})();
</script>
