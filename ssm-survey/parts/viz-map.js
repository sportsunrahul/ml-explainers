<script>
(() => {
  const D = {"scanned": "2026-10-09", "records": 2941, "deep": 2488, "title_ssm_only": 453, "yearly": {"2013": 1, "2014": 2, "2015": 2, "2016": 1, "2019": 2, "2020": 5, "2021": 2, "2022": 13, "2023": 31, "2024": 757, "2025": 923, "2026": 749}, "months": ["2023-12", "2024-01", "2024-02", "2024-03", "2024-04", "2024-05", "2024-06", "2024-07", "2024-08", "2024-09", "2024-10", "2024-11", "2024-12", "2025-01", "2025-02", "2025-03", "2025-04", "2025-05", "2025-06", "2025-07", "2025-08", "2025-09", "2025-10", "2025-11", "2025-12", "2026-01", "2026-02", "2026-03", "2026-04", "2026-05", "2026-06", "2026-07", "2026-08", "2026-09", "2026-10"], "counts": [3, 10, 33, 65, 53, 80, 66, 64, 83, 87, 84, 60, 72, 48, 65, 97, 64, 101, 87, 80, 86, 75, 76, 79, 65, 68, 63, 82, 92, 112, 72, 75, 71, 95, 19], "title_only_counts": [2, 2, 3, 6, 8, 12, 10, 8, 7, 13, 9, 14, 8, 4, 10, 11, 5, 13, 12, 11, 10, 8, 12, 3, 5, 11, 10, 14, 8, 12, 10, 8, 9, 12, 3], "domains": [["images and general vision", 469], ["architecture, theory, efficiency", 417], ["medical imaging and health", 332], ["other", 247], ["language and LLMs", 158], ["video and action", 156], ["time series and forecasting", 145], ["3D and point clouds", 143], ["remote sensing", 136], ["audio and speech", 129], ["graphs, molecules and biology", 80], ["robotics, control and RL", 76]]};
  const fig = document.getElementById('viz-map'), svg = document.getElementById('map-svg'), out = document.getElementById('map-readout');
  const S = { view: 'months', u: 1 }, NS = 'http://www.w3.org/2000/svg';
  const MARKS = [['2023-12', 'Mamba'], ['2024-05', 'Mamba-2'], ['2026-03', 'Mamba-3']];
  let cancel = () => {};
  function draw() {
    while (svg.firstChild) svg.removeChild(svg.firstChild);
    const narrow = svg.clientWidth < 520, W = narrow ? 360 : 700, H = narrow ? 290 : 290; svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
    const el = (tag, a, txt) => { const e = document.createElementNS(NS, tag); for (const k in a) e.setAttribute(k, a[k]); if (txt != null) e.textContent = txt; svg.appendChild(e); return e; };
    const txt = (x, y, s, a = {}) => el('text', Object.assign({ x, y, fill: 'var(--muted)', 'font-size': 10, 'font-family': 'system-ui' }, a), s);
    if (S.view === 'months') {
      const x0 = 34, y0 = H - 34, tot = D.counts.map((v, i) => v + D.title_only_counts[i]), mx = Math.max(...tot), bw = (W - x0 - 10) / D.months.length, Y = v => y0 - v / mx * (H - 86);
      [0, 40, 80, 120].forEach(v => { if (v <= mx) { el('line', { x1: x0, x2: W - 6, y1: Y(v), y2: Y(v), stroke: 'var(--rule)' }); txt(x0 - 5, Y(v) + 4, v, { 'text-anchor': 'end' }); } });
      D.months.forEach((m, i) => {
        const a = (y0 - Y(D.counts[i])) * S.u, b = (Y(D.counts[i]) - Y(tot[i])) * S.u, x = x0 + i * bw + 1, w = Math.max(1, bw - 2);
        el('rect', { x, y: y0 - a, width: w, height: a, fill: 'var(--fg)', opacity: .72 });
        el('rect', { x, y: y0 - a - b, width: w, height: b, fill: 'var(--muted)', opacity: .38 });
        if (m.endsWith('-01')) txt(x + bw / 2, y0 + 14, m.slice(0, 4), { 'text-anchor': 'middle' });
      });
      MARKS.forEach(([m, name], k) => { const i = D.months.indexOf(m); if (i < 0) return; const x = x0 + i * bw + bw / 2;
        el('line', { x1: x, x2: x, y1: 34 + k * 12, y2: y0, stroke: 'var(--yellow)', 'stroke-dasharray': '3 3', opacity: .9 });
        txt(x + 3, 32 + k * 12, name, { fill: 'var(--yellow)' }); });
      txt(x0, 14, narrow ? 'arXiv papers per month' : 'arXiv papers per month: solid = Mamba or selective SSM, light = title-only "state space model"', { 'font-size': 11 });
    } else {
      const rows = D.domains, mx = rows[0][1], x0 = narrow ? 128 : 220, rh = (H - 34) / rows.length;
      rows.forEach(([name, n], i) => { const y = 22 + i * rh, w = (W - x0 - 46) * n / mx * S.u;
        el('rect', { x: x0, y: y + 3, width: w, height: rh - 7, fill: name === 'other' ? 'var(--rule)' : 'var(--fg)', opacity: name === 'other' ? 1 : .72, rx: 2 });
        txt(x0 - 6, y + rh / 2 + 3, narrow ? name.split(',')[0].replace(' and ', ' & ') : name, { 'text-anchor': 'end', fill: 'var(--fg)', 'font-size': narrow ? 8.5 : 10.5 });
        txt(x0 + w + 5, y + rh / 2 + 3, n); });
      txt(8, 12, 'title-keyword split of the solid set (heuristic, first matching rule wins)', { 'font-size': 11 });
    }
  }
  function set(v) {
    S.view = v; fig.querySelectorAll('[data-view]').forEach(b => b.setAttribute('aria-pressed', String(b.dataset.view === v)));
    cancel(); S.u = 0; cancel = PE.animate(600, u => { S.u = u; draw(); });
    const dom = Object.fromEntries(D.domains);
    out.textContent = v === 'months'
      ? `${D.deep.toLocaleString('en-US')} papers in the solid set: ${D.yearly['2024']} in 2024, ${D.yearly['2025']} in 2025, ${D.yearly['2026']} in 2026 up to the scan; ${D.title_ssm_only} more match only the title query.`
      : `Largest groups: general vision (${dom['images and general vision']}), architecture and theory (${dom['architecture, theory, efficiency']}), medical imaging (${dom['medical imaging and health']}); language and LLMs: ${dom['language and LLMs']}.`;
  }
  fig.querySelectorAll('[data-view]').forEach(b => b.addEventListener('click', () => set(b.dataset.view)));
  document.getElementById('btn-map-reset').addEventListener('click', () => set('months'));
  new ResizeObserver(draw).observe(svg); PE.onTheme(draw); set('months');
})();
</script>
