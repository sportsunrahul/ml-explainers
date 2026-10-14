<script>
(() => {
  const D = {"scanned": "2026-10-08", "total": 569, "yearly": {"2022": 1, "2023": 7, "2024": 38, "2025": 76, "2026": 447}, "months": ["2024-02", "2024-03", "2024-04", "2024-05", "2024-06", "2024-07", "2024-08", "2024-09", "2024-10", "2024-11", "2024-12", "2025-01", "2025-02", "2025-03", "2025-04", "2025-05", "2025-06", "2025-07", "2025-08", "2025-09", "2025-10", "2025-11", "2025-12", "2026-01", "2026-02", "2026-03", "2026-04", "2026-05", "2026-06", "2026-07", "2026-08", "2026-09", "2026-10"], "counts": [1, 4, 1, 5, 3, 2, 3, 5, 7, 4, 3, 3, 6, 2, 6, 3, 8, 7, 2, 10, 8, 11, 10, 17, 26, 27, 20, 46, 53, 61, 69, 97, 31], "domains": [["world models, planning, robotics", 196], ["other", 73], ["video", 73], ["theory, analysis, recipes", 46], ["graphs, molecules, science", 32], ["audio, speech, music", 31], ["medical, bio-signals", 28], ["language, reasoning, LLMs", 24], ["time series, tabular, finance", 20], ["remote sensing, wireless", 16], ["images, general vision", 14], ["3D and point clouds", 11], ["generation", 5]], "queries": ["search_query=ti:JEPA&sortBy=submittedDate&sortOrder=descending", "search_query=abs:%22joint-embedding%20predictive%22%20OR%20abs:%22joint%20embedding%20predictive%22&sortBy=submittedDate&sortOrder=descending", "search_query=abs:JEPA%20OR%20abs:JEPAs&sortBy=submittedDate&sortOrder=descending"]};
  const fig = document.getElementById('viz-map'), svg = document.getElementById('map-svg'), out = document.getElementById('map-readout');
  const S = { view: 'months', u: 1 }, NS = 'http://www.w3.org/2000/svg';
  function draw() {
    while (svg.firstChild) svg.removeChild(svg.firstChild);
    const narrow = svg.clientWidth < 520, W = narrow ? 360 : 700, H = narrow ? 260 : 280; svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
    const el = (tag, a, txt) => { const e = document.createElementNS(NS, tag); for (const k in a) e.setAttribute(k, a[k]); if (txt != null) e.textContent = txt; svg.appendChild(e); return e; };
    if (S.view === 'months') {
      const x0 = 36, y0 = H - 34, mx = Math.max(...D.counts), bw = (W - x0 - 12) / D.months.length, Y = v => y0 - v / mx * (H - 70);
      [0, 20, 40, 60].forEach(v => { if (v <= mx) { el('line', { x1: x0, x2: W - 8, y1: Y(v), y2: Y(v), stroke: 'var(--rule)' }); el('text', { x: x0 - 5, y: Y(v) + 4, 'text-anchor': 'end', fill: 'var(--muted)', 'font-size': 10, 'font-family': 'system-ui' }, v); } });
      D.months.forEach((m, i) => { const h = (y0 - Y(D.counts[i])) * S.u; el('rect', { x: x0 + i * bw + 1, y: y0 - h, width: Math.max(1, bw - 2), height: h, fill: m.startsWith('2026') ? 'var(--blue)' : 'var(--brown)', opacity: .9 });
        if (m.endsWith('-01')) el('text', { x: x0 + i * bw + bw / 2, y: y0 + 14, 'text-anchor': 'middle', fill: 'var(--muted)', 'font-size': 10, 'font-family': 'system-ui' }, m.slice(0, 4)); });
      el('text', { x: x0, y: 14, fill: 'var(--muted)', 'font-size': 11, 'font-family': 'system-ui' }, 'JEPA papers first posted on arXiv, per month (last bar: partial October 2026)');
    } else {
      const rows = D.domains.slice(0, 10), mx = rows[0][1], x0 = narrow ? 120 : 210, rh = (H - 40) / rows.length;
      rows.forEach(([name, n], i) => { const y = 24 + i * rh, w = (W - x0 - 50) * n / mx * S.u; el('rect', { x: x0, y: y + 3, width: w, height: rh - 8, fill: name === 'other' ? 'var(--rule)' : 'var(--blue)', opacity: .9, rx: 2 });
        el('text', { x: x0 - 6, y: y + rh / 2 + 3, 'text-anchor': 'end', fill: 'var(--fg)', 'font-size': narrow ? 8.5 : 10.5, 'font-family': 'system-ui' }, narrow ? name.split(',')[0] : name);
        el('text', { x: x0 + w + 5, y: y + rh / 2 + 3, fill: 'var(--muted)', 'font-size': 10, 'font-family': 'system-ui' }, n); });
      el('text', { x: 8, y: 14, fill: 'var(--muted)', 'font-size': 11, 'font-family': 'system-ui' }, 'title-keyword classification of the same papers (heuristic)');
    }
  }
  function set(v) { S.view = v; fig.querySelectorAll('[data-view]').forEach(b => b.setAttribute('aria-pressed', String(b.dataset.view === v))); S.u = 0; PE.animate(600, u => { S.u = u; draw(); });
    out.textContent = v === 'months' ? `${D.total} papers in all; ${D.yearly['2024']} in 2024, ${D.yearly['2025']} in 2025, ${D.yearly['2026']} so far in 2026.` : `Largest title-keyword group: ${D.domains[0][0]} (${D.domains[0][1]} papers); ${D.domains.find(d => d[0] === 'other')[1]} titles name no domain.`; }
  fig.querySelectorAll('[data-view]').forEach(b => b.addEventListener('click', () => set(b.dataset.view)));
  document.getElementById('btn-map-reset').addEventListener('click', () => set('months'));
  new ResizeObserver(draw).observe(svg); PE.onTheme(draw); set('months');
})();
</script>
