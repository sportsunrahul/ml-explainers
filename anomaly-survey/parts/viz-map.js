<script>
(() => {
  // ch-map: papers per year in the industrial visual subset, with one keyword family highlighted. DATA is generated
  // from notes/map-data.json (do not hand-edit); year totals carry data-claim ids from notes/claims/map.json.
  const DATA = {"years": ["2018", "2019", "2020", "2021", "2022", "2023", "2024", "2025", "2026"], "total": [5, 9, 18, 43, 56, 101, 156, 201, 191], "fam": {"vlm": {"label": "vision-language / zero-shot", "counts": [0, 0, 0, 0, 0, 13, 42, 75, 72]}, "fewshot": {"label": "few-shot", "counts": [0, 0, 1, 2, 1, 14, 21, 31, 33]}, "multiclass": {"label": "multi-class / unified", "counts": [0, 0, 0, 1, 1, 7, 37, 50, 46]}, "synthetic": {"label": "synthetic anomalies", "counts": [0, 1, 3, 7, 11, 19, 26, 50, 36]}, "memory": {"label": "memory bank / kNN", "counts": [0, 0, 1, 2, 2, 13, 17, 25, 20]}, "flow": {"label": "normalizing flow", "counts": [0, 0, 1, 3, 4, 3, 3, 2, 3]}, "st": {"label": "student-teacher / distillation", "counts": [0, 1, 0, 1, 5, 5, 18, 5, 17]}, "recon": {"label": "reconstruction", "counts": [2, 3, 9, 11, 20, 33, 37, 39, 49]}, "diffusion": {"label": "diffusion", "counts": [0, 0, 0, 0, 0, 6, 13, 21, 17]}, "3d": {"label": "3D / point cloud / depth", "counts": [0, 0, 0, 2, 7, 11, 25, 36, 42]}, "logical": {"label": "logical anomalies", "counts": [0, 0, 0, 1, 2, 8, 12, 13, 16]}, "edge": {"label": "edge / real-time / latency", "counts": [1, 1, 0, 3, 3, 11, 21, 26, 32]}}};
  const LANES = [
    ['Benchmarks', [['MVTec AD', 2019.45, 'ch-benchmark', 'r', 'b'], ['VisA', 2022.57, 'ch-benchmark', 'r', 'b'], ['MVTec AD 2', 2025.24, 'ch-benchmark', 'l', 'b']]],
    ['Feature embedding', [['SPADE', 2020.34, 'ch-spade', 'a', 'b'], ['PaDiM', 2020.88, 'ch-padim', 'b', 'a'], ['PatchCore', 2021.45, 'ch-patchcore', 'r', 'b'], ['AnomalyDINO', 2024.39, 'ch-dinov2', 'r', 'b']]],
    ['Normalizing flow', [['FastFlow', 2021.87, 'ch-fastflow', 'r', 'b']]],
    ['Synthetic defects', [['CutPaste', 2021.27, 'ch-synthetic', 'a', 'b'], ['DRAEM', 2021.63, 'ch-synthetic', 'b', 'a']]],
    ['Student-teacher', [['EfficientAD', 2023.23, 'ch-efficientad', 'r', 'b']]],
    ['Multi-class', [['UniAD', 2022.44, 'ch-uniad', 'r', 'b'], ['Dinomaly', 2024.39, 'ch-dinov2', 'r', 'b']]],
    ['Vision-language', [['WinCLIP', 2023.23, 'ch-winclip', 'a', 'b'], ['AnomalyCLIP', 2023.83, 'ch-anomalyclip', 'b', 'a']]],
  ];
  const NS = 'http://www.w3.org/2000/svg';
  const fig = document.getElementById('viz-map'), svg = document.getElementById('svg-map'), lanes = document.getElementById('svg-lanes'), out = fig.querySelector('.readout');
  svg.style.fontFamily = lanes.style.fontFamily = 'system-ui, sans-serif';
  const KEYS = ['vlm', 'multiclass', 'fewshot', 'synthetic', 'recon', 'memory', 'st', 'flow', '3d', 'edge'];
  const S = { fam: null, u: 0 };
  const el = (tag, at, parent, text) => { const e = document.createElementNS(NS, tag); for (const k in at) e.setAttribute(k, at[k]); if (text != null) e.textContent = text; parent.appendChild(e); return e; };
  function drawBars() {
    const narrow = fig.clientWidth < 520, W = narrow ? 360 : 640, H = narrow ? 310 : 290, m = { l: 40, r: 12, t: 50, b: 34 };
    svg.setAttribute('viewBox', `0 0 ${W} ${H}`); svg.innerHTML = '';
    const n = DATA.years.length, bw = (W - m.l - m.r) / n, ymax = 220, Y = v => H - m.b - v / ymax * (H - m.t - m.b);
    [0, 50, 100, 150, 200].forEach(v => { el('line', { x1: m.l, x2: W - m.r, y1: Y(v), y2: Y(v), stroke: 'var(--rule)' }, svg); el('text', { x: m.l - 6, y: Y(v) + 4, 'text-anchor': 'end', 'font-size': 11, fill: 'var(--muted)' }, svg, v); });
    el('text', { x: m.l, y: 14, 'font-size': 11, fill: 'var(--muted)' }, svg, 'papers per year');
    const fam = S.fam ? DATA.fam[S.fam].counts : null;
    DATA.years.forEach((y, i) => {
      const x = m.l + i * bw + bw * .14, w = bw * .72, t = DATA.total[i];
      el('rect', { x, y: Y(t), width: w, height: Y(0) - Y(t), fill: 'var(--panel2)', stroke: 'var(--muted)', 'stroke-width': 1, 'stroke-dasharray': y === '2026' ? '3 2' : 'none' }, svg);
      if (fam) { const v = fam[i] * S.u; el('rect', { x: x + w * .18, y: Y(v), width: w * .64, height: Y(0) - Y(v), fill: 'var(--fg)', opacity: .85 }, svg);
        if (fam[i] && S.u > .99) el('text', { x: x + w / 2, y: Y(v) - 4, 'text-anchor': 'middle', 'font-size': 10, fill: 'var(--fg)', 'font-weight': 600 }, svg, fam[i]); }
      el('text', { x: x + w / 2, y: Y(t) - (fam ? 16 : 5), 'text-anchor': 'middle', 'font-size': 10, fill: 'var(--muted)', 'data-claim': 'map-year-' + y }, svg, t);
      el('text', { x: x + w / 2, y: H - m.b + 15, 'text-anchor': 'middle', 'font-size': narrow ? 10 : 11, fill: 'var(--muted)' }, svg, narrow ? "'" + y.slice(2) : y);
    });
    if (fam) el('text', { x: m.l, y: 31, 'font-size': 11, fill: 'var(--fg)', 'font-weight': 600 }, svg, 'highlighted: ' + DATA.fam[S.fam].label + ' (keyword match)');
  }
  function drawLanes() {
    const narrow = fig.clientWidth < 520, W = narrow ? 360 : 640, lh = narrow ? 54 : 40, m = { l: narrow ? 10 : 150, r: 16, t: 14, b: 26 }, H = m.t + LANES.length * lh + m.b;
    lanes.setAttribute('viewBox', `0 0 ${W} ${H}`); lanes.innerHTML = '';
    const X = t => m.l + (t - 2019) / (2026 - 2019) * (W - m.l - m.r);
    for (let y = 2019; y <= 2026; y++) { el('line', { x1: X(y), x2: X(y), y1: m.t, y2: H - m.b, stroke: 'var(--rule)' }, lanes);
      el('text', { x: X(y), y: H - 8, 'text-anchor': 'middle', 'font-size': narrow ? 10 : 11, fill: 'var(--muted)' }, lanes, narrow ? "'" + String(y).slice(2) : y); }
    LANES.forEach(([name, ps], r) => {
      const cy = m.t + r * lh + lh * (narrow ? .58 : .5);
      el('line', { x1: m.l, x2: W - m.r, y1: cy, y2: cy, stroke: 'var(--rule)', 'stroke-width': 1 }, lanes);
      el('text', { x: narrow ? m.l : m.l - 10, y: narrow ? cy - 22 : cy + 4, 'text-anchor': narrow ? 'start' : 'end', 'font-size': 11, fill: 'var(--muted)' }, lanes, name);
      ps.forEach(([label, t, href, pd, pn]) => {
        const a = el('a', { href: '#' + href }, lanes), x = X(t), pos = narrow ? pn : pd;
        el('circle', { cx: x, cy, r: 4.5, fill: 'var(--fg)' }, a);
        const at = { r: [x + 8, cy + 4, 'start'], l: [x - 8, cy + 4, 'end'], a: [x, cy - 9, 'middle'], b: [x, cy + 17, 'middle'] }[pos];
        el('text', { x: at[0], y: at[1], 'text-anchor': at[2], 'font-size': narrow ? 9.5 : 11, fill: 'var(--fg)' }, a, label);
      });
    });
  }
  function say() {
    if (!S.fam) { out.textContent = `All ${DATA.total.reduce((a, b) => a + b, 0)} papers in the subset, by year: ` + DATA.years.map((y, i) => `${y}: ${DATA.total[i]}`).join(', ') + ' (2026 to the scan date).'; return; }
    const f = DATA.fam[S.fam], first = DATA.years.find((y, i) => f.counts[i] > 0);
    out.textContent = `${f.label}: ${f.counts.reduce((a, b) => a + b, 0)} keyword matches, first in ${first}. By year: ` + DATA.years.map((y, i) => `${y}: ${f.counts[i]}`).join(', ') + '.';
  }
  function set(fam) {
    S.fam = fam; KEYS.forEach(k => document.getElementById('btn-map-' + k).setAttribute('aria-pressed', k === fam));
    PE.animate(600, u => { S.u = u; drawBars(); }, () => { S.u = 1; drawBars(); }); say();
  }
  KEYS.forEach(k => document.getElementById('btn-map-' + k).addEventListener('click', () => set(k)));
  document.getElementById('btn-map-reset').addEventListener('click', () => set(null));
  new ResizeObserver(() => { drawBars(); drawLanes(); }).observe(fig);
  S.u = 1; drawBars(); drawLanes(); say();
})();
</script>
