<style>
/* Charts in this group are SVG; let the page scroll over them on touch, and show the focus ring. */
#map-svg, #tree-svg { touch-action: pan-y; }
figure.viz svg.chart:focus-visible { outline: 2px solid var(--yellow); outline-offset: 2px; }
</style>
<script>
(() => {
  // arXiv scan of 2026-10-09: notes/data/scan-stats.json (claims in notes/claims/scan.json). Months absent there
  // are zero; every month in the range has a claim field-month-YYYY-MM in notes/claims/page-field.json.
  const YEARS = [[2017, 2], [2018, 6], [2019, 2], [2020, 5], [2021, 12], [2022, 38], [2023, 93], [2024, 110], [2025, 116], [2026, 138]];
  const BYM = { '2017-10': 1, '2017-12': 1, '2018-03': 2, '2018-04': 1, '2018-05': 1, '2018-07': 1, '2018-11': 1, '2019-06': 1, '2019-12': 1,
    '2020-01': 1, '2020-02': 1, '2020-10': 2, '2020-11': 1, '2021-02': 1, '2021-03': 1, '2021-04': 3, '2021-08': 1, '2021-09': 2, '2021-11': 2,
    '2021-12': 2, '2022-01': 2, '2022-03': 5, '2022-04': 2, '2022-05': 2, '2022-06': 2, '2022-07': 7, '2022-08': 3, '2022-09': 2, '2022-10': 2,
    '2022-11': 6, '2022-12': 5, '2023-01': 2, '2023-02': 3, '2023-03': 14, '2023-04': 6, '2023-05': 4, '2023-06': 5, '2023-07': 6, '2023-08': 5,
    '2023-09': 9, '2023-10': 12, '2023-11': 9, '2023-12': 18, '2024-01': 3, '2024-02': 5, '2024-03': 14, '2024-04': 12, '2024-05': 6,
    '2024-06': 12, '2024-07': 13, '2024-08': 8, '2024-09': 6, '2024-10': 11, '2024-11': 12, '2024-12': 8, '2025-01': 6, '2025-02': 4,
    '2025-03': 20, '2025-04': 6, '2025-05': 9, '2025-06': 10, '2025-07': 17, '2025-08': 8, '2025-09': 11, '2025-10': 9, '2025-11': 8,
    '2025-12': 8, '2026-01': 9, '2026-02': 8, '2026-03': 24, '2026-04': 19, '2026-05': 21, '2026-06': 15, '2026-07': 12, '2026-08': 15,
    '2026-09': 13, '2026-10': 2 };
  const MONTHS = [];
  for (let y = 2017, m = 10; y < 2026 || (y === 2026 && m <= 10); m === 12 ? (y++, m = 1) : m++) {
    const k = `${y}-${String(m).padStart(2, '0')}`; MONTHS.push([k, BYM[k] || 0]);
  }
  // Title-keyword domains, sorted by count (scan-stats.json "domains"; claim ids scan-dom-*).
  const DOMS = [['3D / LiDAR', 63, '3d-lidar'], ['few-shot / adaptation / benchmarks', 57, 'few-shot-adaptation-benchmarks'],
    ['open-world / unknowns', 53, 'open-world-unknowns'], ['real-time / efficient', 38, 'real-time-efficient'],
    ['remote sensing / aerial', 27, 'remote-sensing-aerial'], ['video / tracking', 25, 'video-tracking'], ['robotics', 20, 'robotics'],
    ['driving', 12, 'driving'], ['medical', 8, 'medical'], ['industrial / defect', 3, 'industrial-defect']];
  // The six main papers of this page, by the month of their first arXiv version (claims *-v1-date / *-published).
  const SPINE = [['ViLD', '2021-04'], ['GLIP', '2021-12'], ['OWL-ViT', '2022-05'], ['Grounding DINO', '2023-03'], ['OWLv2', '2023-06'], ['YOLO-World', '2024-01']];
  const MAXM = Math.max(...MONTHS.map(m => m[1]));

  const fig = document.getElementById('viz-map'), svg = document.getElementById('map-svg'), out = document.getElementById('map-readout');
  const NS = 'http://www.w3.org/2000/svg', MC = document.createElement('canvas').getContext('2d');
  const tw = (s, fs, wt = 400) => { MC.font = `${wt} ${fs}px system-ui, sans-serif`; return MC.measureText(s).width; };
  const c = (id, v) => `<span data-claim="${id}">${v}</span>`;
  const S = { view: 'years', u: 1, hover: -1, sel: -1 };
  let stop = () => {}, lastW = 0, geo = null;

  function el(tag, a, txt, parent = svg) {
    const e = document.createElementNS(NS, tag);
    for (const k in a) e.setAttribute(k, a[k]);
    if (txt != null) e.textContent = txt;
    parent.appendChild(e); return e;
  }
  const T = (x, y, s, o = {}) => el('text', Object.assign({ x, y, 'font-family': 'system-ui, sans-serif', 'font-size': o.fs || 11,
    fill: o.fill, 'font-weight': o.wt || 400, 'text-anchor': o.anchor || 'start' }, o.claim ? { 'data-claim': o.claim } : {}), s);
  // A bar with rounded data end, anchored to the baseline.
  const barPath = (x, y, w, base) => { const r = Math.min(3, w / 2, Math.max(0, base - y));
    return `M${x},${base} V${y + r} Q${x},${y} ${x + r},${y} H${x + w - r} Q${x + w},${y} ${x + w},${y + r} V${base} Z`; };
  const hbarPath = (x, y, w, h) => { const r = Math.min(3, h / 2, Math.max(0, w));
    return `M${x},${y} H${x + w - r} Q${x + w},${y} ${x + w},${y + r} V${y + h - r} Q${x + w},${y + h} ${x + w - r},${y + h} H${x} Z`; };

  function draw() {
    while (svg.firstChild) svg.removeChild(svg.firstChild);
    const k = DK.C(), W = Math.max(280, svg.clientWidth), narrow = W < 520, u = S.u;
    const ink = DK.mixh(k.bg, k.fg, .5), faint = DK.mixh(k.bg, k.fg, .22), cur = S.hover >= 0 ? S.hover : S.sel;
    let H;
    if (S.view === 'years') {
      H = narrow ? 270 : 300;
      const x0 = 8, x1 = W - 8, top = 64, base = H - 28, slot = (x1 - x0) / YEARS.length, bw = slot * (narrow ? .7 : .58);
      const Y = v => base - v / 138 * (base - top);
      geo = { x0, slot, n: YEARS.length };
      T(x0, 16, cur >= 0 ? `${YEARS[cur][0]}: ${YEARS[cur][1]} papers` : (narrow ? 'Papers per year' : 'Papers first posted on arXiv, per year'), { fill: k.mu, fs: 12 });
      el('line', { x1: x0, x2: x1, y1: base + .5, y2: base + .5, stroke: k.rule });
      YEARS.forEach(([y, n], i) => {
        const x = x0 + i * slot + (slot - bw) / 2, yb = PE.lerp(base, Y(n), u), part = y === 2026, on = i === cur;
        el('path', { d: barPath(x, yb, bw, base), fill: on ? k.fg : (part ? faint : ink),
          stroke: part && !on ? ink : 'none', 'stroke-dasharray': '3 3', 'stroke-width': 1 });
        T(x + bw / 2, base + 16, narrow ? `'${String(y).slice(2)}` : String(y), { fill: k.mu, fs: narrow ? 10.5 : 11, anchor: 'middle' });
        const lab = T(x + bw / 2, yb - 6, String(n), { fill: on ? k.fg : k.mu, fs: narrow ? 10.5 : 11.5, wt: 600, anchor: 'middle', claim: 'scan-year-' + y });
        lab.setAttribute('opacity', u);
        if (part) { const p = T(x + bw, yb - 21, 'partial, to ', { fill: k.mu, fs: 10.5, anchor: 'end' });
          const ts = el('tspan', { 'data-claim': 'scan-date' }, '2026-10-09', p); p.setAttribute('opacity', u); void ts; }
      });
    } else if (S.view === 'months') {
      H = narrow ? 300 : 320;
      const x0 = 28, x1 = W - 8, base = H - 26, slot = (x1 - x0) / MONTHS.length;
      const fs = narrow ? 10.5 : 11, rowH = 15, gap = 6;
      // marker labels: greedy rows so no two labels overlap
      const rows = [], marks = SPINE.map(([name, m]) => {
        const i = MONTHS.findIndex(x => x[0] === m), cx = x0 + (i + .5) * slot, w = tw(name, fs, 600);
        let lx = cx - 3, anchor = 'start'; if (lx + w > x1) { lx = cx + 3 - w; anchor = 'end'; }
        let r = 0; while (rows[r] != null && rows[r] > lx - gap) r++; rows[r] = lx + w;
        return { name, i, cx, r, anchor, tx: anchor === 'start' ? lx : cx + 3 };
      });
      const band = 40 + rows.length * rowH, top = band + 14;
      const Y = v => base - v / MAXM * (base - top);
      geo = { x0, slot, n: MONTHS.length };
      T(8, 16, cur >= 0 ? `${MONTHS[cur][0]}: ${MONTHS[cur][1]} papers` : (narrow ? 'Papers per month' : 'Papers first posted on arXiv, per month; the six main papers of this page are marked'), { fill: k.mu, fs: 12 });
      [0, 10, 20].forEach(v => { el('line', { x1: x0, x2: x1, y1: Y(v) + .5, y2: Y(v) + .5, stroke: k.rule });
        T(x0 - 5, Y(v) + 4, String(v), { fill: k.mu, fs: 10.5, anchor: 'end' }); });
      MONTHS.forEach(([m, n], i) => {
        const x = x0 + i * slot, bw = Math.max(1, slot - (slot > 5 ? 1.5 : .6)), yb = PE.lerp(base, Y(n), u), on = i === cur;
        if (n > 0) el('path', { d: barPath(x, yb, bw, base), fill: on ? k.fg : (m === '2026-10' ? faint : ink) });
        if (on && n === 0) el('line', { x1: x + bw / 2, x2: x + bw / 2, y1: base - 6, y2: base, stroke: k.fg, 'stroke-width': 2 });
        if (m.endsWith('-01')) { el('line', { x1: x, x2: x, y1: base, y2: base + 4, stroke: k.mu });
          T(x, base + 16, narrow ? `'${m.slice(2, 4)}` : m.slice(0, 4), { fill: k.mu, fs: 10.5, anchor: 'middle' }); }
      });
      marks.forEach(mk => {
        const ly = 34 + mk.r * rowH, n = MONTHS[mk.i][1];
        el('line', { x1: mk.cx, x2: mk.cx, y1: ly + 4, y2: PE.lerp(base, Y(n), u) - 3, stroke: k.mu, 'stroke-width': 1, opacity: .8 * u });
        const t = T(mk.tx, ly, mk.name, { fill: k.fg, fs, wt: 600, anchor: mk.anchor }); t.setAttribute('opacity', u);
      });
    } else {
      geo = null;
      const rh = narrow ? 34 : 25, top = 40, lw = narrow ? 0 : Math.min(250, W * .32);
      H = top + DOMS.length * rh + 44;
      const bx = narrow ? 8 : lw + 14, bmax = W - bx - 40;
      T(8, 16, narrow ? 'Title keyword groups (a heuristic)' : 'Title keyword groups (a heuristic: one title can match several groups)', { fill: k.mu, fs: 12 });
      DOMS.forEach(([name, n, id], i) => {
        const ind = id === 'industrial-defect', y = top + i * rh, bh = narrow ? 9 : 12, by = narrow ? y + 15 : y + (rh - bh) / 2;
        const w = Math.max(2, bmax * n / 63 * u);
        if (narrow) T(8, y + 11, name, { fill: k.fg, fs: 11, wt: ind ? 700 : 400 });
        else T(lw, by + bh / 2 + 4, name, { fill: k.fg, fs: 11.5, wt: ind ? 700 : 400, anchor: 'end' });
        el('path', { d: hbarPath(bx, by, w, bh), fill: ind ? k.fg : ink });
        T(bx + w + 6, by + bh / 2 + 4, String(n), { fill: ind ? k.fg : k.mu, fs: 11.5, wt: ind ? 700 : 600, claim: 'scan-dom-' + id }).setAttribute('opacity', u);
      });
      const ty = top + DOMS.length * rh + 14;
      el('line', { x1: 8, x2: W - 8, y1: ty + .5, y2: ty + .5, stroke: k.rule });
      const t = T(8, ty + 20, 'No domain keyword in the title: ', { fill: k.mu, fs: 11.5 });
      el('tspan', { 'data-claim': 'scan-dom-none', fill: k.fg, 'font-weight': 600 }, '269', t);
      el('tspan', {}, ' titles', t);
    }
    svg.setAttribute('viewBox', `0 0 ${W} ${H}`); svg.style.height = H + 'px';
  }

  function sayView() {
    out.innerHTML = {
      years: `${c('scan-total', 522)} papers in the scan: ${c('scan-year-2021', 12)} in 2021, ${c('scan-year-2024', 110)} in 2024, ${c('scan-year-2025', 116)} in 2025 and ${c('scan-year-2026', 138)} in 2026 up to ${c('scan-date', '2026-10-09')}.`,
      months: `The same papers per month. March was the busiest month of 2024, 2025 and 2026, with ${c('field-month-2024-03', 14)}, ${c('field-month-2025-03', 20)} and ${c('field-month-2026-03', 24)} papers.`,
      domains: `${c('scan-dom-3d-lidar', 63)} titles match 3D or LiDAR; ${c('scan-dom-industrial-defect', 3)} of ${c('scan-total', 522)} match industry, defects, manufacturing or inspection: BioDet (arXiv 2510.21000), a field-inspection grasping robot (2603.07866) and IndusAgent (2605.20682).`,
    }[S.view];
  }
  function sayCursor() {
    const i = S.sel;
    if (S.view === 'years') { const [y, n] = YEARS[i];
      out.innerHTML = `${y}: ${c('scan-year-' + y, n)} papers first posted on arXiv${y === 2026 ? `, up to ${c('scan-date', '2026-10-09')}` : ''}.`; }
    else { const [m, n] = MONTHS[i], mk = SPINE.find(s => s[1] === m);
      out.innerHTML = `${m}: ${c('field-month-' + m, n)} papers first posted${m === '2026-10' ? `, up to ${c('scan-date', '2026-10-09')}` : ''}.${mk ? ` ${mk[0]} first appeared this month.` : ''}`; }
  }
  // Buttons SET a view; pressing one twice equals pressing it once.
  function set(view) {
    S.view = view; S.hover = -1; S.sel = -1;
    ['years', 'months', 'domains'].forEach(v => document.getElementById('btn-map-' + v).setAttribute('aria-pressed', String(v === view)));
    stop(); S.u = 0; stop = PE.animate(700, t => { S.u = t; draw(); });
    sayView();
  }
  const idxAt = e => { if (!geo) return -1; const r = svg.getBoundingClientRect(), x = (e.clientX - r.left) * (svg.viewBox.baseVal.width / r.width);
    const i = Math.floor((x - geo.x0) / geo.slot); return i >= 0 && i < geo.n ? i : -1; };
  ['years', 'months', 'domains'].forEach(v => document.getElementById('btn-map-' + v).addEventListener('click', () => set(v)));
  document.getElementById('btn-map-reset').addEventListener('click', () => set('years'));
  svg.addEventListener('pointermove', e => { const i = idxAt(e); if (i !== S.hover) { S.hover = i; draw(); } });
  svg.addEventListener('pointerleave', () => { if (S.hover >= 0) { S.hover = -1; draw(); } });
  svg.addEventListener('pointerdown', e => { const i = idxAt(e); if (i < 0) return; S.sel = i; draw(); sayCursor(); });
  svg.addEventListener('keydown', e => {
    if (!geo) return; const d = { ArrowLeft: -1, ArrowRight: 1, ArrowUp: 1, ArrowDown: -1 }[e.key];
    if (d == null && e.key !== 'Home' && e.key !== 'End') return; e.preventDefault();
    S.hover = -1; S.sel = e.key === 'Home' ? 0 : e.key === 'End' ? geo.n - 1 : PE.clamp((S.sel < 0 ? (d > 0 ? -1 : geo.n) : S.sel) + d, 0, geo.n - 1);
    draw(); sayCursor();
  });
  new ResizeObserver(() => { const w = svg.clientWidth; if (w !== lastW) { lastW = w; draw(); } }).observe(svg);
  PE.onTheme(draw);
  set('years');
})();
</script>
