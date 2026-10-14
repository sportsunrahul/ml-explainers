/* FIG.map: the two literatures on arXiv (ch-map). Monthly counts of records returned by two arXiv API queries (2026-10-09),
   2015 onward, and a title-keyword split (a heuristic: a title can match several groups or none). Computed from the scan. */
FIG.map = (el, o = {}) => {
  const D = DATA.scan, S = { q: 'both' };
  const cv = PE.canvas(el, o.aspect || (w => w < 620 ? 1.25 : .44), draw);
  function draw(c, W, H) {
    const C = { fg: PE.css('--fg'), mu: PE.css('--muted'), blue: PE.css('--blue'), red: PE.css('--red'), yel: PE.css('--yellow') };
    c.clearRect(0, 0, W, H); H = Math.max(140, H - (o.padBottom || 0));
    const narrow = W < 620, A = narrow ? { x: 40, y: 20, w: W - 54, h: H * .5 - 50 } : { x: 44, y: 20, w: W * .62 - 60, h: H - 60 };
    const al = D['active-learning'], ln = D['label-noise'], n = al.months.length;
    const sm = a => a.map((_, i) => { let s = 0, k = 0; for (let j = Math.max(0, i - 2); j <= i; j++) { s += a[j]; k++; } return s / k; });
    const mx = Math.max(...sm(al.counts), ...sm(ln.counts));
    const yr = k => 2015 + k;
    const ax = PE.axes(c, A, [0, n - 1], [0, mx * 1.1], { xt: [0, 24, 48, 72, 96, 120, n - 1], xf: i => al.months[i].slice(0, 4) + (i === n - 1 ? '-10' : ''), yt: [0, 20, 40], xl: 'month (3-month average); 2026 runs to 9 October' });
    if (S.q !== 'noise') PE.polyline(c, sm(al.counts).map((v, i) => [ax.X(i), ax.Y(v)]), C.blue, 2.2);
    if (S.q !== 'al') PE.polyline(c, sm(ln.counts).map((v, i) => [ax.X(i), ax.Y(v)]), C.red, 2.2);
    PE.label(c, 'papers per month', A.x, A.y - 6, C.mu, 11);
    if (S.q !== 'noise') PE.label(c, 'active learning', ax.X(n * .55), ax.Y(sm(al.counts)[Math.floor(n * .55)]) - 14, C.blue, 11);
    if (S.q !== 'al') PE.label(c, 'noisy labels', ax.X(n * .32), ax.Y(sm(ln.counts)[Math.floor(n * .32)]) + 18, C.red, 11);
    const B = narrow ? { x: 8, y: H * .5 + 6, w: W - 16, h: H * .5 - 10 } : { x: W * .62 + 10, y: 20, w: W * .38 - 18, h: H - 40 };
    const groups = (S.q === 'noise' ? [] : [['active learning', al, C.blue]]).concat(S.q === 'al' ? [] : [['noisy labels', ln, C.red]]);
    let y = B.y; const kmax = Math.max(...groups.flatMap(g => g[1].kw.map(k => k[1])));
    groups.forEach(([lab, g, col]) => { PE.label(c, `${lab}: titles mentioning`, B.x, y + 10, col, 11); y += 18;
      g.kw.forEach(([k, v]) => { const w = (B.w - 190) * v / kmax; PE.label(c, k, B.x + 140, y + 10, C.fg, 10, 'right'); c.fillStyle = col; c.fillRect(B.x + 146, y + 2, w, 10); PE.label(c, String(v), B.x + 150 + w, y + 11, C.mu, 10); y += 15; });
      y += 10; });
    o.say && o.say(`Active learning: ${al.n.toLocaleString('en-US')} records (${al.y2020} in 2020, ${al.y2025} in 2025). Noisy labels: ${ln.n.toLocaleString('en-US')} records (${ln.y2020} in 2020, ${ln.y2025} in 2025). ${D.overlap} records are in both.`);
  }
  PE.onTheme(() => cv.redraw());
  return { query(q) { S.q = q; cv.redraw(); }, reset() { S.q = 'both'; cv.redraw(); }, redraw: () => cv.redraw(), destroy() { cv.destroy && cv.destroy(); } };
};
