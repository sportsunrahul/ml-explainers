/* ch-map: arXiv counts per year from notes/scan/arxiv-counts-2026-10-13.json (same numbers as notes/claims/page-map.json,
   ids scan-<series>-<year>). Bars use neutral colors: the five key colors carry meanings this chart has no claim to. */
KIT.MAPSCAN = {
  years: [2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026],
  date: '2026-10-13',
  data: {"title":[93,182,183,432,853,1197,1299,1331,1324,1056],"abs":[4,6,17,241,690,1203,1501,1705,1710,1060],"graph":[0,0,2,17,57,102,118,125,115,48],"text":[4,2,3,17,56,137,155,171,170,104],"image":[26,55,49,100,142,207,219,220,240,173],"medical":[0,0,3,10,19,26,39,36,45,27],"video":[2,1,4,22,56,80,65,69,54,62],"rec":[0,0,0,3,14,35,49,42,40,20],"anomaly":[0,0,0,1,12,13,16,17,23,13],"supervised":[2,1,2,57,142,183,172,128,115,70]},
  names: { image: 'image, visual or vision', text: 'sentence, text or language', graph: 'graph', video: 'video, audio or speech',
    medical: 'medical, clinical or radiology', rec: 'recommendation or recommender', anomaly: 'anomaly', supervised: 'supervised' },
  short: { image: 'images', text: 'text', graph: 'graph', video: 'video or audio', medical: 'medical terms', rec: 'recommendation',
    anomaly: 'anomaly', supervised: 'supervised' },
};
// A claim span for one scan count: the page check and the reviewer match its text against page-map.json.
KIT.mapClaim = (series, year) => '<span data-claim="scan-' + series + '-' + year + '">' + KIT.MAPSCAN.data[series][KIT.MAPSCAN.years.indexOf(year)] + '</span>';

// State: view 'words' (title vs abstract, side by side) or 'domain' (all titles behind, one slice in front, share line).
KIT.mapchart = (canvas, opts = {}) => {
  const Y = KIT.MAPSCAN.years, D = KIT.MAPSCAN.data, TOP = 1800, SHARE_TOP = .3;
  const target = s => ({ g: s.view === 'words' ? 1 : 0, front: (s.view === 'words' ? D.abs : D[s.dom]).slice(), share: s.view === 'words' ? 0 : 1 });
  const state = { view: 'words', dom: 'image' };
  let cur = target(state), cancel = () => {};
  const glow = opts.fig ? PE.linkPulses(opts.fig, () => cv.redraw()) : () => 0;
  const draw = (ctx, W, H) => {
    if (!W) return;
    const narrow = KIT.narrow(W), fg = PE.css('--fg'), mut = PE.css('--muted'), rule = PE.css('--rule'), p2 = PE.css('--panel2');
    ctx.clearRect(0, 0, W, H);
    const b = { x: narrow ? 46 : 52, y: narrow ? 64 : 50, w: 0, h: 0 };
    b.w = W - b.x - (narrow ? 34 : 46); b.h = H - b.y - 30;
    const ax = PE.axes(ctx, b, [0, Y.length], [0, TOP], { yt: [0, 500, 1000, 1500], yl: 'papers per year', ylo: narrow ? 38 : 42 });
    const bw = b.w / Y.length, g = cur.g, back = PE.mix(p2, mut, g);
    const gAll = glow('all'), gSlice = glow('slice'), gShare = glow('share');
    Y.forEach((yr, i) => {
      const x0 = b.x + i * bw + bw * .12, inner = bw * .76, wBack = PE.lerp(inner, inner / 2, g);
      const hb = b.h * D.title[i] / TOP, hf = b.h * cur.front[i] / TOP, xf = PE.lerp(x0, x0 + inner / 2, g), wf = PE.lerp(inner * .62, inner / 2, g);
      ctx.fillStyle = back; ctx.fillRect(x0, b.y + b.h - hb, wBack, hb);
      ctx.fillStyle = fg; ctx.fillRect(xf + (1 - g) * (inner - wf) / 2, b.y + b.h - hf, wf, hf);
      if (gAll) { ctx.save(); ctx.globalAlpha = gAll; ctx.strokeStyle = fg; ctx.lineWidth = 2; ctx.strokeRect(x0 - 1, b.y + b.h - hb - 1, wBack + 2, hb + 2); ctx.restore(); }
      if (gSlice) { const hs = b.h * D[state.dom][i] / TOP; ctx.save(); ctx.globalAlpha = gSlice; ctx.strokeStyle = fg; ctx.lineWidth = 2.5;
        ctx.strokeRect(x0 + inner * .19 - 2, b.y + b.h - hs - 3, inner * .62 + 4, hs + 3); ctx.restore(); }
      if (yr === 2026) { ctx.save(); ctx.setLineDash([3, 3]); ctx.strokeStyle = mut; ctx.lineWidth = 1; ctx.strokeRect(x0 - 3, b.y + b.h - Math.max(hb, hf) - 4, inner + 6, Math.max(hb, hf) + 4); ctx.restore();
        PE.label(ctx, 'partial', x0 + inner / 2, b.y + b.h - Math.max(hb, hf) - 9, mut, 10, 'center', 400); }
      PE.label(ctx, narrow ? "'" + String(yr).slice(2) : String(yr), x0 + inner / 2, b.y + b.h + 15, mut, narrow ? 10 : 11, 'center', 400);
    });
    // Share of all titles held by the slice, on its own right-hand axis.
    const sa = Math.max(cur.share, gShare, gSlice * (1 - cur.share));
    if (sa > .01) {
      const sl = state.view === 'words' ? D.image : cur.front, sy = v => b.y + b.h - b.h * v / SHARE_TOP;
      const pts = Y.map((_, i) => [b.x + (i + .5) * bw, sy(sl[i] / D.title[i])]);
      PE.polyline(ctx, pts, mut, 2 + 2 * gShare, [5, 4], sa);
      pts.forEach(([x, y]) => PE.dot(ctx, x, y, 3.2 + 2 * gShare, mut, sa));
      ctx.save(); ctx.globalAlpha = sa; [0, .1, .2, .3].forEach(v => PE.label(ctx, Math.round(v * 100) + '%', b.x + b.w + 5, sy(v) + 4, mut, 10, 'left', 400)); ctx.restore();
    }
    // Legend.
    const lx = b.x, ly = narrow ? 16 : 18, sw = (x, y, c, t) => { ctx.fillStyle = c; ctx.fillRect(x, y - 9, 11, 11); PE.label(ctx, t, x + 16, y, fg, narrow ? 11 : 12, 'left', 500); };
    if (state.view === 'words') {
      sw(lx, ly, back, 'title contains "contrastive"');
      sw(narrow ? lx : lx + 230, narrow ? ly + 18 : ly, fg, 'abstract contains "contrastive learning"');
    } else {
      sw(lx, ly, back, 'all "contrastive" titles');
      sw(narrow ? lx : lx + 190, narrow ? ly + 18 : ly, fg, 'titles that also name ' + (narrow ? KIT.MAPSCAN.short[state.dom] : KIT.MAPSCAN.names[state.dom]));
      ctx.save(); ctx.setLineDash([5, 4]); ctx.strokeStyle = mut; ctx.lineWidth = 2; ctx.beginPath();
      const sx = narrow ? lx : lx, sy2 = narrow ? ly + 36 : ly + 20; ctx.moveTo(sx, sy2 - 4); ctx.lineTo(sx + 11, sy2 - 4); ctx.stroke(); ctx.restore();
      PE.label(ctx, 'share of all titles (right axis)', sx + 16, sy2, mut, narrow ? 11 : 12, 'left', 400);
    }
  };
  const cv = PE.canvas(canvas, w => KIT.narrow(w) ? 1.05 : .5, draw);
  const set = next => {
    Object.assign(state, next); cancel();
    const from = { g: cur.g, share: cur.share, front: cur.front.slice() }, to = target(state);
    cancel = PE.animate(700, u => {
      cur = { g: PE.lerp(from.g, to.g, u), share: PE.lerp(from.share, to.share, u), front: from.front.map((v, i) => PE.lerp(v, to.front[i], u)) };
      cv.redraw();
    });
  };
  return { set, state, redraw: cv.redraw, resize: cv.resize, destroy: () => { cancel(); cv.destroy(); } };
};
// Readout HTML, written once per click; every count is a claim span.
KIT.mapText = s => {
  const c = KIT.mapClaim, yrs = [2019, 2020, 2021, 2025];
  if (s.view === 'words')
    return 'Titles with "contrastive": ' + yrs.map(y => c('title', y) + ' in ' + y).join(', ') + '. Abstracts with "contrastive learning": ' +
      yrs.map(y => c('abs', y)).join(', ') + ' in the same years. The abstract count passes the title count in 2022 (' + c('abs', 2022) + ' vs ' + c('title', 2022) + ').';
  return 'Titles with "contrastive" that also name ' + KIT.MAPSCAN.names[s.dom] + ': ' +
    [2019, 2021, 2025].map(y => c(s.dom, y) + ' of ' + c('title', y) + ' in ' + y).join(', ') +
    '. The 2026 bar stops at <span data-claim="scan-date">' + KIT.MAPSCAN.date + '</span>.';
};
