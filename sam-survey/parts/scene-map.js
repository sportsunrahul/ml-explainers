/* ch-map: the arXiv scan of 2026-10-09 (title or abstract contains "segment anything"). Tabs: by month (with the
   SAM 2 and SAM 3 release months marked) and by topic (a title-keyword heuristic). Sourced counts. */
SCENES.map = ctx => {
  const MONTHS = {"2023-04": 41, "2023-05": 38, "2023-06": 39, "2023-07": 18, "2023-08": 29, "2023-09": 20, "2023-10": 28, "2023-11": 39, "2023-12": 50, "2024-01": 37, "2024-02": 29, "2024-03": 53, "2024-04": 32, "2024-05": 26, "2024-06": 37, "2024-07": 48, "2024-08": 63, "2024-09": 50, "2024-10": 33, "2024-11": 45, "2024-12": 37, "2025-01": 26, "2025-02": 26, "2025-03": 43, "2025-04": 42, "2025-05": 41, "2025-06": 38, "2025-07": 41, "2025-08": 44, "2025-09": 40, "2025-10": 22, "2025-11": 39, "2025-12": 24, "2026-01": 34, "2026-02": 15, "2026-03": 30, "2026-04": 32, "2026-05": 33, "2026-06": 22, "2026-07": 21, "2026-08": 13, "2026-09": 20, "2026-10": 5};
  const DOMAINS = {"remote sensing": 82, "other": 592, "medical": 387, "robustness and evaluation": 57, "efficient": 98, "3D": 98, "video and tracking": 95, "industrial": 34};
  const ORDER = ["industrial", "medical", "remote sensing", "3D", "video and tracking", "efficient", "robustness and evaluation", "other"];
  const MARKS = { '2023-04': 'SAM', '2024-08': 'SAM 2', '2025-11': 'SAM 3 and SAM 3D' };
  const keys = Object.keys(MONTHS), S = { step: 0, u: 1, hover: -1 };
  let cancel = () => {};
  const cv = PE.canvas(el(), S3ASPECT(ctx.stage), draw);
  function el() { return mkCanvas(ctx.stage); }
  const offMove = listen(cv.ctx.canvas, 'pointermove', e => { if (S.step) return; const r = cv.ctx.canvas.getBoundingClientRect(), x = e.clientX - r.left; const b = box(r.width, r.height); const i = Math.floor((x - b.x) / (b.w / keys.length)); S.hover = i >= 0 && i < keys.length ? i : -1; cv.redraw(); readout(); });
  const offLeave = listen(cv.ctx.canvas, 'pointerleave', () => { S.hover = -1; cv.redraw(); readout(); });
  function box(W, H) { const narrow = W < 620, bottom = ctx.reserve || 0; return { x: narrow ? 34 : 48, y: 44, w: W - (narrow ? 44 : 70), h: H - bottom - 44 - (narrow ? 54 : 50), narrow }; }
  function draw(c, W, H) {
    c.clearRect(0, 0, W, H); const b = box(W, H), u = S.u, G = PE.css('--green'), B = PE.css('--blue'), M = PE.css('--muted'), Y = PE.css('--yellow'), F = PE.css('--fg');
    if (S.step === 0) {
      const max = 70, ax = PE.axes(c, b, [0, keys.length], [0, max], { yt: [0, 20, 40, 60], yl: b.narrow ? '' : 'papers per month', ylo: 36 });
      const bw = b.w / keys.length;
      keys.forEach((k, i) => { const v = MONTHS[k], h = (ax.Y(0) - ax.Y(v)) * u, part = k >= '2026-09';
        c.fillStyle = part ? PE.css('--brown') : B; c.globalAlpha = S.hover === i ? 1 : .75; c.fillRect(b.x + i * bw + 1, ax.Y(0) - h, Math.max(1, bw - 2), h); c.globalAlpha = 1;
        if (k.endsWith('-01') || i === 0) PE.label(c, i === 0 && !k.endsWith('-01') ? k.slice(0, 4) + '-04' : k.slice(0, 4), b.x + i * bw, b.y + b.h + 16, M, 11, 'left', 400); });
      Object.entries(MARKS).forEach(([k, name], j) => { const i = keys.indexOf(k); if (i < 0) return; const x = b.x + (i + .5) * bw;
        c.save(); c.strokeStyle = Y; c.setLineDash([3, 3]); c.beginPath(); c.moveTo(x, b.y - 4 + j * 0); c.lineTo(x, b.y + b.h); c.stroke(); c.restore();
        PE.label(c, name, x + (i > keys.length * .7 ? -4 : 4), b.y - 8 + (j % 2) * 14, Y, b.narrow ? 10 : 12, i > keys.length * .7 ? 'right' : 'left', 600); });
      PE.label(c, 'brown: months still being indexed', b.x + b.w, b.y + b.h + 32, PE.css('--brown'), 11, 'right', 400);
    } else {
      const items = ORDER.map(k => [k, DOMAINS[k] || 0]).sort((a, b) => b[1] - a[1]), max = items[0][1];
      const lab = b.narrow ? 152 : 170, bh = Math.min(30, (b.h + 30) / items.length - 8);
      items.forEach(([k, v], i) => { const y = b.y + i * (bh + 8), w = (b.w - lab - 40) * v / max * u;
        PE.label(c, k, b.x - (b.narrow ? 24 : 30) + lab - 8, y + bh / 2 + 4, k === 'other' ? M : F, b.narrow ? 11 : 13, 'right', 500);
        c.fillStyle = k === 'industrial' ? G : k === 'other' ? PE.css('--rule') : B; c.globalAlpha = .85; c.fillRect(b.x - (b.narrow ? 24 : 30) + lab, y, w, bh); c.globalAlpha = 1;
        PE.label(c, String(v), b.x - (b.narrow ? 24 : 30) + lab + w + 6, y + bh / 2 + 4, M, 12, 'left', 600); });
    }
  }
  function readout() { if (S.step === 0) { const k = keys[S.hover]; ctx.hud(badge('sourced', k ? `${k}: ${MONTHS[k]} papers mention "segment anything" in the title or abstract.` : 'Papers per month whose title or abstract contains "segment anything", April 2023 to 9 October 2026. Hover a bar for its month.')); }
    else ctx.hud(badge('sourced', 'The same papers split by keywords in their titles. Each paper counts once, in the first matching group (industrial, medical, remote sensing, 3D, video, efficient, robustness). A heuristic, not a classification.')); }
  function animateIn() { cancel(); cancel = PE.animate(700, t => { S.u = t; cv.redraw(); }); }
  readout();
  return { flat: true, step(n) { S.step = n; S.hover = -1; readout(); animateIn(); }, resize: () => cv.resize(), destroy: () => { cancel(); offMove(); offLeave(); cv.destroy(); } };
};
