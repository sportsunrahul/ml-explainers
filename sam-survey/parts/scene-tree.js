/* ch-family / ch-timeline: the lineage on one time axis. Lanes by line of work; each paper at its arXiv v1 date
   (sourced, notes/claims/timeline.json); lines show "builds on" (our reading of each paper's method section).
   Tabs zoom: the whole span, the years before SAM, the years after. Hover or tap a dot for its name and date. */
SCENES.tree = ctx => {
  const LANES = [{"lane": "Segmentation models", "items": [{"k": "fcn", "n": "FCN", "d": "2014-11-14"}, {"k": "unet", "n": "U-Net", "d": "2015-05-18"}, {"k": "maskrcnn", "n": "Mask R-CNN", "d": "2017-03-20"}, {"k": "maskformer", "n": "MaskFormer", "d": "2021-07-13"}, {"k": "mask2former", "n": "Mask2Former", "d": "2021-12-02"}]}, {"lane": "Interactive segmentation", "items": [{"k": "dios", "n": "DIOS", "d": "2016-03-13"}, {"k": "ritm", "n": "RITM", "d": "2021-02-12"}, {"k": "focalclick", "n": "FocalClick", "d": "2022-04-06"}, {"k": "simpleclick", "n": "SimpleClick", "d": "2022-10-20"}]}, {"lane": "Video memory", "items": [{"k": "stm", "n": "STM", "d": "2019-04-01"}, {"k": "xmem", "n": "XMem", "d": "2022-07-14"}, {"k": "cutie", "n": "Cutie", "d": "2023-10-19"}]}, {"lane": "Backbones", "items": [{"k": "vit", "n": "ViT", "d": "2020-10-22"}, {"k": "mae", "n": "MAE", "d": "2021-11-11"}, {"k": "vitdet", "n": "ViTDet", "d": "2022-03-30"}, {"k": "hiera", "n": "Hiera", "d": "2023-06-01"}]}, {"lane": "The SAM line", "items": [{"k": "sam", "n": "SAM", "d": "2023-04-05"}, {"k": "sam2", "n": "SAM 2", "d": "2024-08-01"}, {"k": "sam3", "n": "SAM 3", "d": "2025-11-20"}, {"k": "sam3d", "n": "SAM 3D", "d": "2025-11-20"}]}, {"lane": "Faster", "items": [{"k": "mobilesam", "n": "MobileSAM", "d": "2023-06-25"}, {"k": "efficientsam", "n": "EfficientSAM", "d": "2023-12-01"}, {"k": "mobilesamv2", "n": "MobileSAMv2", "d": "2023-12-15"}, {"k": "efficientvitsam", "n": "EfficientViT-SAM", "d": "2024-02-07"}, {"k": "edgetam", "n": "EdgeTAM", "d": "2025-01-13"}, {"k": "efficientsam3", "n": "EfficientSAM3", "d": "2025-11-19"}]}, {"lane": "Sharper, adapted, composed", "items": [{"k": "groundingdino", "n": "Grounding DINO", "d": "2023-03-09"}, {"k": "seem", "n": "SEEM", "d": "2023-04-13"}, {"k": "samadapter", "n": "SAM-Adapter", "d": "2023-04-18"}, {"k": "medsam", "n": "MedSAM", "d": "2023-04-24"}, {"k": "trackanything", "n": "Track Anything", "d": "2023-04-24"}, {"k": "hqsam", "n": "HQ-SAM", "d": "2023-06-02"}, {"k": "groundedsam", "n": "Grounded SAM", "d": "2024-01-25"}, {"k": "sam2long", "n": "SAM2Long", "d": "2024-10-21"}, {"k": "samurai", "n": "SAMURAI", "d": "2024-11-18"}, {"k": "medsam2", "n": "MedSAM2", "d": "2025-04-04"}]}, {"lane": "Inspection and failure studies", "items": [{"k": "notperfect", "n": "Not Always Perfect", "d": "2023-04-12"}, {"k": "concealed", "n": "Concealed scenes", "d": "2023-04-12"}, {"k": "saa", "n": "Segment Any Anomaly", "d": "2023-05-18"}, {"k": "industrialsam", "n": "Industrial SAM", "d": "2023-07-24"}, {"k": "samdefect", "n": "SAM in Defect Detection", "d": "2023-11-17"}, {"k": "kairosad", "n": "KairosAD", "d": "2025-05-30"}, {"k": "boxes2pixels", "n": "Boxes2Pixels", "d": "2026-04-13"}]}];
  const EDGES = [["mask2former", "sam3"], ["ritm", "sam"], ["simpleclick", "sam"], ["mae", "sam"], ["vitdet", "sam"], ["sam", "sam2"], ["xmem", "sam2"], ["hiera", "sam2"], ["sam2", "sam3"], ["sam", "sam3d"], ["sam", "mobilesam"], ["sam", "efficientsam"], ["sam", "hqsam"], ["sam", "medsam"], ["sam", "samadapter"], ["groundingdino", "groundedsam"], ["sam", "groundedsam"], ["xmem", "trackanything"], ["sam", "trackanything"], ["sam2", "edgetam"], ["sam2", "sam2long"], ["sam2", "samurai"], ["sam2", "medsam2"], ["sam3", "efficientsam3"], ["sam", "saa"], ["mobilesam", "kairosad"], ["sam", "boxes2pixels"]];
  const RANGES = [[2014.6, 2026.6], [2014.6, 2023.4], [2023.1, 2026.6]];
  const el = mkCanvas(ctx.stage), S = { step: 0, u: 1, from: RANGES[0], hover: null };
  const yr = d => { const [y, m, dd] = d.split('-').map(Number); return y + (m - 1) / 12 + (dd - 1) / 365; };
  const all = []; LANES.forEach((L, li) => L.items.forEach(it => all.push({ ...it, li, t: yr(it.d) })));
  const byK = Object.fromEntries(all.map(a => [a.k, a]));
  let cancel = () => {}, pos = [];
  const cv = PE.canvas(el, S3ASPECT(ctx.stage), draw);
  function draw(c, W, H) {
    c.clearRect(0, 0, W, H); const narrow = W < 620, bottom = ctx.reserve || 0, left = narrow ? 8 : 190, top = 40, right = 16;
    const r = S.from, X = t => left + (t - r[0]) / (r[1] - r[0]) * (W - left - right), lh = (H - bottom - top - 30) / LANES.length;
    const M = PE.css('--muted'), R = PE.css('--rule');
    for (let y = Math.ceil(r[0]); y <= r[1]; y++) { const x = X(y); c.strokeStyle = R; c.lineWidth = 1; c.beginPath(); c.moveTo(x, top - 6); c.lineTo(x, H - bottom - 26); c.stroke();
      if (!narrow || y % 2 === 0 || r[1] - r[0] < 6) PE.label(c, String(y), x, H - bottom - 10, M, 11, 'center', 400); }
    LANES.forEach((L, li) => { const y = top + (li + .5) * lh; c.strokeStyle = R; c.setLineDash([2, 4]); c.beginPath(); c.moveTo(left, y); c.lineTo(W - right, y); c.stroke(); c.setLineDash([]);
      PE.label(c, L.lane, narrow ? 8 : left - 10, narrow ? y - lh / 2 + 11 : y + 4, li === 4 ? PE.css('--green') : M, narrow ? 10 : 12, narrow ? 'left' : 'right', li === 4 ? 700 : 500); });
    pos = all.map(a => ({ a, x: X(a.t), y: top + (a.li + .5) * lh + (narrow ? 5 : 0) }));
    const P = Object.fromEntries(pos.map(p => [p.a.k, p]));
    c.save(); c.beginPath(); c.rect(left - 6, 0, W - left - right + 12, H); c.clip(); c.globalAlpha = .45 * S.u; EDGES.forEach(([f, t]) => { const p = P[f], q = P[t]; if (!p || !q) return; const vis = x => x >= left - 4 && x <= W - right + 4; if (!vis(p.x) && !vis(q.x)) return;
      c.strokeStyle = t.startsWith('sam') && f.startsWith('sam') ? PE.css('--green') : PE.css('--blue'); c.lineWidth = 1.2; c.beginPath(); c.moveTo(p.x, p.y); c.bezierCurveTo((p.x + q.x) / 2, p.y, (p.x + q.x) / 2, q.y, q.x, q.y); c.stroke(); }); c.restore();
    const rows = {}; // per lane: occupied [x0, x1] intervals per row; rows alternate above / below the line
    pos.forEach(p => { if (p.x < left - 4 || p.x > W - right + 4) return; const spine = p.a.li === 4, hov = S.hover === p.a.k;
      PE.dot(c, p.x, p.y, spine ? 7 : 4.5, spine ? PE.css('--green') : PE.css('--blue'), hov ? 1 : .9); if (hov) PE.ring(c, p.x, p.y, 10, PE.css('--yellow'), 2); });
    pos.slice().sort((a, b) => a.x - b.x).forEach(p => { if (p.x < left - 4 || p.x > W - right + 4) return; const spine = p.a.li === 4, hov = S.hover === p.a.k;
      const show = hov || spine || (!narrow && (S.step !== 0 || p.a.t < 2023.2)); if (!show) return;
      const fs = spine ? 13 : 11; c.font = `${spine ? 700 : 500} ${fs}px system-ui, sans-serif`; const w = c.measureText(p.a.n).width;
      const alignR = p.x + w > W - right, x0 = alignR ? p.x - w : p.x, x1 = alignR ? p.x : p.x + w;
      const R = rows[p.a.li] = rows[p.a.li] || []; let r = 0; const maxR = Math.max(1, Math.floor((lh - 10) / 12));
      while (r < maxR && (R[r] || []).some(([a, b]) => x0 < b + 6 && x1 > a - 6)) r++;
      if (r >= maxR) { if (!hov) return; r = 0; }
      (R[r] = R[r] || []).push([x0, x1]);
      const ly = r % 2 === 0 ? p.y - 9 - 12 * (r >> 1) : p.y + 17 + 12 * (r >> 1);
      PE.label(c, p.a.n, alignR ? p.x : p.x, ly, spine ? PE.css('--green') : PE.css('--fg'), fs, alignR ? 'right' : 'left', spine ? 700 : 500); });
  }
  function readout() { const a = S.hover && byK[S.hover];
    ctx.hud(badge('sourced', a ? `${a.n}: first posted to arXiv on ${a.d}. Lane: ${LANES[a.li].lane}.` : ['The papers behind this page at their first arXiv dates. Green: the SAM line. Lines: "builds on", our reading of each method section. Hover or tap a dot.', 'Before SAM: segmentation models with fixed classes, interactive segmentation, video memory, and the backbones SAM would reuse.', 'After SAM: three years of descendants. Most adapt SAM to a domain, make it faster, or compose it with other models.'][S.step])); }
  function pick(e) { const r = el.getBoundingClientRect(), x = e.clientX - r.left, y = e.clientY - r.top; let best = null, bd = 16;
    pos.forEach(p => { const d = Math.hypot(p.x - x, p.y - y); if (d < bd) { bd = d; best = p.a.k; } }); if (best !== S.hover) { S.hover = best; cv.redraw(); readout(); } }
  const offs = [listen(el, 'pointermove', pick), listen(el, 'click', pick), listen(el, 'pointerleave', () => { S.hover = null; cv.redraw(); readout(); })];
  function zoom(n) { const a = S.from.slice(), b = RANGES[n]; cancel(); cancel = PE.animate(700, u => { S.from = [PE.lerp(a[0], b[0], u), PE.lerp(a[1], b[1], u)]; cv.redraw(); }); }
  readout();
  return { flat: true, step(n) { S.step = n; S.hover = null; readout(); zoom(n); }, resize: () => cv.resize(), destroy: () => { cancel(); offs.forEach(f => f()); cv.destroy(); } };
};
