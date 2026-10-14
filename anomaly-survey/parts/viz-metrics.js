<script>
(() => {
  // ch-benchmark: image ROC, pixel ROC and PRO (to pixel FPR 0.3) on a 16-part toy test set (BENCH_KIT).
  // Checked in build/toy-bench.mjs: find both -> image AUROC 0.922, pixel AUROC 0.886, AU-PRO 0.847;
  // miss the tiny hole -> pixel AUROC 0.882 (barely moves), AU-PRO 0.752; miss the large stain -> pixel AUROC 0.757, AU-PRO 0.751.
  const fig = document.getElementById('viz-metrics'), el = document.getElementById('cv-metrics'), out = fig.querySelector('.readout');
  const slider = document.getElementById('rng-metrics-t'), sOut = document.getElementById('out-metrics-t');
  const { items, featured } = BENCH_KIT.build();
  const base = items.map(it => BENCH_KIT.pixels(it));
  const states = {};
  function state(miss) {
    if (states[miss]) return states[miss];
    const maps = base.slice(); maps[featured] = BENCH_KIT.pixels(items[featured], miss);
    return states[miss] = { maps, g: BENCH_KIT.grade(items, maps) };
  }
  const TLO = .03, THI = 1.8, toT = v => TLO * Math.pow(THI / TLO, v / 100), toV = t => 100 * Math.log(t / TLO) / Math.log(THI / TLO);
  const DEF = { miss: 0, t: .1 };
  const S = { miss: 0, t: .1, curve: null };
  const sub = (pts, n) => { const o = []; for (let i = 0; i < n; i++) o.push(pts[Math.round(i / (n - 1) * (pts.length - 1))]); return o; };
  const NPT = 160;
  S.curve = { pix: sub(state(0).g.pixRoc, NPT), pro: sub(state(0).g.pro, NPT), a: [state(0).g.imgAuroc, state(0).g.pixAuroc, state(0).g.aupro] };
  // ground-truth outlines of the featured part
  const lab = items[featured].img.lab, N = TOY.N;
  const edge = []; for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) { const l = lab[y * N + x]; if (!l) continue;
    if (x === 0 || y === 0 || x === N - 1 || y === N - 1 || lab[y * N + x - 1] !== l || lab[y * N + x + 1] !== l || lab[(y - 1) * N + x] !== l || lab[(y + 1) * N + x] !== l) edge.push([x, y, l]); }
  let heat = null, heatKey = '';
  const glow = PE.linkPulses(fig, () => view.redraw());
  const narrowDims = W => { const s = Math.min(W * .56, 220), cw = (W - 34 - 24 - 18) / 2; return { s, cw, H: 30 + s + 58 + 40 + 62 + cw + 62 + cw + 46 }; };
  const view = PE.canvas(el, w => w < 560 ? narrowDims(w).H / w : .43, (ctx, W, H) => draw(ctx, W, H));

  function layout(W, H) {
    if (W < 560) {
      const { s, cw } = narrowDims(W), part = { x: (W - s) / 2, y: 30, s };
      const strip = { x: 34, y: part.y + s + 58, w: W - 52, h: 40 }, top = strip.y + strip.h + 62;
      return { narrow: true, part, strip, charts: [{ x: 34, y: top, w: cw, h: cw }, { x: 34 + cw + 24, y: top, w: cw, h: cw }, { x: 34, y: top + cw + 62, w: W - 52, h: cw }] };
    }
    const s = Math.min(W * .3, H * .78), part = { x: 18, y: 38, s };
    const x0 = W * .4, wR = W - x0 - 14, strip = { x: x0, y: 40, w: wR, h: 44 };
    const gap = 26, cw = (wR - 2 * gap) / 3, ch = Math.min(cw, H - 170 - 46);
    return { narrow: false, part, strip, charts: [0, 1, 2].map(k => ({ x: x0 + k * (cw + gap), y: 170, w: cw, h: ch })) };
  }
  function chart(ctx, b, pts, xr, title, area, cur, C, hl) {
    const xt = xr[1] === 1 ? [0, .5, 1] : [0, .1, .2, .3];
    const A = PE.axes(ctx, b, xr, [0, 1], { xt, yt: [0, .5, 1], xl: xr[1] === 1 ? 'false positive rate' : 'pixel FPR (to 0.3)', ylo: 26 });
    if (hl) { ctx.save(); ctx.strokeStyle = C.red; ctx.globalAlpha = .5 * hl; ctx.lineWidth = 3; ctx.strokeRect(b.x - 3, b.y - 3, b.w + 6, b.h + 6); ctx.restore(); }
    const vis = []; for (let i = 0; i < pts.length; i++) { const [x, y] = pts[i]; if (x <= xr[1]) { vis.push([x, y]); continue; }
      const [x0, y0] = pts[i - 1]; vis.push([xr[1], y0 + (y - y0) * (xr[1] - x0) / (x - x0)]); break; }
    ctx.save(); ctx.beginPath(); ctx.moveTo(A.X(xr[0]), A.Y(0)); vis.forEach(([x, y]) => ctx.lineTo(A.X(x), A.Y(y)));
    ctx.lineTo(A.X(vis.at(-1)[0]), A.Y(0)); ctx.closePath(); ctx.fillStyle = C.red; ctx.globalAlpha = .14; ctx.fill(); ctx.restore();
    PE.polyline(ctx, vis.map(([x, y]) => [A.X(x), A.Y(y)]), C.red, 2);
    if (cur && cur[0] <= xr[1]) PE.dot(ctx, A.X(cur[0]), A.Y(cur[1]), 4.5, C.yellow);
    PE.label(ctx, title, b.x, b.y - 22, C.fg, 12);
    PE.label(ctx, area, b.x, b.y - 8, C.red, 12);
  }
  function draw(ctx, W, H) {
    if (!W) return;
    const L = layout(W, H), C = { blue: PE.css('--blue'), red: PE.css('--red'), yellow: PE.css('--yellow'), muted: PE.css('--muted'), rule: PE.css('--rule'), fg: PE.css('--fg') };
    ctx.clearRect(0, 0, W, H);
    const st = state(S.miss), a = BENCH_KIT.at(items, st.maps, S.t);
    // featured part: red where the pixel score is over the threshold; dashed outlines = ground-truth regions
    const hk = S.miss + ':' + S.t.toFixed(4) + C.red;
    if (hk !== heatKey) { heat = TOY.heatBitmap(st.maps[featured].map, S.t * .999, S.t * 2.2, C.red); heatKey = hk; }
    TOY.draw(ctx, items[featured].img, L.part.x, L.part.y, L.part.s, { heat, heatAlpha: .85 + .15 * glow('pred') });
    const k = L.part.s / N, gr = glow('region');
    ctx.save(); ctx.fillStyle = C.fg; edge.forEach(([x, y], i) => { if ((x + y) % 3 === 0 || gr > .05) { ctx.globalAlpha = .9; ctx.fillRect(L.part.x + x * k, L.part.y + y * k, k * (1 + gr), k * (1 + gr)); } }); ctx.restore();
    PE.label(ctx, L.narrow ? 'featured part' : 'featured part: a large stain and a tiny hole', L.part.x, L.part.y - 14, C.muted, 12);
    PE.label(ctx, 'dotted: true defect regions', L.part.x, L.part.y + L.part.s + 20, C.fg, 11);
    PE.label(ctx, 'red: pixels over the threshold', L.part.x, L.part.y + L.part.s + 35, C.red, 11);
    // image-level strip: each test part is one dot at its image score (log axis)
    const sx = v => L.strip.x + Math.log(v / TLO) / Math.log(THI / TLO) * L.strip.w, sy = L.strip.y + L.strip.h / 2, gi = glow('image') + glow('pair');
    ctx.save(); ctx.strokeStyle = C.rule; ctx.beginPath(); ctx.moveTo(L.strip.x, sy); ctx.lineTo(L.strip.x + L.strip.w, sy); ctx.stroke(); ctx.restore();
    st.g.imgS.forEach((v, i) => { const good = items[i].good, x = sx(PE.clamp(v, TLO, THI)), y = sy + (good ? -9 : 9) + ((i * 7) % 5 - 2) * 1.5;
      PE.dot(ctx, x, y, 4 + gi, good ? C.blue : C.red, .9);
      if (i === featured) { ctx.save(); ctx.strokeStyle = C.fg; ctx.lineWidth = 1.2; ctx.beginPath(); ctx.arc(x, y, 7, 0, 7); ctx.stroke(); ctx.restore(); } });
    const tx = sx(S.t); ctx.save(); ctx.strokeStyle = C.yellow; ctx.lineWidth = 2; ctx.beginPath(); ctx.moveTo(tx, L.strip.y - 6); ctx.lineTo(tx, L.strip.y + L.strip.h + 6); ctx.stroke(); ctx.restore();
    PE.label(ctx, L.narrow ? 'image scores, 16 parts (log scale)' : 'image scores of 16 test parts (log scale): blue good, red defective', L.strip.x, L.strip.y - 12, C.muted, 11);
    PE.label(ctx, `threshold ${S.t.toFixed(2)}`, PE.clamp(tx + 6, L.strip.x, L.strip.x + L.strip.w - 80), L.strip.y + L.strip.h + 18, C.yellow, 11);
    // curves
    const [c0, c1, c2] = L.charts, A = S.curve.a;
    chart(ctx, c0, st.g.imgRoc, [0, 1], 'Image ROC', `AUROC ${A[0].toFixed(2)}`, [a.imgFpr, a.imgTpr], C, glow('image'));
    chart(ctx, c1, S.curve.pix, [0, 1], 'Pixel ROC', `AUROC ${A[1].toFixed(2)}`, [a.fpr, a.tpr], C, glow('pixel'));
    chart(ctx, c2, S.curve.pro, [0, .3], 'PRO curve', `AU-PRO ${A[2].toFixed(2)}`, [a.fpr, a.pro], C, glow('fpr'));
  }
  function say() {
    const st = state(S.miss), a = BENCH_KIT.at(items, st.maps, S.t), g = st.g, pct = v => (100 * v).toFixed(v < .1 ? 1 : 0) + '%';
    const what = S.miss === 2 ? 'The tiny hole is missed. ' : S.miss === 1 ? 'The large stain is missed. ' : '';
    out.textContent = `${what}Threshold ${S.t.toFixed(2)}. Image level: ${a.rejBad} of 8 defective parts rejected, ${a.rejGood} of 8 good parts rejected. Pixel level: ${pct(a.tpr)} of defect pixels flagged at a ${pct(a.fpr)} false positive rate; mean region overlap (PRO) ${a.pro.toFixed(2)}. Areas: image AUROC ${g.imgAuroc.toFixed(2)}, pixel AUROC ${g.pixAuroc.toFixed(2)}, AU-PRO ${g.aupro.toFixed(2)}.`;
  }
  function setMiss(m) {
    S.miss = m; ['find', 'miss-tiny', 'miss-large'].forEach((id, i) => document.getElementById('btn-metrics-' + id).setAttribute('aria-pressed', [0, 2, 1][i] === m));
    const from = S.curve, g = state(m).g, to = { pix: sub(g.pixRoc, NPT), pro: sub(g.pro, NPT), a: [g.imgAuroc, g.pixAuroc, g.aupro] };
    PE.animate(700, u => { S.curve = { pix: from.pix.map((p, i) => [PE.lerp(p[0], to.pix[i][0], u), PE.lerp(p[1], to.pix[i][1], u)]),
      pro: from.pro.map((p, i) => [PE.lerp(p[0], to.pro[i][0], u), PE.lerp(p[1], to.pro[i][1], u)]), a: from.a.map((v, i) => PE.lerp(v, to.a[i], u)) }; view.redraw(); },
      () => { S.curve = to; view.redraw(); });
    say();
  }
  function setT(t, animate) {
    const a0 = S.t; slider.value = toV(t); sOut.textContent = t.toFixed(2);
    if (!animate) { S.t = t; view.redraw(); say(); return; }
    PE.animate(500, u => { S.t = a0 * Math.pow(t / a0, u); view.redraw(); }, () => { S.t = t; view.redraw(); say(); });
  }
  document.getElementById('btn-metrics-find').addEventListener('click', () => setMiss(0));
  document.getElementById('btn-metrics-miss-tiny').addEventListener('click', () => setMiss(2));
  document.getElementById('btn-metrics-miss-large').addEventListener('click', () => setMiss(1));
  slider.addEventListener('input', () => setT(toT(+slider.value), false));
  document.getElementById('btn-metrics-reset').addEventListener('click', () => { setMiss(DEF.miss); setT(DEF.t, true); });
  slider.value = toV(DEF.t); say();
})();
</script>
