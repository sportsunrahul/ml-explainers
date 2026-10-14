/* ch-efficient. Tabs: 0 where the time goes (MobileSAM's own timings, one GPU); 1 distil the encoder (toy: a small
   student can only make an affine map, so it matches the teacher's embeddings up to a residual); 2 one table, one
   setup (EfficientViT-SAM Table 1: throughput on one A100 with TensorRT fp16, and COCO mAP with ViTDet boxes; every row of the table). */
SCENES.efficient = ctx => {
  const el = mkCanvas(ctx.stage), S = { step: 0, u: 1, a: 0 };
  const TIME = [['SAM ViT-H encoder', 452, '~452ms'], ['MobileSAM encoder', 8, '~8ms'], ['mask decoder', 4, '4ms']];
  const T1 = [['SAM-ViT-H', 11, '46.5'], ['MobileSAM', 278, '38.7'], ['EfficientSAM', 183, '44.4'], ['EdgeSAM', 449, '42.1'], ['EfficientViT-SAM-L0', 762, '45.7'], ['EfficientViT-SAM-L1', 638, '46.2'], ['EfficientViT-SAM-L2', 538, '46.6'], ['EfficientViT-SAM-XL0', 278, '47.5'], ['EfficientViT-SAM-XL1', 182, '47.8']];
  // distillation toy: inputs x on a ring; teacher = a curved map; student = best affine map (closed form), blended in by S.a
  const r = PE.rng(4), N = 36, X = [], Tt = [], S0 = [];
  for (let i = 0; i < N; i++) { const t = i / N * Math.PI * 2, x = [Math.cos(t), Math.sin(t)]; X.push(x); Tt.push([x[0] * 1.1 + .35 * Math.sin(2 * t), x[1] * .8 + .3 * Math.cos(3 * t)]); S0.push([(r() - .5) * .6, (r() - .5) * .6]); }
  const fit = (() => { // least squares T ~ A x + b
    const m = [0, 0], tm = [0, 0]; X.forEach((x, i) => { m[0] += x[0] / N; m[1] += x[1] / N; tm[0] += Tt[i][0] / N; tm[1] += Tt[i][1] / N; });
    let sxx = 0, sxy = 0, syy = 0; const c = [[0, 0], [0, 0]];
    X.forEach((x, i) => { const dx = x[0] - m[0], dy = x[1] - m[1]; sxx += dx * dx; sxy += dx * dy; syy += dy * dy; for (let k = 0; k < 2; k++) { c[k][0] += (Tt[i][k] - tm[k]) * dx; c[k][1] += (Tt[i][k] - tm[k]) * dy; } });
    const det = sxx * syy - sxy * sxy, inv = [[syy / det, -sxy / det], [-sxy / det, sxx / det]];
    const A = c.map(row => [row[0] * inv[0][0] + row[1] * inv[1][0], row[0] * inv[0][1] + row[1] * inv[1][1]]);
    return X.map(x => [0, 1].map(k => A[k][0] * (x[0] - m[0]) + A[k][1] * (x[1] - m[1]) + tm[k]));
  })();
  const stud = () => S0.map((p, i) => [PE.lerp(p[0], fit[i][0], S.a), PE.lerp(p[1], fit[i][1], S.a)]);
  const mse = () => { const s = stud(); return s.reduce((acc, p, i) => acc + (p[0] - Tt[i][0]) ** 2 + (p[1] - Tt[i][1]) ** 2, 0) / N; };
  let cancel = () => {};
  const cv = PE.canvas(el, S3ASPECT(ctx.stage), draw);
  const trainBtn = ctx.q('[data-train]');
  const offs = [listen(trainBtn, 'click', () => { if (S.step !== 1) { const t = ctx.q('[role="tab"][data-step="1"]'); if (t) t.click(); } const a0 = S.a, a1 = Math.min(1, a0 + .34 + (a0 > .6 ? 1 : 0)); cancel(); cancel = PE.animate(700, u => { S.a = PE.lerp(a0, a1, u); cv.redraw(); }, readout); })];
  function draw(c, W, H) {
    c.clearRect(0, 0, W, H); const narrow = W < 620, bottom = ctx.reserve || 0, u = S.u, M = PE.css('--muted'), F = PE.css('--fg'), B = PE.css('--blue'), G = PE.css('--green');
    if (S.step === 0) {
      const lab = narrow ? 130 : 180, b = { x: 12 + lab, y: 70, w: W - lab - 90, h: H - bottom - 110 }, bh = Math.min(40, b.h / 4);
      PE.label(c, 'time per image, as MobileSAM measured it on one GPU (ms)', 12, 50, M, 12, 'left', 400);
      TIME.forEach(([n, v, s], i) => { const y = b.y + i * (bh + 18), w = b.w * v / 452 * u; PE.label(c, n, b.x - 10, y + bh / 2 + 4, F, narrow ? 11 : 13, 'right', 500);
        c.fillStyle = i === 0 ? B : i === 1 ? G : PE.css('--yellow'); c.globalAlpha = .85; c.fillRect(b.x, y, Math.max(2, w), bh); c.globalAlpha = 1; PE.label(c, s, b.x + Math.max(2, w) + 6, y + bh / 2 + 4, M, 12, 'left', 600); });
    } else if (S.step === 1) {
      const sc = Math.min(W, H - bottom) / 4.2, cx = W / 2, cy = (H - bottom) / 2 + 10, P = p => [cx + p[0] * sc, cy - p[1] * sc], s = stud();
      Tt.forEach((t, i) => { const [x, y] = P(t), [x2, y2] = P(s[i]); c.strokeStyle = PE.css('--red'); c.globalAlpha = .6; c.lineWidth = 1; c.beginPath(); c.moveTo(x, y); c.lineTo(x2, y2); c.stroke(); c.globalAlpha = 1; PE.ring(c, x, y, 4.5, B, 2); PE.dot(c, x2, y2, 3.5, G); });
      PE.label(c, 'blue rings: the big encoder\'s embeddings (teacher)', 12, 50, B, 12, 'left', 500); PE.label(c, 'green dots: the small encoder\'s (student); red: the MSE it minimises', 12, 66, G, 12, 'left', 500);
    } else {
      const lab = narrow ? 128 : 190, b = { x: 12 + lab, y: 66, w: W - lab - (narrow ? 70 : 120), h: H - bottom - 100 }, bh = Math.min(30, b.h / T1.length - 8), lx = v => Math.log10(v) / Math.log10(1000);
      if (narrow) { PE.label(c, 'images per second, one A100 (TensorRT, fp16),', 12, 38, M, 10, 'left', 400); PE.label(c, 'log scale; COCO mAP (ViTDet boxes)', 12, 51, M, 10, 'left', 400); }
      else PE.label(c, 'images per second on one A100 (TensorRT, fp16), log scale; COCO mAP with ViTDet boxes', 12, 50, M, 12, 'left', 400);
      T1.forEach(([n, v, ap], i) => { const y = b.y + i * (bh + 8), w = b.w * lx(v) * u; PE.label(c, narrow ? n.replace('EfficientViT-SAM-', 'EffViT-SAM-') : n, b.x - 10, y + bh / 2 + 4, F, narrow ? 10 : 12, 'right', 500);
        c.fillStyle = i === 0 ? B : n.startsWith('EfficientViT') ? G : PE.css('--muted'); c.globalAlpha = .8; c.fillRect(b.x, y, w, bh); c.globalAlpha = 1;
        if (narrow && ap) { PE.label(c, String(v), b.x + w + 6, y + bh / 2 - 1, M, 11, 'left', 600); PE.label(c, `mAP ${ap}`, b.x + w + 6, y + bh / 2 + 11, M, 10, 'left', 500); }
        else PE.label(c, v + (ap ? `   mAP ${ap}` : ''), b.x + w + 6, y + bh / 2 + 4, M, 11, 'left', 600); });
    }
  }
  const text = ['On one GPU, SAM\'s ViT-H encoder took about 452 ms per image and the mask decoder 4 ms. Almost all of the cost is the encoder, so that is what the efficient variants replace.',
    null, 'EfficientViT-SAM\'s Table 1 times several fast SAMs on the same hardware. SAM-ViT-H runs 11 images per second; EfficientViT-SAM-L2 runs 538 at a COCO mAP of 46.6 against SAM\'s 46.5. All nine rows of the table are shown.'];
  function readout() { if (S.step === 1) ctx.hud(badge('toy', `A small student can only learn a simpler map. Training pulls its embeddings toward the teacher's until the error stops falling: MSE ${mse().toFixed(3)}. Press "Train the student".`));
    else ctx.hud(badge('sourced', text[S.step])); }
  function animateIn() { cancel(); cancel = PE.animate(700, t => { S.u = t; cv.redraw(); }); }
  readout();
  return { flat: true, step(n) { S.step = n; readout(); animateIn(); }, resize: () => cv.resize(), destroy: () => { cancel(); offs.forEach(f => f()); cv.destroy(); } };
};
