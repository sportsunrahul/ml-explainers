<script>
(() => {
  const fig = document.getElementById('viz-boxes'), el = document.getElementById('boxes-canvas');
  const out = document.getElementById('boxes-readout'), slider = document.getElementById('rng-boxes-bins'), binsOut = document.getElementById('boxes-bins-val');
  const rowQ = document.getElementById('boxes-qwen'), rowF = document.getElementById('boxes-florence'), rowM = document.getElementById('boxes-molmo');
  const IW = 640, IH = 400;   // the toy image, in pixels
  const OBJ = [
    { name: 'the dog', b: [88, 205, 272, 342] }, { name: 'the ball', b: [309, 296, 353, 340] },
    { name: 'the sign', b: [296, 64, 404, 132] }, { name: 'the tree', b: [460, 55, 610, 300] }];
  const APPLES = [[565, 100], [500, 110], [530, 165]];   // already in Molmo's order: top-down, then left-to-right
  const INIT = { box: [120, 150, 330, 250], pt: [420, 330], B: 1000 };
  const S = { box: INIT.box.slice(), pt: INIT.pt.slice(), B: INIT.B, sel: 'tl', count: 0, verdict: 0 };
  const glow = PE.linkPulses(fig, () => cv.redraw());
  let stop = () => {};

  const binsFromSlider = v => Math.round(4 * Math.pow(250, v / 1000));
  const sliderFromBins = B => Math.round(1000 * Math.log(B / 4) / Math.log(250));
  const q = (v, side, B) => PE.clamp(Math.floor(v / side * B), 0, B - 1);     // bin index 0..B-1 (the toy's convention)
  const dec = (k, side, B) => (k + .5) * side / B;                            // decode at the bin centre
  const norm = b => [Math.min(b[0], b[2]), Math.min(b[1], b[3]), Math.max(b[0], b[2]), Math.max(b[1], b[3])];
  const iou = (a, b) => { const w = Math.max(0, Math.min(a[2], b[2]) - Math.max(a[0], b[0])), h = Math.max(0, Math.min(a[3], b[3]) - Math.max(a[1], b[1]));
    const I = w * h, U = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - I; return U > 0 ? I / U : 0; };
  const refOf = b => { let best = null, bi = .3; OBJ.forEach(o => { const v = iou(b, o.b); if (v > bi) { bi = v; best = o.name; } }); return best || 'this region'; };
  const under = p => { const o = OBJ.find(o => p[0] >= o.b[0] - 6 && p[0] <= o.b[2] + 6 && p[1] >= o.b[1] - 6 && p[1] <= o.b[3] + 6); return o ? o.name : 'this spot'; };
  const decoded = () => { const b = norm(S.box), B = Math.round(S.B); return [dec(q(b[0], IW, B), IW, B), dec(q(b[1], IH, B), IH, B), dec(q(b[2], IW, B), IW, B), dec(q(b[3], IH, B), IH, B)]; };

  const cv = PE.canvas(el, IH / IW, (ctx, W, H) => {
    const Bl = PE.css('--blue'), Y = PE.css('--yellow'), Rd = PE.css('--red'), Br = PE.css('--brown'), F = PE.css('--fg'), M = PE.css('--muted'),
      BG = PE.css('--bg'), P2 = PE.css('--panel'), bg = BG.length === 7 ? BG : '#16181d';
    const s = W / IW, X = x => x * s, Yp = y => y * s, small = W < 560;
    ctx.fillStyle = BG; ctx.fillRect(0, 0, W, H);
    // The scene is pixels, so it is drawn in the vision colour.
    const fillA = u => PE.mix(bg, Bl, u);
    ctx.fillStyle = fillA(.08); ctx.fillRect(0, Yp(300), W, H - Yp(300));
    ctx.fillStyle = fillA(.35); ctx.fillRect(X(522), Yp(170), X(28), Yp(130));
    ctx.fillStyle = fillA(.22); ctx.beginPath(); ctx.arc(X(535), Yp(128), X(76), 0, 7); ctx.fill();
    APPLES.forEach(([x, y]) => PE.dot(ctx, X(x), Yp(y), X(10), fillA(.75)));
    ctx.fillStyle = fillA(.35); ctx.fillRect(X(346), Yp(132), X(6), Yp(168));
    ctx.fillStyle = fillA(.3); ctx.fillRect(X(296), Yp(64), X(108), Yp(68));
    PE.label(ctx, 'OPEN', X(350), Yp(108), F, Math.max(10, 22 * s), 'center');
    ctx.fillStyle = fillA(.55); ctx.beginPath(); ctx.ellipse(X(178), Yp(288), X(82), Yp(40), 0, 0, 7); ctx.fill();
    ctx.beginPath(); ctx.arc(X(248), Yp(240), X(30), 0, 7); ctx.fill();
    [[118, 320], [150, 322], [205, 322], [236, 318]].forEach(([x, y]) => ctx.fillRect(X(x), Yp(y), X(9), Yp(22)));
    ctx.fillStyle = fillA(.85); ctx.beginPath(); ctx.arc(X(331), Yp(318), X(22), 0, 7); ctx.fill();
    // The image frame: the model's input x.
    const gi = glow('input'); ctx.strokeStyle = gi > 0 ? Bl : PE.css('--rule'); ctx.lineWidth = 1 + 4 * gi; ctx.strokeRect(.5, .5, W - 1, H - 1);

    // Bin grid (brown): only drawn when bins are coarse enough to see.
    const B = Math.round(S.B);
    if (B <= 64) { ctx.save(); ctx.strokeStyle = Br; ctx.globalAlpha = .35; ctx.lineWidth = 1; ctx.beginPath();
      for (let k = 1; k < B; k++) { ctx.moveTo(X(k * IW / B), 0); ctx.lineTo(X(k * IW / B), H); ctx.moveTo(0, Yp(k * IH / B)); ctx.lineTo(W, Yp(k * IH / B)); }
      ctx.stroke(); ctx.restore(); }

    const gt = glow('target');
    if (S.count < 1) {
      // The written box (yellow: it is text the model outputs) and what the tokens decode back to (red dashes).
      const b = norm(S.box), d = decoded();
      ctx.save(); ctx.globalAlpha = 1 - S.count;
      ctx.strokeStyle = Rd; ctx.lineWidth = 2; ctx.setLineDash([6, 4]); ctx.strokeRect(X(d[0]), Yp(d[1]), X(d[2] - d[0]), Yp(d[3] - d[1])); ctx.setLineDash([]);
      ctx.strokeStyle = Y; ctx.lineWidth = 2.5 + 3 * gt; ctx.strokeRect(X(b[0]), Yp(b[1]), X(b[2] - b[0]), Yp(b[3] - b[1]));
      [['tl', S.box[0], S.box[1]], ['br', S.box[2], S.box[3]]].forEach(([k, x, y]) => {
        ctx.fillStyle = Y; ctx.fillRect(X(x) - 6, Yp(y) - 6, 12, 12);
        if (S.sel === k) { ctx.strokeStyle = F; ctx.lineWidth = 2; ctx.strokeRect(X(x) - 9, Yp(y) - 9, 18, 18); } });
      const ty = b[1] * s > 20 ? Yp(b[1]) - 7 : Yp(b[3]) + 16;
      PE.label(ctx, refOf(b), X(b[0]), ty, Y, small ? 11 : 12.5);
      // The Molmo point.
      const [px, py] = S.pt; PE.dot(ctx, X(px), Yp(py), 7 + 3 * gt, Y);
      ctx.strokeStyle = BG; ctx.lineWidth = 2; ctx.beginPath(); ctx.arc(X(px), Yp(py), 7 + 3 * gt, 0, 7); ctx.stroke();
      if (S.sel === 'pt') { ctx.strokeStyle = F; ctx.lineWidth = 2; ctx.beginPath(); ctx.arc(X(px), Yp(py), 12, 0, 7); ctx.stroke(); }
      PE.label(ctx, 'point', X(px) + 12, Yp(py) + 4, Y, small ? 11 : 12);
      ctx.restore();
    }
    if (S.count > 0) {
      // Point-then-count: numbered points, drawn one after another.
      APPLES.forEach(([x, y], i) => { const a = PE.clamp(S.count * APPLES.length - i, 0, 1); if (!a) return;
        ctx.save(); ctx.globalAlpha = a; PE.dot(ctx, X(x), Yp(y), 9 + 3 * gt, Y);
        PE.label(ctx, String(i + 1), X(x), Yp(y) + 4, BG, 11, 'center'); ctx.restore(); });
    }
    if (S.verdict > 0) {
      // Molmo's own answer to the prediction: Table 4d (and 4a), CountBenchQA / PixMo-Count.
      ctx.save(); ctx.globalAlpha = S.verdict;
      const pw = Math.min(W - 16, 420), ph = small ? 150 : 176, x0 = (W - pw) / 2, y0 = (H - ph) / 2;
      ctx.fillStyle = P2; ctx.fillRect(x0, y0, pw, ph); ctx.strokeStyle = PE.css('--rule'); ctx.strokeRect(x0, y0, pw, ph);
      const fs = small ? 10.5 : 12;
      PE.label(ctx, "Molmo's ablation (counting accuracy, %)", x0 + 10, y0 + 20, F, fs + 1);
      const rows = [['points as plain-text digits', 89.4, 86.3, Y], ['1000 special point tokens', 85.8, 80.9, Br]];
      const bx = x0 + 10, bw = pw - 20 - (small ? 120 : 132), rh = small ? 16 : 19;
      rows.forEach(([name, a, b, col], r) => {
        const yy = y0 + 34 + r * (rh * 2 + 20);
        PE.label(ctx, name, bx, yy + 8, F, fs);
        [[a, 'CountBenchQA'], [b, 'PixMo-Count']].forEach(([v, tag], j) => {
          const y = yy + 12 + j * rh; ctx.fillStyle = col; ctx.fillRect(bx, y, bw * v / 100 * S.verdict, rh - 5);
          PE.label(ctx, v.toFixed(1) + ' ' + tag, bx + bw * v / 100 + 5, y + rh - 7, M, fs - 1); });
      });
      ctx.restore();
    }
  });

  const f1 = v => v.toFixed(1);
  function render() {
    const b = norm(S.box), B = Math.round(S.B), ref = refOf(b);
    const qd = b.map((v, i) => Math.floor(v / (i % 2 ? IH : IW) * 1000));          // Qwen-VL: [0, 1000), plain digits
    const lk = b.map((v, i) => q(v, i % 2 ? IH : IW, B));
    rowQ.textContent = `<ref>${ref}</ref><box>(${qd[0]},${qd[1]}),(${qd[2]},${qd[3]})</box>`;
    rowF.textContent = `${ref}${lk.map(k => `<loc_${k}>`).join('')}` + (B === 1000 ? '' : `   (${B} bins here; the paper uses 1,000)`);
    if (S.count >= 1) {
      const attrs = APPLES.map(([x, y], i) => `x${i + 1}="${f1(x / IW * 100)}" y${i + 1}="${f1(y / IH * 100)}"`).join(' ');
      rowM.textContent = `<points ${attrs} alt="apples">apples</points> ... so there are ${APPLES.length}.`;
    } else { const nm = under(S.pt); rowM.textContent = `<point x="${f1(S.pt[0] / IW * 100)}" y="${f1(S.pt[1] / IH * 100)}" alt="${nm}">${nm}</point>`; }
    binsOut.textContent = String(B); slider.value = sliderFromBins(B);
    const d = decoded(), err = Math.max(...d.map((v, i) => Math.abs(v - b[i]))), bound = IW / (2 * B);
    if (S.verdict >= 1) out.textContent = 'Molmo tried both on the same pointing data: coordinates written as plain-text digits beat 1000 new special point tokens on both counting benchmarks.';
    else if (S.count >= 1) out.textContent = `Point, then count: Molmo writes one numbered point per apple, top to bottom and left to right, so the count (${APPLES.length}) is just the last number it wrote.`;
    else out.textContent = `With ${B} bins, the box around "${ref}" decodes back (red dashes) at most ${err.toFixed(1)} px from where it was drawn; the worst case for this ${IW}-pixel-wide image is ${bound.toFixed(2)} px.`;
  }
  const redraw = () => { cv.redraw(); render(); };

  function tween(to, ms = 700) {
    stop(); const from = { box: S.box.slice(), pt: S.pt.slice(), B: S.B, count: S.count, verdict: S.verdict };
    const lB0 = Math.log(from.B), lB1 = Math.log(to.B ?? from.B);
    stop = PE.animate(ms, u => {
      if (to.box) S.box = from.box.map((v, i) => PE.lerp(v, to.box[i], u));
      if (to.pt) S.pt = from.pt.map((v, i) => PE.lerp(v, to.pt[i], u));
      if (to.B) S.B = Math.exp(PE.lerp(lB0, lB1, u));
      if (to.count !== undefined) S.count = PE.lerp(from.count, to.count, u);
      if (to.verdict !== undefined) S.verdict = PE.lerp(from.verdict, to.verdict, u);
      redraw(); }, () => { if (to.B) S.B = to.B; redraw(); });
  }
  const on = (id, f) => document.getElementById(id).addEventListener('click', f);
  on('btn-boxes-dog', () => tween({ box: OBJ[0].b.slice(), count: 0, verdict: 0 }));
  on('btn-boxes-sign', () => tween({ box: OBJ[2].b.slice(), count: 0, verdict: 0 }));
  on('btn-boxes-point', () => tween({ pt: [331, 318], count: 0, verdict: 0 }));
  on('btn-boxes-count', () => tween({ count: 1, verdict: 0 }, 1200));
  on('btn-boxes-coarse', () => tween({ B: 8, verdict: 0 }));
  on('btn-boxes-fine', () => tween({ B: 1000, verdict: 0 }));
  on('btn-boxes-verdict', () => tween({ verdict: 1 }, 800));
  on('btn-boxes-reset', () => { stop(); S.box = INIT.box.slice(); S.pt = INIT.pt.slice(); S.B = INIT.B; S.count = 0; S.verdict = 0; S.sel = 'tl'; redraw(); });
  slider.addEventListener('input', () => { stop(); S.B = binsFromSlider(+slider.value); redraw(); });

  // Drag a corner, the box body, or the point; or focus the canvas, Enter picks a handle, arrows move it.
  const local = e => { const r = el.getBoundingClientRect(), s = r.width / IW; return [(e.clientX - r.left) / s, (e.clientY - r.top) / s, s]; };
  let drag = null;
  el.addEventListener('pointerdown', e => {
    const [x, y, s] = local(e), rr = 16 / s, d2 = (a, b) => (a - x) ** 2 + (b - y) ** 2, b = norm(S.box);
    if (S.count > 0 || S.verdict > 0) { stop(); S.count = 0; S.verdict = 0; }
    if (d2(S.box[0], S.box[1]) < rr * rr) drag = { k: 'tl' };
    else if (d2(S.box[2], S.box[3]) < rr * rr) drag = { k: 'br' };
    else if (d2(S.pt[0], S.pt[1]) < rr * rr) drag = { k: 'pt' };
    else if (x > b[0] && x < b[2] && y > b[1] && y < b[3]) drag = { k: 'body', x, y, b0: S.box.slice() };
    else return;
    stop(); if (drag.k !== 'body') S.sel = drag.k; el.setPointerCapture(e.pointerId); redraw();
  });
  el.addEventListener('pointermove', e => { if (!drag) return; const [x0, y0] = local(e), x = PE.clamp(x0, 0, IW - 1), y = PE.clamp(y0, 0, IH - 1);
    if (drag.k === 'tl') { S.box[0] = x; S.box[1] = y; } else if (drag.k === 'br') { S.box[2] = x; S.box[3] = y; }
    else if (drag.k === 'pt') S.pt = [x, y];
    else { const w = drag.b0[2] - drag.b0[0], h = drag.b0[3] - drag.b0[1], nx = PE.clamp(drag.b0[0] + x0 - drag.x, Math.max(0, -w), IW - 1 - Math.max(0, w)),
      ny = PE.clamp(drag.b0[1] + y0 - drag.y, Math.max(0, -h), IH - 1 - Math.max(0, h)); S.box = [nx, ny, nx + w, ny + h]; }
    redraw(); });
  el.addEventListener('pointerup', () => { drag = null; });
  el.addEventListener('keydown', e => {
    if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); S.sel = { tl: 'br', br: 'pt', pt: 'tl' }[S.sel]; redraw(); return; }
    const dv = { ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowUp: [0, -1], ArrowDown: [0, 1] }[e.key]; if (!dv) return;
    e.preventDefault(); stop(); S.count = 0; S.verdict = 0; const st = e.shiftKey ? 1 : 4;
    const mv = (xi, yi, arr) => { arr[xi] = PE.clamp(arr[xi] + dv[0] * st, 0, IW - 1); arr[yi] = PE.clamp(arr[yi] + dv[1] * st, 0, IH - 1); };
    if (S.sel === 'tl') mv(0, 1, S.box); else if (S.sel === 'br') mv(2, 3, S.box); else mv(0, 1, S.pt);
    redraw();
  });
  render();
})();
</script>
