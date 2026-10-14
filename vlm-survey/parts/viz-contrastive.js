<script>
(() => {
  const fig = document.getElementById('viz-contrastive'), el = document.getElementById('contrastive-canvas');
  const slider = document.getElementById('rng-contrastive-tau'), tauOut = document.getElementById('contrastive-tau-val');
  const out = document.getElementById('contrastive-readout');
  const names = ['dog', 'cat', 'car', 'pizza', 'tree'], N = names.length;
  // A photo is matched only when its own caption is strictly closest: argmax would hand an exact tie to the first caption.
  const wins = (row, i) => row.every((s, j) => j === i || row[i] - s > 1e-9);
  // A fixed random start, so every visit (and every cue replay) begins from the same scramble.
  const r0 = PE.rng(65), init = { th: names.map(() => r0() * 2 * Math.PI), ph: names.map(() => r0() * 2 * Math.PI) };
  const S = { th: init.th.slice(), ph: init.ph.slice(), tau: .07, sel: null, zs: 0, trained: false };
  const glow = PE.linkPulses(fig, () => cv.redraw());
  let stop = () => {};
  const tauFromSlider = v => Math.pow(10, v / 100);            // slider -150..0 -> tau 0.03..1
  const sliderFromTau = t => Math.round(100 * Math.log10(t));

  function trainFrames(tau) {
    let th = init.th.slice(), ph = init.ph.slice(); const F = [[th, ph]];
    const lr = .05;
    for (let k = 0; k < 400; k++) { const g = PE.contrast(th, ph, tau);
      th = th.map((v, i) => v - lr * g.gth[i]); ph = ph.map((v, i) => v - lr * g.gph[i]); if (k % 4 === 3) F.push([th, ph]); }
    return F;
  }
  // Zero-shot probe: an image the model never saw. Toy assumption: a trained encoder puts it near the dog photos.
  const probeAngle = () => S.th[0] + .35;

  const layout = (W, H) => {
    const stacked = W < 560;
    if (stacked) { const R = W * .27; return { stacked, cx: W / 2, cy: R + 34, R, mx: 64, my: 2 * R + 118, cell: Math.min(44, (W - 84) / N) }; }
    const R = Math.min(H * .36, W * .19), cx = R + 82, mx = cx + R + 118; return { stacked, cx, cy: H / 2 + 4, R, mx, my: 66, cell: Math.min(46, (W - mx - 16) / N) };
  };

  const cv = PE.canvas(el, w => w < 560 ? 1.8 : .56, (ctx, W, H) => {
    const B = PE.css('--blue'), Y = PE.css('--yellow'), G = PE.css('--green'), Rd = PE.css('--red'), Br = PE.css('--brown'),
      F = PE.css('--fg'), M = PE.css('--muted'), Rule = PE.css('--rule'), BG = PE.css('--bg');
    ctx.fillStyle = BG; ctx.fillRect(0, 0, W, H);
    const Lh = layout(W, H), { cx, cy, R } = Lh;
    const P = a => [cx + R * Math.cos(a), cy - R * Math.sin(a)];
    const res = PE.contrast(S.th, S.ph, S.tau), Sm = res.S;
    // The shared embedding space: every image and every caption is a direction (a point on the circle).
    ctx.strokeStyle = Rule; ctx.lineWidth = 1.5; ctx.beginPath(); ctx.arc(cx, cy, R, 0, 7); ctx.stroke();
    PE.label(ctx, 'shared embedding space', cx, cy + 4, M, 11, 'center');
    const gp = glow('pos'), gn = glow('neg');
    // Each image draws a line to the caption it currently finds most similar: green if right, red if wrong.
    for (let i = 0; i < N; i++) {
      let best = 0; for (let j = 1; j < N; j++) if (Sm[i][j] > Sm[i][best]) best = j;
      const [x1, y1] = P(S.th[i]), [x2, y2] = P(S.ph[best]), ok = best === i && wins(Sm[i], i);
      PE.polyline(ctx, [[x1, y1], [x2, y2]], ok ? G : Rd, 2 + 2 * (ok ? gp : gn), ok ? null : [5, 4], .8);
    }
    const lab = (a, text, color, rr) => { const [x, y] = [cx + rr * Math.cos(a), cy - rr * Math.sin(a)];
      PE.label(ctx, text, x, y + 4, color, Lh.stacked ? 10.5 : 11.5, Math.cos(a) > .3 ? 'left' : Math.cos(a) < -.3 ? 'right' : 'center'); };
    for (let j = 0; j < N; j++) { const [x, y] = P(S.ph[j]), s = 7 + (S.sel === 'p' + j ? 3 : 0);
      ctx.fillStyle = Y; ctx.beginPath(); ctx.moveTo(x, y - s); ctx.lineTo(x + s, y); ctx.lineTo(x, y + s); ctx.lineTo(x - s, y); ctx.closePath(); ctx.fill();
      lab(S.ph[j], '"' + names[j] + '"', Y, R - 26); }
    for (let i = 0; i < N; i++) { const [x, y] = P(S.th[i]); PE.dot(ctx, x, y, 7 + (S.sel === 't' + i ? 3 : 0), B);
      lab(S.th[i], names[i] + ' photo', B, R + 14); }
    if (S.zs > 0) { const a = probeAngle(), [x, y] = P(a); ctx.save(); ctx.globalAlpha = S.zs;
      ctx.strokeStyle = B; ctx.lineWidth = 2.5; ctx.beginPath(); ctx.arc(x, y, 9, 0, 7); ctx.stroke();
      lab(a, 'new photo?', F, R + 14); ctx.restore(); }

    // Right: the N x N table the loss actually sees.
    const { mx, my, cell } = Lh, gs = glow('sim'), gt = glow('tau');
    if (S.zs > 0) {
      // Zero-shot: the class names ARE the classifier. Probabilities over "a photo of a {label}".
      const pr = PE.softmax(S.ph.map(p => Math.cos(probeAngle() - p) / S.tau)), bw = cell * N - 40;
      PE.label(ctx, 'new photo vs "a photo of a {label}"', Lh.stacked ? 8 : mx - 50, my - 30, F, 12);
      pr.forEach((p, j) => { const y = my + j * cell * .78, h = cell * .6;
        PE.label(ctx, names[j], mx - 8, y + h * .7, Y, 11.5, 'right');
        ctx.fillStyle = Rule; ctx.fillRect(mx, y, bw, h);
        ctx.globalAlpha = S.zs; ctx.fillStyle = j === 0 ? G : Br; ctx.fillRect(mx, y, bw * p, h); ctx.globalAlpha = 1;
        PE.label(ctx, (100 * p).toFixed(0) + '%', mx + bw + 6, y + h * .7, M, 11); });
    } else {
      PE.label(ctx, 'similarity: rows photos, columns captions', Lh.stacked ? 8 : mx - 50, my - 30, F, 12);
      for (let i = 0; i < N; i++) {
        const row = PE.softmax(Sm[i].map(s => s / S.tau));
        PE.label(ctx, names[i], mx - 6, my + i * cell + cell * .6, B, 11, 'right');
        for (let j = 0; j < N; j++) {
          const x = mx + j * cell, y = my + i * cell, diag = i === j, p = row[j];
          ctx.fillStyle = PE.mix(BG.length === 7 ? BG : '#16181d', diag ? G : Br, .12 + .88 * p); ctx.fillRect(x + 1, y + 1, cell - 2, cell - 2);
          if (diag) { ctx.strokeStyle = G; ctx.lineWidth = 1.5 + 2.5 * gp; ctx.strokeRect(x + 1.5, y + 1.5, cell - 3, cell - 3); }
          else if (gn > 0) { ctx.strokeStyle = Br; ctx.lineWidth = 2.5 * gn; ctx.strokeRect(x + 1.5, y + 1.5, cell - 3, cell - 3); }
          if (gs > 0) { ctx.strokeStyle = F; ctx.globalAlpha = gs; ctx.lineWidth = 1.5; ctx.strokeRect(x + 3, y + 3, cell - 6, cell - 6); ctx.globalAlpha = 1; }
          PE.label(ctx, Sm[i][j].toFixed(2), x + cell / 2, y + cell * .58, F, cell < 40 ? 9.5 : 10.5, 'center');
        }
      }
      for (let j = 0; j < N; j++) PE.label(ctx, names[j], mx + j * cell + cell / 2, my - 6, Y, 11, 'center');
      // Loss bar: red, shrinking as training succeeds.
      const ly = my + N * cell + 28, bw = cell * N, frac = Math.min(1, res.L / 6);
      PE.label(ctx, 'loss ' + res.L.toFixed(2), mx, ly - 5, Rd, 12);
      ctx.fillStyle = Rule; ctx.fillRect(mx, ly, bw, 12); ctx.fillStyle = Rd; ctx.fillRect(mx, ly, bw * frac, 12);
      PE.label(ctx, 'temperature ' + S.tau.toFixed(2) + '; color = row softmax', Lh.stacked ? 8 : mx - 50, ly + 30, gt > 0 ? F : M, 11 + gt);
    }
  });

  function readout() {
    const res = PE.contrast(S.th, S.ph, S.tau);
    let right = 0; res.S.forEach((row, i) => { if (wins(row, i)) right++; });
    if (S.zs > 0) {
      const pr = PE.softmax(S.ph.map(p => Math.cos(probeAngle() - p) / S.tau));
      out.textContent = `Zero-shot: the new photo is compared with the caption "a photo of a {label}" for each class name. It is called "${names[pr.indexOf(Math.max(...pr))]}" with ${(100 * Math.max(...pr)).toFixed(0)}% of the probability. No classifier was trained for these classes; the text is the classifier.`;
      return;
    }
    out.textContent = `${right} of ${N} photos currently sit closest to their own caption. Loss ${res.L.toFixed(2)} at temperature ${S.tau.toFixed(2)}. ` +
      (right === N && res.L < .1 ? 'With the loss this low the gradient has all but vanished: training slows to a crawl.' : S.trained ? 'Training moved every point at once: matched pairs pull together, mismatched pairs push apart.' : 'Drag any photo or caption around the circle and watch the table and the loss.');
  }
  function tweenTo(th, ph, ms, done) {
    stop(); const a0 = S.th.slice(), b0 = S.ph.slice();
    stop = PE.animate(ms, u => { S.th = a0.map((v, i) => PE.lerp(v, th[i], u)); S.ph = b0.map((v, i) => PE.lerp(v, ph[i], u)); cv.redraw(); readout(); }, done);
  }
  function setTau(t) { S.tau = t; slider.value = sliderFromTau(t); tauOut.textContent = t.toFixed(2); }
  function scramble() { S.zs = 0; S.trained = false; tweenTo(init.th, init.ph, 700); }
  function train(then) {
    stop(); S.zs = 0; S.th = init.th.slice(); S.ph = init.ph.slice(); S.trained = true;
    const Fr = trainFrames(S.tau);
    stop = PE.animate(2600, u => { const f = u * (Fr.length - 1), k = Math.floor(f), w = f - k, A = Fr[k], Bn = Fr[Math.min(k + 1, Fr.length - 1)];
      S.th = A[0].map((v, i) => PE.lerp(v, Bn[0][i], w)); S.ph = A[1].map((v, i) => PE.lerp(v, Bn[1][i], w)); cv.redraw(); readout(); }, then);
  }
  document.getElementById('btn-contrastive-before').addEventListener('click', scramble);
  document.getElementById('btn-contrastive-train').addEventListener('click', () => train());
  document.getElementById('btn-contrastive-hot').addEventListener('click', () => { setTau(1); train(); });
  document.getElementById('btn-contrastive-cold').addEventListener('click', () => { setTau(.07); train(); });
  document.getElementById('btn-contrastive-zeroshot').addEventListener('click', () => {
    stop(); const Fr = trainFrames(S.tau), last = Fr[Fr.length - 1]; S.th = last[0].slice(); S.ph = last[1].slice(); S.trained = true;
    stop = PE.animate(700, u => { S.zs = u; cv.redraw(); readout(); });
  });
  document.getElementById('btn-contrastive-reset').addEventListener('click', () => {
    stop(); setTau(.07); S.th = init.th.slice(); S.ph = init.ph.slice(); S.zs = 0; S.sel = null; S.trained = false; cv.redraw(); readout(); });
  slider.addEventListener('input', () => { S.tau = tauFromSlider(+slider.value); tauOut.textContent = S.tau.toFixed(2); cv.redraw(); readout(); });

  // Drag a point around the circle (pointer), or Tab to the canvas, pick with Up/Down, rotate with Left/Right.
  const keyOrder = names.flatMap((_, i) => ['t' + i, 'p' + i]);
  const angleOf = k => k[0] === 't' ? S.th[+k.slice(1)] : S.ph[+k.slice(1)];
  const setAngle = (k, a) => { if (k[0] === 't') S.th[+k.slice(1)] = a; else S.ph[+k.slice(1)] = a; };
  let dragging = false;
  const local = e => { const b = el.getBoundingClientRect(); return [e.clientX - b.left, e.clientY - b.top, b.width, b.height]; };
  el.addEventListener('pointerdown', e => { const [x, y, W, H] = local(e), L = layout(W, H); let best = null, bd = 18 * 18;
    keyOrder.forEach(k => { const a = angleOf(k), px = L.cx + L.R * Math.cos(a), py = L.cy - L.R * Math.sin(a), d = (px - x) ** 2 + (py - y) ** 2; if (d < bd) { bd = d; best = k; } });
    if (best) { stop(); S.zs = 0; S.sel = best; dragging = true; el.setPointerCapture(e.pointerId); cv.redraw(); } });
  el.addEventListener('pointermove', e => { if (!dragging) return; const [x, y, W, H] = local(e), L = layout(W, H);
    setAngle(S.sel, Math.atan2(L.cy - y, x - L.cx)); cv.redraw(); readout(); });
  el.addEventListener('pointerup', () => { dragging = false; });
  el.addEventListener('keydown', e => {
    if (e.key === 'ArrowUp' || e.key === 'ArrowDown') { e.preventDefault(); const i = keyOrder.indexOf(S.sel);
      S.sel = keyOrder[(i + (e.key === 'ArrowUp' ? 1 : keyOrder.length - 1) + (i < 0 ? 1 : 0)) % keyOrder.length]; cv.redraw(); }
    if ((e.key === 'ArrowLeft' || e.key === 'ArrowRight') && S.sel) { e.preventDefault(); stop(); S.zs = 0;
      setAngle(S.sel, angleOf(S.sel) + (e.key === 'ArrowLeft' ? .08 : -.08)); cv.redraw(); readout(); }
  });
  setTau(.07); readout();
})();
</script>
