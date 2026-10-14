<script>
(() => {
  const fig = document.getElementById('viz-resampler'), el = document.getElementById('resampler-canvas'), mel = document.getElementById('resampler-mask-canvas');
  const rT = document.getElementById('rng-resampler-frames'), rS = document.getElementById('rng-resampler-side');
  const tOut = document.getElementById('resampler-frames-val'), sOut = document.getElementById('resampler-side-val');
  const out = document.getElementById('resampler-readout');
  const LAT = 64;   // Flamingo Sec. 2.1: a fixed number of visual outputs (64)
  const S = { T: 1, side: 10, shown: 1, mask: 0 };   // shown = frames drawn (tweened); mask 0 = last preceding image, 1 = all previous
  const glow = PE.linkPulses(fig, () => { cv.redraw(); mv.redraw(); });
  let stopA = () => {}, stopM = () => {};
  const fmt = n => n.toLocaleString('en-US');

  const cv = PE.canvas(el, w => w < 560 ? .95 : .42, (ctx, W, H) => {
    const B = PE.css('--blue'), G = PE.css('--green'), Br = PE.css('--brown'), F = PE.css('--fg'), M = PE.css('--muted'), Rule = PE.css('--rule'), BG = PE.css('--bg');
    ctx.fillStyle = BG; ctx.fillRect(0, 0, W, H);
    const st = W < 560, gR = glow('resampler'), gL = glow('latents'), gFt = glow('features');
    // Left: T frames of side x side features, stacked like cards. Only temporal embeddings are added.
    const fx = st ? 16 : 22, fy = st ? 40 : 46, fs = st ? Math.min(120, W * .36) : Math.min(130, H * .45);
    const n = Math.max(1, Math.round(S.shown)), off = Math.min(5, 60 / Math.max(1, n));
    PE.label(ctx, S.T === 1 ? 'one image = one frame' : S.T + ' frames', fx, fy - 18, B, 12);
    for (let k = Math.min(n, 30) - 1; k >= 0; k--) {
      const x = fx + k * off, y = fy + k * off;
      ctx.fillStyle = BG; ctx.fillRect(x, y, fs, fs); ctx.strokeStyle = B; ctx.lineWidth = 1 + 2 * gFt; ctx.strokeRect(x, y, fs, fs);
      if (k === 0) { const c = fs / S.side; ctx.fillStyle = B; ctx.globalAlpha = .55 + .4 * gFt;
        for (let i = 0; i < S.side; i++) for (let j = 0; j < S.side; j++) ctx.fillRect(x + i * c + c * .15, y + j * c + c * .15, c * .7, c * .7);
        ctx.globalAlpha = 1; }
    }
    const lastOff = (Math.min(n, 30) - 1) * off;
    PE.label(ctx, '+ time embedding t1' + (S.T > 1 ? '..t' + S.T : ''), fx, fy + fs + lastOff + 18, B, 11);
    PE.label(ctx, S.side + ' x ' + S.side + ' features per frame', fx, fy + fs + lastOff + 33, M, 11);
    // Middle: the resampler.
    const N = S.T * S.side * S.side;
    const bx = st ? W * .55 : W * .42, bw = st ? W * .4 : W * .2, by = st ? fy : H * .25, bh = st ? 64 : H * .42;
    ctx.strokeStyle = G; ctx.lineWidth = 2 + 2 * gR; ctx.strokeRect(bx, by, bw, bh);
    PE.label(ctx, 'Perceiver', bx + bw / 2, by + bh / 2 - 6, G, 12.5, 'center');
    PE.label(ctx, 'Resampler', bx + bw / 2, by + bh / 2 + 10, G, 12.5, 'center');
    PE.label(ctx, 'keys/values: ' + fmt(N) + ' + 64', bx + bw / 2, by + bh + 16, M, 11, 'center');
    if (!st) PE.arrow(ctx, fx + fs + lastOff + 10, by + bh / 2, bx - 6, by + bh / 2, B, 2, 9);
    else PE.arrow(ctx, fx + fs + 8, fy + fs / 2, bx - 6, by + bh / 2, B, 2, 9);
    // Right (or below): 64 latents, always 8 x 8.
    const cs = st ? Math.min(15, (W - 60) / 16) : Math.min(16, H * .07), gx = st ? W * .55 + (W * .4 - 8 * cs) / 2 : W * .7, gy = st ? by + bh + 62 : H * .5 - 4 * cs;
    PE.arrow(ctx, st ? bx + bw / 2 : bx + bw + 6, st ? by + bh + 24 : by + bh / 2, st ? bx + bw / 2 : gx - 8, st ? gy - 26 : by + bh / 2, G, 2, 9);
    for (let i = 0; i < 8; i++) for (let j = 0; j < 8; j++) { ctx.fillStyle = G; ctx.globalAlpha = .8; ctx.fillRect(gx + i * cs + 1, gy + j * cs + 1, cs - 2, cs - 2); }
    ctx.globalAlpha = 1;
    if (gL > 0) { ctx.strokeStyle = F; ctx.lineWidth = 1 + 2 * gL; ctx.strokeRect(gx - 3, gy - 3, 8 * cs + 6, 8 * cs + 6); }
    PE.label(ctx, '64 visual tokens', gx + 4 * cs, gy - 10, G, 12, 'center');
    PE.label(ctx, 'to every gated xattn layer', gx + 4 * cs, gy + 8 * cs + 16, M, 11, 'center');
    PE.label(ctx, fmt(N) + ' in, 64 out', gx + 4 * cs, gy + 8 * cs + 32, F, 12, 'center');
  });

  // The interleaved sequence and which image each text chunk may cross-attend to.
  const seq = [['t', 'Photos:'], ['i', 1], ['t', 'a dog.'], ['i', 2], ['t', 'a cat.'], ['i', 3], ['t', 'a ...']];
  const textIdx = seq.map((s, k) => s[0] === 't' ? k : -1).filter(k => k >= 0);
  const lastImg = k => { let m = 0; seq.slice(0, k).forEach(s => { if (s[0] === 'i') m = s[1]; }); return m; };
  const mv = PE.canvas(mel, w => w < 560 ? .86 : .42, (ctx, W, H) => {
    const B = PE.css('--blue'), Y = PE.css('--yellow'), G = PE.css('--green'), Br = PE.css('--brown'), F = PE.css('--fg'), M = PE.css('--muted'), Rule = PE.css('--rule'), BG = PE.css('--bg');
    ctx.fillStyle = BG; ctx.fillRect(0, 0, W, H);
    const gM = glow('mask'), st = W < 560;
    const cw = (W - 24) / seq.length, cy = st ? 30 : 34, ch = 26;
    ctx.font = '600 11px system-ui, sans-serif';
    const cx = k => 12 + k * cw;
    seq.forEach(([k, v], i) => { const x = cx(i) + 2, w = cw - 4;
      ctx.strokeStyle = k === 'i' ? B : Y; ctx.lineWidth = 1.5; ctx.strokeRect(x, cy, w, ch);
      PE.label(ctx, k === 'i' ? 'img ' + v : v, x + w / 2, cy + 17, k === 'i' ? B : Y, st ? 10 : 11.5, 'center'); });
    PE.label(ctx, 'interleaved sequence', 12, cy - 12, M, 11);
    // Matrix: rows = text chunks, columns = images. Green = this chunk's cross-attention may see that image.
    const mx = st ? 86 : W * .34, my = cy + ch + (st ? 58 : 66), cell = Math.min(st ? 34 : 38, (H - my - 16) / textIdx.length);
    PE.label(ctx, 'may cross-attend to', mx + 1.5 * cell, my - 26, M, 11, 'center');
    [1, 2, 3].forEach(j => PE.label(ctx, 'img ' + j, mx + (j - .5) * cell, my - 8, B, 10.5, 'center'));
    textIdx.forEach((k, r) => { const y = my + r * cell, li = lastImg(k);
      PE.label(ctx, seq[k][1], mx - 8, y + cell * .62, Y, 11, 'right');
      [1, 2, 3].forEach(j => { const vis0 = j === li ? 1 : 0, vis1 = j <= li ? 1 : 0, v = PE.lerp(vis0, vis1, S.mask);
        ctx.fillStyle = PE.mix(BG, G, .1 + .8 * v);
        ctx.fillRect(mx + (j - 1) * cell + 1, y + 1, cell - 2, cell - 2);
        if (gM > 0 && v > .5) { ctx.strokeStyle = F; ctx.lineWidth = 2 * gM; ctx.strokeRect(mx + (j - 1) * cell + 2, y + 2, cell - 4, cell - 4); } });
      if (li === 0) PE.label(ctx, 'no image yet', mx + 3 * cell + 10, y + cell * .62, M, 10.5); });
    // Arcs from each text chunk to the image(s) it sees, over the chips.
    textIdx.forEach(k => { const li = lastImg(k); [1, 2, 3].forEach(j => { const v = PE.lerp(j === li ? 1 : 0, j <= li ? 1 : 0, S.mask); if (v < .03) return;
      const ik = seq.findIndex(s => s[0] === 'i' && s[1] === j), x1 = cx(k) + cw / 2, x2 = cx(ik) + cw / 2, yb = cy + ch + 3;
      ctx.save(); ctx.globalAlpha = v; ctx.strokeStyle = G; ctx.lineWidth = 2; ctx.beginPath(); ctx.moveTo(x1, yb);
      ctx.quadraticCurveTo((x1 + x2) / 2, yb + 6 + 3 * (k - ik), x2, yb); ctx.stroke(); ctx.restore(); }); });
    PE.label(ctx, S.mask < .5 ? 'Flamingo: only the last preceding image' : 'ablation: all previous images', st ? 12 : W * .34 + 3 * 38 + 110, st ? H - 8 : my + 14, S.mask < .5 ? G : F, 11.5, st ? 'left' : 'left');
  });

  function readout() {
    const N = S.T * S.side * S.side;
    out.textContent = `${S.T} frame${S.T > 1 ? 's' : ''} x ${S.side * S.side} features = ${fmt(N)} visual features in. Each of the 64 latents attends over ${fmt(N + LAT)} keys (the features plus the latents); 64 tokens come out whatever the input size. ` +
      (S.maskTarget < .5 ? 'Masking: each text chunk cross-attends only to the image just before it; text before the first image sees no image.' : 'Masking (ablation): each text chunk cross-attends to every earlier image.');
  }
  function setFrames(T) {
    stopA(); S.T = T; rT.value = T; tOut.textContent = T; const s0 = S.shown;
    stopA = PE.animate(700, u => { S.shown = PE.lerp(s0, T, u); cv.redraw(); }); readout();
  }
  function setMask(m) { stopM(); const m0 = S.mask; stopM = PE.animate(700, u => { S.mask = PE.lerp(m0, m, u); mv.redraw(); }); S.maskTarget = m; readout(); }
  document.getElementById('btn-resampler-image').addEventListener('click', () => setFrames(1));
  document.getElementById('btn-resampler-train').addEventListener('click', () => setFrames(8));
  document.getElementById('btn-resampler-infer').addEventListener('click', () => setFrames(30));
  document.getElementById('btn-resampler-last').addEventListener('click', () => setMask(0));
  document.getElementById('btn-resampler-all').addEventListener('click', () => setMask(1));
  document.getElementById('btn-resampler-reset').addEventListener('click', () => { stopA(); stopM(); S.T = 1; S.shown = 1; S.side = 10; S.mask = 0; S.maskTarget = 0;
    rT.value = 1; rS.value = 10; tOut.textContent = 1; sOut.textContent = 10 * 10; cv.redraw(); mv.redraw(); readout(); });
  rT.addEventListener('input', () => { stopA(); S.T = +rT.value; S.shown = S.T; tOut.textContent = S.T; cv.redraw(); readout(); });
  rS.addEventListener('input', () => { S.side = +rS.value; sOut.textContent = S.side * S.side; cv.redraw(); readout(); });
  S.maskTarget = 0; readout();
})();
</script>
