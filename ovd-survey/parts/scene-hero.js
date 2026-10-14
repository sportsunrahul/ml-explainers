<script>
/* SceneHero(el, { aspect, bottom }): the page's core idea in one motion, shared by the hero and the atlas overview.
   A closed-set detector boxes four objects, its fixed class rows name three and call the forklift
   background; the rows become text embeddings, "forklift" is typed, and the box is claimed; then the tap
   strip draws: where text enters each of the five models. Toy scores. */
const SceneHero = (el, o = {}) => {
  const ROWS = ['person', 'pallet', 'traffic cone', 'car', 'chair', 'dog'];
  const SCORE = { person: '0.93', pallet: '0.88', 'traffic cone': '0.90' };
  const T_END = 7600;
  let t = T_END, raf = 0;
  const ph = (a, b) => PE.clamp((t - a) / (b - a), 0, 1);
  const ease = u => u < .5 ? 2 * u * u : 1 - Math.pow(-2 * u + 2, 2) / 2;
  function draw(ctx, w, h) {
    const k = DK.C(); ctx.clearRect(0, 0, w, h); h -= o.bottom || 0;
    const L = layout(w, h); if (!L) return;
    const { narrow, sx, sy, sw, sh, fs, pos, rowH } = L;
    // scene
    DK.drawScene(ctx, DK.WAREHOUSE, sx, sy, sw, sh, k, ease(ph(0, 400)));
    // classifier block
    const uBlock = ease(ph(1600, 2200)), uText = ease(ph(3400, 4200));
    if (uBlock > 0) {
      ctx.save(); ctx.globalAlpha = uBlock; const ox = (1 - uBlock) * 30;
      ctx.font = `600 ${fs}px system-ui, sans-serif`; ctx.textBaseline = 'middle';
      ctx.fillStyle = PE.mix(k.mu, k.Y, uText);
      ctx.fillText(uText < .5 ? 'classifier: one learned row per class' : 'text encoder: one vector per name', L.tx + ox, L.ty);
      ctx.font = `400 ${fs - 1}px system-ui, sans-serif`; ctx.fillStyle = k.mu;
      ctx.fillText(uText < .5 ? 'fixed at training time' : 'any name, at any time', L.tx + ox, L.ty + 18);
      ROWS.forEach((name, i) => {
        const p = pos(i), col = DK.mixh(DK.mixh(k.bg, k.fg, .45), k.Y, uText);
        for (let j = 0; j < 8; j++) { const hh = 4 + 9 * Math.abs(Math.sin(i * 2.1 + j * 1.3)); ctx.fillStyle = col; ctx.fillRect(p.x + ox + j * 6, p.y + 6 - hh, 4, hh); }
        ctx.font = `500 ${fs}px system-ui, sans-serif`; ctx.fillStyle = uText > .5 ? k.Y : k.fg; ctx.fillText(name, p.x + ox + 56, p.y);
      });
      ctx.restore();
    }
    // typed prompt and its new row
    const nType = Math.floor(ph(4200, 5000) * 8.99), uDrop = ease(ph(5000, 5400));
    if (t >= 4200) {
      const p = pos(ROWS.length), word = 'forklift'.slice(0, nType), y = p.y;
      ctx.save(); ctx.font = `500 ${fs}px system-ui, sans-serif`; ctx.textBaseline = 'middle';
      const pw = Math.max(ctx.measureText('forklift').width + 26, 70);
      DK.rr(ctx, p.x + 52, y - 12, pw, 24, 12); ctx.strokeStyle = k.Y; ctx.lineWidth = 1.2; ctx.stroke();
      ctx.fillStyle = k.Y; ctx.fillText(word, p.x + 64, y);
      if (uDrop < 1 && t < 5000) ctx.fillRect(p.x + 64 + ctx.measureText(word).width + 2, y - 7, 1.5, 14);
      if (uDrop > 0) for (let j = 0; j < 8; j++) { const hh = 4 + 9 * Math.abs(Math.sin(9 * 2.1 + j * 1.3)); ctx.globalAlpha = uDrop; ctx.fillStyle = k.Y; ctx.fillRect(p.x + j * 6, y + 6 - hh, 4, hh); }
      ctx.restore();
    }
    // boxes, links, tags
    DK.WAREHOUSE.filter(o => o.name).forEach((o, i) => {
      const a = ease(ph(400 + i * 280, 700 + i * 280)); if (a <= 0) return;
      const [x, y, bw, bh] = DK.boxOf(o, sx, sy, sw, sh);
      const fork = o.row < 0, uLink = ease(ph(2200 + i * 180, 2700 + i * 180));
      let col = k.B, dash = null;
      if (!fork && uLink > 0) col = PE.mix(k.B, k.G, uLink);
      if (fork && t > 3000 && t < 5400) { col = PE.mix(k.B, k.Br, ease(ph(3000, 3300))); dash = [5, 4]; }
      const uClaim = ease(ph(5400, 5800));
      if (fork && uClaim > 0) { col = PE.mix(k.Br, k.G, uClaim); dash = uClaim > .5 ? null : [5, 4]; }
      DK.brackets(ctx, x, y, bw, bh, col, { alpha: a, lw: 2.2, dash });
      // link from the box to its row
      if (uLink > 0 && uBlock > 0) {
        const tp = pos(fork ? (uClaim > 0 ? ROWS.length : 3) : o.row);
        const lc = fork ? (uClaim > 0 ? k.G : k.Br) : k.G, la = fork && uClaim <= 0 ? .45 * uLink * (1 - ease(ph(3300, 3600))) : (fork ? uClaim : uLink) * .7;
        if (la > 0) { ctx.save(); ctx.globalAlpha = la; ctx.strokeStyle = lc; ctx.lineWidth = 1.2; ctx.setLineDash([3, 4]);
          ctx.beginPath(); if (narrow) { ctx.moveTo(x + bw / 2, y + bh); ctx.lineTo(tp.x + 20, tp.y - 9); } else { ctx.moveTo(x + bw, y + bh * .3); ctx.lineTo(tp.x - 4, tp.y); }
          ctx.stroke(); ctx.restore(); }
      }
      if (!fork && uLink > .6) DK.tag(ctx, x, y, o.name, SCORE[o.name], k.G, k, { size: narrow ? 10 : 12, alpha: (uLink - .6) / .4 });
      if (fork && t > 3200 && uClaim <= 0) DK.tag(ctx, x, y, 'background', null, k.Br, k, { size: narrow ? 10 : 12, nameCol: k.Br, alpha: ease(ph(3200, 3500)) });
      if (fork && uClaim > 0) DK.tag(ctx, x, y, 'forklift', '0.81', k.G, k, { size: narrow ? 10 : 12, alpha: uClaim });
    });
    // the strip
    const uS = ph(5800, T_END);
    if (uS > 0) { ctx.save(); ctx.translate(0, h - L.stripH); DK.strip(ctx, w, L.stripH, k, 'all', uS); ctx.restore(); }
  }
  // Desktop: scene left, classifier column right, strip below. Phone: scene, then the rows in two columns, then the strip.
  const STRIP_D = 150, STRIP_N = 158, ROW_N = 24;
  const aspect = w => w < 640 ? (30 + .56 * w + 16 + 44 + 4 * ROW_N + 18 + STRIP_N) / w : .5;
  function layout(w, h) {
    if (w < 120 || h < 200) return null;
    const narrow = w < 640, fs = narrow ? 11 : 13;
    if (!narrow) {
      const stripH = STRIP_D, top = h - stripH - 18, sx = 0, sy = 14, sw = w * .58, sh = top - 22, bx = sw + w * .05;
      const rowH = Math.min(30, (sh - 70) / (ROWS.length + 1.4)), r0 = sy + 52;
      return { narrow, fs, stripH, sx, sy, sw, sh, rowH, tx: bx, ty: sy + 14,
        pos: i => ({ x: bx, y: i < ROWS.length ? r0 + i * rowH : r0 + i * rowH + rowH * .4 }) };
    }
    const stripH = STRIP_N, sx = 0, sy = 30, sw = w, sh = .56 * w, by = sy + sh + 16, colW = w / 2;
    return { narrow, fs, stripH, sx, sy, sw, sh, rowH: ROW_N, tx: 4, ty: by + 6,
      pos: i => i < ROWS.length ? { x: 4 + Math.floor(i / 3) * colW, y: by + 44 + (i % 3) * ROW_N } : { x: 4, y: by + 44 + 3 * ROW_N + 6 } };
  }
  const cv = PE.canvas(el, o.aspect || aspect, draw);
  function play() {
    cancelAnimationFrame(raf);
    if (PE.instant()) { t = T_END; cv.redraw(); return; }
    const t0 = performance.now();
    const f = now => { t = Math.min(T_END, now - t0); cv.redraw(); if (t < T_END) raf = requestAnimationFrame(f); };
    raf = requestAnimationFrame(f);
  }
  return { play, redraw: cv.redraw, destroy: () => { cancelAnimationFrame(raf); cv.destroy && cv.destroy(); } };
};
</script>
