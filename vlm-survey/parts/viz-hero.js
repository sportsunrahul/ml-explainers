<script>
(() => {
  const el = document.getElementById('hero-canvas'), btn = document.getElementById('btn-hero-replay');
  const words = ['a', 'dog', 'catching', 'a', 'frisbee'];
  const T_END = 5.2;
  let t = 0, raf = 0;
  const cl = (x, a = 0, b = 1) => Math.max(a, Math.min(b, x));
  const ease = x => { x = cl(x); return x < .5 ? 4 * x * x * x : 1 - Math.pow(-2 * x + 2, 3) / 2; };
  const seg = (a, b) => ease((t - a) / (b - a));
  // Two virtual stages: wide (960 x 365) and narrow (360 x 600); everything is laid out in virtual units.
  const L = wide => wide ? {
    vw: 960, vh: 365, photo: { x: 34, y: 34, s: 176 }, cap: { x: 34, y: 262, fs: 18 },
    row: { x: 288, y: 113, s: 14, g: 4 }, chips: { x: 288, y: 249, h: 26, fs: 13 },
    encI: { x: 612, y: 88, w: 112, h: 64 }, encT: { x: 612, y: 230, w: 112, h: 64 },
    circ: { x: 852, y: 178, r: 88 }, fs: 13, angI: 166, angT: 190,
  } : {
    vw: 360, vh: 620, photo: { x: 18, y: 18, s: 136 }, cap: { x: 170, y: 52, fs: 16 },
    row: { x: 18, y: 214, s: 13, g: 4 }, chips: { x: 18, y: 268, h: 24, fs: 12 },
    encI: { x: 18, y: 336, w: 150, h: 50 }, encT: { x: 192, y: 336, w: 150, h: 50 },
    circ: { x: 180, y: 492, r: 66 }, fs: 12, angI: 99, angT: 81,
  };
  let C = {};
  const tint = (c, a) => PE.mix(C.BG.length === 7 ? C.BG : '#16181d', c.length === 7 ? c : '#58c4dd', a);
  function rr(ctx, x, y, w, h, r, fill, stroke, lw = 1.5) {
    r = Math.min(r, w / 2, h / 2); ctx.beginPath(); ctx.moveTo(x + r, y); ctx.arcTo(x + w, y, x + w, y + h, r); ctx.arcTo(x + w, y + h, x, y + h, r);
    ctx.arcTo(x, y + h, x, y, r); ctx.arcTo(x, y, x + w, y, r); ctx.closePath();
    if (fill) { ctx.fillStyle = fill; ctx.fill(); } if (stroke) { ctx.strokeStyle = stroke; ctx.lineWidth = lw; ctx.stroke(); }
  }
  // The photo, in vision blue only: sky, ground, a dog leaping for a frisbee.
  function photo(ctx, x, y, s) {
    ctx.fillStyle = tint(C.B, .2); ctx.fillRect(x, y, s, s);
    ctx.fillStyle = tint(C.B, .42); ctx.fillRect(x, y + s * .72, s, s * .28);
    ctx.fillStyle = C.B;
    ctx.beginPath(); ctx.ellipse(x + s * .44, y + s * .58, s * .2, s * .1, -.35, 0, 7); ctx.fill();
    ctx.beginPath(); ctx.arc(x + s * .65, y + s * .43, s * .085, 0, 7); ctx.fill();
    ctx.beginPath(); ctx.moveTo(x + s * .7, y + s * .36); ctx.lineTo(x + s * .72, y + s * .27); ctx.lineTo(x + s * .76, y + s * .38); ctx.fill();
    ctx.lineWidth = s * .04; ctx.strokeStyle = C.B; ctx.lineCap = 'round';
    for (const [a, b, c2, d] of [[.32, .64, .22, .76], [.4, .66, .36, .8], [.52, .64, .6, .76], [.56, .6, .66, .7]]) {
      ctx.beginPath(); ctx.moveTo(x + s * a, y + s * b); ctx.lineTo(x + s * c2, y + s * d); ctx.stroke(); }
    ctx.beginPath(); ctx.moveTo(x + s * .26, y + s * .54); ctx.quadraticCurveTo(x + s * .14, y + s * .44, x + s * .18, y + s * .36); ctx.stroke();
    ctx.fillStyle = tint(C.B, .7); ctx.beginPath(); ctx.ellipse(x + s * .82, y + s * .2, s * .1, s * .035, -.25, 0, 7); ctx.fill();
    ctx.fillStyle = tint(C.B, .55); ctx.beginPath(); ctx.arc(x + s * .16, y + s * .16, s * .06, 0, 7); ctx.fill();
  }
  // Draw the first fraction u of a polyline, with an arrowhead at its current end.
  function path(ctx, pts, u, col) {
    const lens = pts.slice(1).map((p, i) => Math.hypot(p[0] - pts[i][0], p[1] - pts[i][1])), tot = lens.reduce((a, b) => a + b, 0);
    let left = u * tot; const out = [pts[0]];
    for (let i = 0; i < lens.length && left > 0; i++) { const f = Math.min(1, left / lens[i]); out.push([PE.lerp(pts[i][0], pts[i + 1][0], f), PE.lerp(pts[i][1], pts[i + 1][1], f)]); left -= lens[i]; }
    if (out.length < 2) return;
    const a = out[out.length - 2], b = out[out.length - 1];
    PE.polyline(ctx, out.slice(0, -1).concat([[PE.lerp(a[0], b[0], .999), PE.lerp(a[1], b[1], .999)]]), col, 1.6, null, .8);
    PE.arrow(ctx, a[0], a[1], b[0], b[1], col, 1.6, 7, .8);
  }
  const font = (ctx, fs, w = 600) => { ctx.font = `${w} ${fs}px system-ui, sans-serif`; };
  const lab = (ctx, s, x, y, col, fs, al = 'left', alpha = 1) => { ctx.save(); ctx.globalAlpha = alpha; PE.label(ctx, s, x, y, col, fs, al); ctx.restore(); };
  function diamond(ctx, x, y, r, col) { ctx.fillStyle = col; ctx.beginPath(); ctx.moveTo(x, y - r); ctx.lineTo(x + r, y); ctx.lineTo(x, y + r); ctx.lineTo(x - r, y); ctx.closePath(); ctx.fill(); }

  const cv = PE.canvas(el, w => w >= 620 ? 365 / 960 : 620 / 360, (ctx, W, H) => {
    C = { B: PE.css('--blue'), Y: PE.css('--yellow'), G: PE.css('--green'), Br: PE.css('--brown'), F: PE.css('--fg'), M: PE.css('--muted'), R: PE.css('--rule'), BG: PE.css('--bg') };
    ctx.clearRect(0, 0, W, H);
    const wide = W >= 620, S = L(wide), k = W / S.vw;
    ctx.save(); ctx.scale(k, k);
    const P = S.photo, n = 4, ts = P.s / n;
    const aIn = seg(0, .6), split = seg(.6, 1.3), gap = 5 * split;
    // A: the photo and its caption.
    const leave = i => seg(1.3 + i * .045, 2.1 + i * .045);
    const ghost = .28 * seg(1.4, 2.4);
    if (ghost > 0) { ctx.save(); ctx.globalAlpha = ghost; photo(ctx, P.x, P.y, P.s); ctx.restore(); }
    // B + C: tiles separate, then fly to the token row and turn into solid tokens.
    for (let i = 0; i < n * n; i++) {
      const r = Math.floor(i / n), c = i % n, u = leave(i);
      const sx = P.x + c * ts + (c - 1.5) * gap, sy = P.y + r * ts + (r - 1.5) * gap;
      const dx = S.row.x + i * (S.row.s + S.row.g), dy = S.row.y;
      const x = PE.lerp(sx, dx, u), y = PE.lerp(sy, dy, u) - Math.sin(Math.PI * u) * 26, s = PE.lerp(ts, S.row.s, u);
      ctx.save(); ctx.globalAlpha = aIn;
      ctx.beginPath(); ctx.rect(x, y, s, s); ctx.clip();
      ctx.translate(x - c * ts * (s / ts), y - r * ts * (s / ts)); ctx.scale(s / ts, s / ts); photo(ctx, 0, 0, P.s);
      ctx.restore();
      const solid = cl((u - .55) / .45);
      if (solid > 0) { ctx.save(); ctx.globalAlpha = solid; rr(ctx, x, y, s, s, 2.5, C.B); ctx.restore(); }
    }
    // Caption: words separate, then become word tokens.
    font(ctx, S.cap.fs); const sp = ctx.measureText(' ').width;
    let cx0 = S.cap.x; const capPos = words.map(w => { const p = cx0; cx0 += ctx.measureText(w).width + sp; return p; });
    font(ctx, S.chips.fs); let kx = S.chips.x; const chipPos = words.map(w => { const wd = ctx.measureText(w).width + 14, p = kx; kx += wd + 5; return [p, wd]; });
    if (ghost > 0) { font(ctx, S.cap.fs); ctx.save(); ctx.globalAlpha = ghost; ctx.fillStyle = C.Y; ctx.fillText(words.join(' '), S.cap.x, S.cap.y); ctx.restore(); }
    words.forEach((w, j) => {
      const u = seg(1.5 + j * .08, 2.3 + j * .08), wsp = 3 * split * j;
      const x = PE.lerp(capPos[j] + wsp, chipPos[j][0] + 7, u), y = PE.lerp(S.cap.y, S.chips.y + S.chips.h * .68, u) - Math.sin(Math.PI * u) * 22;
      const chip = cl((u - .5) / .5);
      ctx.save(); ctx.globalAlpha = aIn;
      if (chip > 0) { ctx.globalAlpha = aIn * chip; rr(ctx, x - 7, y - S.chips.h * .68, chipPos[j][1], S.chips.h, 6, tint(C.Y, .3), C.Y, 1.2); ctx.globalAlpha = aIn; }
      font(ctx, PE.lerp(S.cap.fs, S.chips.fs, u)); ctx.fillStyle = C.Y; ctx.fillText(w, x, y);
      ctx.restore();
    });
    const rowEnd = S.row.x + n * n * (S.row.s + S.row.g) - S.row.g, chipEnd = kx - 5;
    lab(ctx, 'each patch becomes a token', S.row.x, S.row.y - 12, C.M, S.fs, 'left', seg(1.9, 2.5));
    lab(ctx, 'each word becomes a token', S.chips.x, S.chips.y - 12, C.M, S.fs, 'left', seg(2.0, 2.6));
    // D: rows feed their encoders.
    const encA = seg(2.4, 3.0), arr1 = seg(2.5, 3.2);
    const E = [[S.encI, C.B, 'image', 'encoder', rowEnd, S.row.y + S.row.s / 2], [S.encT, C.Y, 'text', 'encoder', chipEnd, S.chips.y + S.chips.h / 2]];
    E.forEach(([e, col, l1, l2, fx, fy]) => {
      let x1, y1, x2, y2;
      let pts;
      if (wide) pts = [[fx + 10, fy], [e.x - 6, e.y + e.h / 2]];
      else if (e === S.encI) pts = [[S.row.x - 3, S.row.y + S.row.s / 2], [8, S.row.y + S.row.s / 2], [8, e.y + e.h / 2], [e.x - 4, e.y + e.h / 2]];
      else pts = [[fx + 6, fy], [e.x + e.w / 2, fy], [e.x + e.w / 2, e.y - 6]];
      if (arr1 > 0) path(ctx, pts, arr1, col);
      ctx.save(); ctx.globalAlpha = encA; rr(ctx, e.x, e.y, e.w, e.h, 9, tint(col, .14), col, 1.6);
      if (wide) { lab(ctx, l1, e.x + e.w / 2, e.y + e.h / 2 - 3, col, S.fs, 'center'); lab(ctx, l2, e.x + e.w / 2, e.y + e.h / 2 + 13, col, S.fs, 'center'); }
      else lab(ctx, l1 + ' ' + l2, e.x + e.w / 2, e.y + e.h / 2 + 4, col, S.fs, 'center');
      ctx.restore();
    });
    // E: one vector each lands in the shared space; the other points are the rest of the batch.
    const Cc = S.circ, circA = seg(3.0, 3.6);
    ctx.save(); ctx.globalAlpha = circA; ctx.strokeStyle = C.R; ctx.lineWidth = 1.5; ctx.beginPath(); ctx.arc(Cc.x, Cc.y, Cc.r, 0, 7); ctx.stroke();
    (wide ? [118, 62, 12, 322, 248, 286] : [160, 205, 250, 295, 340, 20]).forEach((a, q) => { const ar = a * Math.PI / 180, px = Cc.x + Cc.r * Math.cos(ar), py = Cc.y - Cc.r * Math.sin(ar);
      if (q % 2) diamond(ctx, px, py, 4.5, tint(C.Br, .8)); else PE.dot(ctx, px, py, 4, tint(C.Br, .8)); });
    lab(ctx, 'one shared space', Cc.x, Cc.y + 4, C.M, S.fs, 'center');
    ctx.restore();
    const angI = S.angI * Math.PI / 180, angT = S.angT * Math.PI / 180;
    const ptI = [Cc.x + Cc.r * Math.cos(angI), Cc.y - Cc.r * Math.sin(angI)], ptT = [Cc.x + Cc.r * Math.cos(angT), Cc.y - Cc.r * Math.sin(angT)];
    const go = seg(3.3, 4.3);
    [[S.encI, C.B, ptI, 0], [S.encT, C.Y, ptT, 1]].forEach(([e, col, pt, isT]) => {
      const x1 = wide ? e.x + e.w + 6 : e.x + e.w / 2, y1 = wide ? e.y + e.h / 2 : e.y + e.h + 6;
      if (go > 0) PE.arrow(ctx, x1, y1, PE.lerp(x1, pt[0], go), PE.lerp(y1, pt[1], go), col, 1.4, 7, .45);
      if (go > 0) { const x = PE.lerp(x1, pt[0], go), y = PE.lerp(y1, pt[1], go); if (isT) diamond(ctx, x, y, 7, col); else PE.dot(ctx, x, y, 6.5, col); }
    });
    // F: the match.
    const m = seg(4.3, 5.0);
    if (m > 0) {
      PE.polyline(ctx, [ptI, [PE.lerp(ptI[0], ptT[0], m), PE.lerp(ptI[1], ptT[1], m)]], C.G, 3);
      ctx.save(); ctx.globalAlpha = .35 * m; ctx.strokeStyle = C.G; ctx.lineWidth = 2; ctx.beginPath(); ctx.arc((ptI[0] + ptT[0]) / 2, (ptI[1] + ptT[1]) / 2, 15, 0, 7); ctx.stroke(); ctx.restore();
      const lx = wide ? Cc.x - Cc.r - 4 : Cc.x, ly = Cc.y + Cc.r + 22, al = wide ? 'left' : 'center';
      lab(ctx, 'a photo and its caption', wide ? Cc.x : lx, ly, C.G, S.fs, 'center', m);
      lab(ctx, 'point the same way', wide ? Cc.x : lx, ly + S.fs + 4, C.G, S.fs, 'center', m);
    }
    ctx.restore();
  });
  function play() {
    cancelAnimationFrame(raf);
    if (PE.instant()) { t = T_END; cv.redraw(); return; }
    const t0 = performance.now();
    const f = now => { t = (now - t0) / 1000; cv.redraw(); if (t < T_END) raf = requestAnimationFrame(f); };
    raf = requestAnimationFrame(f);
  }
  btn.addEventListener('click', play);
  play();
})();
</script>
