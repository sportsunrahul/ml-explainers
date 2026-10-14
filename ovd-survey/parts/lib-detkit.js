<script>
/* DK: the detection drawing kit shared by the long page and the atlas (the atlas build inlines this file).
   Boxes are corner brackets, matches carry a tag, scenes are flat drawn factory objects, and the tap strip
   shows where text enters the network. Needs PE (css, mix, lerp, clamp, arrow, label). ASCII only. */
const DK = (() => {
  const C = () => {
    const c = n => PE.css(n);
    return { bg: c('--bg'), fg: c('--fg'), mu: c('--muted'), rule: c('--rule'), p: c('--panel'), p2: c('--panel2'),
      B: c('--blue'), Y: c('--yellow'), G: c('--green'), R: c('--red'), Br: c('--brown') };
  };
  // Like PE.mix but returns "#rrggbb", so results can be mixed again (PE.mix returns rgb(), which it cannot parse).
  function mixh(c1, c2, u) {
    const p = c => [1, 3, 5].map(i => parseInt(c.slice(i, i + 2), 16)), a = p(c1), b = p(c2);
    return '#' + a.map((v, i) => Math.round(v + (b[i] - v) * u).toString(16).padStart(2, '0')).join('');
  }
  function rr(ctx, x, y, w, h, r) {
    r = Math.min(r, w / 2, h / 2); ctx.beginPath(); ctx.moveTo(x + r, y); ctx.arcTo(x + w, y, x + w, y + h, r);
    ctx.arcTo(x + w, y + h, x, y + h, r); ctx.arcTo(x, y + h, x, y, r); ctx.arcTo(x, y, x + w, y, r); ctx.closePath();
  }
  // Four L-shaped corners. o: { lw, len (fraction of the short side), alpha, dash }
  function brackets(ctx, x, y, w, h, col, o = {}) {
    const lw = o.lw || 2, L = Math.max(5, Math.min(w, h) * (o.len || .28));
    ctx.save(); ctx.globalAlpha = o.alpha == null ? 1 : o.alpha; ctx.strokeStyle = col; ctx.lineWidth = lw; ctx.lineCap = 'round';
    if (o.dash) ctx.setLineDash(o.dash);
    ctx.beginPath();
    [[x, y, 1, 1], [x + w, y, -1, 1], [x, y + h, 1, -1], [x + w, y + h, -1, -1]].forEach(([cx, cy, sx, sy]) => {
      ctx.moveTo(cx + sx * L, cy); ctx.lineTo(cx, cy); ctx.lineTo(cx, cy + sy * L); });
    ctx.stroke(); ctx.restore();
  }
  // A tag above a box: the name in yellow (it came from text), the score in fg; border in the box color.
  function tag(ctx, x, y, name, score, col, k, o = {}) {
    const fs = o.size || 12, a = o.alpha == null ? 1 : o.alpha;
    ctx.save(); ctx.globalAlpha = a; ctx.font = `600 ${fs}px system-ui, sans-serif`;
    const s = score == null ? '' : ' ' + score, wn = ctx.measureText(name).width, ws = ctx.measureText(s).width;
    const w = wn + ws + fs * .9, h = fs * 1.55, maxX = o.maxX || ctx.canvas.width / (devicePixelRatio || 1) - 2;
    const x0 = Math.max(2, Math.min(o.align === 'right' ? x - w : x, maxX - w)), y0 = y - h - 3;
    rr(ctx, x0, y0, w, h, h / 2); ctx.fillStyle = k.bg; ctx.fill(); ctx.strokeStyle = col; ctx.lineWidth = 1.2; ctx.stroke();
    ctx.textBaseline = 'middle'; ctx.fillStyle = o.nameCol || k.Y; ctx.fillText(name, x0 + fs * .45, y0 + h / 2 + .5);
    ctx.fillStyle = k.fg; ctx.fillText(s, x0 + fs * .45 + wn, y0 + h / 2 + .5); ctx.restore();
    return { x: x0, y: y0, w, h };
  }
  // Flat factory objects drawn inside a box (x, y, w, h). Neutral ink so overlays carry the color.
  const ink = k => ({ f: PE.mix(k.bg, k.fg, .16), s: PE.mix(k.bg, k.fg, .42), d: PE.mix(k.bg, k.fg, .3) });
  const OBJ = {
    person(ctx, x, y, w, h, k) { const i = ink(k); ctx.fillStyle = i.f; ctx.strokeStyle = i.s; ctx.lineWidth = 1.4;
      ctx.beginPath(); ctx.arc(x + w / 2, y + h * .11, Math.min(w, h) * .13, 0, 7); ctx.fill(); ctx.stroke();
      rr(ctx, x + w * .22, y + h * .25, w * .56, h * .4, w * .14); ctx.fill(); ctx.stroke();
      ctx.fillRect(x + w * .27, y + h * .63, w * .17, h * .37); ctx.fillRect(x + w * .56, y + h * .63, w * .17, h * .37);
      ctx.fillStyle = i.d; ctx.fillRect(x + w * .2, y + h * .02, w * .6, h * .07); },
    forklift(ctx, x, y, w, h, k) { const i = ink(k); ctx.fillStyle = i.f; ctx.strokeStyle = i.s; ctx.lineWidth = 1.4;
      ctx.fillRect(x + w * .05, y, w * .05, h * .88); ctx.fillRect(x, y + h * .78, w * .38, h * .05);
      rr(ctx, x + w * .3, y + h * .38, w * .62, h * .42, 4); ctx.fill(); ctx.stroke();
      ctx.strokeRect(x + w * .44, y + h * .06, w * .34, h * .34); ctx.fillStyle = i.d; ctx.fillRect(x + w * .8, y + h * .48, w * .2, h * .3);
      ctx.fillStyle = i.s; [[.42, .86], [.8, .86]].forEach(([u, v]) => { ctx.beginPath(); ctx.arc(x + w * u, y + h * v, h * .12, 0, 7); ctx.fill(); }); },
    pallet(ctx, x, y, w, h, k) { const i = ink(k); ctx.fillStyle = i.f; ctx.strokeStyle = i.s; ctx.lineWidth = 1.4;
      const ph = h * .16; [[0, 0, .48, .42], [.5, .12, .5, .3], [.14, .42, .6, .42]].forEach(([u, v, a, b]) => {
        ctx.fillRect(x + w * u, y + h * v, w * a, h * b); ctx.strokeRect(x + w * u, y + h * v, w * a, h * b); });
      ctx.fillStyle = i.d; ctx.fillRect(x, y + h - ph, w, ph * .45); [0, .45, .9].forEach(u => ctx.fillRect(x + w * u, y + h - ph * .6, w * .1, ph * .6)); },
    cone(ctx, x, y, w, h, k) { const i = ink(k); ctx.fillStyle = i.f; ctx.strokeStyle = i.s; ctx.lineWidth = 1.4;
      ctx.beginPath(); ctx.moveTo(x + w / 2, y); ctx.lineTo(x + w * .85, y + h * .88); ctx.lineTo(x + w * .15, y + h * .88); ctx.closePath(); ctx.fill(); ctx.stroke();
      ctx.fillStyle = i.d; ctx.fillRect(x + w * .3, y + h * .45, w * .4, h * .1); ctx.fillRect(x, y + h * .88, w, h * .12); },
    crate(ctx, x, y, w, h, k) { const i = ink(k); ctx.fillStyle = i.f; ctx.strokeStyle = i.s; ctx.lineWidth = 1.4;
      ctx.fillRect(x, y, w, h); ctx.strokeRect(x, y, w, h); ctx.beginPath(); ctx.moveTo(x, y); ctx.lineTo(x + w, y + h); ctx.moveTo(x + w, y); ctx.lineTo(x, y + h); ctx.stroke(); },
    shelf(ctx, x, y, w, h, k) { const i = ink(k); ctx.strokeStyle = i.d; ctx.lineWidth = 2;
      ctx.strokeRect(x, y, w, h); [.33, .66].forEach(v => { ctx.beginPath(); ctx.moveTo(x, y + h * v); ctx.lineTo(x + w, y + h * v); ctx.stroke(); }); },
    panel(ctx, x, y, w, h, k) { const i = ink(k); ctx.fillStyle = i.f; ctx.strokeStyle = i.s; ctx.lineWidth = 1.4; rr(ctx, x, y, w, h, 3); ctx.fill(); ctx.stroke(); },
    scratch(ctx, x, y, w, h, k) { const i = ink(k); ctx.strokeStyle = i.s; ctx.lineWidth = 1.6; ctx.beginPath(); ctx.moveTo(x, y + h * .7);
      ctx.bezierCurveTo(x + w * .3, y + h * .5, x + w * .6, y + h * .55, x + w, y + h * .2); ctx.stroke(); },
    pores(ctx, x, y, w, h, k) { const i = ink(k); ctx.fillStyle = i.s; const r = PE.rng(7);
      for (let n = 0; n < 9; n++) { ctx.beginPath(); ctx.arc(x + w * (.1 + .8 * r()), y + h * (.25 + .5 * r()), 1 + 2.2 * r(), 0, 7); ctx.fill(); } },
  };
  // Callers may pass a non-positive box while a canvas is still being sized; skip those instead of throwing.
  Object.keys(OBJ).forEach(k => { const f = OBJ[k]; OBJ[k] = (ctx, x, y, w, h, kk) => { if (w > 0 && h > 0) f(ctx, x, y, w, h, kk); }; });
  // The warehouse scene in unit coordinates (x, y, w, h in [0, 1]).
  const WAREHOUSE = [
    { kind: 'shelf', box: [.02, .12, .2, .68], bg: true },
    { kind: 'person', box: [.24, .3, .11, .58], name: 'person', row: 0 },
    { kind: 'pallet', box: [.4, .52, .2, .36], name: 'pallet', row: 1 },
    { kind: 'forklift', box: [.63, .3, .26, .58], name: 'forklift', row: -1 },
    { kind: 'cone', box: [.91, .68, .07, .2], name: 'traffic cone', row: 2 },
  ];
  function drawScene(ctx, objs, X, Y, W, H, k, alpha = 1) {
    ctx.save(); ctx.globalAlpha = alpha; ctx.strokeStyle = PE.mix(k.bg, k.fg, .22); ctx.lineWidth = 1.2;
    ctx.beginPath(); ctx.moveTo(X, Y + H * .885); ctx.lineTo(X + W, Y + H * .885); ctx.stroke();
    objs.forEach(o => { const [u, v, a, b] = o.box; OBJ[o.kind](ctx, X + u * W, Y + v * H, a * W, b * H, k); });
    ctx.restore();
  }
  const boxOf = (o, X, Y, W, H, pad = .012) => { const [u, v, a, b] = o.box; return [X + (u - pad) * W, Y + (v - pad) * H, (a + 2 * pad) * W, (b + 2 * pad) * H]; };

  // Tap strip: the image pipeline (blue lane) and the text lane (yellow) above; taps show where text enters.
  const NODES = [['image'], ['backbone'], ['encoder', 'or neck'], ['proposals', 'or queries'], ['region', 'features'], ['score']];
  const TAPS = [
    { node: 2, models: ['glip', 'gdino', 'yolow'], text: ['GLIP, Grounding DINO,', 'YOLO-World'] },
    { node: 3, models: ['gdino'], text: ['Grounding', 'DINO'] },
    { node: 4, models: ['gdino'], text: ['Grounding', 'DINO'] },
    { node: 5, models: ['vild', 'owl', 'glip', 'gdino', 'yolow'], text: ['all five;', 'ViLD, OWL:', 'only here'] },
  ];
  // u in [0, 1] draws the strip progressively (taps last). active: model key to highlight, or 'all'.
  function strip(ctx, w, h, k, active = 'all', u = 1, o = {}) {
    if (w < 240 || h < 100) return null;
    const narrow = w < 560, n = NODES.length, padL = narrow ? 4 : 8, padR = narrow ? 4 : 8;
    const step = (w - padL - padR) / n, bw = Math.min(step - (narrow ? 6 : 14), 132), bh = narrow ? 40 : 44;
    const yT = narrow ? 30 : 26, yB = h - bh - (narrow ? 6 : 8), fs = narrow ? 10 : 12.5;
    const cx = i => padL + step * (i + .5);
    const ease = (a, b) => PE.clamp((u - a) / (b - a), 0, 1);
    ctx.save(); ctx.font = `600 ${fs}px system-ui, sans-serif`; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
    // image lane
    const uL = ease(0, .45);
    for (let i = 0; i < n; i++) {
      const a = PE.clamp(uL * n - i, 0, 1); if (a <= 0) continue;
      ctx.globalAlpha = a; const x = cx(i) - bw / 2;
      rr(ctx, x, yB, bw, bh, 8); ctx.fillStyle = PE.mix(k.bg, k.B, .14); ctx.fill(); ctx.strokeStyle = k.B; ctx.lineWidth = 1.3; ctx.stroke();
      ctx.fillStyle = k.fg; const L = NODES[i];
      if (L.length === 1) ctx.fillText(L[0], cx(i), yB + bh / 2); else { ctx.fillText(L[0], cx(i), yB + bh / 2 - fs * .62); ctx.fillStyle = k.mu; ctx.fillText(L[1], cx(i), yB + bh / 2 + fs * .62); }
      if (i < n - 1) PE.arrow(ctx, cx(i) + bw / 2 + 1, yB + bh / 2, cx(i + 1) - bw / 2 - 2, yB + bh / 2, k.B, 1.4, 6, a);
    }
    ctx.globalAlpha = 1;
    // text lane
    const uT = ease(.4, .6);
    if (uT > 0) {
      ctx.globalAlpha = uT; const x0 = cx(0) - bw / 2, tw = Math.min(bw, narrow ? 54 : 110);
      rr(ctx, x0, yT - 13, tw, 26, 13); ctx.strokeStyle = k.Y; ctx.lineWidth = 1.2; ctx.stroke();
      ctx.fillStyle = k.Y; ctx.fillText(narrow ? 'prompt' : 'text prompt', x0 + tw / 2, yT);
      ctx.strokeStyle = k.Y; ctx.lineWidth = 2; ctx.beginPath(); ctx.moveTo(x0 + tw + 2, yT); ctx.lineTo(PE.lerp(x0 + tw + 2, cx(n - 1), uT), yT); ctx.stroke();
      ctx.globalAlpha = 1;
    }
    // taps
    TAPS.forEach((t, j) => {
      const a = ease(.6 + j * .08, .72 + j * .08); if (a <= 0) return;
      const on = active === 'all' || t.models.includes(active), x = cx(t.node), col = on ? k.Y : PE.mix(k.bg, k.Y, .3);
      ctx.globalAlpha = a; PE.arrow(ctx, x, yT, x, PE.lerp(yT, yB - 3, a), col, on ? 2.4 : 1.4, 8, 1);
      ctx.beginPath(); ctx.arc(x, yT, on ? 4.5 : 3, 0, 7); ctx.fillStyle = col; ctx.fill();
      if (!narrow && (o.labels !== false)) {
        ctx.textAlign = 'left'; ctx.font = `${on ? 600 : 400} 11.5px system-ui, sans-serif`; ctx.fillStyle = on ? k.fg : k.mu;
        const y0 = (yT + yB) / 2 - 6 - (t.text.length - 2) * 7;
        t.text.forEach((s, m) => ctx.fillText(s, x + 8, y0 + m * 14)); ctx.textAlign = 'center'; ctx.font = `600 ${fs}px system-ui, sans-serif`;
      }
      ctx.globalAlpha = 1;
    });
    ctx.restore();
    return { cx, yB, bh, bw, yT };
  }
  // Mount every .tapstrip[data-model] on the page as a static canvas.
  function mountStrips(root = document) {
    root.querySelectorAll('.tapstrip[data-model]').forEach(div => {
      const el = document.createElement('canvas'); div.prepend(el);
      PE.canvas(el, w => (w < 560 ? 158 : 150) / w, (ctx, w, h) => { ctx.clearRect(0, 0, w, h); strip(ctx, w, h, C(), div.dataset.model); });
    });
  }
  return { C, mixh, rr, brackets, tag, OBJ, WAREHOUSE, drawScene, boxOf, strip, mountStrips, NODES, TAPS };
})();
</script>
