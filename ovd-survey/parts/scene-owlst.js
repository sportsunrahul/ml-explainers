<script>
/* SceneOwlST(el, o): OWLv2's self-training pipeline as a five-stage toy, shared by the long page (viz-owlst) and the
   atlas (chapter "owl", step 3). Stages: web image + alt text -> word N-grams as queries -> teacher boxes, each
   keeping its best query -> keep confident boxes, drop images with none -> train a student on mosaics. The N-gram
   count is computed from the toy sentence by the paper's rule (all word N-grams up to 10 words, minus those made
   only of stop words); the boxes and scores are made up. The closing arrow is drawn dashed: the paper annotates once.
   o: { aspect, bottom, onSay }. API: set(stage), play(), reset(), redraw(), resize(), destroy(), stage(). */
const SceneOwlST = (el, o = {}) => {
  const STAGES = ['web', 'ngrams', 'teacher', 'filter', 'student'];
  const NODE = ['web image', 'N-gram queries', 'teacher boxes', 'keep confident', 'train student'];
  const ALT = 'Toyota forklift lifting a wooden pallet in our warehouse', ALT2 = 'Happy Monday from the whole team!';
  const STOP = new Set(['a', 'an', 'the', 'in', 'our', 'of', 'on', 'from', 'for', 'with', 'to', 'and']);
  const words = ALT.toLowerCase().split(' '), NG = [];
  for (let L = 1; L <= Math.min(10, words.length); L++) for (let i = 0; i + L <= words.length; i++) {
    const w = words.slice(i, i + L); NG.push({ s: w.join(' '), stop: w.every(x => STOP.has(x)), L }); }
  const KEPT = NG.filter(q => !q.stop).length;
  const IMG = [
    { kind: 'shelf', box: [.03, .1, .2, .72], bg: true },
    { kind: 'pallet', box: [.29, .56, .16, .22] },
    { kind: 'forklift', box: [.43, .25, .32, .62] },
    { kind: 'person', box: [.82, .3, .1, .57] },
  ];
  // Toy teacher output: each box's best query and its score.
  const BOXES = [
    { i: 2, q: 'toyota forklift', p: .62, keep: true }, { i: 1, q: 'wooden pallet', p: .47, keep: true },
    { i: 0, q: 'our warehouse', p: .13, keep: false }, { i: 3, q: 'warehouse', p: .06, keep: false }];
  const RANK = [['toyota forklift', .62], ['forklift', .58], ['toyota forklift lifting', .55], ['forklift lifting', .41], ['toyota', .22]];
  const IMG2 = [{ kind: 'shelf', box: [.55, .14, .32, .72], bg: true }, { kind: 'crate', box: [.16, .5, .26, .36] }];
  const TILES = [
    { objs: [{ kind: 'cone', box: [.38, .4, .26, .48] }], tag: 'safety cone', b: [.34, .36, .34, .56] },
    { objs: [{ kind: 'crate', box: [.25, .42, .45, .46] }], tag: 'wooden crate', b: [.21, .38, .53, .54] },
    { objs: [{ kind: 'person', box: [.4, .14, .22, .74] }], tag: 'hard hat worker', b: [.36, .1, .3, .82] },
  ];
  let S = { st: 0, u: 1 }, stop = () => {}, timer = 0;

  function geo(w, h) {
    const hh = h - (o.bottom || 0), narrow = w < 620, top = narrow ? 86 : 100;
    if (!narrow) {
      let iw = w * .5, ih = iw * .62; const maxH = hh - top - 44; if (ih > maxH) { ih = maxH; iw = ih / .62; }
      const t2 = top + Math.max(0, (hh - top - ih - 44) * .3);
      return { narrow, top, ix: 14, iy: t2, iw, ih, px: 14 + iw + 26, py: t2, pw: w - (14 + iw + 26) - 12, ph: ih + 30, fs: 12 };
    }
    let iw = w - 20, ih = iw * .58; const maxH = hh - top - 40 - 170; if (ih > maxH && maxH > 110) { ih = maxH; iw = ih / .58; }
    return { narrow, top, ix: (w - iw) / 2, iy: top, iw, ih, px: 10, py: top + ih + 34, pw: w - 20, ph: hh - (top + ih + 34) - 4, fs: 11 };
  }
  const aspect = o.aspect || (w => (w < 620 ? (86 + (w - 20) * .58 + 34 + 176) / w : (100 + w * .5 * .62 + 44) / w));
  const frame = (ctx, x, y, w, h, k, objs) => { DK.rr(ctx, x, y, w, h, 6); ctx.fillStyle = k.p; ctx.fill(); ctx.strokeStyle = k.rule; ctx.lineWidth = 1; ctx.stroke();
    DK.drawScene(ctx, objs, x + 4, y + 4, w - 8, h - 8, k); };
  const bx = (objs, i, x, y, w, h) => DK.boxOf(objs[i], x + 4, y + 4, w - 8, h - 8, .015);

  function strip(ctx, w, k, G) {
    const n = STAGES.length, pad = 12, step = (w - 2 * pad) / n, y = G.narrow ? 38 : 46;
    ctx.save(); ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
    for (let i = 0; i < n; i++) {
      const cx = pad + step * (i + .5), on = i === S.st, done = i < S.st, col = on ? k.G : done ? k.fg : k.mu;
      if (G.narrow) { ctx.beginPath(); ctx.arc(cx, y, 11, 0, 7); ctx.fillStyle = on ? k.G : k.p2; ctx.fill(); ctx.strokeStyle = col; ctx.lineWidth = 1.3; ctx.stroke();
        ctx.fillStyle = on ? k.bg : col; ctx.font = '600 11px system-ui, sans-serif'; ctx.fillText(String(i + 1), cx, y + .5); }
      else { const bw = Math.min(step - 18, 150); DK.rr(ctx, cx - bw / 2, y - 14, bw, 28, 14); ctx.fillStyle = on ? DK.mixh(k.bg, k.G, .18) : k.p; ctx.fill();
        ctx.strokeStyle = col; ctx.lineWidth = on ? 2 : 1; ctx.stroke(); ctx.fillStyle = col; ctx.font = `${on ? 600 : 500} 12px system-ui, sans-serif`; ctx.fillText(`${i + 1} ${NODE[i]}`, cx, y + .5); }
      if (i < n - 1) { const x1 = cx + (G.narrow ? 13 : Math.min(step - 18, 150) / 2 + 2), x2 = pad + step * (i + 1.5) - (G.narrow ? 13 : Math.min(step - 18, 150) / 2 + 2);
        PE.arrow(ctx, x1, y, x2, y, done || on ? k.fg : k.mu, 1.2, 6, .8); }
    }
    // the closing arrow: the student could annotate the web again, but the paper annotates once
    const xa = pad + step * (n - .5), xb = pad + step * 2.5, yb = y + (G.narrow ? 14 : 16);
    ctx.save(); ctx.setLineDash([3, 4]); ctx.strokeStyle = k.mu; ctx.lineWidth = 1.1; ctx.beginPath(); ctx.moveTo(xa, yb); ctx.quadraticCurveTo((xa + xb) / 2, yb + 22, xb, yb + 2); ctx.stroke(); ctx.restore();
    ctx.fillStyle = k.mu; ctx.font = `400 ${G.narrow ? 10 : 11}px system-ui, sans-serif`; ctx.fillText('a second round: not done in the paper', (xa + xb) / 2, yb + 20);
    if (G.narrow) { ctx.fillStyle = k.fg; ctx.font = '600 11.5px system-ui, sans-serif'; ctx.textAlign = 'left'; }
    ctx.restore();
  }

  function chip(ctx, x, y, s, k, fs, struck, a = 1) {
    ctx.save(); ctx.globalAlpha = a; ctx.font = `500 ${fs}px system-ui, sans-serif`; const w = ctx.measureText(s).width + 14;
    DK.rr(ctx, x, y - 10, w, 20, 10); ctx.strokeStyle = struck ? k.Br : k.Y; ctx.lineWidth = 1; if (struck) ctx.setLineDash([3, 2]); ctx.stroke(); ctx.setLineDash([]);
    ctx.fillStyle = struck ? k.Br : k.Y; ctx.textBaseline = 'middle'; ctx.fillText(s, x + 7, y + .5);
    if (struck) { ctx.strokeStyle = k.Br; ctx.beginPath(); ctx.moveTo(x + 5, y); ctx.lineTo(x + w - 5, y); ctx.stroke(); }
    ctx.restore(); return w;
  }

  function draw(ctx, w, h) {
    const k = DK.C(); ctx.clearRect(0, 0, w, h); if (w < 160 || h < 160) return;
    const G = geo(w, h), { ix, iy, iw, ih, px, py, pw, fs, narrow } = G, st = STAGES[S.st], u = S.u;
    strip(ctx, w, k, G);
    const text = (s, x, y, col, size = fs, wt = 500, al = 'left') => { ctx.save(); ctx.font = `${wt} ${size}px system-ui, sans-serif`; ctx.fillStyle = col; ctx.textAlign = al; ctx.textBaseline = 'middle'; ctx.fillText(s, x, y); ctx.restore(); };
    const wrap = (s, x, y, maxW, col, size = fs, lh = size * 1.45) => { ctx.save(); ctx.font = `400 ${size}px system-ui, sans-serif`; let line = '', yy = y;
      s.split(' ').forEach(wd => { const t = line ? line + ' ' + wd : wd; if (ctx.measureText(t).width > maxW && line) { text(line, x, yy, col, size, 400); line = wd; yy += lh; } else line = t; });
      if (line) text(line, x, yy, col, size, 400); ctx.restore(); return yy + lh; };
    if (st !== 'student') {
      ctx.save(); ctx.globalAlpha = st === 'ngrams' ? .55 : 1; frame(ctx, ix, iy, iw, ih, k, IMG); ctx.restore();
      ctx.save(); ctx.font = `italic 400 ${fs}px system-ui, sans-serif`; ctx.fillStyle = k.Y; ctx.textBaseline = 'middle';
      let s = `alt text: "${ALT}"`; while (ctx.measureText(s).width > iw && s.length > 20) s = s.slice(0, -5) + '..."';
      ctx.fillText(s, ix, iy + ih + 16); ctx.restore();
    }
    if (st === 'web') {
      text('A web image and its alt text', px, py + 8, k.fg, fs + .5, 600);
      wrap('The alt text is a noisy caption. Here it names the forklift and the pallet. It says nothing about the person or the shelf. Nobody drew a box.', px, py + 34, pw, k.mu);
    }
    if (st === 'ngrams') {
      text(`${NG.length} N-grams, ${NG.length - KEPT} only stop words: ${KEPT} queries`, px, py + 8, k.fg, fs + .5, 600);
      let x = px, y = py + 36; const show = NG.filter(q => q.L <= 2).concat(NG.filter(q => q.L === 3).slice(0, 3), [NG[NG.length - 1]]);
      let shown = 0;
      for (const [m, q] of show.entries()) {
        ctx.save(); ctx.font = `500 ${fs - .5}px system-ui, sans-serif`; let s = q.s; while (ctx.measureText(s).width + 14 > pw && s.length > 8) s = s.slice(0, -4) + '...'; const cw = ctx.measureText(s).width + 14; ctx.restore();
        if (x + cw > px + pw) { x = px; y += 25; }
        if (y > py + G.ph - 18) break;
        x += chip(ctx, x, y, s, k, fs - .5, q.stop, PE.clamp(u * show.length * 1.4 - m, 0, 1)) + 6; shown++;
      }
      if (shown < NG.length) text(`+ ${NG.length - shown} more, up to the whole sentence`, px, Math.min(y + 26, py + G.ph - 4), k.mu, fs - .5, 400);
    }
    if (st === 'teacher' || st === 'filter') {
      BOXES.forEach((b, m) => {
        const a = st === 'teacher' ? PE.clamp(u * 5 - m, 0, 1) : 1, [x, y, bw, bh] = bx(IMG, b.i, ix, iy, iw, ih);
        const drop = st === 'filter' && !b.keep, kept = st === 'filter' && b.keep;
        const col = drop ? DK.mixh(k.B, k.Br, u) : kept ? DK.mixh(k.B, k.G, u) : k.B;
        DK.brackets(ctx, x, y, bw, bh, col, { alpha: a * (drop ? 1 - .35 * u : 1), lw: 2, dash: drop && u > .5 ? [4, 3] : null });
        if (a > .3) DK.tag(ctx, x, y + (b.i === 0 ? 14 : 0), drop && u > .5 ? 'dropped' : b.q, drop && u > .5 ? null : b.p.toFixed(2), col, k,
          { size: narrow ? 9.5 : 11, alpha: a, nameCol: drop && u > .5 ? k.Br : k.Y, maxX: ix + iw });
      });
    }
    if (st === 'teacher') {
      text('Forklift box: every query scored', px, py + 8, k.fg, fs + .5, 600);
      const lab = Math.min(pw * .55, narrow ? 150 : 170), bwM = pw - lab - 40;
      RANK.forEach(([q, p], m) => { const y = py + 34 + m * (narrow ? 21 : 24), a = PE.clamp(u * 6 - m, 0, 1);
        ctx.save(); ctx.globalAlpha = a; text(q, px, y, k.Y, fs - .5); ctx.fillStyle = m === 0 ? k.G : DK.mixh(k.bg, k.fg, .35); ctx.fillRect(px + lab, y - 5, bwM * p, 10);
        text(p.toFixed(2), px + lab + bwM * p + 5, y, k.mu, fs - 1, 400); ctx.restore(); });
      wrap('The box came from the image alone; the queries only name it. It keeps its best query as a hard label.', px, py + 34 + 5 * (narrow ? 21 : 24) + 6, pw, k.mu, fs - .5);
    }
    if (st === 'filter') {
      text('Another image: no confident box', px, py + 8, k.fg, fs + .5, 600);
      const tw = Math.min(pw * .62, narrow ? 150 : 210), th = tw * .62, ty = py + 26;
      frame(ctx, px, ty, tw, th, k, IMG2);
      [[1, '"monday" 0.09'], [0, '"whole team" 0.05']].forEach(([i, s]) => { const [x, y, bw, bh] = bx(IMG2, i, px, ty, tw, th); DK.brackets(ctx, x, y, bw, bh, k.Br, { lw: 1.6, dash: [4, 3] });
        ctx.save(); ctx.font = `500 ${fs - 1.5}px system-ui, sans-serif`; ctx.lineWidth = 3.5; ctx.strokeStyle = k.p; ctx.lineJoin = 'round'; ctx.textBaseline = 'middle'; ctx.strokeText(s, x + 2, y - 7); ctx.restore(); text(s, x + 2, y - 7, k.Br, fs - 1.5, 500); });
      ctx.save(); ctx.globalAlpha = .7 * u; ctx.strokeStyle = k.Br; ctx.lineWidth = 2.5; ctx.beginPath(); ctx.moveTo(px + 6, ty + 6); ctx.lineTo(px + tw - 6, ty + th - 6); ctx.moveTo(px + tw - 6, ty + 6); ctx.lineTo(px + 6, ty + th - 6); ctx.stroke(); ctx.restore();
      ctx.save(); ctx.font = `italic 400 ${fs - 1}px system-ui, sans-serif`; ctx.fillStyle = k.Y; ctx.textBaseline = 'middle'; ctx.fillText(`"${ALT2}"`, px, ty + th + 14); ctx.restore();
      wrap('Weak boxes go. An image with no confident box goes entirely.', px, ty + th + 36, pw, k.mu, fs - .5);
    }
    if (st === 'student') {
      // a 2 x 2 mosaic of pseudo-labelled images feeds a new detector
      const ms = Math.min(iw, ih), t = ms / 2, mx = ix, my = iy;
      const tiles = [{ objs: IMG, img: true }].concat(TILES);
      tiles.forEach((tl, m) => { const x = mx + (m % 2) * t, y = my + Math.floor(m / 2) * t, a = PE.clamp(u * 4 - m, 0, 1);
        ctx.save(); ctx.globalAlpha = a; frame(ctx, x + 2, y + 2, t - 4, t - 4, k, tl.objs);
        if (tl.img) BOXES.filter(b => b.keep).forEach(b => { const [bx0, by0, bw, bh] = bx(IMG, b.i, x + 2, y + 2, t - 4, t - 4); DK.brackets(ctx, bx0, by0, bw, bh, k.G, { lw: 1.6 });
          DK.tag(ctx, bx0, by0, b.q, null, k.G, k, { size: 9.5, maxX: x + t - 2 }); });
        else { const [u0, v0, a0, b0] = tl.b, X0 = x + 6 + u0 * (t - 12), Y0 = y + 6 + v0 * (t - 12); DK.brackets(ctx, X0, Y0, a0 * (t - 12), b0 * (t - 12), k.G, { lw: 1.6 });
          DK.tag(ctx, X0, Y0 + 14, tl.tag, null, k.G, k, { size: 9.5, maxX: x + t - 2 }); }
        ctx.restore(); });
      text('one training example: a mosaic of pseudo-labelled images', mx, my + ms + 16, k.mu, fs - .5, 400);
      const sx = narrow ? px : px + 8, sy = narrow ? py + 4 : py + 20, sw = Math.min(pw - 16, 240), shh = 58;
      if (!narrow) PE.arrow(ctx, mx + ms + 6, my + ms / 2, sx - 6, sy + shh / 2, k.G, 2, 8, u);
      ctx.save(); ctx.globalAlpha = u; DK.rr(ctx, sx, sy, sw, shh, 9); ctx.fillStyle = DK.mixh(k.bg, k.B, .14); ctx.fill(); ctx.strokeStyle = k.B; ctx.lineWidth = 1.5; ctx.stroke();
      text('student: a new OWL model', sx + 12, sy + 20, k.fg, fs + .5, 600); text('CLIP towers, fresh heads', sx + 12, sy + 40, k.mu, fs - .5, 400); ctx.restore();
      const ny = sy + shh + 22; ['trains with OWL-ViT\'s losses', 'drops flat patches (training only)', 'losses only on likely objects', 'many images per example'].forEach((s, m) => {
        ctx.save(); ctx.globalAlpha = PE.clamp(u * 5 - m - 1, 0, 1); PE.dot(ctx, sx + 5, ny + m * 21, 2.5, k.B); text(s, sx + 14, ny + m * 21, k.fg, fs - .5, 400); ctx.restore(); });
    }
  }
  const cv = PE.canvas(el, aspect, draw);
  const SAY = {
    web: 'Step 1: a web image and its alt text. The text names a forklift and a pallet; it says nothing about the person.',
    ngrams: `Step 2: every word N-gram of the alt text, up to 10 words: ${NG.length} here. The ${NG.length - KEPT} made only of stop words go, so ${KEPT} queries remain. No parser decides which words are objects.`,
    teacher: 'Step 3: the teacher (OWL-ViT L/14) boxes the image without the queries, then scores every box against every query. Each box keeps its best query: the forklift box becomes "toyota forklift" (0.62 in this toy).',
    filter: 'Step 4: weak boxes go (the shelf at 0.13, the person at 0.06 in this toy). The second image has no confident box, so all of it goes.',
    student: 'Step 5: a new detector trains on the surviving boxes, many images per example. The paper annotates the web once; the student does not relabel it.',
  };
  const say = () => o.onSay && o.onSay(SAY[STAGES[S.st]]);
  function set(stage) { clearTimeout(timer); stop(); const i = Math.max(0, STAGES.indexOf(stage)); S.st = i; S.u = 0;
    stop = PE.animate(800, v => { S.u = v; cv.redraw(); }); say(); }
  // Atlas: walk through all five stages and stop on the last one.
  function play() { clearTimeout(timer); let i = 0; const next = () => { set(STAGES[i]); i++; if (i < STAGES.length && !PE.instant()) timer = setTimeout(next, 2200); else if (i < STAGES.length) { set(STAGES[STAGES.length - 1]); } }; next(); }
  function reset() { clearTimeout(timer); stop(); S = { st: 0, u: 1 }; cv.redraw(); say(); }
  say();
  return { set, play, reset, redraw: cv.redraw, resize: () => (cv.resize ? cv.resize() : cv.redraw()), stage: () => STAGES[S.st],
    destroy: () => { clearTimeout(timer); stop(); cv.destroy && cv.destroy(); } };
};
</script>
