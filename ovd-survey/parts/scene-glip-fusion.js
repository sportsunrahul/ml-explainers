<script>
/* GLIP fusion toy, shared by the long page and the atlas. An aquarium grid of region features (5 dims:
   object, flat, black-white, fin, water) and the prompt "fish. penguin. stingray" run through six rounds of
   one-head cross-attention in the form of GLIP Eq. 4-6: each region attends over the tokens (softmax over
   tokens) and adds the tokens' values; each token attends over the regions (softmax over regions) and adds
   their features; both sides are renormalized. The DyHead and BERT layers themselves are the identity here.
   Token values stand in for the learned W(v), W(out): a class prototype in image space. Late fusion skips the
   exchange, so the features never change. The map shows, per region, the share of its score that goes to
   one token: a softmax over the prompt's tokens of cos(region, token). Parameters were checked in node:
   stingray regions 0.64 -> 0.87, fish 0.28 -> 0.06 over six deep layers; late stays at layer 0.
   o: { aspect(w) -> h/w, glow(word) -> 0..1, onSay(text), onPick(layer), bottom (px kept free) } */
const SceneGlipFusion = (el, o = {}) => {
  const glow = o.glow || (() => 0), bottom = o.bottom || 0;
  const GW = 16, GH = 10, L = 6, BETA = 16, BETA_I = 16, G1 = 1, G2 = .4, BS = 4;
  const BASE = { water: [.05, 0, 0, 0, 1], ray: [.4, .75, 0, .5, .1], peng: [.4, 0, .85, .2, .1], fish: [.4, .05, 0, .9, .1] };
  const SEQ = ['[CLS]', 'fish', '.', 'penguin', '.', 'stingray', '[SEP]'];
  const KEY = [[.3, 0, 0, 0, .6], [.3, 0, 0, .9, 0], [.1, 0, 0, 0, .4], [.3, 0, .8, .3, 0], [.1, 0, 0, 0, .4], [.3, .5, 0, .6, 0], [.05, 0, 0, 0, 1]];
  const VAL = [[0, 0, 0, 0, 0], [.2, 0, 0, 1, 0], [0, 0, 0, 0, 0], [.2, 0, 1, 0, 0], [0, 0, 0, 0, 0], [.2, 1, 0, .1, 0], [0, 0, 0, 0, .3]];
  const WORD = { ray: 5, peng: 3 }, KIND = { ray: 'stingray', peng: 'penguin', fish: 'fish' };
  const INST = [['ray', 0], ['peng', 1], ['peng', 2], ['fish', 3], ['fish', 4], ['fish', 5]];
  const instAt = (x, y) => {
    if (((x + .5 - 4.5) / 2.8) ** 2 + ((y + .5 - 6.6) / 1.5) ** 2 < 1) return 0;
    if (x >= 10 && x <= 11 && y >= 1 && y <= 4) return 1;
    if (x >= 13 && x <= 14 && y >= 2 && y <= 5) return 2;
    if (y === 2 && (x === 2 || x === 3)) return 3;
    if (y === 3 && (x === 7 || x === 8)) return 4;
    if (y === 8 && (x === 12 || x === 13)) return 5;
    return -1;
  };
  const nrm = v => { const s = Math.hypot(...v) || 1; return v.map(x => x / s); };
  const dot = (a, b) => a.reduce((s, x, i) => s + x * b[i], 0);
  const r = PE.rng(11), gs = PE.gauss(r), cells = [];
  for (let y = 0; y < GH; y++) for (let x = 0; x < GW; x++) {
    const i = instAt(x, y), kd = i < 0 ? 'water' : INST[i][0];
    cells.push({ x, y, i, kd, f: nrm(BASE[kd].map(v => v + .08 * gs())) });
  }
  function run(deep) {
    let O = cells.map(c => c.f.slice()), T = KEY.map(nrm); const out = [];
    for (let i = 0; i <= L; i++) {
      out.push({ O: O.map(v => v.slice()), T: T.map(v => v.slice()) });
      if (i === L || !deep) continue;
      const A = O.map(f => T.map(t => dot(f, t)));
      const t2i = A.map(row => { const a = PE.softmax(row.map(x => BETA * x)); return VAL[0].map((_, d) => a.reduce((s, wt, m) => s + wt * VAL[m][d], 0)); });
      const i2t = T.map((_, m) => { const b = PE.softmax(A.map(row => BETA_I * row[m])); return O[0].map((_, d) => b.reduce((s, wt, n) => s + wt * O[n][d], 0)); });
      O = O.map((f, n) => nrm(f.map((v, d) => v + G1 * t2i[n][d])));
      T = T.map((t, m) => nrm(t.map((v, d) => v + G2 * i2t[m][d])));
    }
    return out;
  }
  const RUN = { late: run(false), deep: run(true) };
  const shareMap = (lay, j) => lay.O.map(f => PE.softmax(lay.T.map(t => BS * dot(f, t)))[j]);
  const attnMap = (lay, j) => lay.O.map(f => PE.softmax(lay.T.map(t => BETA * dot(f, t)))[j]);
  const SH = {}; for (const m of ['late', 'deep']) for (const w of ['ray', 'peng']) SH[m + w] = RUN[m].map(lay => shareMap(lay, WORD[w]));
  const mean = a => a.reduce((s, x) => s + x, 0) / a.length;
  function stats(m, w, i) {
    const sh = SH[m + w][i], by = {}; cells.forEach((c, n) => { (by[c.kd] = by[c.kd] || []).push(sh[n]); });
    const others = ['ray', 'peng', 'fish'].filter(kd => kd !== w).map(kd => [kd, mean(by[kd])]).sort((a, b) => b[1] - a[1]);
    return { tgt: mean(by[w]), other: others[0][1], otherKind: others[0][0] };
  }
  const S = { mode: 'late', word: 'ray', layer: L, u: 1, from: null, disp: SH.lateray.map(a => a.slice()) };
  let stop = () => {};

  function geo(w, h) {
    const hh = h - bottom, narrow = w < 620, labL = narrow ? 4 : 58;
    const colW = (w - labL - 8) / (L + 1), cx = i => labL + 4 + colW * (i + .5);
    const tw = colW - (narrow ? 6 : 14), th = tw * GH / GW;
    const yText = narrow ? 52 : 34, textH = narrow ? 22 : 30, yThumb = yText + textH + (narrow ? 30 : 40);
    const topB = yThumb + th + (narrow ? 32 : 22);
    let big, ch;
    if (narrow) {
      const bw = w - 16, bh = Math.min(bw * GH / GW, (hh - topB - 30) * .6);
      big = { x: (w - bh * GW / GH) / 2, y: topB + 28, w: bh * GW / GH, h: bh };
      ch = { x: 40, y: big.y + bh + 36, w: w - 56, h: Math.max(60, hh - (big.y + bh + 36) - 34) };
    } else {
      const avail = hh - topB - 40, bw = Math.min(w * .5, avail * GW / GH), bh = bw * GH / GW;
      big = { x: 12, y: topB + 18, w: bw, h: bh };
      ch = { x: bw + 76, y: big.y + 8, w: w - bw - 100, h: bh - 40 };
    }
    return { hh, narrow, labL, colW, cx, tw, th, yText, textH, yThumb, topB, big, ch };
  }
  const heat = (k, v) => DK.mixh(k.bg, k.G, PE.clamp(v, 0, 1) * .92);
  function drawMap(ctx, k, x0, y0, w, h, vals, o2 = {}) {
    const cw = w / GW, chh = h / GH;
    cells.forEach((c, n) => { ctx.fillStyle = heat(k, vals[n]); ctx.fillRect(x0 + c.x * cw, y0 + c.y * chh, Math.ceil(cw), Math.ceil(chh)); });
    if (!o2.outline) return;
    const ink = DK.mixh(k.bg, k.fg, .6);
    ctx.save(); ctx.strokeStyle = ink; ctx.lineWidth = o2.lw || 1.4;
    ctx.beginPath(); ctx.ellipse(x0 + 4.5 * cw, y0 + 6.6 * chh, 2.6 * cw, 1.35 * chh, 0, 0, 7); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(x0 + 7.1 * cw, y0 + 6.6 * chh); ctx.lineTo(x0 + 9.2 * cw, y0 + 6.2 * chh); ctx.stroke();
    [[10, 1], [13, 2]].forEach(([px, py]) => { DK.rr(ctx, x0 + (px + .15) * cw, y0 + (py + .1) * chh, 1.7 * cw, 3.8 * chh, cw * .7); ctx.stroke();
      ctx.beginPath(); ctx.arc(x0 + (px + 1) * cw, y0 + (py + .8) * chh, cw * .35, 0, 7); ctx.stroke(); });
    [[2, 2], [7, 3], [12, 8]].forEach(([px, py]) => { ctx.beginPath(); ctx.ellipse(x0 + (px + 1) * cw, y0 + (py + .5) * chh, .85 * cw, .35 * chh, 0, 0, 7); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(x0 + (px + .15) * cw, y0 + (py + .5) * chh); ctx.lineTo(x0 + (px - .3) * cw, y0 + (py + .15) * chh); ctx.lineTo(x0 + (px - .3) * cw, y0 + (py + .85) * chh); ctx.closePath(); ctx.stroke(); });
    ctx.restore();
  }
  function draw(ctx, w, h) {
    const k = DK.C(); ctx.clearRect(0, 0, w, h); if (w < 80 || h < 80) return;
    const g = geo(w, h), { narrow, cx, tw, th, yText, textH, yThumb, big, ch } = g, deep = S.mode === 'deep';
    const gX = glow('xmha'), gI = glow('image'), gTx = glow('text'), gT2 = glow('t2i'), gI2 = glow('i2t'), gA = glow('attn');
    const lay = RUN[S.mode], j = WORD[S.word], fs = narrow ? 10 : 11.5;
    // row labels
    ctx.save(); ctx.font = `600 ${fs}px system-ui, sans-serif`; ctx.textBaseline = 'middle';
    if (narrow) { ctx.fillStyle = k.Y; ctx.fillText(`text: "${KIND[S.word]}" token P^i`, 6, yText - 16); ctx.fillStyle = k.B; ctx.fillText('image O^i: share of each region\'s score', 6, yThumb + th + 27); }
    else { ctx.fillStyle = k.Y; ctx.fillText('text P^i', 4, yText + textH / 2); ctx.fillStyle = k.B; ctx.fillText('image O^i', 4, yThumb + th / 2); }
    ctx.restore();
    // exchange arrows between layer i and i+1
    for (let i = 0; i < L; i++) {
      const xa = cx(i), xb = cx(i + 1), yT = yText + textH + 2, yB = yThumb - 3;
      PE.arrow(ctx, xa + tw / 2 * .35, yText + textH / 2, xb - tw / 2 * .35 - 4, yText + textH / 2, DK.mixh(k.bg, k.Y, .5), 1.2, 6);
      if (deep || gX > .02) {
        const a = deep ? 1 : .35 * gX;
        PE.arrow(ctx, xa + 6, yT, xb - 10, yB, k.Y, 1.6 + 2 * (gX + gT2), 7, a);
        PE.arrow(ctx, xa + 6, yB, xb - 10, yT, k.B, 1.6 + 2 * (gX + gI2), 7, a);
      }
    }
    ctx.save(); ctx.font = `500 ${narrow ? 10 : 11}px system-ui, sans-serif`; ctx.fillStyle = deep ? k.fg : k.mu; ctx.textAlign = 'center';
    if (!narrow) ctx.fillText(deep ? 'deep fusion: X-MHA between every pair of layers' : 'late fusion: no exchange between the lanes', (cx(2) + cx(4)) / 2, yText - 8);
    ctx.restore();
    // text row: the token's vector as five bars
    for (let i = 0; i <= L; i++) {
      const t = lay[i].T[j], x = cx(i), bw = Math.min(tw, 70), y0 = yText;
      ctx.save(); DK.rr(ctx, x - bw / 2, y0, bw, textH, 6); ctx.fillStyle = k.bg; ctx.fill(); ctx.strokeStyle = k.Y; ctx.lineWidth = 1 + 1.6 * gTx; ctx.stroke();
      const n = 5, bwi = (bw - 14) / n;
      t.forEach((v, d) => { const hb = Math.max(1, (textH - 8) * Math.max(0, v)); ctx.fillStyle = DK.mixh(k.bg, k.Y, .85); ctx.fillRect(x - bw / 2 + 7 + d * bwi + 1, y0 + textH - 4 - hb, bwi - 2, hb); });
      ctx.restore();
    }
    if (!narrow) { ctx.save(); ctx.font = `600 ${fs}px system-ui, sans-serif`; ctx.fillStyle = k.Y; ctx.textAlign = 'center'; ctx.fillText(`"${KIND[S.word]}"`, cx(0), yText - 8); ctx.restore(); }
    // image row: thumbnails
    for (let i = 0; i <= L; i++) {
      const x = cx(i) - tw / 2;
      drawMap(ctx, k, x, yThumb, tw, th, S.disp[i]);
      ctx.save(); ctx.strokeStyle = i === S.layer ? k.fg : k.B; ctx.lineWidth = i === S.layer ? 2.4 : 1 + 1.6 * gI; ctx.strokeRect(x, yThumb, tw, th); ctx.restore();
      ctx.save(); ctx.font = `${i === S.layer ? 700 : 500} ${narrow ? 10 : 11}px system-ui, sans-serif`; ctx.fillStyle = i === S.layer ? k.fg : k.mu; ctx.textAlign = 'center';
      ctx.fillText(narrow ? String(i) : (i === 0 ? 'layer 0' : String(i)), cx(i), yThumb + th + 13); ctx.restore();
    }
    // big map of the selected layer, with detections
    const vals = S.disp[S.layer];
    drawMap(ctx, k, big.x, big.y, big.w, big.h, vals, { outline: true, lw: narrow ? 1.2 : 1.6 });
    ctx.save(); ctx.strokeStyle = k.B; ctx.lineWidth = 1.2; ctx.strokeRect(big.x, big.y, big.w, big.h); ctx.restore();
    ctx.save(); ctx.font = `600 ${fs}px system-ui, sans-serif`; ctx.fillStyle = k.fg; ctx.textBaseline = 'alphabetic';
    ctx.fillText(`Layer ${S.layer}: share of each region's score on "${KIND[S.word]}"`, big.x, big.y - 7); ctx.restore();
    const cw = big.w / GW, chh = big.h / GH;
    INST.forEach(([kd], ii) => {
      const ns = cells.map((c, n) => [c, n]).filter(([c]) => c.i === ii), v = mean(ns.map(([, n]) => vals[n]));
      if (v < .5) return;
      const xs = ns.map(([c]) => c.x), ys = ns.map(([c]) => c.y), x0 = big.x + Math.min(...xs) * cw - 3, y0 = big.y + Math.min(...ys) * chh - 3;
      const bw = (Math.max(...xs) - Math.min(...xs) + 1) * cw + 6, bh = (Math.max(...ys) - Math.min(...ys) + 1) * chh + 6;
      DK.brackets(ctx, x0, y0, bw, bh, k.G, { lw: 2.2 });
      const room = y0 - 24 >= big.y;
      DK.tag(ctx, room ? x0 : x0 - 4, room ? y0 : y0 + 26, KIND[S.word], v.toFixed(2), k.G, k, { size: narrow ? 10 : 11, maxX: big.x + big.w, align: room ? 'left' : 'right' });
    });
    if (gA > .02) {
      const at = attnMap(lay[S.layer], j); ctx.save(); ctx.globalAlpha = gA;
      cells.forEach((c, n) => { if (at[n] > .02) PE.dot(ctx, big.x + (c.x + .5) * cw, big.y + (c.y + .5) * chh, 1 + Math.min(cw, chh) * .42 * at[n], k.Y); });
      ctx.restore();
    }
    // chart: share on the word's own regions vs the best other object, per layer, both modes
    const ax = PE.axes(ctx, ch, [0, L], [0, 1], { xt: [0, 1, 2, 3, 4, 5, 6], yt: [0, .5, 1], xl: 'layer', yl: 'mean share', ylo: 30 });
    for (const m of ['late', 'deep']) {
      const on = m === S.mode, dash = m === 'late' ? [5, 4] : null, pts = i => stats(m, S.word, i);
      const P1 = [], P2 = []; for (let i = 0; i <= L; i++) { const s = pts(i); P1.push([ax.X(i), ax.Y(s.tgt)]); P2.push([ax.X(i), ax.Y(s.other)]); }
      PE.polyline(ctx, P1, k.G, on ? 2.6 : 1.4, dash, on ? 1 : .35);
      PE.polyline(ctx, P2, k.mu, on ? 2.2 : 1.2, dash, on ? 1 : .35);
      if (on) { const s = pts(0); ctx.save(); ctx.font = `600 ${narrow ? 10 : 11}px system-ui, sans-serif`; ctx.textAlign = 'left';
        ctx.fillStyle = k.G; ctx.fillText(KIND[S.word] + ' regions', ax.X(.1), ax.Y(s.tgt) - 7);
        ctx.fillStyle = k.mu; ctx.fillText('best other object: ' + KIND[s.otherKind], ax.X(.1), ax.Y(s.other) - 7); ctx.restore(); }
    }
    ctx.save(); ctx.font = `500 ${narrow ? 10 : 11}px system-ui, sans-serif`; ctx.fillStyle = k.mu; ctx.textAlign = 'left';
    ctx.fillText('solid: deep fusion; dashed: late', ch.x, ch.y - 8); ctx.restore();
  }
  function height(w) {
    const g = geo(w, 4000), narrow = w < 620;
    return narrow ? g.topB + 28 + (w - 16) * GH / GW * .78 + 36 + 150 + 34 : g.topB + 18 + Math.min(w * .5 * GH / GW, 330) + 34;
  }
  const cv = PE.canvas(el, o.aspect || (w => height(w) / w), draw);
  function describe() {
    const s = stats(S.mode, S.word, S.layer), s0 = stats(S.mode, S.word, 0), word = KIND[S.word], oth = KIND[s.otherKind];
    if (S.mode === 'late') return `Late fusion: the image features never see the prompt, so every layer shows the layer-0 map. The ${word} regions put ${s.tgt.toFixed(2)} of their score on "${word}"; the ${oth} regions still put ${s.other.toFixed(2)} there.`;
    return `Deep fusion, layer ${S.layer}: the ${word} regions put ${s.tgt.toFixed(2)} of their score on "${word}" and the ${oth} regions ${s.other.toFixed(2)} (layer 0: ${s0.tgt.toFixed(2)} and ${s0.other.toFixed(2)}). The image features now depend on the prompt.`;
  }
  const say = () => o.onSay && o.onSay(describe());
  function retarget(ms = 900) {
    const from = S.disp.map(a => a.slice()), to = SH[S.mode + S.word];
    stop(); stop = PE.animate(ms, u => {
      for (let i = 0; i <= L; i++) { const ui = PE.clamp((u - i * .07) / (1 - L * .07), 0, 1); S.disp[i] = from[i].map((v, n) => PE.lerp(v, to[i][n], ui)); }
      cv.redraw();
    });
  }
  // Mode and word are states: setting one twice equals setting it once.
  function setMode(m) { if (S.mode !== m) { S.mode = m; retarget(); } say(); }
  function setWord(wd) { if (S.word !== wd) { S.word = wd; retarget(600); } say(); }
  function setLayer(i) { S.layer = PE.clamp(i, 0, L); cv.redraw(); say(); o.onPick && o.onPick(S.layer); }
  function reset() { stop(); Object.assign(S, { mode: 'late', word: 'ray', layer: L }); S.disp = SH.lateray.map(a => a.slice()); cv.redraw(); say(); }
  const onClick = e => {
    const r2 = el.getBoundingClientRect(), x = e.clientX - r2.left, y = e.clientY - r2.top, g = geo(r2.width, r2.height);
    if (y < g.yThumb - 4 || y > g.yThumb + g.th + 16) return;
    for (let i = 0; i <= L; i++) if (Math.abs(x - g.cx(i)) <= g.colW / 2) { setLayer(i); return; }
  };
  const onKey = e => { const d = { ArrowLeft: -1, ArrowRight: 1 }[e.key]; if (!d) return; e.preventDefault(); setLayer(S.layer + d); };
  el.addEventListener('click', onClick); el.addEventListener('keydown', onKey);
  say();
  return { setMode, setWord, setLayer, reset, redraw: cv.redraw, resize: () => cv.resize && cv.resize(),
    mode: () => S.mode, word: () => S.word,
    destroy: () => { stop(); el.removeEventListener('click', onClick); el.removeEventListener('keydown', onKey); cv.destroy && cv.destroy(); } };
};
</script>
