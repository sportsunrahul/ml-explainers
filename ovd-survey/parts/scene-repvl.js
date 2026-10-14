<script>
/* Chapter 10 toy, shared by the long page and the atlas. One feature map of a plain YOLO over the warehouse
   (every cell predicts; task-aligned assignment picks a few positives per object), then YOLO-World's text gate in
   the neck (Eq. 2: sigmoid of the max over prompt similarities, one gate per cell), then the fold (App. Eq. 4): the
   text encoder runs once and its output becomes 1x1 conv weights, while I-Pooling still reads 27 pooled tokens
   from each image. Toy grid (16 x 10 cells), toy similarities, toy scores.
   o: { aspect(w) -> h/w, glow(word) -> 0..1, onSay(text), bottom (px kept free at the bottom) }
   API: set('dense' | 'gate' | 'reparam'), setVocab(4 | 3), reset(), redraw(), resize(), state(), destroy() */
const SceneRepvl = (el, o = {}) => {
  const NX = 16, NY = 10, glow = o.glow || (() => 0), bottom = o.bottom || 0;
  const NOUNS = ['forklift', 'person', 'pallet', 'traffic cone'];
  const ROWS = ['person', 'pallet', 'traffic cone', 'car', 'chair'];
  const OBJS = DK.WAREHOUSE.filter(x => x.name);
  const sig = z => 1 / (1 + Math.exp(-z));
  const rnd = PE.rng(10);
  const CELLS = [];
  for (let j = 0; j < NY; j++) for (let i = 0; i < NX; i++) {
    const u = (i + .5) / NX, v = (j + .5) / NY;
    let own = null, d = 9;
    OBJS.forEach(ob => { const [x, y, w, h] = ob.box, dd = Math.max(Math.abs(u - x - w / 2) / (w / 2), Math.abs(v - y - h / 2) / (h / 2));
      if (dd <= 1 && dd < d) { d = dd; own = ob.name; } });
    const sim = NOUNS.map(n => own === n ? 3.3 - 2.3 * d * d + .3 * (rnd() - .5) : own ? -1.3 + .4 * (rnd() - .5) : -2.4 + .8 * (rnd() - .5));
    const x = Array.from({ length: 8 }, () => .25 + .75 * rnd());
    const jit = Array.from({ length: 4 }, () => rnd() - .5);
    CELLS.push({ i, j, u, v, own, d, sim, x, jit, pos: false });
  }
  // Toy task-aligned assignment: the three cells nearest each object's centre become its positives.
  OBJS.forEach(ob => CELLS.filter(c => c.own === ob.name).sort((a, b) => a.d - b.d).slice(0, 3).forEach(c => { c.pos = true; }));
  const SEL0 = 5 * NX + 12;
  const reset0 = () => ({ mode: 'dense', vocab: 4, uG: 0, uF: 0, vw: [1, 1, 1, 1], sel: SEL0 });
  let S = reset0(), stopM = () => {}, stopV = () => {};

  const effSim = c => c.sim.map((s, n) => PE.lerp(-12, s, S.vw[n]));
  const best = c => { const e = effSim(c); let m = 0; e.forEach((s, n) => { if (s > e[m]) m = n; }); return { n: m, s: e[m] }; };
  const gate = c => sig(best(c).s);
  const rowScores = c => ROWS.map(r => { const n = NOUNS.indexOf(r); return n >= 0 ? sig(c.sim[n] - .4) : sig(-3.2 + c.jit[0]); });
  function predBox(c) {
    const ob = OBJS.find(b => b.name === c.own);
    if (!ob) return [c.u - .035, c.v - .06, .07, .12];
    const [x, y, w, h] = ob.box, e = .1 + .25 * c.d;
    return [x + w * e * c.jit[0], y + h * e * c.jit[1], w * (1 + e * c.jit[2]), h * (1 + e * c.jit[3])];
  }

  function geo(w, h) {
    const hh = h - bottom, narrow = w < 620;
    if (!narrow) {
      const imgW = Math.min(w * .54, Math.max(120, (hh - 160) * 1.55)), imgH = imgW / 1.55;
      const A = { x: 12, y: 30, w: imgW, h: imgH };
      const B = { x: 12, y: A.y + imgH + 18, w: imgW, h: hh - A.y - imgH - 26 };
      const P = { x: imgW + 40, y: 30, w: w - imgW - 52, h: hh - 38 };
      return { narrow, hh, A, B, P, fs: 12, sm: 11 };
    }
    let imgW = w - 16, imgH = Math.min(imgW / 1.55, hh * .4);
    imgW = imgH * 1.55;
    const A = { x: (w - imgW) / 2, y: 22, w: imgW, h: imgH }, rest = hh - A.y - imgH - 12;
    const showB = rest >= 360;
    const P = { x: 8, y: A.y + imgH + 14, w: w - 16, h: showB ? rest - 128 : rest };
    const B = showB ? { x: 8, y: P.y + P.h + 6, w: w - 16, h: 118 } : null;
    return { narrow, hh, A, B, P, fs: 11, sm: 10 };
  }
  const font = (wt, px) => `${wt} ${px}px system-ui, sans-serif`;
  const halo = (ctx, k, s, x, y) => { ctx.lineWidth = 3.5; ctx.strokeStyle = k.bg; ctx.lineJoin = 'round'; ctx.strokeText(s, x, y); ctx.fillText(s, x, y); };

  function drawImage(ctx, k, G) {
    const { x: X, y: Y, w: W, h: H } = G.A, cw = W / NX, ch = H / NY, uG = S.uG;
    ctx.save();
    ctx.fillStyle = DK.mixh(k.bg, k.p, .6); ctx.fillRect(X, Y, W, H);
    DK.drawScene(ctx, DK.WAREHOUSE, X, Y, W, H, k, 1 - .5 * uG);
    const gG = glow('gate');
    if (uG > 0) CELLS.forEach(c => { ctx.globalAlpha = uG * (.08 + .62 * gate(c)) + .25 * gG * gate(c); ctx.fillStyle = k.B; ctx.fillRect(X + c.i * cw + .5, Y + c.j * ch + .5, cw - 1, ch - 1); });
    ctx.globalAlpha = .55; ctx.strokeStyle = k.rule; ctx.lineWidth = 1; ctx.beginPath();
    for (let i = 0; i <= NX; i++) { ctx.moveTo(X + i * cw, Y); ctx.lineTo(X + i * cw, Y + H); }
    for (let j = 0; j <= NY; j++) { ctx.moveTo(X, Y + j * ch); ctx.lineTo(X + W, Y + j * ch); }
    ctx.stroke(); ctx.globalAlpha = 1;
    const gC = glow('cells'), dotA = 1 - .6 * uG;
    CELLS.forEach(c => {
      const px = X + (c.i + .5) * cw, py = Y + (c.j + .5) * ch;
      if (c.pos) PE.dot(ctx, px, py, 3.4 + 2 * gC, k.G, Math.max(dotA, gC));
      else if (c.own) PE.dot(ctx, px, py, 2.2 + 1.5 * gC, k.B, Math.max(dotA, gC));
      else PE.dot(ctx, px, py, 1.4 + gC, k.mu, Math.max(dotA * .8, gC));
    });
    // detections: closed-set rows in the dense state, prompts once text is in
    const gS = glow('score');
    OBJS.forEach(ob => {
      const [bx, by, bw, bh] = DK.boxOf(ob, X, Y, W, H);
      let name = null, sc = 0, fromText = uG > .5;
      if (!fromText) { const n = ROWS.indexOf(ob.name); if (n >= 0) { const c = CELLS.filter(q => q.own === ob.name).sort((a, b) => a.d - b.d)[0]; name = ob.name; sc = rowScores(c)[n]; } }
      else { const c = CELLS.filter(q => q.own === ob.name).sort((a, b) => a.d - b.d)[0], bb = best(c); sc = sig(bb.s - .9); if (sc > .3) name = NOUNS[bb.n]; }
      if (name) {
        DK.brackets(ctx, bx, by, bw, bh, k.G, { lw: 2 + 1.5 * gS });
        DK.tag(ctx, bx, by, name, sc.toFixed(2), k.G, k, { size: G.sm, nameCol: fromText ? k.Y : k.fg, maxX: X + W });
      } else {
        DK.brackets(ctx, bx, by, bw, bh, k.Br, { dash: [4, 3], alpha: .9 });
        DK.tag(ctx, bx, by, 'background', null, k.Br, k, { size: G.sm, nameCol: k.Br, maxX: X + W });
      }
    });
    // the selected cell, and in the dense state its own predicted box with the four edge distances
    const c = CELLS[S.sel], cx0 = X + c.i * cw, cy0 = Y + c.j * ch;
    if (uG < 1) {
      const [u, v, a, b] = predBox(c), bx = X + u * W, by = Y + v * H, bw = a * W, bh = b * H, px = cx0 + cw / 2, py = cy0 + ch / 2, al = 1 - uG;
      DK.brackets(ctx, bx, by, bw, bh, k.B, { lw: 2, alpha: al, len: .18 });
      ctx.save(); ctx.globalAlpha = al; ctx.strokeStyle = k.B; ctx.lineWidth = 1.2; ctx.setLineDash([3, 2]); ctx.beginPath();
      ctx.moveTo(px, py); ctx.lineTo(bx, py); ctx.moveTo(px, py); ctx.lineTo(bx + bw, py); ctx.moveTo(px, py); ctx.lineTo(px, by); ctx.moveTo(px, py); ctx.lineTo(px, by + bh); ctx.stroke(); ctx.restore();
    }
    ctx.strokeStyle = k.fg; ctx.lineWidth = 2; ctx.strokeRect(cx0 + 1, cy0 + 1, cw - 2, ch - 2);
    ctx.restore();
  }

  function drawPAN(ctx, k, B, G) {
    const { x, y, w, h } = B, fs = G.sm, A0 = ctx.globalAlpha;
    ctx.save(); ctx.font = font(600, G.fs); ctx.fillStyle = k.fg; ctx.textBaseline = 'middle';
    ctx.fillText('PAN neck: a top-down path, then a bottom-up path', x, y + 8);
    const rowH = Math.min(28, (h - 40) / 3), y0 = y + 26, cols = [x + 4, x + w * .28, x + w * .46, x + w * .64];
    ctx.font = font(400, fs); ctx.fillStyle = k.mu;
    ['backbone', 'top-down', 'bottom-up'].forEach((s, i) => { ctx.textAlign = i ? 'center' : 'left'; ctx.fillText(s, i ? cols[i] : cols[0], y0 - 2); });
    ctx.textAlign = 'left';
    const lv = [5, 4, 3], sz = { 5: 7, 4: 10, 3: 13 }, side = { 5: '20x20', 4: '40x40', 3: '80x80' };
    lv.forEach((l, r) => {
      const yy = y0 + 12 + (r + .5) * rowH, s = sz[l];
      ctx.fillStyle = DK.mixh(k.bg, k.B, .35); ctx.strokeStyle = k.B; ctx.lineWidth = 1.2;
      ctx.fillRect(cols[0], yy - s / 2, s, s); ctx.strokeRect(cols[0], yy - s / 2, s, s);
      ctx.fillStyle = k.fg; ctx.font = font(600, fs); ctx.fillText('C' + l, cols[0] + 18, yy);
      PE.arrow(ctx, cols[0] + 40, yy, cols[1] - 7, yy, k.B, 1.2, 6, A0);
      [1, 2].forEach(ci => { PE.dot(ctx, cols[ci], yy, 4.5, k.B, A0); }); ctx.globalAlpha = A0;
      PE.arrow(ctx, cols[1] + 7, yy, cols[2] - 7, yy, k.B, 1.2, 6, A0);
      PE.arrow(ctx, cols[2] + 7, yy, cols[3] - 4, yy, k.B, 1.2, 6, A0);
      ctx.fillStyle = k.fg; ctx.fillText('P' + l, cols[3], yy); ctx.fillStyle = k.mu; ctx.font = font(400, fs); ctx.fillText(side[l], cols[3] + 22, yy);
      if (r < 2) PE.arrow(ctx, cols[1], yy + 6, cols[1], yy + rowH - 6, k.B, 1.4, 6, A0);
      if (r > 0) PE.arrow(ctx, cols[2], yy - 6, cols[2], yy - rowH + 6, k.B, 1.4, 6, A0);
    });
    ctx.fillStyle = k.mu; ctx.font = font(400, fs);
    const ty = y0 + 12 + 3 * rowH + 10;
    if (ty < y + h + 4) ctx.fillText('every cell of P3, P4, P5 predicts: 8400 cells at a 640 input (our reading)', x, ty);
    ctx.restore();
  }

  function drawIPool(ctx, k, B, G) {
    const { x, y, w, h } = B, fs = G.sm, gP = glow('pooled'), A0 = ctx.globalAlpha;
    ctx.save(); ctx.font = font(600, G.fs); ctx.fillStyle = k.fg; ctx.textBaseline = 'middle';
    ctx.fillText('I-Pooling: the text also looks at the image', x, y + 8);
    ctx.font = font(400, fs); ctx.fillStyle = k.mu;
    ctx.fillText('max-pool P3, P4, P5 to 3x3 each: 27 tokens', x, y + 26);
    const s = Math.min(9, (h - 50) / 3.6), g0 = y + 40;
    ['P3', 'P4', 'P5'].forEach((name, m) => {
      const gx = x + m * (3 * s + 22);
      for (let a = 0; a < 3; a++) for (let b = 0; b < 3; b++) {
        // pool the toy grid in 3 x 3 regions (offset per level so the three maps differ a little)
        const i0 = Math.floor(b * NX / 3), i1 = Math.floor((b + 1) * NX / 3), j0 = Math.floor(a * NY / 3), j1 = Math.floor((a + 1) * NY / 3);
        let mx = 0; CELLS.forEach(c => { if (c.i >= i0 && c.i < i1 && c.j >= j0 && c.j < j1) mx = Math.max(mx, (.3 + .7 * gate(c)) * (c.x[m] + c.x[m + 3]) / 2); });
        ctx.fillStyle = DK.mixh(k.bg, k.B, PE.clamp(.15 + .8 * mx, 0, 1)); ctx.fillRect(gx + b * s, g0 + a * s, s - 1, s - 1);
      }
      if (gP > 0) { ctx.strokeStyle = k.Y; ctx.globalAlpha = A0 * gP; ctx.lineWidth = 2; ctx.strokeRect(gx - 2, g0 - 2, 3 * s + 3, 3 * s + 3); ctx.globalAlpha = A0; }
      ctx.fillStyle = k.mu; ctx.textAlign = 'center'; ctx.fillText(name, gx + 1.5 * s, g0 + 3 * s + 9); ctx.textAlign = 'left';
    });
    const ax = x + 3 * (3 * s + 22) + 2, ay = g0 + 1.5 * s;
    PE.arrow(ctx, ax, ay, ax + 26, ay, k.G, 1.6, 7, A0);
    ctx.font = font(600, fs); ctx.fillStyle = k.Y; ctx.fillText("W' = W + attention", ax + 32, ay - 7);
    ctx.font = font(400, fs); ctx.fillStyle = S.uF > .5 ? k.fg : k.mu;
    ctx.fillText(S.uF > .5 ? (G.narrow ? 'still runs per image' : 'still runs on every image after folding') : (G.narrow ? 'text queries 27 tokens' : 'text queries, 27 image tokens'), ax + 32, ay + 8);
    ctx.restore();
  }

  function drawDensePanel(ctx, k, P, G, al) {
    const c = CELLS[S.sel], fs = G.fs, sm = G.sm;
    ctx.save(); ctx.globalAlpha = al; ctx.textBaseline = 'middle';
    let y = P.y + 8;
    ctx.font = font(600, fs + 1); ctx.fillStyle = k.fg; ctx.fillText('Plain YOLO: every cell predicts', P.x, y); y += 20;
    ctx.font = font(400, sm); ctx.fillStyle = k.mu; ctx.fillText(`Selected cell: column ${c.i + 1}, row ${c.j + 1} (white square)`, P.x, y); y += 22;
    ctx.font = font(600, fs); ctx.fillStyle = k.B; ctx.fillText('Box branch', P.x, y);
    ctx.font = font(400, sm); ctx.fillStyle = k.fg; ctx.fillText('four distances to the box edges', P.x + (G.narrow ? 74 : 82), y); y += 22;
    ctx.font = font(600, fs); ctx.fillStyle = k.B; ctx.fillText('Class branch', P.x, y);
    ctx.font = font(400, sm); ctx.fillStyle = k.fg; ctx.fillText('one score per class row', P.x + (G.narrow ? 74 : 82), y); y += 18;
    const sc = rowScores(c), mx = Math.max(...sc), lab = G.narrow ? 76 : 86, bw = P.w - lab - 44, rh = G.narrow ? 15 : 17;
    ROWS.forEach((r, n) => {
      const yy = y + n * rh + rh / 2;
      ctx.font = font(500, sm); ctx.fillStyle = k.fg; ctx.fillText(r, P.x + 6, yy);
      ctx.fillStyle = PE.mix(k.bg, k.fg, .14); ctx.fillRect(P.x + lab, yy - 5, bw, 10);
      ctx.fillStyle = sc[n] === mx && mx > .3 ? k.G : PE.mix(k.bg, k.fg, .4); ctx.fillRect(P.x + lab, yy - 5, Math.max(1, bw * sc[n]), 10);
      ctx.fillStyle = k.mu; ctx.fillText(sc[n].toFixed(2), P.x + lab + bw + 6, yy);
    });
    y += ROWS.length * rh + 12;
    ctx.font = font(600, sm);
    if (mx > .3) { ctx.fillStyle = k.G; ctx.fillText(`best row: ${ROWS[sc.indexOf(mx)]}`, P.x + 6, y); }
    else { ctx.fillStyle = k.Br; ctx.fillText(c.own === 'forklift' ? 'no row fires: no class "forklift" exists' : 'no row fires: background', P.x + 6, y); }
    y += 24;
    if (y + 44 < P.y + P.h) {
      ctx.font = font(400, sm);
      [[k.G, 3.4, 'positive: picked by task-aligned assignment'], [k.B, 2.2, 'inside an object, not picked'], [k.mu, 1.4, 'background']].forEach(([col, rad, s], m) => {
        PE.dot(ctx, P.x + 10, y + m * 16, rad + 1, col, al); ctx.globalAlpha = al; ctx.fillStyle = k.mu; ctx.fillText(s, P.x + 22, y + m * 16); });
    }
    ctx.restore();
  }

  function drawTextPanel(ctx, k, P, G, al) {
    const c = CELLS[S.sel], fs = G.fs, sm = G.sm, uF = S.uF, gT = glow('text'), gCv = glow('conv');
    ctx.save(); ctx.globalAlpha = al; ctx.textBaseline = 'middle';
    let y = P.y + 8;
    ctx.font = font(600, fs + 1); ctx.fillStyle = k.fg; ctx.fillText(uF > .5 ? 'Folded: the prompts are now weights' : 'Text in the neck: one gate per cell', P.x, y); y += 22;
    // row A: text encoder -> W
    const encW = Math.min(150, P.w * .4), encH = 52, wx = P.x + encW + 30, rowH = 12;
    ctx.save(); ctx.globalAlpha = al * (1 - .62 * uF);
    DK.rr(ctx, P.x, y, encW, encH, 8); ctx.fillStyle = DK.mixh(k.bg, k.Br, .12); ctx.fill(); ctx.setLineDash([4, 3]); ctx.strokeStyle = k.Br; ctx.lineWidth = 1.3; ctx.stroke(); ctx.setLineDash([]);
    ctx.font = font(600, sm); ctx.fillStyle = k.Y; ctx.textAlign = 'center'; ctx.fillText('CLIP text encoder', P.x + encW / 2, y + 17);
    ctx.font = font(400, sm); ctx.fillStyle = k.Br; ctx.fillText('frozen', P.x + encW / 2, y + 34);
    ctx.textAlign = 'left'; ctx.restore();
    if (uF > 0) { ctx.save(); ctx.globalAlpha = al * uF; ctx.font = font(600, sm); ctx.fillStyle = k.Br; ctx.textAlign = 'center';
      ctx.fillText('ran once, offline', P.x + encW / 2, y + encH + 10); ctx.restore(); }
    PE.arrow(ctx, P.x + encW + 4, y + encH / 2, wx - 6, y + encH / 2, k.Y, 1.6, 7, al * (1 - .5 * uF));
    const wy = y + encH / 2 - (NOUNS.length * rowH) / 2, nameW = G.narrow ? 64 : 72;
    NOUNS.forEach((n, m) => {
      const yy = wy + m * rowH + rowH / 2, a = .25 + .75 * S.vw[m];
      ctx.save(); ctx.globalAlpha = al * a; ctx.font = font(500, sm - (G.narrow ? 1 : 0)); ctx.fillStyle = k.Y; ctx.fillText(n, wx, yy);
      for (let q = 0; q < 8; q++) { const hq = 2 + 7 * Math.abs(Math.sin(m * 2.3 + q * 1.7)); ctx.fillRect(wx + nameW + q * 6, yy + 4 - hq, 4, hq); }
      if (S.vw[m] < .5) { ctx.strokeStyle = k.mu; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(wx - 2, yy); ctx.lineTo(wx + nameW + 48, yy); ctx.stroke(); }
      ctx.restore();
    });
    const fx = wx - 6, fy = wy - 6, fw = nameW + 60, fh = NOUNS.length * rowH + 12;
    if (gT > 0) { ctx.save(); ctx.globalAlpha = gT; ctx.strokeStyle = k.Y; ctx.lineWidth = 2; DK.rr(ctx, fx, fy, fw, fh, 6); ctx.stroke(); ctx.restore(); }
    if (uF > 0 || gCv > 0) {
      ctx.save(); ctx.globalAlpha = al * Math.max(uF, gCv); ctx.strokeStyle = k.fg; ctx.lineWidth = 1.4 + 1.6 * gCv; DK.rr(ctx, fx, fy, fw, fh, 6); ctx.stroke();
      ctx.font = font(600, sm); ctx.fillStyle = k.fg; ctx.fillText('1x1 conv weights', fx, fy - 8);
      ctx.font = font(400, sm); ctx.fillStyle = k.mu; ctx.fillText('C x D x 1 x 1', fx, fy + fh + 9); ctx.restore();
    }
    ctx.font = font(400, sm); ctx.fillStyle = k.mu; ctx.textAlign = 'right';
    if (uF < .5) ctx.fillText('W: one row per prompt', P.x + P.w, y - 9);
    ctx.textAlign = 'left';
    y += encH + 34;
    // row B: this cell's similarity to each prompt, max, sigmoid
    ctx.font = font(600, fs); ctx.fillStyle = k.fg;
    ctx.fillText(uF > .5 ? 'Selected cell: same numbers, from the 1x1 conv' : 'Selected cell: x . w for each prompt', P.x, y); y += 16;
    const e = effSim(c), bb = best(c), lab = G.narrow ? 76 : 86, bwid = P.w - lab - 40, z0 = P.x + lab + bwid * .45, unit = bwid * .55 / 3.6, rh = G.narrow ? 15 : 17;
    const gSi = glow('sims'), gMx = glow('max'), gG = glow('gate'), gF = glow('features');
    NOUNS.forEach((n, m) => {
      const yy = y + m * rh + rh / 2, s = PE.clamp(e[m], -3.6, 3.6), on = S.vw[m] > .5, isMax = m === bb.n;
      ctx.save(); ctx.globalAlpha = al * (on ? 1 : .35);
      ctx.font = font(500, sm); ctx.fillStyle = k.Y; ctx.fillText(n, P.x + 6, yy);
      ctx.fillStyle = isMax ? k.G : PE.mix(k.bg, k.fg, .4);
      const L = s * unit; ctx.fillRect(Math.min(z0, z0 + L), yy - 5 - gSi * 1.5, Math.max(1, Math.abs(L)), 10 + gSi * 3);
      if (isMax && gMx > 0) { ctx.strokeStyle = k.G; ctx.lineWidth = 2; ctx.globalAlpha = al * gMx; ctx.strokeRect(Math.min(z0, z0 + L) - 3, yy - 8, Math.abs(L) + 6, 16); ctx.globalAlpha = al; }
      ctx.fillStyle = k.mu; ctx.fillText(on ? e[m].toFixed(1) : 'not a prompt', z0 + Math.max(L, 0) + 6, yy);
      ctx.restore();
    });
    ctx.save(); ctx.globalAlpha = al * .7; ctx.strokeStyle = k.mu; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(z0, y); ctx.lineTo(z0, y + NOUNS.length * rh); ctx.stroke(); ctx.restore();
    y += NOUNS.length * rh + 12;
    const g = sig(bb.s);
    ctx.font = font(600, sm + (gG > 0 ? 1 : 0)); ctx.fillStyle = k.fg;
    const t1 = `max: ${NOUNS[bb.n]} ${bb.s.toFixed(1)}`, t1w = ctx.measureText(t1).width;
    ctx.fillText(t1, P.x + 6, y);
    PE.arrow(ctx, P.x + 12 + t1w, y, P.x + 34 + t1w, y, k.mu, 1.2, 6, al);
    ctx.fillStyle = g > .5 ? k.G : k.Br; ctx.fillText(`gate = sigmoid = ${g.toFixed(2)}`, P.x + 40 + t1w, y);
    if (gG > 0) { ctx.save(); ctx.globalAlpha = gG; ctx.strokeStyle = k.G; ctx.lineWidth = 2; const tw = ctx.measureText(`gate = sigmoid = ${g.toFixed(2)}`).width; ctx.strokeRect(P.x + 36 + t1w, y - 9, tw + 8, 18); ctx.restore(); }
    y += 24;
    // row C: x -> gate * x
    if (y + 24 < P.y + P.h) {
      ctx.font = font(600, sm); ctx.fillStyle = k.B; ctx.fillText("features x", P.x + 6, y + 2);
      const sx0 = P.x + (G.narrow ? 76 : 86), bwq = G.narrow ? 4 : 5, hmax = 18 + 4 * gF;
      c.x.forEach((v, q) => { ctx.fillStyle = PE.mix(k.bg, k.B, .5); ctx.fillRect(sx0 + q * (bwq + 2), y + 9 - v * hmax, bwq, v * hmax); });
      const ax = sx0 + 8 * (bwq + 2) + 6; PE.arrow(ctx, ax, y + 1, ax + 22, y + 1, k.B, 1.3, 6, al);
      c.x.forEach((v, q) => { ctx.fillStyle = k.B; ctx.fillRect(ax + 28 + q * (bwq + 2), y + 9 - v * g * hmax, bwq, Math.max(1, v * g * hmax)); });
      ctx.fillStyle = k.B; ctx.fillText("x' = gate x", ax + 34 + 8 * (bwq + 2), y + 2);
    }
    ctx.restore();
  }

  function draw(ctx, w, h) {
    const k = DK.C(); ctx.clearRect(0, 0, w, h); if (w < 80 || h < 80) return;
    const G = geo(w, h);
    drawImage(ctx, k, G);
    // the two panels hand over in sequence (one fades out, then the other fades in), so their text never overlaps
    // An equation term that names a text-panel part shows that panel briefly, even in the plain-YOLO step.
    const peek = Math.max(...['sims', 'max', 'features', 'text', 'conv', 'pooled'].map(glow));
    const aT = Math.max(PE.clamp(2 * S.uG - 1, 0, 1), peek), aD = Math.min(PE.clamp(1 - 2 * S.uG, 0, 1), 1 - peek);
    if (aD > 0) drawDensePanel(ctx, k, G.P, G, aD);
    if (aT > 0) drawTextPanel(ctx, k, G.P, G, aT);
    if (G.B && G.B.h > 70) {
      if (aD > 0) { ctx.save(); ctx.globalAlpha = aD; drawPAN(ctx, k, G.B, G); ctx.restore(); }
      if (aT > 0) { ctx.save(); ctx.globalAlpha = aT; drawIPool(ctx, k, G.B, G); ctx.restore(); }
    }
  }
  const cv = PE.canvas(el, o.aspect || (w => (w < 620 ? (w / 1.55 + 430) / w : .6)), draw);

  function describe() {
    const c = CELLS[S.sel], where = c.own ? `on the ${c.own}` : 'on the background';
    if (S.mode === 'dense') {
      const sc = rowScores(c), mx = Math.max(...sc);
      return `Plain YOLO, closed set: every cell predicts a box and one score per class row. The selected cell lies ${where}; ` +
        (mx > .3 ? `its best row is "${ROWS[sc.indexOf(mx)]}" (${mx.toFixed(2)}).` : 'no row fires, so it counts as background. There is no "forklift" row.');
    }
    const bb = best(c), g = sig(bb.s), drop = S.vocab === 3 ? ' "forklift" is not a prompt now.' : '';
    if (S.mode === 'gate') return `Text in the neck: the selected cell (${where}) matches "${NOUNS[bb.n]}" best (${bb.s.toFixed(1)}), so its gate is ${g.toFixed(2)}.${drop} ` +
      (g > .5 ? 'Its features pass almost unchanged; background cells are damped.' : 'The neck damps this cell before the head sees it.');
    return `Folded: the text encoder ran once, offline, and its ${S.vocab} vectors are now 1x1 conv weights. The selected cell's gate is still ${g.toFixed(2)}, the same number as before. I-Pooling still reads 27 pooled tokens from each image.`;
  }
  const say = () => o.onSay && o.onSay(describe());
  function tweenMode(toG, toF) {
    const g0 = S.uG, f0 = S.uF; stopM();
    stopM = PE.animate(800, u => { S.uG = PE.lerp(g0, toG, u); S.uF = PE.lerp(f0, toF, u); cv.redraw(); });
  }
  function set(mode) {
    S.mode = mode;
    if (mode === 'dense') tweenMode(0, 0); else if (mode === 'gate') tweenMode(1, 0); else tweenMode(1, 1);
    say();
  }
  function setVocab(n) {
    S.vocab = n; const from = S.vw[0], to = n === 3 ? 0 : 1; stopV();
    stopV = PE.animate(700, u => { S.vw[0] = PE.lerp(from, to, u); cv.redraw(); });
    say();
  }
  function reset() { stopM(); stopV(); S = reset0(); cv.redraw(); say(); }
  const cellAt = e => { const r = el.getBoundingClientRect(), G = geo(r.width, r.height), x = e.clientX - r.left - G.A.x, y = e.clientY - r.top - G.A.y;
    if (x < 0 || y < 0 || x > G.A.w || y > G.A.h) return -1; return Math.min(NY - 1, Math.floor(y / (G.A.h / NY))) * NX + Math.min(NX - 1, Math.floor(x / (G.A.w / NX))); };
  let drag = false;
  const offs = [
    ['pointerdown', e => { const n = cellAt(e); if (n >= 0) { drag = true; S.sel = n; el.setPointerCapture(e.pointerId); cv.redraw(); } }],
    ['pointermove', e => { if (!drag) return; const n = cellAt(e); if (n >= 0 && n !== S.sel) { S.sel = n; cv.redraw(); } }],
    ['pointerup', () => { if (drag) say(); drag = false; }],
    ['keydown', e => { const d = { ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowUp: [0, -1], ArrowDown: [0, 1] }[e.key]; if (!d) return; e.preventDefault();
      const c = CELLS[S.sel], i = PE.clamp(c.i + d[0], 0, NX - 1), j = PE.clamp(c.j + d[1], 0, NY - 1); S.sel = j * NX + i; cv.redraw(); say(); }],
  ].map(([ev, fn]) => { el.addEventListener(ev, fn); return () => el.removeEventListener(ev, fn); });
  say();
  return { set, setVocab, reset, redraw: cv.redraw, resize: () => (cv.resize ? cv.resize() : cv.redraw()), state: () => ({ mode: S.mode, vocab: S.vocab }),
    destroy: () => { stopM(); stopV(); offs.forEach(f => f()); cv.destroy && cv.destroy(); } };
};
</script>
