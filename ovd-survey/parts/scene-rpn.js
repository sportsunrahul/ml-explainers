<script>
/* SceneRpn: the two-stage detector toy, shared by viz-rpn (ch. 3) and the atlas. A drawn warehouse, a 16 x 10
   feature grid, 9 anchors per cell (3 sizes x 3 shapes), a stand-in "trained" RPN whose objectness is a
   sigmoid of each anchor's best IoU with any object (names never enter), box deltas that move an anchor most
   of the way onto its object, NMS at IoU 0.7, top-k, a 7 x 7 RoIAlign crop and a closed-set region head.
   All numbers are toy values. The forklift has no class row and was background in training, so its
   objectness is scaled down slightly (ViLD Table 1 measures a gap of about 2 AR for novel classes).
     const sc = SceneRpn(canvasEl, { aspect, bottom, glow, onSay });
     sc.set(stage)  'image' | 'features' | 'anchors' | 'objectness' | 'deltas' | 'topk' | 'roialign' | 'head'
     sc.setK(k)     proposals kept, 1..12;  sc.play() steps through all stages (jumps to 'head' under reduced motion)
     sc.reset(); sc.redraw(); sc.resize(); sc.destroy(); sc.stage(); sc.k()
   Pointer: click a cell (early stages) or a proposal (late stages). Keys: arrows move the cell or cycle proposals. */
const SceneRpn = (el, o = {}) => {
  const GC = 16, GR = 10, AR = 1.6, KMAX = 12, K0 = 5;
  const STAGES = ['image', 'features', 'anchors', 'objectness', 'deltas', 'topk', 'roialign', 'head'];
  const glow = o.glow || (() => 0), bottom = o.bottom || 0;
  const OBJS = DK.WAREHOUSE.filter(q => !q.bg), ROWS = ['person', 'pallet', 'traffic cone', 'car', 'chair'];
  const iou = (a, b) => { const x0 = Math.max(a[0], b[0]), y0 = Math.max(a[1], b[1]), x1 = Math.min(a[0] + a[2], b[0] + b[2]), y1 = Math.min(a[1] + a[3], b[1] + b[3]);
    const I = Math.max(0, x1 - x0) * Math.max(0, y1 - y0); return I / (a[2] * a[3] + b[2] * b[3] - I); };
  const best = box => OBJS.reduce((m, q) => { const v = iou(box, q.box); return v > m.v ? { q, v } : m; }, { q: null, v: 0 });
  const toC = b => [b[0] + b[2] / 2, b[1] + b[3] / 2, b[2], b[3]], fromC = c => [c[0] - c[2] / 2, c[1] - c[3] / 2, c[2], c[3]];
  const lerpBox = (a, b, u) => a.map((v, i) => v + (b[i] - v) * u);

  // Anchors, objectness and regressed boxes, computed once.
  const rnd = PE.rng(11), ANCH = [];
  for (let j = 0; j < GR; j++) for (let i = 0; i < GC; i++) for (const s of [.16, .3, .5]) for (const ra of [.5, 1, 2]) {
    const cx = (i + .5) / GC, cy = (j + .5) / GR, w = s * Math.sqrt(ra) / AR, h = s / Math.sqrt(ra), box = [cx - w / 2, cy - h / 2, w, h];
    const m = best(box), nz = () => (rnd() - .5) * .04;
    let obj = 1 / (1 + Math.exp(-(m.v - .38) * 16)) * (m.q && m.q.row < 0 ? .92 : 1) + nz();
    obj = PE.clamp(obj, .01, .99);
    const ref = m.v > .3 ? fromC(lerpBox(toC(box), toC(m.q.box), .85).map(v => v + nz() * .4)) : box.slice();
    ANCH.push({ i, j, box, obj, ref, q: m.q });
  }
  const order = ANCH.map((_, n) => n).sort((x, y) => ANCH[y].obj - ANCH[x].obj), KEEP = [];
  for (const n of order) { if (KEEP.length >= KMAX) break; if (KEEP.every(m => iou(ANCH[m].ref, ANCH[n].ref) < .7)) KEEP.push(n); }
  // Closed-set head on each kept proposal: a name when it covers an object that has a row, background otherwise.
  const HEAD = KEEP.map((n, r) => {
    const a = ANCH[n], m = best(a.ref), named = m.v >= .5 && m.q.row >= 0, rr = PE.rng(40 + r);
    const z = [named ? 1.2 : 3.6, ...ROWS.map(nm => (named && nm === m.q.name ? 3.4 + 3 * m.v : rr() * 1.4))];
    const p = PE.softmax(z), j = p.indexOf(Math.max(...p));
    return { cover: m.v >= .5 ? m.q.name : null, name: j ? ROWS[j - 1] : 'background', p, score: p[j] };
  });
  // A smooth toy feature map for the crop: one bump per object plus faint texture.
  const feat = (u, v) => { let f = .08 + .05 * Math.sin(u * 37) * Math.cos(v * 23);
    OBJS.forEach(q => { const [x, y, w, h] = q.box, dx = (u - x - w / 2) / (w * .6), dy = (v - y - h / 2) / (h * .6); f += .9 * Math.exp(-(dx * dx + dy * dy)); });
    return PE.clamp(f, 0, 1); };

  const S0 = () => ({ stage: 'image', u: 1, k: K0, ci: 12, cj: 5, sel: Math.max(0, KEEP.findIndex(n => ANCH[n].q && ANCH[n].q.row < 0)) });
  let S = S0(), stop = () => {}, timers = [];
  const cellAnch = () => ANCH.filter(a => a.i === S.ci && a.j === S.cj);
  const bestAt = () => cellAnch().reduce((m, a) => (a.obj > m.obj ? a : m));
  const kept = () => KEEP.slice(0, S.k).map(n => ANCH[n]);
  const after = s => STAGES.indexOf(S.stage) >= STAGES.indexOf(s);

  function geo(w, h) {
    const hh = h - bottom, narrow = w < 620, pad = narrow ? 10 : 16;
    let iw, ih, ix, iy, px, py, pw, ph;
    if (narrow) {
      // A short stage (atlas bottom sheet) has no room for the detail panel; the HUD carries the readout instead.
      const room = hh - 40 - (w - 2 * pad) / AR - 18 >= 190;
      iw = w - 2 * pad; ih = iw / AR; if (ih > (room ? hh * .55 : hh - 50)) { ih = room ? hh * .55 : hh - 50; iw = ih * AR; }
      ix = (w - iw) / 2; iy = 40; px = pad; py = iy + ih + 18; pw = room ? w - 2 * pad : 0; ph = hh - py - 6;
    } else {
      pw = PE.clamp(w * .3, 210, 330); iw = w - pw - 3 * pad; ih = iw / AR;
      if (ih > hh - 44) { ih = hh - 44; iw = ih * AR; }
      ix = pad; iy = Math.max(28, (hh - ih) / 2 + 8); px = ix + iw + pad * 1.5; py = iy; pw = w - px - pad; ph = Math.min(ih, hh - py - 6);
    }
    return { narrow, ix, iy, iw, ih, px, py, pw, ph, X: u => ix + u * iw, Y: v => iy + v * ih };
  }
  const font = (sz, wt = 500) => `${wt} ${sz}px system-ui, sans-serif`;

  function draw(ctx, w, h) {
    const k = DK.C(); ctx.clearRect(0, 0, w, h); if (w < 80 || h < 80) return;
    const G = geo(w, h), { ix, iy, iw, ih, X, Y } = G, u = S.u, st = S.stage, fs = G.narrow ? 10.5 : 12;
    const R = b => [X(b[0]), Y(b[1]), b[2] * iw, b[3] * ih];
    ctx.save(); DK.rr(ctx, ix - 4, iy - 4, iw + 8, ih + 8, 8); ctx.fillStyle = k.p; ctx.fill(); ctx.restore();
    const imgA = st === 'image' ? 1 : st === 'features' ? PE.lerp(1, .45, u) : .45;
    DK.drawScene(ctx, DK.WAREHOUSE, ix, iy, iw, ih, k, imgA);
    ctx.save(); ctx.font = font(fs, 600); ctx.fillStyle = k.mu; ctx.textBaseline = 'alphabetic';
    ctx.fillText({ image: 'Input image', features: 'Backbone: one feature vector per cell', anchors: 'Anchors at one cell', objectness: 'RPN objectness, best anchor per cell',
      deltas: 'RPN box deltas', topk: `Top ${S.k} proposals after NMS`, roialign: 'RoIAlign on one proposal', head: 'Second stage: region head names each box' }[st], ix, iy - 9);
    ctx.restore();
    // feature grid
    if (st !== 'image') {
      const ga = (st === 'features' ? u : 1) * (after('topk') ? .35 : .7);
      ctx.save(); ctx.globalAlpha = ga; ctx.strokeStyle = DK.mixh(k.bg, k.B, .5); ctx.lineWidth = 1;
      ctx.beginPath(); for (let i = 0; i <= GC; i++) { ctx.moveTo(X(i / GC), iy); ctx.lineTo(X(i / GC), iy + ih); } for (let j = 0; j <= GR; j++) { ctx.moveTo(ix, Y(j / GR)); ctx.lineTo(ix + iw, Y(j / GR)); } ctx.stroke(); ctx.restore();
    }
    if (st === 'features') { ctx.save(); for (let j = 0; j < GR; j++) for (let i = 0; i < GC; i++) { ctx.globalAlpha = u * .55 * feat((i + .5) / GC, (j + .5) / GR); ctx.fillStyle = k.B; ctx.fillRect(X(i / GC) + 1, Y(j / GR) + 1, iw / GC - 2, ih / GR - 2); } ctx.restore(); }
    // objectness heat: best anchor per cell
    if (st === 'objectness' || st === 'deltas') {
      const ha = st === 'objectness' ? u : 1 - .6 * u;
      ctx.save(); for (let j = 0; j < GR; j++) for (let i = 0; i < GC; i++) { let m = 0; for (let n = (j * GC + i) * 9, e = n + 9; n < e; n++) m = Math.max(m, ANCH[n].obj);
        ctx.globalAlpha = ha * .75 * m; ctx.fillStyle = k.B; ctx.fillRect(X(i / GC) + 1, Y(j / GR) + 1, iw / GC - 2, ih / GR - 2); } ctx.restore();
    }
    const gC = glow('cell');
    if (st === 'anchors' || st === 'objectness' || st === 'deltas') {
      ctx.save(); ctx.strokeStyle = k.Y; ctx.lineWidth = 2 + 2 * gC; ctx.strokeRect(X(S.ci / GC), Y(S.cj / GR), iw / GC, ih / GR); ctx.restore();
      const A = cellAnch(), b0 = bestAt();
      A.forEach(a => {
        const isB = a === b0, al = st === 'anchors' ? u : st === 'objectness' ? .25 + .75 * a.obj : (isB ? 1 - u : .25 * (1 - u));
        const [x, y, ww, hh] = R(a.box); ctx.save(); ctx.globalAlpha = al; ctx.setLineDash([4, 3]);
        ctx.strokeStyle = st === 'anchors' ? k.fg : DK.mixh(DK.mixh(k.bg, k.fg, .35), k.B, a.obj); ctx.lineWidth = isB && st !== 'anchors' ? 2 : 1.2; ctx.strokeRect(x, y, ww, hh); ctx.restore();
      });
      if (st === 'deltas') {
        ANCH.filter(a => a.obj > .6 && a !== b0).forEach(a => { const [x, y, ww, hh] = R(lerpBox(a.box, a.ref, u)); ctx.save(); ctx.globalAlpha = .28; ctx.strokeStyle = k.B; ctx.lineWidth = 1; ctx.strokeRect(x, y, ww, hh); ctx.restore(); });
        const [x, y, ww, hh] = R(lerpBox(b0.box, b0.ref, u)); DK.brackets(ctx, x, y, ww, hh, k.B, { lw: 2.4 });
        const c0 = toC(b0.box), c1 = toC(lerpBox(b0.box, b0.ref, u)); PE.arrow(ctx, X(c0[0]), Y(c0[1]), X(c1[0]), Y(c1[1]), k.B, 1.6, 7, .9);
      }
    }
    // proposals
    if (after('topk')) {
      const P = kept(), gP = glow('proposals'), tagAt = {};
      // one tag per object: its best-ranked proposal, or the selected one if that covers it, so duplicates do not stack
      P.forEach((_, r) => { const c = HEAD[r].cover; if (c && !(c in tagAt)) tagAt[c] = r; });
      if (P[S.sel] && HEAD[S.sel].cover) tagAt[HEAD[S.sel].cover] = S.sel;
      P.forEach((a, r) => {
        const [x, y, ww, hh] = R(a.ref), isSel = r === S.sel;
        if (st === 'topk') {
          const al = PE.clamp(u * S.k - r, 0, 1); DK.brackets(ctx, x, y, ww, hh, k.B, { lw: 2 + gP, alpha: al });
          ctx.save(); ctx.globalAlpha = al; ctx.font = font(fs, 700); ctx.fillStyle = k.B; ctx.fillText(String(r + 1), x + 4, y + fs + 2); ctx.restore();
        } else if (st === 'roialign') {
          DK.brackets(ctx, x, y, ww, hh, k.B, { lw: isSel ? 2.4 : 1.3, alpha: isSel ? 1 : .35 });
          if (isSel) { ctx.save(); ctx.globalAlpha = u; ctx.strokeStyle = k.Y; ctx.lineWidth = 1;
            for (let t = 1; t < 7; t++) { ctx.beginPath(); ctx.moveTo(x + ww * t / 7, y); ctx.lineTo(x + ww * t / 7, y + hh); ctx.moveTo(x, y + hh * t / 7); ctx.lineTo(x + ww, y + hh * t / 7); ctx.stroke(); }
            ctx.fillStyle = k.Y; for (let a2 = 0; a2 < 7; a2++) for (let b2 = 0; b2 < 7; b2++) { ctx.beginPath(); ctx.arc(x + ww * (a2 + .5) / 7, y + hh * (b2 + .5) / 7, 1.4, 0, 7); ctx.fill(); }
            ctx.restore(); }
        } else {
          const H = HEAD[r], bgd = H.name === 'background', col = bgd ? k.Br : k.G, al = isSel ? 1 : .55 + .45 * u;
          DK.brackets(ctx, x, y, ww, hh, col, { lw: isSel ? 2.6 : 1.8, alpha: al, dash: bgd ? [4, 3] : null });
          const showTag = isSel || (H.cover && tagAt[H.cover] === r);
          if (showTag) DK.tag(ctx, x, y, H.name, H.score.toFixed(2), col, k, { size: G.narrow ? 10 : 11, alpha: al * (isSel ? 1 : .9), nameCol: bgd ? k.Br : k.Y, maxX: ix + iw + 4 });
        }
      });
    }
    panel(ctx, k, G, fs);
  }

  // The right-hand (or lower) panel: what the current stage computes for the selected cell or proposal.
  function panel(ctx, k, G, fs) {
    const { px, py, pw, ph } = G, u = S.u, st = S.stage, P = kept(), a = P[S.sel];
    if (pw < 60 || ph < 40) return;
    const title = s => { ctx.font = font(fs, 600); ctx.fillStyle = k.fg; ctx.fillText(s, px, py + 2); };
    const muted = (s, x, y, al = 'left') => { ctx.font = font(fs - .5); ctx.fillStyle = k.mu; ctx.textAlign = al; ctx.fillText(s, x, y); ctx.textAlign = 'left'; };
    ctx.save(); ctx.globalAlpha = Math.min(1, .3 + u); ctx.textBaseline = 'alphabetic';
    if (st === 'image') { muted('Step through the detector', px, py + 2); muted('with the buttons below.', px, py + 2 + fs * 1.5); }
    else if (st === 'features') {
      title('Feature map'); const cw = Math.min(pw * .55, 150), ch = cw / AR;
      for (let l = 3; l >= 0; l--) { const ox = px + l * 9, oy = py + 22 + (3 - l) * 9; ctx.fillStyle = DK.mixh(k.bg, k.B, .12 + .1 * l); ctx.strokeStyle = k.B; ctx.lineWidth = 1; ctx.fillRect(ox, oy, cw, ch); ctx.strokeRect(ox, oy, cw, ch); }
      muted(`${GC} x ${GR} cells x C channels`, px, py + 22 + 27 + ch + fs * 1.6);
      muted('each cell sees one patch of the image', px, py + 22 + 27 + ch + fs * 3);
    } else if (st === 'anchors' || st === 'objectness') {
      title(st === 'anchors' ? `9 anchors at cell (${S.ci + 1}, ${S.cj + 1})` : 'Objectness of each anchor');
      const A = cellAnch().slice().sort((x, y) => (st === 'objectness' ? y.obj - x.obj : 0)), rowH = Math.min(20, (ph - 30) / 9.5);
      A.forEach((an, n) => {
        const y = py + 20 + n * rowH, bw = an.box[2] * G.iw, bh = an.box[3] * G.ih, sc = Math.min(1, (rowH - 4) / bh, 26 / bw);
        ctx.save(); ctx.setLineDash([3, 2]); ctx.strokeStyle = st === 'anchors' ? k.fg : DK.mixh(DK.mixh(k.bg, k.fg, .35), k.B, an.obj); ctx.lineWidth = 1;
        ctx.strokeRect(px + 14 - bw * sc / 2, y + rowH / 2 - bh * sc / 2, bw * sc, bh * sc); ctx.restore();
        const sz = ['small', 'medium', 'large'][Math.floor(ANCH.indexOf(an) % 9 / 3)], sh = ['tall', 'square', 'wide'][ANCH.indexOf(an) % 3];
        if (st === 'anchors') muted(`${sz}, ${sh}`, px + 36, y + rowH / 2 + 4);
        else { const L = (pw - 90) * an.obj; ctx.fillStyle = k.B; ctx.fillRect(px + 36, y + rowH / 2 - 5, Math.max(1, L * u), 10); muted(an.obj.toFixed(2), px + 42 + L * u, y + rowH / 2 + 4); }
      });
    } else if (st === 'deltas') {
      const b0 = bestAt(), c0 = toC(b0.box), c1 = toC(b0.ref);
      const d = [['dx', (c1[0] - c0[0]) / c0[2]], ['dy', (c1[1] - c0[1]) / c0[3]], ['dw', Math.log(c1[2] / c0[2])], ['dh', Math.log(c1[3] / c0[3])]];
      title('Four deltas for the best anchor');
      d.forEach(([n, v], m) => { const y = py + 30 + m * 26, mid = px + pw / 2; ctx.font = font(fs, 600); ctx.fillStyle = k.fg; ctx.fillText(n, px, y + 4);
        ctx.strokeStyle = k.rule; ctx.beginPath(); ctx.moveTo(mid, y - 8); ctx.lineTo(mid, y + 8); ctx.stroke();
        const L = PE.clamp(v, -1, 1) * (pw / 2 - 50) * u; ctx.fillStyle = k.B; ctx.fillRect(Math.min(mid, mid + L), y - 5, Math.abs(L), 10);
        muted((v >= 0 ? '+' : '') + v.toFixed(2), px + pw, y + 4, 'right'); });
      muted('shift x, y; log-scale width, height', px, py + 30 + 4 * 26 + 4);
    } else if (st === 'topk') {
      title('Kept proposals, by objectness'); const rowH = Math.min(19, (ph - 26) / KMAX);
      P.forEach((an, r) => { const y = py + 20 + r * rowH, al = PE.clamp(u * S.k - r, 0, 1); ctx.globalAlpha = al;
        ctx.font = font(fs, 700); ctx.fillStyle = k.B; ctx.fillText(String(r + 1), px, y + rowH / 2 + 4);
        const L = (pw - 150) * an.obj; ctx.fillRect(px + 22, y + rowH / 2 - 4, Math.max(1, L), 8);
        muted(an.obj.toFixed(2), px + 28 + L, y + rowH / 2 + 4);
        muted(HEAD[r].cover ? 'on the ' + HEAD[r].cover : 'background', px + pw, y + rowH / 2 + 4, 'right'); });
      ctx.globalAlpha = 1;
    } else if (a) {
      const cs = Math.min((pw - 10) / 7, (st === 'head' ? ph * .32 : ph - 50) / 7, 22), cx0 = px, cy0 = py + 18, ga = glow('crop');
      title(st === 'roialign' ? `Proposal ${S.sel + 1}: 7 x 7 crop` : `Proposal ${S.sel + 1}: region head`);
      for (let gy = 0; gy < 7; gy++) for (let gx = 0; gx < 7; gx++) {
        const b = a.ref, f = feat(b[0] + b[2] * (gx + .5) / 7, b[1] + b[3] * (gy + .5) / 7);
        ctx.fillStyle = DK.mixh(k.bg, k.B, .15 + .8 * f); ctx.globalAlpha = st === 'roialign' ? PE.clamp(u * 2 - (gx + gy) / 12, 0, 1) : 1;
        ctx.fillRect(cx0 + gx * cs, cy0 + gy * cs, cs - 1.5, cs - 1.5);
      }
      ctx.globalAlpha = 1;
      if (ga) { ctx.strokeStyle = k.Y; ctx.lineWidth = 1 + 2 * ga; ctx.strokeRect(cx0 - 2, cy0 - 2, cs * 7 + 2, cs * 7 + 2); }
      if (st === 'roialign') { muted('7 x 7 x C, the same size for', px, cy0 + cs * 7 + fs * 1.5); muted('every box (toy size)', px, cy0 + cs * 7 + fs * 2.9); }
      else {
        const H = HEAD[S.sel], names = ['background', ...ROWS], y0 = cy0 + cs * 7 + 14, rowH = Math.min(18, (py + ph - y0 - 4) / names.length), lab = 84;
        muted('head -> class scores (closed list)', px, y0 + 2);
        names.forEach((n, m) => { const y = y0 + 10 + m * rowH, isB = H.p[m] === Math.max(...H.p);
          ctx.font = font(fs - .5); ctx.fillStyle = m === 0 ? k.Br : k.fg; ctx.fillText(n, px, y + rowH / 2 + 3);
          const L = Math.max(1, (pw - lab - 40) * H.p[m] * u); ctx.fillStyle = isB ? (m === 0 ? k.Br : k.G) : DK.mixh(k.bg, k.fg, .3); ctx.fillRect(px + lab, y + rowH / 2 - 4, L, 8);
          muted(H.p[m].toFixed(2), px + lab + L + 5, y + rowH / 2 + 3); });
      }
    }
    ctx.restore();
  }

  const cv = PE.canvas(el, o.aspect || (w => (w < 620 ? 1.36 : .47)), draw);
  function describe() {
    const P = kept(), cover = new Set(P.map((_, r) => HEAD[r].cover).filter(Boolean)), a = P[S.sel], H = HEAD[S.sel], b0 = bestAt();
    const onFk = b0.q && b0.q.row < 0 && b0.obj > .5;
    return {
      image: 'A drawn warehouse: a person, a pallet, a forklift and a traffic cone. Step 1 runs the backbone.',
      features: `The backbone turns the image into a grid of feature vectors: here ${GC} x ${GR} cells, each a column of numbers that describes its patch.`,
      anchors: `Cell (${S.ci + 1}, ${S.cj + 1}) carries 9 anchors: three sizes times three shapes. Click another cell, or focus the figure and use the arrow keys.`,
      objectness: `Every anchor at every cell gets an objectness score. Best anchor at this cell: ${b0.obj.toFixed(2)}.` + (onFk ? ' The forklift lights up although no label ever named it.' : ''),
      deltas: 'Each anchor also gets four box deltas: it slides and stretches most of the way onto the object it overlaps.',
      topk: `Sort all ${ANCH.length} anchors by objectness, drop near-duplicates (NMS), keep the top ${S.k}: they cover ${cover.size} of 4 objects` + (cover.has('forklift') ? ', the forklift included.' : '.'),
      roialign: `RoIAlign samples proposal ${S.sel + 1} into a fixed 7 x 7 grid of feature vectors, interpolating between cells instead of rounding to them.`,
      head: a ? `The region head scores proposal ${S.sel + 1} against its fixed class rows: "${H.name}" (${H.score.toFixed(2)}).` + (H.cover === 'forklift' ? ' The forklift was boxed, but the closed list has no row for it, so it is called background.' : '') : '',
    }[S.stage];
  }
  const say = () => o.onSay && o.onSay(describe());
  function set(stage) {
    stop(); S.stage = stage; S.u = 0; say();
    stop = PE.animate(stage === 'topk' ? 900 : 700, v => { S.u = v; cv.redraw(); });
  }
  function setK(n) { S.k = PE.clamp(Math.round(n), 1, KMAX); if (S.sel >= S.k) S.sel = 0; cv.redraw(); if (after('topk')) say(); }
  function clearTimers() { timers.forEach(clearTimeout); timers = []; }
  function play() {
    clearTimers(); if (PE.instant()) { set('head'); return; }
    STAGES.slice(1).forEach((s, n) => timers.push(setTimeout(() => set(s), n * 1300)));
  }
  function reset() { clearTimers(); stop(); S = S0(); cv.redraw(); say(); }

  const pick = e => {
    const r = el.getBoundingClientRect(), G = geo(r.width, r.height), ux = (e.clientX - r.left - G.ix) / G.iw, vy = (e.clientY - r.top - G.iy) / G.ih;
    if (ux < 0 || ux > 1 || vy < 0 || vy > 1) return;
    if (after('topk')) { let bi = -1, bd = 1e9; kept().forEach((a, n) => { const c = toC(a.ref), inside = Math.abs(ux - c[0]) < c[2] / 2 && Math.abs(vy - c[1]) < c[3] / 2, d = Math.hypot(ux - c[0], vy - c[1]) * (inside ? 1 : 3) + c[2] * c[3];
      if (d < bd) { bd = d; bi = n; } }); if (bi >= 0) S.sel = bi; }
    else { S.ci = PE.clamp(Math.floor(ux * GC), 0, GC - 1); S.cj = PE.clamp(Math.floor(vy * GR), 0, GR - 1); }
    cv.redraw(); say();
  };
  const key = e => {
    const d = { ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowUp: [0, -1], ArrowDown: [0, 1] }[e.key]; if (!d) return; e.preventDefault();
    if (after('topk')) S.sel = (S.sel + (d[0] || d[1]) + S.k) % S.k; else { S.ci = PE.clamp(S.ci + d[0], 0, GC - 1); S.cj = PE.clamp(S.cj + d[1], 0, GR - 1); }
    cv.redraw(); say();
  };
  el.addEventListener('pointerdown', pick); el.addEventListener('keydown', key);
  say();
  return { set, setK, play, reset, redraw: cv.redraw, resize: () => cv.resize && cv.resize(), stage: () => S.stage, k: () => S.k,
    destroy: () => { clearTimers(); stop(); el.removeEventListener('pointerdown', pick); el.removeEventListener('keydown', key); cv.destroy && cv.destroy(); } };
};
</script>
