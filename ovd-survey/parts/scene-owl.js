<script>
/* SceneOwl(el, o): OWL-ViT as a toy, shared by the long page (viz-owl) and the atlas (chapter "owl").
   A 16 x 9 token grid over the warehouse scene. Every token predicts one box (faint brackets) and one class
   embedding (6-D toy vectors). A query vector (text, or the embedding of an example box from another image)
   scores every token: p = sigmoid(K (cos - C)). Boxes at or above the threshold turn green. Changing the
   query changes scores only; boxes never move, because the image side ran once (late fusion).
   The constants were tuned in node: forklift passes 0.5 on several tokens, the rare name "delineator" tops
   out near 0.29 on the cone, the example box scores the cone near 0.96.
   o: { aspect(w) -> h/w, bottom (px kept free), glow(word) -> 0..1, onSay(text) }. API: set(mode, q),
   setThr(v), reset(), redraw(), resize(), destroy(), state(). mode: boxes | text | image. */
const SceneOwl = (el, o = {}) => {
  const GX = 16, GY = 9, N = GX * GY, D = 6, BGI = 4, GAP = 5, K = 12, C0 = .72;
  const glow = o.glow || (() => 0), bottom = o.bottom || 0;
  const OBJS = DK.WAREHOUSE.filter(b => b.name).map((b, c) => ({ ...b, c }));
  const QN = { forklift: 'forklift', pallet: 'pallet', rare: 'delineator', image: 'example box' };
  // Toy embeddings: one axis per class, a background axis, and a text-only axis (the modality gap).
  const ax = i => Array.from({ length: D }, (_, j) => +(i === j));
  const add = (a, b, s = 1) => a.map((v, i) => v + s * b[i]);
  const nrm = a => { const L = Math.hypot(...a) || 1; return a.map(v => v / L); };
  const r = PE.rng(11), g = PE.gauss(r), noise = () => Array.from({ length: D }, (_, k) => (k === GAP ? 0 : g() * .22));
  const toks = [];
  for (let j = 0; j < GY; j++) for (let i = 0; i < GX; i++) {
    const u = (i + .5) / GX, v = (j + .5) / GY;
    let best = null, bw = -1;
    OBJS.forEach(b => { const [x0, y0, a, h] = b.box; if (u >= x0 && u <= x0 + a && v >= y0 && v <= y0 + h) {
      const w = Math.max(0, 1 - Math.hypot((u - x0 - a / 2) / (a / 2), (v - y0 - h / 2) / (h / 2)) / Math.SQRT2);
      if (w > bw) { bw = w; best = b; } } });
    toks.push({ i, j, u, v, o: best, w: best ? bw : 0 });
  }
  // DETR-style matching trains one token per object to carry it; its neighbours learn weaker versions.
  OBJS.forEach(b => { let m = null; toks.forEach(t => { if (t.o === b && (!m || t.w > m.w)) m = t; }); if (m) m.matched = true; });
  const rb = PE.rng(5);
  toks.forEach(t => {
    let e = add(ax(BGI), noise());
    if (t.o) { const s = t.matched ? 2.6 : .25 + t.w * t.w; e = add(add(ax(BGI), ax(t.o.c), s), noise()); e = add(e, ax(BGI), -.3 * (t.matched ? 1 : t.w)); }
    t.e = nrm(e);
    // Box: object tokens predict their object's box (jitter shrinks toward the centre); others a small box on their own patch.
    if (t.o) { const [x0, y0, a, h] = t.o.box, jj = (1 - t.w) * .35, q = () => (rb() - .5) * 2 * jj;
      t.box = [x0 + a * q() * .5, y0 + h * q() * .5, a * (1 + q() * .6), h * (1 + q() * .6)]; }
    else { const s = 1 + rb() * .5; t.box = [t.u - s * .5 / GX + (rb() - .5) * .3 / GX, t.v - s * .5 / GY + (rb() - .5) * .3 / GY, s / GX, s / GY]; }
  });
  const Q = {
    forklift: nrm(add(ax(2), ax(GAP), .3)),
    pallet: nrm(add(ax(1), ax(GAP), .3)),
    rare: nrm(add(add(ax(3), ax(GAP), 1.15), ax(0), .35)),
    image: nrm(add(ax(3), ax(BGI), .12)),
  };
  const sig = z => 1 / (1 + Math.exp(-z)), dot = (a, b) => a.reduce((s, x, i) => s + x * b[i], 0);
  const SC = Object.fromEntries(Object.entries(Q).map(([k, q]) => [k, toks.map(t => sig(K * (dot(t.e, q) - C0)))]));
  const center = (() => { let m = 0; toks.forEach((t, i) => { if (t.matched && t.o.name === 'forklift') m = i; }); return m; })();
  const reset0 = () => ({ mode: 'boxes', q: 'forklift', thr: .5, sel: center, uBox: 1, disp: new Array(N).fill(0), from: new Array(N).fill(0), uS: 1 });
  let S = reset0(), stopB = () => {}, stopS = () => {};
  const qKey = () => (S.mode === 'image' ? 'image' : S.q);
  const target = () => (S.mode === 'boxes' ? new Array(N).fill(0) : SC[qKey()]);

  function geo(w, h) {
    const hh = h - bottom, narrow = w < 620;
    if (!narrow) {
      const pw = Math.max(210, Math.min(300, w * .3));
      let sw = w - pw - 40, sh = sw * 9 / 16; const maxH = hh - 52;
      if (sh > maxH) { sh = maxH; sw = sh * 16 / 9; }
      const sx = 14, sy = 34 + Math.max(0, (hh - sh - 52) * .35), px = sx + sw + 26;
      return { narrow, sx, sy, sw, sh, px, py: sy, pw: w - px - 12, ph: Math.max(sh, 230), fs: 12 };
    }
    let sw = w - 16, sh = sw * 9 / 16; const maxS = hh - 34 - 190;
    if (sh > maxS && maxS > 120) { sh = maxS; sw = sh * 16 / 9; }
    const sx = (w - sw) / 2, sy = 30;
    return { narrow, sx, sy, sw, sh, px: 10, py: sy + sh + 14, pw: w - 20, ph: hh - (sy + sh + 14) - 6, fs: 11 };
  }
  const aspect = o.aspect || (w => (w < 620 ? ((w - 16) * 9 / 16 + 30 + 14 + 178) / w : ((w - Math.max(210, Math.min(300, w * .3)) - 40) * 9 / 16 + 52) / w));

  function draw(ctx, w, h) {
    const k = DK.C(); ctx.clearRect(0, 0, w, h); if (w < 120 || h < 120) return;
    const G = geo(w, h), { sx, sy, sw, sh, fs, narrow } = G, cw = sw / GX, ch = sh / GY;
    const X = u => sx + u * sw, Y = v => sy + v * sh;
    const halo = (s, x, y) => { ctx.lineWidth = 4; ctx.strokeStyle = k.bg; ctx.lineJoin = 'round'; ctx.strokeText(s, x, y); ctx.fillText(s, x, y); };
    ctx.save(); ctx.font = `600 ${fs + .5}px system-ui, sans-serif`; ctx.fillStyle = k.fg; ctx.textBaseline = 'alphabetic';
    ctx.fillText(`${GX} x ${GY} = ${N} tokens, one box each`, sx, sy - 12); ctx.restore();
    DK.drawScene(ctx, DK.WAREHOUSE, sx, sy, sw, sh, k);
    // score heat per token
    const gS = glow('scores'), gT = glow('tokens'), gB = glow('boxes');
    if (S.mode !== 'boxes' || S.uS < 1) toks.forEach((t, n) => { const p = S.disp[n]; if (p < .02) return;
      ctx.fillStyle = k.G; ctx.globalAlpha = Math.min(.6, (.42 + .25 * gS) * p); ctx.fillRect(X(t.u) - cw / 2, Y(t.v) - ch / 2, cw, ch); });
    ctx.globalAlpha = 1;
    // token grid
    ctx.save(); ctx.strokeStyle = gT > 0 ? DK.mixh(k.rule, k.B, gT) : k.rule; ctx.lineWidth = 1 + gT; ctx.globalAlpha = .9;
    ctx.beginPath(); for (let i = 0; i <= GX; i++) { ctx.moveTo(sx + i * cw, sy); ctx.lineTo(sx + i * cw, sy + sh); }
    for (let j = 0; j <= GY; j++) { ctx.moveTo(sx, sy + j * ch); ctx.lineTo(sx + sw, sy + j * ch); } ctx.stroke(); ctx.restore();
    // every token's box, drawn faintly, appearing in token order
    const pass = S.mode === 'boxes' ? [] : toks.map((t, n) => S.disp[n] >= S.thr);
    toks.forEach((t, n) => {
      const a = PE.clamp(S.uBox * (N + 30) - n, 0, 30) / 30; if (a <= 0 || pass[n]) return;
      const [u, v, bw, bh] = t.box; DK.brackets(ctx, X(u), Y(v), bw * sw, bh * sh, k.B, { alpha: a * (.17 + .45 * gB), lw: 1, len: .22 });
    });
    // boxes that pass the threshold, one tag per object (the top token), plus the selected token's box
    const tagged = new Set();
    toks.map((t, n) => n).filter(n => pass[n]).sort((a, b) => S.disp[b] - S.disp[a]).forEach(n => {
      const t = toks[n], [u, v, bw, bh] = t.box, x = X(u), y = Y(v);
      DK.brackets(ctx, x, y, bw * sw, bh * sh, k.G, { lw: 2.2 });
      const owner = t.o ? t.o.name : 'cell' + n;
      if (!tagged.has(owner)) { tagged.add(owner); DK.tag(ctx, x, y, S.mode === 'image' ? 'example' : QN[S.q], S.disp[n].toFixed(2), k.G, k, { size: narrow ? 10 : 11.5, maxX: sx + sw + (narrow ? 0 : 20) }); }
    });
    const st = toks[S.sel], [su, sv, sbw, sbh] = st.box;
    if (!pass[S.sel]) DK.brackets(ctx, X(su), Y(sv), sbw * sw, sbh * sh, k.B, { lw: 2.2 });
    ctx.strokeStyle = k.fg; ctx.lineWidth = 2; ctx.strokeRect(X(st.u) - cw / 2 + 1, Y(st.v) - ch / 2 + 1, cw - 2, ch - 2);
    drawPanel(ctx, G, k, halo);
  }

  function drawPanel(ctx, G, k) {
    const { px, py, pw, fs, narrow } = G, gQ = glow('queries'), gS = glow('scores');
    const keys = ['forklift', 'pallet', 'rare', 'image'], active = S.mode === 'boxes' ? null : qKey();
    ctx.save(); ctx.textBaseline = 'middle'; ctx.font = `600 ${fs + .5}px system-ui, sans-serif`; ctx.fillStyle = k.fg;
    ctx.fillText(S.mode === 'boxes' ? 'Queries: none picked yet' : 'Queries, and the selected token\'s score', px, py + 6);
    const cols = narrow ? 2 : 1, rowH = narrow ? 30 : 34, colW = pw / cols, y0 = py + 30;
    keys.forEach((q, m) => {
      const cx = px + (m % cols) * colW, cy = y0 + Math.floor(m / cols) * rowH, on = q === active;
      ctx.globalAlpha = S.mode === 'boxes' ? .55 : (on ? 1 : .7);
      let pillW;
      if (q === 'image') {
        // a thumbnail from another image: a cone with the example box drawn on it
        const tw = narrow ? 34 : 40, th = rowH - 8; DK.rr(ctx, cx, cy - th / 2, tw, th, 4); ctx.fillStyle = k.p2; ctx.fill();
        ctx.strokeStyle = on ? k.G : k.rule; ctx.lineWidth = on ? 2 : 1; ctx.stroke();
        DK.OBJ.cone(ctx, cx + tw * .32, cy - th / 2 + 3, tw * .36, th - 6, k); DK.brackets(ctx, cx + tw * .26, cy - th / 2 + 2, tw * .48, th - 4, k.B, { lw: 1.4 });
        ctx.font = `500 ${fs}px system-ui, sans-serif`; ctx.fillStyle = k.B; ctx.fillText(narrow ? 'example' : 'example box', cx + tw + 6, cy); pillW = tw + 6 + ctx.measureText(narrow ? 'example' : 'example box').width;
      } else {
        ctx.font = `500 ${fs}px system-ui, sans-serif`; const s = QN[q], tw = ctx.measureText(s).width + 18;
        DK.rr(ctx, cx, cy - 11, tw, 22, 11); ctx.strokeStyle = on ? k.G : k.Y; ctx.lineWidth = on ? 2 + gQ : 1.1 + gQ; ctx.stroke();
        ctx.fillStyle = k.Y; ctx.fillText(s, cx + 9, cy); pillW = tw;
      }
      if (S.mode !== 'boxes') {
        const p = SC[q][S.sel], bx = cx + (narrow ? Math.min(pillW + 6, colW * .55) : 128), bwMax = Math.max(16, colW - (bx - cx) - 40);
        ctx.fillStyle = p >= S.thr ? k.G : DK.mixh(k.bg, k.fg, .35); ctx.fillRect(bx, cy - 4, Math.max(1.5, bwMax * p), 8);
        ctx.fillStyle = k.mu; ctx.font = `500 ${fs - .5}px system-ui, sans-serif`; ctx.fillText(p.toFixed(2), bx + Math.max(1.5, bwMax * p) + 5, cy);
      }
      ctx.globalAlpha = 1;
    });
    // strip: every token's score against the active query, with the threshold
    const yS = y0 + Math.ceil(keys.length / cols) * rowH + (narrow ? 34 : 44), x0 = px + 6, x1 = px + pw - 10;
    ctx.font = `600 ${fs}px system-ui, sans-serif`; ctx.fillStyle = k.fg;
    ctx.fillText(active ? `All ${N} tokens against "${QN[active]}"` : `All ${N} tokens: no scores without a query`, px, yS - 34);
    ctx.strokeStyle = k.rule; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(x0, yS + 12); ctx.lineTo(x1, yS + 12); ctx.stroke();
    ctx.fillStyle = k.mu; ctx.font = `400 ${fs - 1}px system-ui, sans-serif`; ctx.textAlign = 'center';
    [0, .5, 1].forEach(v => ctx.fillText(String(v), PE.lerp(x0, x1, v), yS + 24));
    if (active || S.uS < 1) { const jr = PE.rng(3);
      toks.forEach((t, n) => { const p = S.disp[n], yy = yS + (jr() - .5) * 16; PE.dot(ctx, PE.lerp(x0, x1, p), yy, 2.6 + (n === S.sel ? 1.5 : 0) + gS, p >= S.thr && active ? k.G : DK.mixh(k.bg, k.fg, .45)); }); }
    const xt = PE.lerp(x0, x1, S.thr); ctx.strokeStyle = k.fg; ctx.setLineDash([3, 3]); ctx.beginPath(); ctx.moveTo(xt, yS - 16); ctx.lineTo(xt, yS + 13); ctx.stroke(); ctx.setLineDash([]);
    ctx.fillStyle = k.fg; ctx.textAlign = xt > x1 - 60 ? 'right' : 'left'; ctx.fillText(`threshold ${S.thr.toFixed(2)}`, xt + (xt > x1 - 60 ? -4 : 4), yS - 16);
    ctx.restore();
  }

  const cv = PE.canvas(el, aspect, draw);
  function describe() {
    const t = toks[S.sel], where = t.o ? `on the ${t.o.name}` : 'on background';
    if (S.mode === 'boxes') return `No query yet. Each of the ${N} tokens predicted one box and one embedding; nothing has a name. The outlined token sits ${where} and predicts the solid blue box.`;
    const s = SC[qKey()], n = s.filter(p => p >= S.thr).length, bi = s.indexOf(Math.max(...s)), bt = toks[bi];
    const name = S.mode === 'image' ? 'the example box (a cone from another image)' : `"${QN[S.q]}"`;
    return `Query ${name}: ${n} of ${N} tokens score at least ${S.thr.toFixed(2)}. Best: ${s[bi].toFixed(2)}, ${bt.o ? 'on the ' + bt.o.name : 'on background'}. No box moved; only the scores changed.`;
  }
  const say = () => o.onSay && o.onSay(describe());
  function tweenScores() {
    stopS(); S.from = S.disp.slice(); S.uS = 0; const to = target();
    stopS = PE.animate(650, u => { S.uS = u; S.disp = S.from.map((f, n) => PE.lerp(f, to[n], u)); cv.redraw(); }, () => { S.uS = 1; });
  }
  // Each call sets a state: calling it twice equals calling it once (the box sweep restarts, ending the same).
  function set(mode, q) {
    S.mode = mode; if (mode === 'text' && q) S.q = q;
    if (mode === 'boxes') { stopB(); S.uBox = 0; stopB = PE.animate(900, u => { S.uBox = u; cv.redraw(); }); }
    else if (S.uBox < 1) { stopB(); S.uBox = 1; }
    tweenScores(); say();
  }
  function setThr(v) { S.thr = PE.clamp(+v, 0, 1); cv.redraw(); say(); }
  function reset() { stopB(); stopS(); S = reset0(); cv.redraw(); say(); }
  function pick(e) { const rc = el.getBoundingClientRect(), G = geo(rc.width, rc.height);
    const i = Math.floor((e.clientX - rc.left - G.sx) / (G.sw / GX)), j = Math.floor((e.clientY - rc.top - G.sy) / (G.sh / GY));
    if (i >= 0 && i < GX && j >= 0 && j < GY) { S.sel = j * GX + i; cv.redraw(); say(); } }
  const offs = [
    ['pointerdown', pick],
    ['keydown', e => { const d = { ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowUp: [0, -1], ArrowDown: [0, 1] }[e.key]; if (!d) return; e.preventDefault();
      const i = PE.clamp(S.sel % GX + d[0], 0, GX - 1), j = PE.clamp(Math.floor(S.sel / GX) + d[1], 0, GY - 1); S.sel = j * GX + i; cv.redraw(); say(); }],
  ].map(([ev, fn]) => { el.addEventListener(ev, fn); return () => el.removeEventListener(ev, fn); });
  say();
  return { set, setThr, reset, redraw: cv.redraw, resize: () => (cv.resize ? cv.resize() : cv.redraw()),
    state: () => ({ mode: S.mode, q: S.q, thr: S.thr }),
    destroy: () => { stopB(); stopS(); offs.forEach(f => f()); cv.destroy && cv.destroy(); } };
};
</script>
