<script>
/* Grounding DINO toy, shared by the long page (viz-qsel) and the atlas (chapter gdino). The drawn warehouse
   sits under a 16 x 9 grid of image tokens; the prompt's words are text tokens. Toy features: an image token is
   its overlap with each object kind plus background, a word is a fixed vector over the same kinds, so the scores
   X_I X_T^T are real dot products. Stages: 'fusion' (text-to-image attention lines), 'qsel' (heat = max over the
   text tokens; the top-K tokens seed anchor boxes), 'decoder' (anchors refine; final scores against the text).
   o: { aspect(w) -> h/w, glow(word) -> 0..1, onSay(text), bottom (px kept free at the bottom), k (top-K) } */
const SceneGdino = (el, o = {}) => {
  const GX = 16, GY = 9, N = GX * GY, glow = o.glow || (() => 0), bottom = o.bottom || 0;
  // dims: person, pallet, forklift, cone, shelf, background
  const KIND = { person: 0, pallet: 1, forklift: 2, cone: 3, shelf: 4 };
  const WORD = {
    forklift: [0, .15, .95, 0, .05, 0], pallet: [0, .95, .2, 0, .15, 0], person: [.95, 0, .1, .05, 0, 0],
    traffic: [.05, 0, .2, .6, 0, .1], cone: [0, 0, 0, .92, 0, .05], ladder: [.05, .1, .05, 0, .45, .05] };
  const PROMPTS = { forklift: [['forklift']], 'pallet-person': [['pallet'], ['person']], cone: [['traffic', 'cone']], ladder: [['ladder']] };
  const OBJS = DK.WAREHOUSE, NAMEOF = { person: 'person', pallet: 'pallet', forklift: 'forklift', cone: 'traffic cone', shelf: 'shelf' };
  const dotv = (a, b) => a.reduce((s, v, i) => s + v * b[i], 0);
  const g = PE.gauss(PE.rng(11)), feats = [], owner = [];
  for (let gy = 0; gy < GY; gy++) for (let gx = 0; gx < GX; gx++) {
    const f = [0, 0, 0, 0, 0, 0];
    const ov = OBJS.map(ob => { let n = 0; const [x, y, w, h] = ob.box;
      for (let a = 0; a < 4; a++) for (let b = 0; b < 4; b++) { const u = (gx + (a + .5) / 4) / GX, v = (gy + (b + .5) / 4) / GY; if (u >= x && u <= x + w && v >= y && v <= y + h) n++; }
      return n / 16; });
    let s = 0; OBJS.forEach((ob, j) => { f[KIND[ob.kind]] += ov[j]; s += ov[j]; }); f[5] += Math.max(0, 1 - s);
    for (let d = 0; d < 6; d++) f[d] += .05 * g();
    const best = ov.indexOf(Math.max(...ov)); feats.push(f); owner.push(ov[best] > .2 ? best : -1);
  }
  const words = p => PROMPTS[p].flat();
  const heatOf = p => feats.map(f => Math.max(...words(p).map(w => dotv(f, WORD[w]))));
  const order = hm => [...hm.keys()].sort((a, b) => hm[b] - hm[a]);
  const cellBox = i => { const gx = i % GX, gy = Math.floor(i / GX); return [gx / GX, gy / GY, 1 / GX, 1 / GY]; };
  // Toy anchor: a box the encoder might predict at the token, halfway between a two-cell box and the object's box.
  const jr = PE.rng(5), jit = Array.from({ length: N }, () => [jr() - .5, jr() - .5, jr() - .5, jr() - .5]);
  function anchorOf(i) {
    const [x, y, w, h] = cellBox(i), cx = x + w / 2, cy = y + h / 2, d = [cx - w, cy - h, 2 * w, 2 * h], j = jit[i];
    if (owner[i] < 0) return d;
    const b = OBJS[owner[i]].box, m = .55;
    return [PE.lerp(d[0], b[0], m) + j[0] * .02, PE.lerp(d[1], b[1], m) + j[1] * .02, PE.lerp(d[2], b[2], m) * (1 + j[2] * .1), PE.lerp(d[3], b[3], m) * (1 + j[3] * .1)];
  }
  function finalOf(i) {
    if (owner[i] < 0) return anchorOf(i);
    const b = OBJS[owner[i]].box, j = jit[i];
    return [b[0] - .006 + j[0] * .008, b[1] - .008 + j[1] * .008, b[2] + .012 + j[2] * .008, b[3] + .014 + j[3] * .008];
  }
  // Final toy scores: the phrase score is the mean over its words of the word's weight on the object's kind;
  // among queries on one object only the best-ranked keeps its score (the decoder learns to drop duplicates).
  function finals(S) {
    const sel = S.sel, seen = {};
    return sel.map(i => {
      if (owner[i] < 0) return { i, p: .04, name: null };
      const kd = KIND[OBJS[owner[i]].kind];
      let best = -1, name = null;
      PROMPTS[S.prompt].forEach(gp => { const s = gp.reduce((a, w) => a + WORD[w][kd], 0) / gp.length; if (s > best) { best = s; name = gp.join(' '); } });
      let p = 1 / (1 + Math.exp(-8 * (best - .55)));
      if (seen[owner[i]]) p *= .12; seen[owner[i]] = 1;
      return { i, p, name };
    });
  }
  const reset0 = () => ({ stage: 'qsel', prompt: 'forklift', K: o.k || 8, uHeat: 1, uSel: 1, uFus: 1, uDec: 1, focus: -1,
    hFrom: heatOf('forklift'), hTo: heatOf('forklift'), sel: [] });
  let S = reset0(), stops = {};
  const pick = () => { S.sel = order(S.hTo).slice(0, S.K); if (S.focus < 0) S.focus = S.sel[0]; };
  pick();
  const heatNow = i => PE.lerp(S.hFrom[i], S.hTo[i], S.uHeat);

  function geo(w, h) {
    const hh = h - bottom, narrow = w < 620;
    if (!narrow && hh / w > .62) {
      // tall and wide (the atlas stage): scene on top, the panel below in two columns
      const sw = Math.max(140, Math.min(w - 20, (hh - 250) * 16 / 9)), sh = sw * 9 / 16, sx = 10, sy = 28, py = sy + sh + 26, cw = (w - 36) / 2;
      return { narrow, split: true, hh, sx, sy, sw, sh, px: 10, px2: 26 + cw, py, pw: cw, ph: hh - py - 2 };
    }
    if (!narrow) {
      const pw0 = PE.clamp(w * .34, 210, 330), maxW = w - pw0 - 32, maxH = hh - 58;
      const sw = Math.max(120, Math.min(maxW, maxH * 16 / 9)), sh = sw * 9 / 16, sx = 10, sy = 28, px = sx + sw + 22;
      return { narrow, hh, sx, sy, sw, sh, px, py: 22, pw: w - px - 8, ph: hh - 26 };
    }
    const sw = Math.max(140, Math.min(w - 16, (hh - 240) * 16 / 9)), sh = sw * 9 / 16, sx = (w - sw) / 2, sy = 24;
    return { narrow, hh, sx, sy, sw, sh, px: 8, py: sy + sh + 26, pw: w - 16, ph: hh - (sy + sh + 26) - 2 };
  }
  const txt = (ctx, s, x, y, col, size = 11.5, weight = 500, align = 'left') => {
    ctx.save(); ctx.font = `${weight} ${size}px system-ui, sans-serif`; ctx.fillStyle = col; ctx.textAlign = align; ctx.textBaseline = 'middle'; ctx.fillText(s, x, y); ctx.restore(); };
  const tw = (ctx, s, size, weight = 600) => { ctx.save(); ctx.font = `${weight} ${size}px system-ui, sans-serif`; const v = ctx.measureText(s).width; ctx.restore(); return v; };
  function chips(ctx, G, k, x0, y0, maxW, gl) {
    // One pill per word; a muted '.' between names. Returns the chip centers (for attention lines) and the next y.
    const fs = 12, hgt = 21, out = []; let x = x0, y = y0;
    PROMPTS[S.prompt].forEach((gp, gi) => {
      gp.forEach(wd => {
        const w = tw(ctx, wd, fs) + 16; if (x + w > x0 + maxW) { x = x0; y += hgt + 6; }
        DK.rr(ctx, x, y, w, hgt, hgt / 2); ctx.fillStyle = DK.mixh(k.bg, k.Y, .12 + .25 * gl); ctx.fill();
        ctx.strokeStyle = k.Y; ctx.lineWidth = 1.2 + 1.6 * gl; ctx.stroke();
        txt(ctx, wd, x + w / 2, y + hgt / 2 + .5, k.Y, fs, 600, 'center'); out.push({ wd, x: x + w / 2, y: y + hgt / 2, x0: x, w }); x += w + 5;
      });
      if (gi < PROMPTS[S.prompt].length - 1) { txt(ctx, '.', x + 1, y + hgt / 2, k.mu, 13, 700); x += 10; }
    });
    return { out, y: y + hgt };
  }
  function draw(ctx, w, h) {
    const k = DK.C(); ctx.clearRect(0, 0, w, h); if (w < 80 || h < 80) return;
    const G = geo(w, h), { sx, sy, sw, sh } = G, cw = sw / GX, ch = sh / GY;
    const X = u => sx + u * sw, Y = v => sy + v * sh;
    const gI = glow('image'), gT = glow('text'), gS = glow('scores'), gM = glow('max'), gK = glow('topk');
    // scene and token grid
    DK.rr(ctx, sx - 1, sy - 1, sw + 2, sh + 2, 6); ctx.fillStyle = k.bg; ctx.fill(); ctx.strokeStyle = k.rule; ctx.lineWidth = 1; ctx.stroke();
    DK.drawScene(ctx, OBJS, sx, sy, sw, sh, k, 1);
    const aHeat = Math.max(S.stage === 'qsel' ? 1 : S.stage === 'decoder' ? .3 : 0, gS, gM);
    if (aHeat > 0) for (let i = 0; i < N; i++) {
      const hn = PE.clamp(heatNow(i), 0, 1), [u, v] = cellBox(i);
      ctx.fillStyle = k.G; ctx.globalAlpha = (.04 + .5 * Math.pow(hn, 1.6)) * aHeat; ctx.fillRect(X(u) + .5, Y(v) + .5, cw - 1, ch - 1);
    }
    ctx.globalAlpha = 1;
    ctx.save(); ctx.strokeStyle = gI > 0 ? k.B : k.mu; ctx.globalAlpha = .22 + .6 * gI; ctx.lineWidth = .6 + 1.2 * gI; ctx.beginPath();
    for (let c = 0; c <= GX; c++) { ctx.moveTo(X(c / GX), sy); ctx.lineTo(X(c / GX), sy + sh); }
    for (let r = 0; r <= GY; r++) { ctx.moveTo(sx, Y(r / GY)); ctx.lineTo(sx + sw, Y(r / GY)); }
    ctx.stroke(); ctx.restore();
    txt(ctx, `image tokens: ${GX} x ${GY} = ${N} (toy; the paper has more than 10,000)`, sx, sy - 13, k.B, G.narrow ? 10.5 : 11.5, 600);
    // selected tokens and their anchors
    const showSel = Math.max(S.stage === 'qsel' ? 1 : S.stage === 'decoder' ? 1 : 0, gK);
    const fin = S.stage === 'decoder' ? finals(S) : null, uD = S.stage === 'decoder' ? S.uDec : 0;
    if (showSel > 0) S.sel.forEach((i, r) => {
      const a = PE.clamp(S.uSel * (S.sel.length + 2) - r, 0, 1) * showSel; if (a <= 0) return;
      const [u, v] = cellBox(i);
      ctx.save(); ctx.globalAlpha = a * (S.stage === 'decoder' ? .45 : 1); ctx.strokeStyle = k.G; ctx.lineWidth = 2 + 1.5 * gK;
      ctx.strokeRect(X(u) + 1.5, Y(v) + 1.5, cw - 3, ch - 3); ctx.restore();
      const A = anchorOf(i), F = finalOf(i), b = S.stage === 'decoder' ? A.map((q, m) => PE.lerp(q, F[m], uD)) : A;
      const done = S.stage === 'decoder' && uD > .85, f = fin && fin[r], win = done && f.p >= .5;
      const col = done ? (win ? k.G : k.Br) : k.B, al = a * (done && !win ? .55 : .9);
      DK.brackets(ctx, X(b[0]), Y(b[1]), b[2] * sw, b[3] * sh, col, { lw: win ? 2.4 : 1.4 + gK, alpha: al, dash: done && !win ? [3, 3] : null, len: .22 });
      if (win) DK.tag(ctx, X(b[0]), Y(b[1]), f.name, f.p.toFixed(2), k.G, k, { size: G.narrow ? 10 : 11, maxX: sx + sw });
    });
    // focused token (its scores are shown in the panel)
    if (S.stage === 'qsel' && S.focus >= 0) { const [u, v] = cellBox(S.focus);
      ctx.save(); ctx.strokeStyle = k.fg; ctx.setLineDash([3, 2]); ctx.lineWidth = 1.4; ctx.strokeRect(X(u) - 1.5, Y(v) - 1.5, cw + 3, ch + 3); ctx.restore(); }
    // panel
    let { px, pw } = G; const yMax = G.py + G.ph, nar = G.narrow, fsz = nar ? 10.5 : 11.5;
    let y = G.py;
    txt(ctx, 'prompt tokens (text features)', px, y + 6, k.mu, fsz); y += 16;
    const C = chips(ctx, G, k, px, y, pw, gT); y = C.y + 12;
    if (S.stage === 'fusion') {
      // text-to-image attention, toy: each word's softmax over image tokens of its score / 0.08; top 10 drawn
      C.out.forEach(c => {
        const sc = feats.map(f => dotv(f, WORD[c.wd]) / .08), m = Math.max(...sc), e = sc.map(s => Math.exp(s - m));
        const top = order(e).slice(0, 10);
        top.forEach((i, r) => { const [u, v] = cellBox(i), a = PE.clamp(S.uFus * 12 - r, 0, 1) * (.2 + .8 * e[i] / e[top[0]]);
          const tx = X(u) + cw / 2, ty = Y(v) + ch / 2, fx = c.x, fy = (nar || G.split) ? c.y - 11 : c.y;
          ctx.save(); ctx.globalAlpha = a * .75; ctx.strokeStyle = k.Y; ctx.lineWidth = 1.2; ctx.beginPath(); ctx.moveTo(fx, fy);
          ctx.lineTo(PE.lerp(fx, tx, PE.clamp(S.uFus * 1.4, 0, 1)), PE.lerp(fy, ty, PE.clamp(S.uFus * 1.4, 0, 1))); ctx.stroke(); ctx.restore();
          ctx.save(); ctx.globalAlpha = a; ctx.strokeStyle = k.B; ctx.lineWidth = 2; ctx.strokeRect(X(u) + 1.5, Y(v) + 1.5, cw - 3, ch - 3); ctx.restore(); });
      });
      if (G.split) { px = G.px2; y = G.py; }
      const L = ['feature enhancer, 6 layers; in each one:', ['image tokens: deformable self-attention', k.B], ['text tokens: self-attention', k.Y],
        ['image-to-text cross-attention', k.fg], ['text-to-image cross-attention (lines)', k.Y], ['an FFN on each side', k.fg]];
      L.forEach((l, j) => { if (y + 14 > yMax) return; const [s, c] = typeof l === 'string' ? [l, k.fg] : l; txt(ctx, (j ? '- ' : '') + s, px, y + 6, c, fsz, j ? 500 : 600); y += 17; });
      y += 6;
      ['Lines: the image tokens each word reads most.', 'After the enhancer, the image features', 'depend on the prompt.'].forEach(s => { if (y + 12 > yMax) return; txt(ctx, s, px, y + 6, k.mu, fsz); y += 16; });
    } else if (S.stage === 'qsel') {
      const f = feats[S.focus], ws = words(S.prompt), sc = ws.map(wd => dotv(f, WORD[wd])), mx = Math.max(...sc), gx = S.focus % GX, gy = Math.floor(S.focus / GX);
      txt(ctx, `scores of the dashed token (${gx}, ${gy})`, px, y + 6, k.mu, fsz); y += 18;
      const lab = 64, bw = Math.max(40, pw - lab - 44);
      ws.forEach((wd, j) => { const v = sc[j], isMax = v === mx;
        txt(ctx, wd, px, y + 7, k.Y, fsz, 600);
        ctx.fillStyle = isMax ? k.G : DK.mixh(k.bg, k.fg, .35); ctx.fillRect(px + lab, y + 1 - gS * 2, Math.max(1, bw * PE.clamp(v, 0, 1.1) / 1.1), 12 + gS * 4);
        if (isMax && gM > 0) { ctx.save(); ctx.strokeStyle = k.G; ctx.lineWidth = 1 + 2 * gM; ctx.strokeRect(px + lab - 2, y - 1, bw * PE.clamp(v, 0, 1.1) / 1.1 + 4, 16); ctx.restore(); }
        txt(ctx, v.toFixed(2), px + lab + bw * PE.clamp(v, 0, 1.1) / 1.1 + 5, y + 7, k.mu, fsz - .5); y += 18; });
      txt(ctx, `max over the words = ${mx.toFixed(2)}: this token's heat`, px, y + 6, k.G, fsz, 600); y += 24;
      if (G.split) { px = G.px2; y = G.py; }
      txt(ctx, `decoder queries: the top ${S.sel.length} of ${N} tokens`, px, y + 6, k.fg, fsz + .5, 600); y += 18;
      const rowH = nar ? 18 : 19, room = Math.max(0, Math.floor((yMax - y - 54) / rowH)), shown = Math.min(S.sel.length, room);
      const lw = Math.round(pw * .56), rw = pw - lw - 4;
      for (let r = 0; r < shown; r++) {
        const i = S.sel[r], a = PE.clamp(S.uSel * (S.sel.length + 2) - r, 0, 1);
        ctx.save(); ctx.globalAlpha = .25 + .75 * a;
        DK.rr(ctx, px, y, lw, rowH - 4, 4); ctx.fillStyle = DK.mixh(k.bg, k.G, .18 + .3 * gK); ctx.fill(); ctx.strokeStyle = k.G; ctx.lineWidth = 1; ctx.stroke();
        txt(ctx, `anchor box from token (${i % GX}, ${Math.floor(i / GX)})`, px + 6, y + (rowH - 4) / 2 + .5, k.fg, nar ? 9.5 : 10.5);
        DK.rr(ctx, px + lw + 4, y, rw, rowH - 4, 4); ctx.setLineDash([3, 2]); ctx.strokeStyle = k.Br; ctx.stroke(); ctx.setLineDash([]);
        txt(ctx, 'content: learned', px + lw + 10, y + (rowH - 4) / 2 + .5, k.Br, nar ? 9.5 : 10.5); ctx.restore();
        y += rowH;
      }
      if (shown < S.sel.length) { txt(ctx, `and ${S.sel.length - shown} more`, px, y + 5, k.mu, fsz - .5); y += 15; }
      y += 4;
      if (y + 28 <= yMax + 4) {
        txt(ctx, 'The token sets only the position part (an anchor box).', px, y + 6, k.mu, fsz - .5);
        txt(ctx, 'The content part is learned, the same for every image.', px, y + 21, k.mu, fsz - .5);
      }
    } else {
      if (G.split) { px = G.px2; y = G.py; }
      const layer = Math.min(6, 1 + Math.floor(S.uDec * 6));
      txt(ctx, `decoder layer ${layer} of 6; in each one:`, px, y + 6, k.fg, fsz, 600); y += 17;
      [['self-attention among the queries', k.fg], ['image cross-attention (deformable)', k.B], ['text cross-attention (new in Grounding DINO)', k.Y], ['an FFN', k.fg]]
        .forEach(([s, c]) => { if (y + 12 > yMax) return; txt(ctx, '- ' + s, px, y + 6, c, fsz); y += 17; });
      y += 6;
      if (S.uDec > .85 && y + 14 <= yMax) {
        const F = finals(S), wins = F.filter(f => f.p >= .5);
        txt(ctx, 'final score: each query against the text tokens', px, y + 6, k.fg, fsz, 600); y += 18;
        wins.forEach(f => { if (y + 12 > yMax) return; txt(ctx, `${f.name} ${f.p.toFixed(2)}: a match`, px, y + 6, k.G, fsz, 600); y += 16; });
        if (y + 12 <= yMax) txt(ctx, `${F.length - wins.length} of ${F.length} queries score low (no object)`, px, y + 6, k.Br, fsz, 600);
      }
    }
  }
  const cv = PE.canvas(el, o.aspect || (w => (w < 620 ? Math.max(.5625 * (w - 16) + 320, 480) / w : Math.max(.355 * w + 46, 340) / w)), draw);
  function counts() { const c = {}; S.sel.forEach(i => { const n = owner[i] < 0 ? 'background' : NAMEOF[OBJS[owner[i]].kind]; c[n] = (c[n] || 0) + 1; });
    return Object.entries(c).sort((a, b) => b[1] - a[1]).map(([n, v]) => `${v} on the ${n}`).join(', '); }
  function describe() {
    const pr = PROMPTS[S.prompt].map(gp => gp.join(' ')).join(' . '), nw = words(S.prompt).length, top = S.hTo[S.sel[0]];
    if (S.stage === 'fusion') return `Feature enhancer (toy): each word of "${pr}" reads the image tokens it matches best, and every image token reads the prompt too, so the image features now depend on it.`;
    if (S.stage === 'decoder') { const F = finals(S), wins = F.filter(f => f.p >= .5);
      return `Decoder (toy): the ${S.sel.length} anchors move over 6 layers, reading the image and the text. ` + (wins.length
        ? `${wins.map(f => `"${f.name}" claims one box (${f.p.toFixed(2)})`).join('; ')}; the other ${F.length - wins.length} queries score low.`
        : `No query scores above 0.5 for "${pr}": the final score rejects all ${F.length}.`); }
    const g0 = S.focus % GX, g1 = Math.floor(S.focus / GX);
    return `Prompt "${pr}": ${nw} word token${nw > 1 ? 's' : ''}. Each of the ${N} image tokens keeps its best score over the words; the top ${S.sel.length} (${counts()}) become anchor boxes. ` +
      (top < .6 ? `The best heat is only ${top.toFixed(2)}: nothing matches well, yet ${S.sel.length} tokens are still selected. ` : '') +
      `Dashed token (${g0}, ${g1}): heat ${S.hTo[S.focus].toFixed(2)}.`;
  }
  const say = () => o.onSay && o.onSay(describe());
  function tween(key, ms, done) { (stops[key] || (() => {}))(); S[key] = 0; stops[key] = PE.animate(ms, u => { S[key] = u; cv.redraw(); }, done); }
  // set({ stage, prompt, k }); every field is a state, so setting it twice equals setting it once.
  function set(p) {
    const st = p.stage || S.stage, pr = p.prompt || S.prompt, K = p.k || S.K;
    const prChanged = pr !== S.prompt, stChanged = st !== S.stage, kChanged = K !== S.K;
    if (!prChanged && !stChanged && !kChanged) { say(); return; }
    if (prChanged) { S.hFrom = S.hFrom.map((_, i) => heatNow(i)); S.hTo = heatOf(pr); S.prompt = pr; S.focus = -1; }
    S.K = K; S.stage = st; pick();
    if (prChanged) { tween('uHeat', 700); tween('uSel', 800); }
    else if (kChanged) { S.uSel = 1; }
    if (stChanged || prChanged) { if (st === 'fusion') tween('uFus', 800); if (st === 'decoder') tween('uDec', 900); }
    cv.redraw(); say();
  }
  function reset() { Object.values(stops).forEach(f => f()); stops = {}; S = reset0(); pick(); cv.redraw(); say(); }
  const cellAt = e => { const r = el.getBoundingClientRect(), G = geo(r.width, r.height), x = e.clientX - r.left, y = e.clientY - r.top;
    if (x < G.sx || y < G.sy || x >= G.sx + G.sw || y >= G.sy + G.sh) return -1;
    return Math.floor((y - G.sy) / (G.sh / GY)) * GX + Math.floor((x - G.sx) / (G.sw / GX)); };
  const offs = [
    ['pointermove', e => { if (S.stage !== 'qsel') return; const i = cellAt(e); if (i >= 0 && i !== S.focus) { S.focus = i; cv.redraw(); } }],
    ['pointerdown', e => { if (S.stage !== 'qsel') return; const i = cellAt(e); if (i >= 0) { S.focus = i; cv.redraw(); say(); } }],
    ['keydown', e => { const d = { ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowUp: [0, -1], ArrowDown: [0, 1] }[e.key];
      if (!d || S.stage !== 'qsel') return; e.preventDefault();
      const gx = PE.clamp(S.focus % GX + d[0], 0, GX - 1), gy = PE.clamp(Math.floor(S.focus / GX) + d[1], 0, GY - 1);
      S.focus = gy * GX + gx; cv.redraw(); say(); }],
  ].map(([ev, fn]) => { el.addEventListener(ev, fn); return () => el.removeEventListener(ev, fn); });
  say();
  return { set, reset, redraw: cv.redraw, resize: () => cv.resize && cv.resize(), state: () => ({ stage: S.stage, prompt: S.prompt, k: S.K }),
    destroy: () => { Object.values(stops).forEach(f => f()); offs.forEach(f => f()); cv.destroy && cv.destroy(); } };
};
</script>
