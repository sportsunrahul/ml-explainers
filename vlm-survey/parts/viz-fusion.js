<script>
(() => {
  const fig = document.getElementById('viz-fusion');
  const el = document.getElementById('fusion-canvas'), vqEl = document.getElementById('fusion-vq-canvas');
  const out = document.getElementById('fusion-readout'), slider = document.getElementById('rng-fusion-k'), kOut = document.getElementById('fusion-k-val');
  const S = { u: 0, mode: 'late', K: 8, rec: null };
  let stopPath = () => {}, stopVq = () => {};
  const glow = PE.linkPulses(fig, () => { cv.redraw(); vq.redraw(); });

  // ---- Toy image and toy vector quantiser (same generator and seeds as build/sim/fusion-vq.cjs). ----
  const SZ = 32, P = 4, G = SZ / P;
  function pix(x, y) {
    const cx = x + .5, cy = y + .5;
    if ((cx - 24) ** 2 + (cy - 7.5) ** 2 < 16) return .95;
    if (cx > 9 && cx < 11.5 && cy > 20 && cy < 25) return .08;
    if (cx > 5 && cx < 15.5 && cy > 16 && cy < 25) return .72;
    if (cy > 9 && cy <= 16 && Math.abs(cx - 10.25) < (cy - 9) * .8) return .42;
    if (cy >= 25) return .5 + .12 * ((x + y) % 2);
    return .12 + .1 * cy / 25;
  }
  const rn = PE.rng(3), img = [];
  for (let y = 0; y < SZ; y++) for (let x = 0; x < SZ; x++) img.push(PE.clamp(pix(x, y) + .06 * (rn() - .5) + .04 * Math.sin(x * .7 + y * .3), 0, 1));
  const patches = [];
  for (let py = 0; py < G; py++) for (let px = 0; px < G; px++) {
    const v = []; for (let y = 0; y < P; y++) for (let x = 0; x < P; x++) v.push(img[(py * P + y) * SZ + px * P + x]); patches.push(v); }
  const d2 = (a, b) => a.reduce((s, v, i) => s + (v - b[i]) ** 2, 0);
  const cache = {};
  function kmeans(K) {
    if (cache[K]) return cache[K];
    const r = PE.rng(7), C = [patches[Math.floor(r() * patches.length)].slice()];
    while (C.length < K) { const d = patches.map(p => Math.min(...C.map(c => d2(p, c)))), T = d.reduce((a, b) => a + b, 0);
      if (T === 0) break; let u = r() * T, i = 0; while ((u -= d[i]) > 0) i++; C.push(patches[i].slice()); }
    let ids = [];
    for (let it = 0; it < 30; it++) {
      ids = patches.map(p => { let b = 0, bd = 1e9; C.forEach((c, k) => { const d = d2(p, c); if (d < bd) { bd = d; b = k; } }); return b; });
      C.forEach((c, k) => { const m = patches.filter((_, i) => ids[i] === k); if (m.length) for (let j = 0; j < c.length; j++) c[j] = m.reduce((s, p) => s + p[j], 0) / m.length; });
    }
    const rec = new Array(SZ * SZ);
    ids.forEach((k, i) => { const px = i % G, py = Math.floor(i / G);
      for (let y = 0; y < P; y++) for (let x = 0; x < P; x++) rec[(py * P + y) * SZ + px * P + x] = C[k][y * P + x]; });
    const rmse = Math.sqrt(rec.reduce((s, v, i) => s + (v - img[i]) ** 2, 0) / rec.length);
    return (cache[K] = { C, ids, rec, rmse });
  }

  // ---- Shared drawing bits ----
  const col = () => ({ B: PE.css('--blue'), Y: PE.css('--yellow'), G: PE.css('--green'), R: PE.css('--red'), Br: PE.css('--brown'),
    F: PE.css('--fg'), M: PE.css('--muted'), Rule: PE.css('--rule'), BG: PE.css('--bg'), Pn: PE.css('--panel2') });
  function rrect(ctx, x, y, w, h, r) { if (w <= 0 || h <= 0) { ctx.beginPath(); return; } ctx.beginPath(); ctx.moveTo(x + r, y); ctx.arcTo(x + w, y, x + w, y + h, r); ctx.arcTo(x + w, y + h, x, y + h, r);
    ctx.arcTo(x, y + h, x, y, r); ctx.arcTo(x, y, x + w, y, r); ctx.closePath(); }
  function box(ctx, c, r, color, lines, alpha, g = 0) {
    if (alpha <= .01) return; ctx.save(); ctx.globalAlpha = alpha;
    rrect(ctx, r.x, r.y, r.w, r.h, 8); ctx.fillStyle = PE.mix(c.BG, color, .16); ctx.fill();
    ctx.strokeStyle = color; ctx.lineWidth = 2 + 3 * g; ctx.stroke();
    lines.forEach((t, i) => PE.label(ctx, t, r.x + r.w / 2, r.y + r.h / 2 + 4 + (i - (lines.length - 1) / 2) * 15, color, 12, 'center'));
    ctx.restore();
  }
  function chip(ctx, c, x, y, text, color, alpha, h = 20) {
    ctx.save(); ctx.font = '600 11px system-ui, sans-serif'; const w = Math.max(20, ctx.measureText(text).width + 10);
    ctx.globalAlpha = alpha; rrect(ctx, x, y - h / 2, w, h, 5); ctx.fillStyle = PE.mix(c.BG, color, .22); ctx.fill();
    ctx.strokeStyle = color; ctx.lineWidth = 1.2; ctx.stroke(); ctx.restore();
    PE.label(ctx, text, x + w / 2, y + 4, color, 11, 'center'); return w;
  }
  function chipRow(ctx, c, x, y, items, alpha, maxX, gap = 4) {
    ctx.save(); ctx.globalAlpha = alpha; let cx = x, cy = y;
    items.forEach(([t, color]) => { ctx.font = '600 11px system-ui, sans-serif'; const w = Math.max(20, ctx.measureText(t).width + 10);
      if (cx + w > maxX) { cx = x; cy += 24; } chip(ctx, c, cx, cy, t, color, alpha); cx += w + gap; });
    ctx.restore(); return cy;
  }
  function pictureIcon(ctx, c, x, y, s, alpha) {
    ctx.save(); ctx.globalAlpha = alpha; const q = s / SZ;
    for (let j = 0; j < SZ; j += 2) for (let i = 0; i < SZ; i += 2) { ctx.fillStyle = PE.mix(c.BG, c.B, img[j * SZ + i]); ctx.fillRect(x + i * q, y + j * q, 2 * q + .5, 2 * q + .5); }
    ctx.strokeStyle = c.B; ctx.lineWidth = 1.5; ctx.strokeRect(x, y, s, s); ctx.restore();
  }

  // ---- Top canvas: the two data paths. u = 0 late fusion, u = 1 early fusion (cross-faded). ----
  const cv = PE.canvas(el, w => w < 560 ? 1.25 : .37, (ctx, W, H) => {
    const c = col(); ctx.fillStyle = c.BG; ctx.fillRect(0, 0, W, H);
    const a = 1 - S.u, b = S.u, st = W < 560, gs = glow('seq'), gv = glow('vocab'), gz = glow('zloss');
    const txt = [['a', c.Y], ['cat', c.Y], ['on', c.Y], ['a', c.Y], ['mat', c.Y]];
    if (!st) {
      const y0 = Math.max(96, H * .36), iy = y0 - 34, isz = 68;
      pictureIcon(ctx, c, 12, iy, isz, 1); PE.label(ctx, 'image', 12 + isz / 2, iy + isz + 16, c.B, 11, 'center');
      // Late: encoder -> connector -> LLM -> text.
      const enc = { x: W * .17, y: y0 - 30, w: W * .15, h: 60 }, con = { x: W * .36, y: y0 - 22, w: W * .12, h: 44 },
        llm = { x: W * .53, y: y0 - 44, w: W * .2, h: 88 };
      if (a > .01) {
        PE.arrow(ctx, 12 + isz + 4, y0, enc.x - 4, y0, c.B, 2, 8, a);
        box(ctx, c, enc, c.B, ['vision', 'encoder'], a); PE.arrow(ctx, enc.x + enc.w + 3, y0, con.x - 4, y0, c.B, 2, 8, a);
        box(ctx, c, con, c.G, ['connector'], a); PE.arrow(ctx, con.x + con.w + 3, y0, llm.x - 4, y0, c.G, 2, 8, a);
        box(ctx, c, llm, c.Y, ['language', 'model'], a);
        chipRow(ctx, c, llm.x, y0 + 80, txt, a, W); PE.arrow(ctx, llm.x + 30, y0 + 68, llm.x + 30, llm.y + llm.h + 3, c.Y, 2, 8, a);
        ctx.save(); ctx.globalAlpha = a; PE.label(ctx, 'text in', llm.x - 6, y0 + 84, c.Y, 11, 'right'); ctx.restore();
        PE.arrow(ctx, llm.x + llm.w + 3, y0, W * .78, y0, c.Y, 2, 8, a);
        ctx.save(); ctx.globalAlpha = a; PE.label(ctx, 'text out', W * .79, y0 - 14, c.Y, 12); ctx.restore();
        chipRow(ctx, c, W * .79, y0 + 4, [['a', c.Y], ['cat', c.Y]], a, W);
        ctx.save(); ctx.globalAlpha = a; PE.label(ctx, 'image out: no path', W * .79, y0 + 40, c.Br, 11);
        PE.label(ctx, 'LATE FUSION: the image joins through a side door', 12, 20, c.M, 11); ctx.restore();
      }
      if (b > .01) {
        const tok = { x: W * .16, y: y0 - 26, w: W * .13, h: 52 }, sx = W * .33;
        PE.arrow(ctx, 12 + isz + 4, y0, tok.x - 4, y0, c.B, 2, 8, b);
        box(ctx, c, tok, c.B, ['image', 'tokenizer'], b);
        PE.arrow(ctx, tok.x + tok.w + 3, y0, sx - 8, y0, c.B, 2, 8, b);
        // One sequence: text tokens and image ids side by side.
        ctx.save(); ctx.globalAlpha = b; PE.label(ctx, 'one sequence', sx, y0 - 20, gs > 0 ? c.F : c.M, 11 + gs); ctx.restore();
        const seq = [...txt.slice(0, 2), ['<img>', c.B], ['4107', c.B], ['88', c.B], ['...', c.B], ['6302', c.B], ['</img>', c.B], ['on', c.Y], ['a', c.Y], ['mat', c.Y]];
        const endY = chipRow(ctx, c, sx, y0, seq, b, W - 8);
        if (gs > 0) { ctx.save(); ctx.globalAlpha = gs * b; ctx.strokeStyle = c.F; ctx.lineWidth = 2; rrect(ctx, sx - 6, y0 - 15, W - sx - 2, endY - y0 + 30, 7); ctx.stroke(); ctx.restore(); }
        const tf = { x: sx, y: endY + 40, w: W * .3, h: 50 };
        PE.arrow(ctx, tf.x + tf.w / 2, endY + 14, tf.x + tf.w / 2, tf.y - 4, c.F, 2, 8, b);
        box(ctx, c, tf, c.Y, ['one transformer', 'reads and writes both'], b);
        // Vocabulary bar: 65,536 entries, 8,192 of them image codes.
        const vy = tf.y + tf.h + 12, vw = tf.w, vfrac = 8192 / 65536;
        ctx.save(); ctx.globalAlpha = b; ctx.fillStyle = c.Y; ctx.fillRect(tf.x, vy, vw * (1 - vfrac), 10); ctx.fillStyle = c.B; ctx.fillRect(tf.x + vw * (1 - vfrac), vy, vw * vfrac, 10);
        if (gv > 0) { ctx.strokeStyle = c.F; ctx.lineWidth = 1 + 2.5 * gv; ctx.strokeRect(tf.x - 2, vy - 2, vw + 4, 14); }
        PE.label(ctx, 'vocabulary: 65,536, 1/8 image codes', tf.x, vy + 25, gv > 0 ? c.F : c.M, 10.5 + gv); ctx.restore();
        // Output: next token from either part of the vocabulary; z-loss sits on this softmax.
        const ox = tf.x + tf.w + 34;
        PE.arrow(ctx, tf.x + tf.w + 3, tf.y + tf.h / 2, ox - 6, tf.y + tf.h / 2, c.F, 2, 8, b);
        ctx.save(); ctx.globalAlpha = b; PE.label(ctx, 'next token: text', ox, tf.y + 10, c.Y, 11.5); PE.label(ctx, 'or image id', ox, tf.y + 26, c.B, 11.5);
        PE.label(ctx, 'softmax over all of it', ox, tf.y + 44, gz > 0 ? c.R : c.M, 10.5 + gz * 1.5);
        if (gz > 0) { ctx.strokeStyle = c.R; ctx.lineWidth = 1 + 2 * gz; rrect(ctx, ox - 6, tf.y - 6, W - ox, 58, 6); ctx.stroke(); }
        PE.label(ctx, 'image ids go back through the', ox, tf.y + 70, c.B, 10.5); PE.label(ctx, "tokenizer's decoder to pixels", ox, tf.y + 84, c.B, 10.5);
        PE.label(ctx, 'EARLY FUSION: the image is written in the same alphabet', 12, 20, c.M, 11); ctx.restore();
      }
    } else {
      // Phone: the same two paths, top to bottom.
      const cx = W / 2, isz = 56;
      pictureIcon(ctx, c, cx - isz / 2, 34, isz, 1); PE.label(ctx, 'image', cx + isz / 2 + 8, 66, c.B, 11);
      const bw = W * .52, r = (y, h) => ({ x: cx - bw / 2, y, w: bw, h });
      if (a > .01) {
        ctx.save(); ctx.globalAlpha = a; PE.label(ctx, 'LATE FUSION', 8, 18, c.M, 11); ctx.restore();
        const enc = r(112, 40), con = r(172, 34), llm = r(236, 60);
        PE.arrow(ctx, cx, 34 + isz + 3, cx, enc.y - 3, c.B, 2, 8, a); box(ctx, c, enc, c.B, ['vision encoder'], a);
        PE.arrow(ctx, cx, enc.y + enc.h + 2, cx, con.y - 3, c.B, 2, 8, a); box(ctx, c, con, c.G, ['connector'], a);
        PE.arrow(ctx, cx, con.y + con.h + 2, cx, llm.y - 3, c.G, 2, 8, a); box(ctx, c, llm, c.Y, ['language model'], a);
        ctx.save(); ctx.globalAlpha = a; PE.label(ctx, 'text in', 8, llm.y + 18, c.Y, 11); ctx.restore();
        chipRow(ctx, c, 8, llm.y + 36, txt.slice(1, 3), a, llm.x - 4, 3);
        PE.arrow(ctx, cx, llm.y + llm.h + 2, cx, 340, c.Y, 2, 8, a);
        ctx.save(); ctx.globalAlpha = a; PE.label(ctx, 'text out', cx - 60, 362, c.Y, 12); ctx.restore();
        chipRow(ctx, c, cx, 358, [['a', c.Y], ['cat', c.Y]], a, W);
        ctx.save(); ctx.globalAlpha = a; PE.label(ctx, 'image out: no path', cx, 396, c.Br, 11, 'center'); ctx.restore();
      }
      if (b > .01) {
        ctx.save(); ctx.globalAlpha = b; PE.label(ctx, 'EARLY FUSION', 8, 18, c.M, 11); ctx.restore();
        const tok = r(112, 40);
        PE.arrow(ctx, cx, 34 + isz + 3, cx, tok.y - 3, c.B, 2, 8, b); box(ctx, c, tok, c.B, ['image tokenizer'], b);
        const seq = [['a', c.Y], ['cat', c.Y], ['<img>', c.B], ['4107', c.B], ['88', c.B], ['...', c.B], ['</img>', c.B], ['on', c.Y], ['a', c.Y], ['mat', c.Y]];
        ctx.save(); ctx.globalAlpha = b; PE.label(ctx, 'one sequence', 8, 176, gs > 0 ? c.F : c.M, 11 + gs); ctx.restore();
        PE.arrow(ctx, cx, tok.y + tok.h + 2, cx, 184, c.B, 2, 8, b);
        if (gs > 0) { ctx.save(); ctx.globalAlpha = gs * b; ctx.strokeStyle = c.F; ctx.lineWidth = 2; rrect(ctx, 4, 186, W - 8, 54, 7); ctx.stroke(); ctx.restore(); }
        const endY = chipRow(ctx, c, 10, 200, seq, b, W - 8, 3);
        const tf = r(endY + 26, 44);
        PE.arrow(ctx, cx, endY + 13, cx, tf.y - 3, c.F, 2, 8, b); box(ctx, c, tf, c.Y, ['one transformer'], b);
        const vy = tf.y + tf.h + 10, vw = tf.w, vfrac = 8192 / 65536;
        ctx.save(); ctx.globalAlpha = b; ctx.fillStyle = c.Y; ctx.fillRect(tf.x, vy, vw * (1 - vfrac), 9); ctx.fillStyle = c.B; ctx.fillRect(tf.x + vw * (1 - vfrac), vy, vw * vfrac, 9);
        if (gv > 0) { ctx.strokeStyle = c.F; ctx.lineWidth = 1 + 2.5 * gv; ctx.strokeRect(tf.x - 2, vy - 2, vw + 4, 13); }
        PE.label(ctx, 'vocabulary: 65,536, 1/8 image codes', cx, vy + 24, gv > 0 ? c.F : c.M, 10.5 + gv, 'center');
        PE.label(ctx, 'next token: text or image id', cx, vy + 44, c.F, 11.5, 'center');
        PE.label(ctx, 'softmax over all of it', cx, vy + 60, gz > 0 ? c.R : c.M, 10.5 + gz * 1.5, 'center');
        if (gz > 0) { ctx.strokeStyle = c.R; ctx.lineWidth = 1 + 2 * gz; rrect(ctx, cx - 100, vy + 30, 200, 38, 6); ctx.stroke(); }
        PE.label(ctx, 'ids -> tokenizer decoder -> pixels', cx, vy + 82, c.B, 10.5, 'center'); ctx.restore();
      }
    }
  });

  // ---- Bottom canvas: the toy tokenizer. ----
  const vq = PE.canvas(vqEl, w => w < 560 ? 1.36 : .36, (ctx, W, H) => {
    const c = col(); ctx.fillStyle = c.BG; ctx.fillRect(0, 0, W, H);
    const cur = kmeans(S.K), rec = S.rec || cur.rec, st = W < 560, gp = glow('patch'), gc = glow('code'), gi = glow('id');
    PE.label(ctx, 'TOY MODEL: k-means codebook on one drawn 32 x 32 image', 8, 16, c.Br, 10.5);
    const gap = st ? 12 : 16, s = st ? Math.min((W - 3 * gap) / 2, 150) : Math.min((W - 5 * gap) / 4, H * .5);
    const pos = st ? [[gap, 40], [2 * gap + s, 40], [gap, 40 + s + 30], [2 * gap + s, 40 + s + 30]]
      : [[gap, 44], [2 * gap + s, 44], [3 * gap + 2 * s, 44], [4 * gap + 3 * s, 44]];
    const q = s / SZ, cell = s / G;
    const paint = (x0, y0, f) => { for (let y = 0; y < SZ; y++) for (let x = 0; x < SZ; x++) { ctx.fillStyle = f(y * SZ + x); ctx.fillRect(x0 + x * q, y0 + y * q, q + .6, q + .6); } };
    const grid = (x0, y0, color, lw) => { ctx.save(); ctx.strokeStyle = color; ctx.lineWidth = lw; ctx.globalAlpha = .55 + .45 * Math.min(1, lw - .5);
      for (let k = 0; k <= G; k++) { ctx.beginPath(); ctx.moveTo(x0 + k * cell, y0); ctx.lineTo(x0 + k * cell, y0 + s); ctx.moveTo(x0, y0 + k * cell); ctx.lineTo(x0 + s, y0 + k * cell); ctx.stroke(); } ctx.restore(); };
    // 1. original with its 8 x 8 patch grid
    let [x0, y0] = pos[0]; paint(x0, y0, i => PE.mix(c.BG, c.B, img[i])); grid(x0, y0, gp > 0 ? c.F : c.Rule, 1 + 2 * gp);
    PE.label(ctx, 'image: 64 patches', x0, y0 - 8, c.B, 11);
    // 2. the ids: the image as 64 "words"
    [x0, y0] = pos[1]; ctx.fillStyle = c.Pn; ctx.fillRect(x0, y0, s, s);
    cur.ids.forEach((k, i) => { const px = i % G, py = Math.floor(i / G);
      PE.label(ctx, String(k), x0 + px * cell + cell / 2, y0 + py * cell + cell / 2 + 4, gi > 0 ? c.F : c.B, Math.min(11, cell * .5) + gi, 'center'); });
    grid(x0, y0, c.Rule, 1); PE.label(ctx, 'ids the model reads', x0, y0 - 8, c.B, 11);
    // 3. reconstruction from the codebook
    [x0, y0] = pos[2]; paint(x0, y0, i => PE.mix(c.BG, c.B, PE.clamp(rec[i], 0, 1))); grid(x0, y0, c.Rule, 1);
    PE.label(ctx, 'rebuilt from ids', x0, y0 - 8, c.B, 11);
    // 4. what was lost
    [x0, y0] = pos[3]; paint(x0, y0, i => PE.mix(c.BG, c.R, PE.clamp(Math.abs(rec[i] - img[i]) * 3, 0, 1)));
    ctx.strokeStyle = c.Rule; ctx.lineWidth = 1; ctx.strokeRect(x0, y0, s, s);
    PE.label(ctx, 'what was lost', x0, y0 - 8, c.R, 11);
    // Codebook: K tiles, each a 4 x 4 patch the tokenizer may use.
    const cy0 = st ? 40 + 2 * s + 60 : 44 + s + 34, avail = W - 2 * gap, per = st ? 16 : 32, ts = Math.min(26, (avail - (per - 1) * 4) / per);
    PE.label(ctx, `codebook: K = ${S.K} tiles (ids 0 to ${S.K - 1})`, gap, cy0 - 8, gc > 0 ? c.F : c.B, 11 + gc);
    cur.C.forEach((tile, k) => { const tx = gap + (k % per) * (ts + 4), ty = cy0 + Math.floor(k / per) * (ts + 4), tq = ts / P;
      for (let y = 0; y < P; y++) for (let x = 0; x < P; x++) { ctx.fillStyle = PE.mix(c.BG, c.B, PE.clamp(tile[y * P + x], 0, 1)); ctx.fillRect(tx + x * tq, ty + y * tq, tq + .5, tq + .5); }
      ctx.strokeStyle = gc > 0 ? c.F : c.Rule; ctx.lineWidth = 1 + 1.5 * gc; ctx.strokeRect(tx, ty, ts, ts); });
  });

  function readout() {
    const r = kmeans(S.K), bits = Math.log2(S.K);
    const path = S.mode === 'late'
      ? 'Late fusion: the image goes encoder, connector, language model; only text comes out.'
      : 'Early fusion: the image becomes ids in the same sequence as the text; one transformer can emit either kind of token.';
    out.textContent = `${path} Toy tokenizer: K = ${S.K} codebook tiles, so each of the 64 patches becomes one id worth ${bits.toFixed(1)} bits; reconstruction error ${(100 * r.rmse).toFixed(1)}% (root mean square, of full brightness).`;
  }
  function setPath(mode) {
    stopPath(); S.mode = mode; const u0 = S.u, u1 = mode === 'early' ? 1 : 0;
    document.getElementById('btn-fusion-late').setAttribute('aria-pressed', mode === 'late');
    document.getElementById('btn-fusion-early').setAttribute('aria-pressed', mode === 'early');
    stopPath = PE.animate(800, u => { S.u = PE.lerp(u0, u1, u); cv.redraw(); }); readout();
  }
  function setK(K, tween) {
    stopVq(); const from = (S.rec || kmeans(S.K).rec).slice(); S.K = K; slider.value = K; kOut.textContent = K;
    const to = kmeans(K).rec; readout();
    if (!tween) { S.rec = null; vq.redraw(); return; }
    stopVq = PE.animate(700, u => { S.rec = from.map((v, i) => PE.lerp(v, to[i], u)); vq.redraw(); }, () => { S.rec = null; vq.redraw(); });
  }
  document.getElementById('btn-fusion-late').addEventListener('click', () => setPath('late'));
  document.getElementById('btn-fusion-early').addEventListener('click', () => setPath('early'));
  document.getElementById('btn-fusion-k2').addEventListener('click', () => setK(2, true));
  document.getElementById('btn-fusion-k32').addEventListener('click', () => setK(32, true));
  document.getElementById('btn-fusion-reset').addEventListener('click', () => {
    stopPath(); stopVq(); S.u = 0; S.mode = 'late'; S.rec = null; setK(8, false); setPath('late'); cv.redraw(); });
  slider.addEventListener('input', () => setK(+slider.value, false));
  setK(8, false); setPath('late');
})();
</script>
