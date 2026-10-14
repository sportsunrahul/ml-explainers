<script>
(() => {
  const fig = document.getElementById('viz-mrope'), el = document.getElementById('mrope-canvas'), out = document.getElementById('mrope-readout');
  // Qwen2-VL's rule: text gets (n, n, n); a visual block starting at s gets (s + frame, s + row, s + column);
  // whatever follows a block starts at the largest id used in it, plus one.
  function build(video) {
    const blocks = [{ kind: 'text', words: ['Read', 'this', 'receipt:'] }, { kind: 'image', F: 1, R: 3, C: 4 }, { kind: 'text', words: ['The', 'total', 'is', '11.25.'] }];
    if (video) blocks.push({ kind: 'text', words: ['Now', 'this', 'clip:'] }, { kind: 'video', F: 2, R: 2, C: 3 }, { kind: 'text', words: ['What', 'moved?'] });
    const toks = []; let n = 0;
    blocks.forEach((b, bi) => {
      b.first = toks.length;
      if (b.kind === 'text') b.words.forEach(w => { toks.push({ kind: 'text', w, bi, m: [n, n, n] }); n++; });
      else { const s = n; let mx = s;
        for (let f = 0; f < b.F; f++) for (let r = 0; r < b.R; r++) for (let c = 0; c < b.C; c++) {
          const m = [s + f, s + r, s + c]; mx = Math.max(mx, ...m); toks.push({ kind: b.kind, f, r, c, bi, m }); }
        b.s = s; b.max = mx; n = mx + 1; }
    });
    toks.forEach((t, i) => { t.one = i; });
    return { blocks, toks, max1: toks.length - 1, maxM: n - 1 };
  }
  const SEQ = { false: build(false), true: build(true) };
  const S = { mode: '1d', from: '1d', u: 1, video: false, vu: 0, sel: 15 };
  const glow = PE.linkPulses(fig, () => cv.redraw());
  let stop = () => {};
  const mctx = el.getContext('2d');
  const tw = (t, px) => { mctx.save(); mctx.font = `600 ${px}px system-ui, sans-serif`; const w = mctx.measureText(t).width; mctx.restore(); return w; };

  // Flow layout: words are pills, an image is a grid of cells, a video is its frames side by side. Wraps to the width.
  function place(W) {
    const small = W < 560, cs = small ? 34 : 40, pad = 12, gapX = small ? 6 : 8, fs = small ? 11.5 : 12.5, seq = SEQ.true;
    const items = []; let x = pad, y = 34, rowH = 0; const rows = [];
    const put = (w, h, obj) => { if (x + w > W - pad && x > pad) { rows.push({ y, h: rowH }); y += rowH + 14; x = pad; rowH = 0; }
      obj.x = x; obj.y = y; obj.w = w; obj.h = h; obj.row = rows.length; items.push(obj); x += w + gapX; rowH = Math.max(rowH, h); };
    seq.blocks.forEach((b, bi) => {
      if (b.kind === 'text') b.words.forEach((w, k) => put(tw(w, fs) + 16, 46, { type: 'pill', t: b.first + k, bi }));
      else put(b.F * b.C * cs + (b.F - 1) * 10, b.R * cs + 18, { type: 'grid', bi, cs, F: b.F, R: b.R, C: b.C });
    });
    rows.push({ y, h: rowH });
    // Vertically center each item in its row.
    items.forEach(it => { const r = rows[it.row]; it.y = r.y + (r.h - it.h) / 2; });
    const cells = {};
    items.forEach(it => {
      if (it.type === 'pill') cells[it.t] = { x: it.x, y: it.y, w: it.w, h: 26, it };
      else { const b = seq.blocks[it.bi]; let t = b.first;
        for (let f = 0; f < it.F; f++) for (let r = 0; r < it.R; r++) for (let c = 0; c < it.C; c++)
          cells[t++] = { x: it.x + f * (it.C * it.cs + 10) + c * it.cs, y: it.y + 18 + r * it.cs, w: it.cs, h: it.cs, it }; }
    });
    return { items, cells, bottom: y + rowH, fs, cs, small };
  }
  const heightFor = W => place(W).bottom + (W < 560 ? 150 : 124);

  const cv = PE.canvas(el, w => heightFor(w) / w, (ctx, W, H) => {
    const B = PE.css('--blue'), Yl = PE.css('--yellow'), F = PE.css('--fg'), M = PE.css('--muted'), Rule = PE.css('--rule'), BG = PE.css('--bg');
    ctx.fillStyle = BG; ctx.fillRect(0, 0, W, H);
    const Lh = place(W), seq = SEQ.true, live = SEQ[S.video], n = live.toks.length, u = S.u;
    const gt = glow('text'), gi = glow('img'), gv = glow('video'), gn = glow('next');
    const idText = (t, mode) => mode === '1d' ? String(t.one) : t.kind === 'text' ? t.m.join(',') : t.m[1] + ',' + t.m[2];
    PE.label(ctx, Lh.small ? 'each token shows its position id' : 'one sequence, in reading order; each token shows its position id', 12, 18, F, Lh.small ? 11 : 12);
    const alphaOf = t => t.bi >= 3 ? S.vu : 1;
    // How much of the M-RoPE-only drawing (t headers, "max + 1" labels) is showing, cross-fading on a mode switch.
    const mA = S.mode === 'm' ? (S.from === 'm' ? 1 : u) : (S.from === 'm' ? 1 - u : 0);
    // Block headers: under M-RoPE all tokens of an image (or of one video frame) share one temporal id t.
    Lh.items.forEach(it => { if (it.type !== 'grid') return; const b = seq.blocks[it.bi], a = it.bi >= 3 ? S.vu : 1; if (a <= 0) return;
      for (let f = 0; f < it.F; f++) { const hx = it.x + f * (it.C * it.cs + 10), name = b.kind === 'image' ? 'image' : 'frame ' + (f + 1);
        ctx.save(); ctx.globalAlpha = a; PE.label(ctx, name, hx, it.y + 12, B, 10.5);
        ctx.globalAlpha = a * mA; PE.label(ctx, 't = ' + (b.s + f), hx + tw(name, 10.5) + 6, it.y + 12, F, 10.5); ctx.restore(); } });
    for (let i = 0; i < seq.toks.length; i++) {
      const t = seq.toks[i], c = Lh.cells[i], a = alphaOf(t); if (a <= 0) continue;
      const sel = i === S.sel && i < n, vis = t.kind !== 'text', g = t.kind === 'text' ? gt : t.kind === 'image' ? gi : gv, col = vis ? B : Yl;
      ctx.save(); ctx.globalAlpha = a;
      ctx.fillStyle = col; ctx.globalAlpha = a * (.13 + .3 * g + (sel ? .2 : 0)); ctx.fillRect(c.x + 1, c.y + 1, c.w - 2, c.h - 2); ctx.globalAlpha = a;
      ctx.strokeStyle = sel ? F : col; ctx.lineWidth = sel ? 2.5 : 1 + 2 * g; ctx.strokeRect(c.x + 1, c.y + 1, c.w - 2, c.h - 2);
      if (!vis) PE.label(ctx, t.w, c.x + c.w / 2, c.y + 17.5, F, Lh.fs, 'center');
      // Ids: cross-fade between the two schemes.
      const lx = c.x + c.w / 2, ly = vis ? c.y + c.h / 2 + 4 : c.y + c.h + 14, fsz = vis ? (Lh.small ? 10 : 11) : 11;
      const A = S.from === S.mode ? 1 : u;
      if (S.from !== S.mode) { ctx.globalAlpha = a * (1 - u); PE.label(ctx, idText(t, S.from), lx, ly, vis ? F : M, fsz, 'center'); }
      ctx.globalAlpha = a * A; PE.label(ctx, idText(t, S.mode), lx, ly, sel ? F : vis ? F : M, fsz, 'center');
      ctx.restore();
    }
    // "max + 1": after each visual block the next token starts one past the largest id inside it.
    if (mA > 0) seq.blocks.forEach((b, bi) => { if (b.kind === 'text' || bi + 1 >= seq.blocks.length) return; const a = (bi >= 3 ? S.vu : 1) * mA; if (a <= 0) return;
      const nx = Lh.cells[seq.blocks[bi + 1].first]; if (!nx) return;
      ctx.save(); ctx.globalAlpha = a;
      PE.label(ctx, `max ${b.max} + 1`, nx.x + nx.w / 2, nx.y - 5, F, 10.5 + 1.5 * gn, 'center'); ctx.restore(); });

    // Id budget: which ids the sequence occupies, colored by who uses them (yellow text, blue vision).
    const y0 = Lh.bottom + 30, bw = W - 24, slot = bw / (SEQ.true.max1 + 1);
    const segs = mode => { const L = []; if (mode === '1d') live.toks.forEach(t => L.push([t.one, t.one + 1, t.kind === 'text' ? Yl : B]));
      else live.blocks.forEach(b => { if (b.kind === 'text') b.words.forEach((w, k) => { const id = live.toks[b.first + k].m[0]; L.push([id, id + 1, Yl]); });
        else L.push([b.s, b.max + 1, B]); }); return L; };
    PE.label(ctx, 'position ids used by the sequence', 12, y0 - 10, F, 12);
    ctx.fillStyle = Rule; ctx.fillRect(12, y0, bw, 16);
    const drawSegs = (mode, a) => { ctx.save(); ctx.globalAlpha = a; segs(mode).forEach(([s0, s1, col]) => { ctx.fillStyle = col; ctx.fillRect(12 + s0 * slot + .5, y0, (s1 - s0) * slot - 1, 16); }); ctx.restore(); };
    if (S.from !== S.mode) drawSegs(S.from, 1 - u); drawSegs(S.mode, S.from !== S.mode ? u : 1);
    const mxId = S.mode === '1d' ? live.max1 : live.maxM;
    PE.label(ctx, `0 to ${mxId}: ${S.mode === '1d' ? 'one id per token' : Lh.small ? 'a block uses as many ids as its longest side' : 'a visual block uses only as many ids as its longest side'}`, 12, y0 + 32, M, Lh.small ? 10.5 : 11.5);
    // The selected token, both ways.
    const t = live.toks[Math.min(S.sel, n - 1)], what = t.kind === 'text' ? `"${t.w}" (text)` : t.kind === 'image' ? `image token, row ${t.r + 1}, column ${t.c + 1}` : `video frame ${t.f + 1}, row ${t.r + 1}, column ${t.c + 1}`;
    const yd = y0 + 60;
    PE.label(ctx, 'selected: ' + what, 12, yd, F, 12);
    PE.label(ctx, `1D RoPE id: ${t.one}`, 12, yd + 22, S.mode === '1d' ? F : M, 12);
    PE.label(ctx, `M-RoPE (t, h, w): (${t.m.join(', ')})`, Lh.small ? 12 : 170, Lh.small ? yd + 42 : yd + 22, S.mode === 'm' ? F : M, 12);
  });

  function readout() {
    const live = SEQ[S.video], t = live.toks[Math.min(S.sel, live.toks.length - 1)];
    const what = t.kind === 'text' ? `the word "${t.w}"` : t.kind === 'image' ? `the image token at row ${t.r + 1}, column ${t.c + 1}` : `the video token in frame ${t.f + 1}, row ${t.r + 1}, column ${t.c + 1}`;
    out.textContent = (S.mode === '1d' ? '1D RoPE: every token gets the next integer. ' : 'M-RoPE: each token gets three ids (time, height, width); text gets the same number three times. ') +
      `Selected: ${what}, 1D id ${t.one}, M-RoPE ids (${t.m.join(', ')}). Highest id in this sequence: ${live.max1} under 1D RoPE, ${live.maxM} under M-RoPE.`;
  }
  function setMode(m) { if (m === S.mode && S.u === 1) return; stop(); S.from = S.mode === m ? S.from : S.mode; S.mode = m; S.u = 0; press(); readout();
    stop = PE.animate(700, v => { S.u = v; cv.redraw(); }, () => { S.from = S.mode; S.u = 1; cv.redraw(); }); }
  function setVideo(on) { if (on === S.video) return; stop(); S.from = S.mode; S.u = 1; const v0 = S.vu; S.video = on; if (!on && S.sel >= SEQ.false.toks.length) S.sel = 15; press(); readout();
    stop = PE.animate(700, v => { S.vu = PE.lerp(v0, on ? 1 : 0, v); cv.redraw(); }); }
  const btn = id => document.getElementById('btn-mrope-' + id);
  function press() { btn('1d').setAttribute('aria-pressed', String(S.mode === '1d')); btn('m').setAttribute('aria-pressed', String(S.mode === 'm'));
    btn('video').setAttribute('aria-pressed', String(S.video)); btn('novideo').setAttribute('aria-pressed', String(!S.video)); }
  btn('1d').addEventListener('click', () => setMode('1d'));
  btn('m').addEventListener('click', () => setMode('m'));
  btn('video').addEventListener('click', () => setVideo(true));
  btn('novideo').addEventListener('click', () => setVideo(false));
  btn('reset').addEventListener('click', () => { stop(); S.mode = S.from = '1d'; S.u = 1; S.video = false; S.vu = 0; S.sel = 15; press(); cv.redraw(); readout(); });

  // Hover or click a token to inspect it; on the focused canvas, Left/Right step through the sequence.
  const hit = e => { const b = el.getBoundingClientRect(), x = e.clientX - b.left, y = e.clientY - b.top, L = place(b.width), n = SEQ[S.video].toks.length;
    for (let i = 0; i < n; i++) { const c = L.cells[i]; if (x >= c.x && x <= c.x + c.w && y >= c.y && y <= c.y + c.h) return i; } return -1; };
  const pick = e => { const i = hit(e); if (i >= 0 && i !== S.sel) { S.sel = i; cv.redraw(); readout(); } };
  el.addEventListener('pointermove', pick); el.addEventListener('pointerdown', pick);
  el.addEventListener('keydown', e => { const d = { ArrowLeft: -1, ArrowRight: 1 }[e.key]; if (!d) return; e.preventDefault();
    const n = SEQ[S.video].toks.length; S.sel = (S.sel + d + n) % n; cv.redraw(); readout(); });
  press(); readout();
})();
</script>
