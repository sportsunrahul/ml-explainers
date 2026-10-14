<script>
(() => {
  const fig = document.getElementById('viz-clipflow'), el = document.getElementById('clipflow-canvas');
  const out = document.getElementById('clipflow-readout');
  const ORDER = ['batch', 'tokens', 'encode', 'pool', 'embed', 'sim', 'loss', 'backward', 'update'];
  const chips = ORDER.map(k => document.getElementById('btn-clipflow-' + k));
  const texts = [...fig.querySelectorAll('.cf-text > div')];
  // Rows of CLIP's "CLIP-ViT hyperparameters" appendix table: patch, input resolution, embedding dim, vision L/W/H, text L/W/H.
  const MODELS = {
    b32: { name: 'ViT-B/32', P: 32, r: 224, de: 512, vl: 12, vw: 768, vh: 12, tl: 12, tw: 512, th: 8 },
    b16: { name: 'ViT-B/16', P: 16, r: 224, de: 512, vl: 12, vw: 768, vh: 12, tl: 12, tw: 512, th: 8 },
    l14: { name: 'ViT-L/14', P: 14, r: 224, de: 768, vl: 24, vw: 1024, vh: 16, tl: 12, tw: 768, th: 12 },
    l336: { name: 'ViT-L/14@336px', P: 14, r: 336, de: 768, vl: 24, vw: 1024, vh: 16, tl: 12, tw: 768, th: 12 },
  };
  const mBtns = Object.keys(MODELS).map(k => document.getElementById('btn-clipflow-' + k));
  const CTX = 76;
  const S = { scene: 'overview', model: 'l14', u: 1, prev: -1 };
  let stop = () => {};
  const x = '×';

  // Box ids: lane 'i' (image) or 't' (text), index 0..4; then 'sim' and 'loss'.
  const STEP = { i0: 0, t0: 0, i1: 1, t1: 1, i2: 2, t2: 2, i3: 3, t3: 3, i4: 4, t4: 4, sim: 5, loss: 6 };
  const TITLES = {
    i0: ['photos'], i1: ['patches', '+ [class]'], i2: ['ViT'], i3: ['[class] out', '+ LN'], i4: ['project', '+ L2 norm'],
    t0: ['captions'], t1: ['BPE tokens', '+ embed'], t2: ['text', 'transformer'], t3: ['[EOS] out', '+ LN'], t4: ['project', '+ L2 norm'],
    sim: ['logits'], loss: ['loss'],
  };
  // What each box learns: shown once the backward pass reaches it. Data boxes have nothing to learn.
  const PARAMS = { i0: null, t0: null, i1: 'E, class, pos, LN', t1: 'token embeddings', i2: 'all layers', t2: 'all layers', i3: 'LN', t3: 'LN', i4: 'W_i', t4: 'W_t', sim: 't', loss: null };
  function shapes(m) {
    const g = m.r / m.P, n = g * g + 1;
    return { i0: `N${x}3${x}${m.r}${x}${m.r}`, i1: `N${x}${n}${x}${m.vw}`, i2: `N${x}${n}${x}${m.vw}`, i3: `N${x}${m.vw}`, i4: `N${x}${m.de}`,
      t0: 'N strings', t1: `N${x}${CTX}${x}${m.tw}`, t2: `N${x}${CTX}${x}${m.tw}`, t3: `N${x}${m.tw}`, t4: `N${x}${m.de}`, sim: `N${x}N`, loss: 'one number' };
  }
  const EDGES = [['i0', 'i1'], ['i1', 'i2'], ['i2', 'i3'], ['i3', 'i4'], ['i4', 'sim'], ['t0', 't1'], ['t1', 't2'], ['t2', 't3'], ['t3', 't4'], ['t4', 'sim'], ['sim', 'loss']];

  function layout(W, H) {
    const R = {};
    if (W >= 600) {
      const cw = (W - 16) / 7, bw = Math.min(cw - 16, 96), bh = 52;
      const yI = 30, yT = H - bh - 30, yM = (yI + yT) / 2;
      for (let k = 0; k < 5; k++) { const cx = 8 + cw * (k + .5); R['i' + k] = { x: cx - bw / 2, y: yI, w: bw, h: bh }; R['t' + k] = { x: cx - bw / 2, y: yT, w: bw, h: bh }; }
      R.sim = { x: 8 + cw * 5.5 - bw / 2, y: yM, w: bw, h: bh }; R.loss = { x: 8 + cw * 6.5 - bw / 2, y: yM, w: bw, h: bh };
      return { R, horiz: true };
    }
    const bw = Math.min(W * .4, 150), bh = 50, rh = (H - 30) / 6.9;
    for (let k = 0; k < 5; k++) { const y = 32 + rh * k; R['i' + k] = { x: W * .27 - bw / 2, y, w: bw, h: bh }; R['t' + k] = { x: W * .73 - bw / 2, y, w: bw, h: bh }; }
    R.sim = { x: W / 2 - bw / 2, y: 32 + rh * 5, w: bw, h: bh }; R.loss = { x: W / 2 - bw / 2, y: 32 + rh * 5.95, w: bw, h: bh };
    return { R, horiz: false };
  }
  let C = {};
  const hex = (c, d) => (c && c.length === 7 ? c : d);
  const tint = (c, a) => PE.mix(hex(C.BG, '#16181d'), c, a);
  function rr(ctx, x0, y0, w, h, r, fill, stroke, lw = 1.5, dash) {
    ctx.beginPath(); ctx.moveTo(x0 + r, y0); ctx.arcTo(x0 + w, y0, x0 + w, y0 + h, r); ctx.arcTo(x0 + w, y0 + h, x0, y0 + h, r);
    ctx.arcTo(x0, y0 + h, x0, y0, r); ctx.arcTo(x0, y0, x0 + w, y0, r); ctx.closePath();
    if (fill) { ctx.fillStyle = fill; ctx.fill(); }
    if (stroke) { ctx.save(); ctx.strokeStyle = stroke; ctx.lineWidth = lw; if (dash) ctx.setLineDash(dash); ctx.stroke(); ctx.restore(); }
  }
  const laneColor = id => id[0] === 'i' ? C.B : id[0] === 't' ? C.Y : id === 'loss' ? C.Rd : C.G;
  // A small drawing inside each box so the box reads as what it holds.
  function glyph(ctx, id, b, col) {
    const cx = b.x + b.w / 2, cy = b.y + b.h / 2;
    ctx.save(); ctx.fillStyle = col; ctx.strokeStyle = col; ctx.lineWidth = 1.3;
    if (id === 'i0') for (let k = 2; k >= 0; k--) { const s = 26; rr(ctx, cx - s / 2 + k * 5 - 5, cy - s / 2 - k * 4 + 4, s, s, 3, tint(C.B, .25 + .15 * (2 - k)), C.B, 1.2); }
    else if (id === 'i1') { const s = 7; for (let r = 0; r < 4; r++) for (let c = 0; c < 4; c++) rr(ctx, cx - 12 + c * (s + 1), cy - 15 + r * (s + 1), s, s, 1, tint(C.B, .55), null);
      rr(ctx, cx - 26, cy - 15, s, s, 1, C.F, null); }
    else if (id === 'i2' || id === 't2') { for (let k = 0; k < 4; k++) rr(ctx, cx - 22, cy - 16 + k * 8, 44, 6, 2, tint(col, .3 + .15 * k), null); }
    else if (id === 'i3' || id === 't3' || id === 'i4' || id === 't4') {
      const unit = id[1] === '4'; for (let k = 0; k < 8; k++) { const h = unit ? 9 : 5 + 10 * Math.abs(Math.sin(k * 1.7 + (id[0] === 'i' ? 0 : 1))); ctx.fillRect(cx - 20 + k * 5.2, cy + 8 - h, 3.6, h); }
      if (unit) { ctx.beginPath(); ctx.arc(cx + 30, cy + 2, 7, 0, 7); ctx.stroke(); } }
    else if (id === 't0') { for (let k = 0; k < 3; k++) rr(ctx, cx - 22, cy - 12 + k * 9, 44 - k * 8, 5, 2, tint(C.Y, .6), null); }
    else if (id === 't1') { for (let k = 0; k < 5; k++) rr(ctx, cx - 24 + k * 10, cy - 8, 8, 12, 2, k === 0 || k === 4 ? C.F : tint(C.Y, .55), null); }
    else if (id === 'sim') { const s = 7; for (let r = 0; r < 4; r++) for (let c = 0; c < 4; c++) rr(ctx, cx - 15 + c * (s + 1), cy - 17 + r * (s + 1), s, s, 1, r === c ? C.G : tint(C.Br, .5), null); }
    else if (id === 'loss') { rr(ctx, cx - 22, cy - 4, 44, 8, 3, tint(C.Rd, .25), null); rr(ctx, cx - 22, cy - 4, 30, 8, 3, C.Rd, null); }
    ctx.restore();
  }
  function edgePts(L, a, b) {
    const A = L.R[a], B = L.R[b];
    if (L.horiz) return [A.x + A.w, A.y + A.h / 2, B.x, B.y + B.h / 2];
    return [A.x + A.w / 2, A.y + A.h + 17, B.x + B.w / 2, B.y - 8 - 12 * TITLES[b].length];
  }

  const cv = PE.canvas(el, w => w >= 600 ? .4 : 2.7, (ctx, W, H) => {
    C = { B: PE.css('--blue'), Y: PE.css('--yellow'), G: PE.css('--green'), Rd: PE.css('--red'), Br: PE.css('--brown'), F: PE.css('--fg'), Mu: PE.css('--muted'), Rule: PE.css('--rule'), BG: PE.css('--bg') };
    ctx.fillStyle = C.BG; ctx.fillRect(0, 0, W, H);
    const L = layout(W, H), m = MODELS[S.model], sh = shapes(m), fs = W < 600 ? 10.5 : 10.5;
    const k = ORDER.indexOf(S.scene), back = S.scene === 'backward' || S.scene === 'update';
    // Forward reveal: boxes up to the current step are shown; the current step fades in with S.u.
    const alphaOf = id => { if (S.scene === 'overview' || back) return 1; const s = STEP[id];
      if (s < k) return .7; if (s === k || (k === 6 && s === 6)) return .25 + .75 * S.u; return .12; };
    // Backward: the gradient starts at the loss and reaches boxes in reverse order as S.u runs 0 -> 1.
    const gradAt = id => { if (S.scene === 'update') return 1; if (S.scene !== 'backward') return 0; const d = (6 - STEP[id]) / 6; return PE.clamp((S.u * 1.15 - d) / .15, 0, 1); };
    for (const [a, b] of EDGES) {
      const [x1, y1, x2, y2] = edgePts(L, a, b), al = Math.min(alphaOf(a), alphaOf(b));
      const col = a[0] === 'i' ? C.B : a[0] === 't' ? C.Y : C.G;
      PE.arrow(ctx, x1 + 2, y1, x2 - 2, y2, col, 1.6, 7, al * .8);
      if (!back && STEP[b] === k && S.u < 1) { const t = S.u; PE.dot(ctx, PE.lerp(x1, x2, t), PE.lerp(y1, y2, t), 4, col); }
      if (back) { const g = Math.min(gradAt(a), gradAt(b) > 0 ? 1 : 0); if (gradAt(b) > 0) {
        const off = L.horiz ? [0, 9] : [9, 0], gg = gradAt(a);
        PE.arrow(ctx, x2 - 2 + off[0], y2 + off[1], PE.lerp(x2, x1, Math.max(gg, .35)) + off[0], PE.lerp(y2, y1, Math.max(gg, .35)) + off[1], C.Rd, 2, 7, .45 + .55 * g); } }
    }
    for (const id of Object.keys(L.R)) {
      const b = L.R[id], al = alphaOf(id), col = laneColor(id), g = gradAt(id);
      ctx.save(); ctx.globalAlpha = al;
      const cur = !back && S.scene !== 'overview' && STEP[id] === k;
      rr(ctx, b.x, b.y, b.w, b.h, 7, tint(col, cur ? .2 : .1), col, cur ? 2.4 : 1.4);
      if (g > 0 && PARAMS[id]) rr(ctx, b.x - 3, b.y - 3, b.w + 6, b.h + 6, 9, null, C.Rd, 2.2 * g);
      glyph(ctx, id, b, col);
      const T = TITLES[id]; T.forEach((t, j) => PE.label(ctx, t, b.x + b.w / 2, b.y - 5 - (T.length - 1 - j) * 12, col, fs, 'center'));
      ctx.restore();
      ctx.save(); ctx.globalAlpha = Math.max(al, .12);
      const below = back && PARAMS[id] ? PARAMS[id] : back && (id === 'i0' || id === 't0') ? 'data, no weights' : sh[id];
      PE.label(ctx, id === 'i2' && !back ? `${m.vl} layers` : id === 't2' && !back ? `${m.tl} layers, masked` : below,
        b.x + b.w / 2, b.y + b.h + 12, back && PARAMS[id] ? C.Rd : back ? C.Mu : C.F, 10, 'center');
      ctx.restore();
    }
  });

  function readout() {
    const m = MODELS[S.model], g = m.r / m.P, sh = shapes(m);
    const lines = {
      overview: `${m.name}: an image of ${m.r} ${x} ${m.r} pixels becomes ${g * g} patch tokens plus one [class] token; a caption becomes at most ${CTX} tokens. Both end as one ${m.de}-number unit vector.`,
      batch: `Step 1: a batch of N image-caption pairs. Images ${sh.i0}; captions are N strings. In CLIP, N = 32,768.`,
      tokens: `Step 2: image cut into ${g} ${x} ${g} = ${g * g} patches of ${m.P} ${x} ${m.P} ${x} 3 = ${m.P * m.P * 3} numbers, each mapped to ${m.vw} numbers, plus the [class] token: ${sh.i1}. Caption: ids ${`N${x}${CTX}`}, embedded to ${sh.t1}.`,
      encode: `Step 3: the vision transformer (${m.vl} layers, width ${m.vw}, ${m.vh} heads) and the text transformer (${m.tl} layers, width ${m.tw}, ${m.th} heads) run side by side. Shapes do not change inside an encoder.`,
      pool: `Step 4: keep one vector per example. Image: the [class] token's output, layer-normalized, ${sh.i3}. Text: the [EOS] token's output, layer-normalized, ${sh.t3}.`,
      embed: `Step 5: W_i is ${m.vw} ${x} ${m.de}, W_t is ${m.tw} ${x} ${m.de}. After the projection both sides are ${`N${x}${m.de}`}, then each row is scaled to length 1.`,
      sim: `Step 6: logits = I_e T_e^T ${x} exp(t): one N ${x} N table. At N = 32,768 that is about 1.07 billion numbers per step.`,
      loss: `Step 7: cross-entropy down every row (which caption?) and every column (which image?), averaged: one number.`,
      backward: `Backward: the gradient starts at the loss and flows right to left into both lanes. Red outlines mark what it updates; nothing in CLIP is frozen.`,
      update: `Update: every red-outlined parameter takes one AdamW step, including the temperature t, then the next batch starts.${S.model === 'l336' ? ' (ViT-L/14@336px is not trained this way from scratch: it is ViT-L/14 continued for one epoch at 336 px.)' : ''}`,
    };
    out.textContent = lines[S.scene];
  }
  function show() {
    texts.forEach(d => { d.hidden = d.dataset.stage !== S.scene; });
    chips.forEach((c, i) => c.setAttribute('aria-pressed', String(ORDER[i] === S.scene)));
    mBtns.forEach(b => b.setAttribute('aria-pressed', String(b.id === 'btn-clipflow-' + S.model)));
    readout();
  }
  function go(scene) {
    stop(); S.scene = scene; S.u = 0; show();
    stop = PE.animate(scene === 'backward' ? 2200 : 800, u => { S.u = u; cv.redraw(); });
  }
  const idx = () => ORDER.indexOf(S.scene);
  ORDER.forEach((k, i) => chips[i].addEventListener('click', () => go(k)));
  Object.keys(MODELS).forEach((k, i) => mBtns[i].addEventListener('click', () => { S.model = k; show(); cv.redraw(); }));
  document.getElementById('btn-clipflow-next').addEventListener('click', () => go(ORDER[Math.min(ORDER.length - 1, idx() + 1)]));
  document.getElementById('btn-clipflow-prev').addEventListener('click', () => { const i = idx(); if (i <= 0) { stop(); S.scene = 'overview'; S.u = 1; show(); cv.redraw(); } else go(ORDER[i - 1]); });
  document.getElementById('btn-clipflow-reset').addEventListener('click', () => { stop(); S.scene = 'overview'; S.model = 'l14'; S.u = 1; show(); cv.redraw(); });
  show();
})();
</script>
