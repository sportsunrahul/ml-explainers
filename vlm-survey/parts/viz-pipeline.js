<script>
(() => {
  const fig = document.getElementById('viz-pipeline'), el = document.getElementById('pipeline-canvas');
  const out = document.getElementById('pipeline-readout');
  const ORDER = ['contrastive', 'sample', 'encode', 'template', 'splice', 'pad', 'pack', 'share', 'targets', 'mask', 'stage1', 'stage2'];
  const chips = ORDER.map(k => document.getElementById('btn-pipeline-' + k));
  const texts = [...fig.querySelectorAll('.pl-text > div')];

  // ---- The toy batch (same definitions as build/sim/pipeline-tokens.cjs). One word = one token. ----
  const NV = 9, REALV = 576;
  const T = (k, s) => s.split(' ').map(w => ({ k, w }));
  const SYS = [{ k: 'sys', w: 'SYS' }], STOP = [{ k: 'stop', w: '###' }], HUM = [{ k: 'role', w: 'Human:' }], ASS = [{ k: 'role', w: 'Assistant:' }], PH = [{ k: 'ph', w: '<image>' }];
  const RAW = {
    A: [...SYS, ...STOP, ...HUM, ...T('q', 'What is the dog doing ?'), ...PH, ...STOP, ...ASS, ...T('a', 'Catching a frisbee .'), ...STOP,
        ...HUM, ...T('q', 'What color is it ?'), ...STOP, ...ASS, ...T('a', 'Red .'), ...STOP],
    B: [...SYS, ...STOP, ...HUM, ...PH, ...T('q', 'Describe the image in detail .'), ...STOP, ...ASS,
        ...T('a', 'A gray cat is curled up asleep on a red sofa . Sunlight from a window falls across its fur , and a blue cushion rests beside it .'), ...STOP],
    C: [...SYS, ...STOP, ...HUM, ...PH, ...T('q', 'What color is the bus ? Answer the question using a single word or phrase .'), ...STOP, ...ASS, ...T('a', 'Yellow'), ...STOP],
  };
  const splice = (seq, nv) => seq.flatMap(t => t.k === 'ph' ? Array.from({ length: nv }, () => ({ k: 'img', w: '' })) : [t]);
  const NAMES = ['A', 'B', 'C'], SEQ = NAMES.map(n => splice(RAW[n], NV)), LEN = SEQ.map(s => s.length);
  const LMAX = Math.max(...LEN), SUM = LEN.reduce((a, b) => a + b, 0), PADS = 3 * LMAX - SUM;
  const isLoss = t => t.k === 'a' || t.k === 'stop';
  const LOSSN = SEQ.flat().filter(isLoss).length;
  const REAL = NAMES.map(n => splice(RAW[n], REALV).length), RLMAX = Math.max(...REAL), RSUM = REAL.reduce((a, b) => a + b, 0);
  const pct = (a, b) => (100 * a / b).toFixed(0) + '%', pct1 = (a, b) => (100 * a / b).toFixed(1) + '%';
  // Molmo toy: three questions about image A.
  const QA = [['What is the dog doing ?', 'Catching a frisbee .'], ['What color is the frisbee ?', 'Red .'], ['Is it indoors ?', 'No .']]
    .map(([q, a]) => [...HUM, ...T('q', q), ...STOP, ...ASS, ...T('a', a), ...STOP]);
  const QAL = QA.map(x => x.length), SHARED = NV + QAL.reduce((a, b) => a + b, 0), NAIVE = QAL.map(l => NV + l);

  const S = { scene: 'overview', u: 1 };
  let stop = () => {};
  const glow = PE.linkPulses(fig, () => cv.redraw());
  const LINKSCENE = { answer: 'mask', visual: 'mask', instruct: 'mask', prev: 'mask', loss: 'mask', trainable: 'stage2', dataset: 'sample', lambda: 'sample', nll: 'sample', sum: 'sample' };
  fig.addEventListener('pe-link', e => { const s = LINKSCENE[e.detail];
    if (s && S.scene !== s && !(s === 'stage2' && S.scene === 'stage1')) go(s); });

  // ---- drawing helpers ----
  let C = {};
  const hex = (c, d) => (c && c.length === 7 ? c : d);
  const tint = (c, a) => PE.mix(hex(C.BG, '#16181d'), c, a);
  function rr(ctx, x, y, w, h, r, fill, stroke, lw = 1.5, dash) {
    if (w <= 0 || h <= 0) return; r = Math.max(0, Math.min(r, w / 2, h / 2)); ctx.beginPath(); ctx.moveTo(x + r, y); ctx.arcTo(x + w, y, x + w, y + h, r); ctx.arcTo(x + w, y + h, x, y + h, r);
    ctx.arcTo(x, y + h, x, y, r); ctx.arcTo(x, y, x + w, y, r); ctx.closePath();
    if (fill) { ctx.fillStyle = fill; ctx.fill(); }
    if (stroke) { ctx.save(); ctx.strokeStyle = stroke; ctx.lineWidth = lw; if (dash) ctx.setLineDash(dash); ctx.stroke(); ctx.restore(); }
  }
  const lab = (ctx, t, x, y, col, fs = 11, al = 'left') => PE.label(ctx, t, x, y, col, fs, al);
  function textW(ctx, t, fs) { ctx.save(); ctx.font = `600 ${fs}px system-ui, sans-serif`; const w = ctx.measureText(t).width; ctx.restore(); return w; }
  // A toy photo, drawn in vision blue only: a sky, a ground line, and a silhouette per subject.
  function thumb(ctx, x, y, s, kind, a = 1) {
    if (s <= 2) return;
    ctx.save(); ctx.globalAlpha = a; rr(ctx, x, y, s, s, 5, tint(C.B, .22), C.B, 1.2);
    ctx.beginPath(); ctx.rect(x + 1, y + 1, s - 2, s - 2); ctx.clip(); ctx.fillStyle = tint(C.B, .45); ctx.fillRect(x, y + s * .7, s, s * .3);
    ctx.fillStyle = C.B;
    if (kind === 'dog') { ctx.beginPath(); ctx.ellipse(x + s * .42, y + s * .62, s * .2, s * .11, 0, 0, 7); ctx.fill(); ctx.beginPath(); ctx.arc(x + s * .64, y + s * .5, s * .09, 0, 7); ctx.fill();
      ctx.fillRect(x + s * .27, y + s * .66, s * .05, s * .14); ctx.fillRect(x + s * .52, y + s * .66, s * .05, s * .14);
      ctx.beginPath(); ctx.ellipse(x + s * .78, y + s * .26, s * .1, s * .035, -.3, 0, 7); ctx.fill(); }
    else if (kind === 'cat') { ctx.beginPath(); ctx.ellipse(x + s * .5, y + s * .64, s * .24, s * .1, 0, 0, 7); ctx.fill(); ctx.beginPath(); ctx.arc(x + s * .3, y + s * .56, s * .08, 0, 7); ctx.fill();
      ctx.beginPath(); ctx.moveTo(x + s * .24, y + s * .5); ctx.lineTo(x + s * .26, y + s * .42); ctx.lineTo(x + s * .31, y + s * .49); ctx.fill(); }
    else if (kind === 'bus') { rr(ctx, x + s * .15, y + s * .38, s * .7, s * .3, 3, C.B); ctx.fillStyle = tint(C.B, .22);
      for (let i = 0; i < 4; i++) ctx.fillRect(x + s * (.2 + i * .16), y + s * .43, s * .1, s * .09); ctx.fillStyle = C.B;
      ctx.beginPath(); ctx.arc(x + s * .3, y + s * .7, s * .06, 0, 7); ctx.arc(x + s * .7, y + s * .7, s * .06, 0, 7); ctx.fill(); }
    else { ctx.fillRect(x + s * .47, y + s * .45, s * .06, s * .27); ctx.beginPath(); ctx.arc(x + s * .5, y + s * .38, s * .17, 0, 7); ctx.fill(); }
    ctx.restore();
  }
  const PHOTO = ['dog', 'cat', 'bus'];
  // Token chips: text tokens are yellow (language), visual tokens blue, padding brown.
  function chipStyle(k) {
    switch (k) {
      case 'sys': return { f: tint(C.Y, .1), s: C.Mu, t: C.Mu };
      case 'stop': return { f: tint(C.Y, .16), s: C.Y, t: C.Y };
      case 'role': return { f: tint(C.Y, .1), s: tint(C.Y, .55), t: C.Y };
      case 'ph': return { f: null, s: C.B, t: C.B, d: [3, 2] };
      case 'img': return { f: C.B, s: null, t: C.BG };
      case 'pad': return { f: tint(C.Br, .45), s: C.Br, t: C.Br };
      default: return { f: tint(C.Y, .3), s: null, t: C.F };
    }
  }
  const cellColor = k => k === 'img' ? C.B : k === 'pad' ? tint(C.Br, .6) : k === 'q' || k === 'a' ? tint(C.Y, .78) : tint(C.Y, .42);
  function chip(ctx, t, x, y, w, h, fs, a = 1, over) {
    const st = over || chipStyle(t.k); ctx.save(); ctx.globalAlpha = a;
    rr(ctx, x, y, w, h, 3, st.f, st.s, 1.1, st.d); if (t.w) lab(ctx, t.w, x + w / 2, y + h - (h - fs) / 2 - 2, st.t, fs, 'center'); ctx.restore();
  }
  const chipW = (ctx, t, fs, h) => t.k === 'img' ? h * .62 : t.k === 'pad' && !t.w ? h * .62 : textW(ctx, t.w, fs) + 8;
  // Wrap a token list into lines; slot widths may be widened by a second row (the targets).
  function flow(ctx, toks, x0, y0, maxW, fs, h, step, widths) {
    const P = []; let x = x0, y = y0;
    toks.forEach((t, i) => { const w = widths ? widths[i] : chipW(ctx, t, fs, h), g = t.k === 'img' ? 1.5 : 3;
      if (x + w > x0 + maxW && x > x0) { x = x0; y += step; } P.push({ x, y, w }); x += w + g; });
    return P;
  }
  function cellRow(ctx, toks, x, y, cw, ch, a = 1, colorOf) {
    ctx.save(); ctx.globalAlpha = a; toks.forEach((t, i) => { ctx.fillStyle = colorOf ? colorOf(t) : cellColor(t.k); ctx.fillRect(x + i * cw, y, Math.max(1, cw - (cw > 4 ? 1 : .4)), ch); }); ctx.restore();
  }
  const ease = (u, a, b) => PE.clamp((u - a) / (b - a), 0, 1);
  // Word-wrapped canvas text; returns the y of the last line.
  function para(ctx, t, x, y, maxW, col, fs, lh = fs * 1.35) {
    let line = ''; t.split(' ').forEach(w => { const tryL = line ? line + ' ' + w : w;
      if (textW(ctx, tryL, fs) > maxW && line) { lab(ctx, line, x, y, col, fs); y += lh; line = w; } else line = tryL; });
    if (line) lab(ctx, line, x, y, col, fs); return y;
  }
  const fit = (ctx, t, fs, maxW, min) => Math.max(min, Math.min(fs, fs * maxW / textW(ctx, t, fs)));
  function heading(ctx, W, t, sub) { lab(ctx, t, 12, 22, C.F, fit(ctx, t, W < 560 ? 12.5 : 14, W - 24, 10));
    if (sub) lab(ctx, sub, 12, 39, C.Mu, fit(ctx, sub, W < 560 ? 10 : 11, W - 24, 8.5)); }

  // ---- scenes ----
  const DRAW = {};
  // Reset view: the whole generative step as a map of its 7 stages.
  DRAW.overview = (ctx, W, H, u) => {
    const nar = W < 560;
    heading(ctx, W, 'One generative training step: 7 stages', nar ? 'press a stage above to zoom in' : 'press a stage above to zoom in; A is the contrastive step for comparison');
    const st = ['sample', 'image to tokens', 'text to tokens', 'splice', 'batch', 'loss mask', 'backward'];
    const boxes = [];
    if (nar) { const top = 54, bh = (H - top - 8) / 7; st.forEach((n, i) => boxes.push({ x: 10, y: top + i * bh, w: W - 20, h: bh - 8 })); }
    else { const top = 58, bw = (W - 24 - 3 * 14) / 4, bh = (H - top - 16) / 2 - 12;
      st.forEach((n, i) => { const r = i < 4 ? 0 : 1, c = i < 4 ? i : i - 4; boxes.push({ x: 12 + c * (bw + 14) + (r ? (bw + 14) / 2 : 0), y: top + r * (bh + 24), w: bw, h: bh }); }); }
    boxes.forEach((b, i) => {
      const a = ease(u, i * .1, i * .1 + .35); if (a <= 0) return;
      ctx.save(); ctx.globalAlpha = a; rr(ctx, b.x, b.y, b.w, b.h, 8, C.P, C.Rule, 1.2);
      PE.dot(ctx, b.x + 14, b.y + 14, 10, C.F); lab(ctx, String(i + 1), b.x + 14, b.y + 18, C.BG, 11, 'center');
      lab(ctx, st[i], b.x + 30, b.y + 18, C.F, nar ? 11.5 : 12);
      // Station drawings: the icon area is to the right (narrow) or below (wide).
      const ix = nar ? b.x + 150 : b.x + 10, iy = nar ? b.y + 4 : b.y + 30, iw = nar ? b.w - 160 : b.w - 20, ih = nar ? b.h - 8 : b.h - 40;
      icon(ctx, i, ix, iy, iw, ih); ctx.restore();
      if (i < 6) { const n = boxes[i + 1]; ctx.save(); ctx.globalAlpha = a;
        if (nar) PE.arrow(ctx, b.x + 14, b.y + 26, b.x + 14, n.y + 4, C.Mu, 1.3, 6);
        else if (i === 3) PE.arrow(ctx, b.x + b.w / 2, b.y + b.h + 2, n.x + n.w - 20, n.y - 2, C.Mu, 1.3, 7);
        else PE.arrow(ctx, b.x + b.w + 2, b.y + b.h / 2, n.x - 2, n.y + n.h / 2, C.Mu, 1.3, 7);
        ctx.restore(); }
    });
    function icon(ctx, i, x, y, w, h) {
      const cy = y + h / 2;
      if (i === 0) { [158, 83, 72, 40].forEach((v, k) => { const bh = h * .8 * v / 158; ctx.fillStyle = k ? tint(C.F, .35) : tint(C.F, .6); ctx.fillRect(x + k * 14, y + h - bh, 10, bh); });
        lab(ctx, 'datasets', x + 62, cy + 4, C.Mu, 10); }
      else if (i === 1) { const s = Math.min(h, 34); thumb(ctx, x, cy - s / 2, s, 'dog'); PE.arrow(ctx, x + s + 4, cy, x + s + 18, cy, C.Mu, 1, 5);
        for (let k = 0; k < 6; k++) { ctx.fillStyle = C.B; ctx.fillRect(x + s + 22 + k * 9, cy - 5, 7, 10); } }
      else if (i === 2) { const fs = 9.5, hh = 14; let xx = x; [{ k: 'role', w: 'Human:' }, { k: 'q', w: 'What' }, { k: 'q', w: 'is' }, { k: 'stop', w: '###' }].forEach(t => { const cw = chipW(ctx, t, fs, hh); if (xx + cw < x + w) chip(ctx, t, xx, cy - hh / 2, cw, hh, fs); xx += cw + 3; }); }
      else if (i === 3) { const toks = [...RAW.A.slice(0, 4), ...Array(5).fill({ k: 'img', w: '' }), ...RAW.A.slice(9, 12)]; cellRow(ctx, toks, x, cy - 6, Math.min(8, w / toks.length), 12); }
      else if (i === 4) { SEQ.forEach((s, r) => cellRow(ctx, [...s, ...Array(LMAX - s.length).fill({ k: 'pad' })], x, y + h * .15 + r * h * .26, Math.min(3, w / LMAX), h * .2)); }
      else if (i === 5) { const cw = Math.min(8, w / 14); SEQ[0].slice(10, 24).forEach((t, k) => { ctx.fillStyle = isLoss(t) ? C.R : tint(C.Br, .5); ctx.fillRect(x + k * cw, cy - 5, cw - 1, 10); }); }
      else { rr(ctx, x, cy - 9, 26, 18, 3, tint(C.Br, .35), C.Br, 1); rr(ctx, x + 34, cy - 9, 22, 18, 3, tint(C.G, .3), C.G, 1); rr(ctx, x + 64, cy - 9, 34, 18, 3, tint(C.Y, .3), C.Y, 1);
        PE.arrow(ctx, x + 98, cy - 13, x + 40, cy - 13, C.R, 1.3, 5); }
    }
  };

  // Mode A: CLIP's contrastive step. Two encoders in parallel, one N x N table, every other caption a negative.
  DRAW.contrastive = (ctx, W, H, u) => {
    const nar = W < 560, N = 4, capt = ['a dog', 'a cat', 'a bus', 'a tree'], ph = ['dog', 'cat', 'bus', 'tree'];
    heading(ctx, W, 'Mode A: a contrastive step (CLIP)', 'N pairs in, two encoders side by side, one N x N table out');
    const s = nar ? 30 : 36, x0 = 12, gap = 6, rowW = N * (s + gap), ey = [56, nar ? 118 : 138];
    const ex = x0 + rowW + 14, ew = nar ? 78 : 104, vx = ex + ew + 16;
    const a1 = ease(u, 0, .35), a2 = ease(u, .3, .6), a3 = ease(u, .55, .9), a4 = ease(u, .85, 1);
    ph.forEach((p, i) => thumb(ctx, x0 + i * (s + gap), ey[0], s, p));
    capt.forEach((c, i) => chip(ctx, { k: 'q', w: c }, x0 + i * (s + gap) - 2, ey[1] + 6, s + 4, 20, nar ? 9 : 10));
    rr(ctx, ex, ey[0] - 4, ew, s + 8, 6, tint(C.B, .18), C.B, 1.5); lab(ctx, 'image encoder', ex + ew / 2, ey[0] + s / 2 + 4, C.B, nar ? 10 : 11, 'center');
    rr(ctx, ex, ey[1] - 2, ew, 32, 6, tint(C.Y, .18), C.Y, 1.5); lab(ctx, 'text encoder', ex + ew / 2, ey[1] + 18, C.Y, nar ? 10 : 11, 'center');
    for (let r = 0; r < 2; r++) { PE.arrow(ctx, x0 + rowW - 2, ey[r] + 14, ex - 3, ey[r] + 14, C.Mu, 1.2, 6);
      for (let i = 0; i < N; i++) { const t = PE.clamp(a1 * 1.3 - i * .1, 0, 1); if (t > 0 && t < 1) PE.dot(ctx, PE.lerp(x0 + i * (s + gap) + s / 2, ex + ew / 2, t), ey[r] + 14, 3, r ? C.Y : C.B); } }
    // Embedding vectors: one short bar per example, blue for images, yellow for captions.
    for (let r = 0; r < 2; r++) for (let i = 0; i < N; i++) { ctx.save(); ctx.globalAlpha = a2;
      ctx.fillStyle = r ? C.Y : C.B; ctx.fillRect(vx, ey[r] - 2 + i * (r ? 8 : (s + 8) / 4), (nar ? 26 : 34) * a2, 5); ctx.restore(); }
    if (a2 > 0) { lab(ctx, 'N vectors each', vx - 4, ey[1] + 48, C.Mu, 10); PE.arrow(ctx, ex + ew + 2, ey[0] + 14, vx - 2, ey[0] + 14, C.Mu, 1, 5); PE.arrow(ctx, ex + ew + 2, ey[1] + 14, vx - 2, ey[1] + 14, C.Mu, 1, 5); }
    // The N x N table: green diagonal (the true pairs), brown negatives, one red loss.
    const tx = nar ? 64 : vx + 100, ty = nar ? ey[1] + 74 : 66, cell = nar ? Math.min(40, (W - 90) / N) : Math.min(46, (W - tx - 20) / N);
    for (let i = 0; i < N; i++) { lab(ctx, ph[i], tx - 6, ty + i * cell + cell * .6, C.B, 10, 'right'); lab(ctx, ph[i], tx + i * cell + cell / 2, ty - 6, C.Y, 10, 'center'); }
    for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) { const a = PE.clamp(a3 * 1.6 - i * .15, 0, 1); ctx.save(); ctx.globalAlpha = a;
      ctx.fillStyle = i === j ? tint(C.G, .75) : tint(C.Br, .38); ctx.fillRect(tx + j * cell + 1, ty + i * cell + 1, cell - 2, cell - 2); ctx.restore(); }
    const ly = ty + N * cell + 22; lab(ctx, 'loss: in each row and column, pick the green cell', tx - (nar ? 52 : 0), ly - 6, C.R, 10.5);
    ctx.fillStyle = C.Rule; ctx.fillRect(tx - (nar ? 52 : 0), ly, cell * N + (nar ? 52 : 0), 9); ctx.fillStyle = C.R; ctx.fillRect(tx - (nar ? 52 : 0), ly, (cell * N + (nar ? 52 : 0)) * .7 * a4, 9);
    para(ctx, 'Every other caption in the batch is a negative, so batch size is part of the loss.', nar ? 12 : tx - 52, ly + 34, nar ? W - 24 : W - tx + 40, C.Mu, nar ? 10 : 11);
  };

  // Stage 1: where the examples come from.
  const MIX = [['LLaVA-Instruct', 158], ['VG', 86], ['VQAv2', 83], ['OCRVQA', 80], ['GQA', 72], ['A-OKVQA', 66], ['RefCOCO', 48], ['ShareGPT', 40], ['TextCaps', 22], ['OKVQA', 9]];
  DRAW.sample = (ctx, W, H, u) => {
    const nar = W < 560;
    heading(ctx, W, '1. Sample: draw examples from a mixture of datasets', nar ? 'LLaVA-1.5 mix: 10 splits, 665K examples' : 'LLaVA-1.5 instruction mix: 10 splits, 665K examples, each example equally likely');
    const gd = glow('dataset'), gl = glow('lambda'), gs = glow('sum'), gn = glow('nll');
    const x0 = 12, nameW = nar ? 76 : 96, bw = nar ? W * .3 : W * .26, rh = nar ? 12 : 14, y0 = 54;
    MIX.forEach(([n, v], i) => { const y = y0 + i * rh;
      lab(ctx, n, x0 + nameW - 6, y + rh - 3, i === 0 || i === 2 ? C.F : C.Mu, nar ? 9 : 10, 'right');
      ctx.fillStyle = tint(C.F, i === 0 || i === 2 ? .6 : .3); ctx.fillRect(x0 + nameW, y + 2, bw * v / 158 * ease(u, 0, .3), rh - 4);
      if (gd > 0) { ctx.strokeStyle = C.F; ctx.lineWidth = 2 * gd; ctx.strokeRect(x0 + nameW - 2, y + 1, bw + 4, rh - 2); }
      lab(ctx, v + 'K', x0 + nameW + bw * v / 158 + 4, y + rh - 3, C.Mu, 9); });
    // The three toy examples leave their datasets and land in this step's batch.
    const tx = nar ? 12 : W * .52, tw = nar ? W - 24 : W * .46, ch = nar ? 26 : 36, cy0 = nar ? y0 + MIX.length * rh + 20 : y0 + 14, cgap = nar ? 4 : 6;
    lab(ctx, 'this batch (toy: 3 examples)', tx, cy0 - 10, C.F, nar ? 10 : 11);
    const cards = [['A', 'dog', 'conversation', 0], ['B', 'cat', 'detailed description', 0], ['C', 'bus', 'short-answer VQA', 2]];
    cards.forEach(([n, p, kind, src], i) => {
      const t = ease(u, .25 + i * .15, .55 + i * .15), sx = x0 + nameW + 10, sy = y0 + src * rh, dx = tx, dy = cy0 + i * (ch + cgap);
      const x = PE.lerp(sx, dx, t), y = PE.lerp(sy, dy, t); if (t <= 0) return;
      ctx.save(); ctx.globalAlpha = Math.min(1, t * 2); rr(ctx, x, y, nar ? tw : tw, ch, 5, C.P, C.Rule, 1);
      thumb(ctx, x + 3, y + 3, ch - 6, p); lab(ctx, n + ': ' + kind, x + ch + 2, y + (nar ? 16 : 15), C.F, nar ? 9.5 : 10.5);
      lab(ctx, i < 2 ? 'from LLaVA-Instruct' : 'from VQAv2', nar ? x + tw - 6 : x + ch + 2, y + (nar ? 16 : 29), C.Mu, nar ? 9 : 10, nar ? 'right' : 'left'); ctx.restore(); });
    // Flamingo mixes at the loss instead: one batch from each dataset, losses weighted by lambda, one update.
    const fy = nar ? cy0 + 3 * (ch + cgap) + 22 : y0 + MIX.length * rh + 28, a = ease(u, .55, 1);
    ctx.save(); ctx.globalAlpha = a;
    ctx.strokeStyle = C.Rule; ctx.beginPath(); ctx.moveTo(12, fy - 12); ctx.lineTo(W - 12, fy - 12); ctx.stroke();
    lab(ctx, nar ? 'Flamingo mixes at the loss instead' : 'Flamingo mixes at the loss: one batch from each dataset per update', 12, fy + 4, C.F, nar ? 10.5 : 12);
    const F4 = [['M3W', 1, '1.0'], ['ALIGN', .2, '0.2'], ['LTIP', .2, '0.2'], ['VTP', .03, '0.03']], fr = nar ? 19 : 22, lx = nar ? 56 : 70, lw = nar ? W * .38 : W * .3;
    F4.forEach(([n, l, ls], i) => { const y = fy + 14 + i * fr;
      lab(ctx, n, lx - 8, y + fr - 8, C.F, nar ? 9.5 : 10.5, 'right');
      for (let k = 0; k < 3; k++) { ctx.fillStyle = tint(C.F, .4); ctx.fillRect(lx + k * 7, y + 3, 5, fr - 8); }
      PE.arrow(ctx, lx + 24, y + fr / 2 - 1, lx + 38, y + fr / 2 - 1, C.Mu, 1, 5);
      ctx.fillStyle = C.R; ctx.globalAlpha = a * (gn > 0 ? 1 : .55); ctx.fillRect(lx + 42, y + 3, 16, fr - 8); ctx.globalAlpha = a;
      lab(ctx, '\u00d7 ' + ls, lx + 64, y + fr - 8, gl > 0 ? C.F : C.R, (nar ? 9.5 : 10.5) + 2 * gl);
      ctx.fillStyle = C.R; ctx.fillRect(lx + 100, y + 3, lw * .5 * l, fr - 8);
      PE.arrow(ctx, lx + 104 + lw * .5 * l, y + fr / 2 - 1, lx + 118 + lw * .5, fy + 14 + 2 * fr, tint(C.R, .6), 1, 5); });
    const sx = lx + 124 + lw * .5, sy = fy + 14 + 2 * fr;
    PE.dot(ctx, sx + 8, sy, 11 + 3 * gs, C.R); lab(ctx, '\u03a3', sx + 8, sy + 5, C.BG, 13, 'center');
    PE.arrow(ctx, sx + 22, sy, sx + 40, sy, C.R, 1.5, 6); lab(ctx, nar ? 'update' : 'one update', sx + 44, sy + 4, C.F, nar ? 10 : 11);
    const ry = fy + 14 + 4 * fr + 16;
    lab(ctx, 'Flamingo-3B overall score: accumulate 70.7, round-robin 62.9', 12, ry, C.Mu, nar ? 9.5 : 10.5);
    ctx.restore();
  };

  // Stage 2: image -> resize -> patch grid -> frozen encoder -> connector -> visual tokens.
  DRAW.encode = (ctx, W, H, u) => {
    const nar = W < 560;
    heading(ctx, W, '2. Image to tokens (LLaVA-1.5 sizes)', 'resize, cut into 14 px patches, encode, project');
    const a1 = ease(u, 0, .25), a2 = ease(u, .15, .5), a3 = ease(u, .45, .75), a4 = ease(u, .7, 1);
    const s = nar ? Math.min(118, W * .34) : Math.min(150, H * .36), y1 = 58, gx = nar ? 12 + 58 : 120;
    thumb(ctx, 12, y1 + s / 2 - 24, 48, 'dog');
    PE.arrow(ctx, 64, y1 + s / 2, gx - 4, y1 + s / 2, C.Mu, 1.2, 6);
    ctx.save(); ctx.globalAlpha = a1; thumb(ctx, gx, y1, s, 'dog');
    const n = Math.round(24 * a2); ctx.strokeStyle = C.BG; ctx.lineWidth = .6; ctx.beginPath();
    for (let i = 1; i < n; i++) { const d = s * i / 24; ctx.moveTo(gx + d, y1); ctx.lineTo(gx + d, y1 + s); ctx.moveTo(gx, y1 + d); ctx.lineTo(gx + s, y1 + d); }
    ctx.stroke(); ctx.restore();
    lab(ctx, '336 x 336 px', gx, y1 + s + 14, C.B, 10.5); lab(ctx, '24 x 24 = 576 patches', gx, y1 + s + 28, C.B, 10.5);
    // Frozen encoder (brown), then the trainable connector (green).
    const row2 = nar ? y1 + s + 52 : y1, ex = nar ? 12 : gx + s + 28, ew = nar ? W * .42 : W * .2, eh = nar ? 52 : 64;
    const eY = nar ? row2 : y1 + s / 2 - eh / 2;
    ctx.save(); ctx.globalAlpha = a2; rr(ctx, ex, eY, ew, eh, 7, tint(C.Br, .25), C.Br, 1.8);
    lab(ctx, 'vision encoder', ex + ew / 2, eY + 22, C.B, nar ? 10.5 : 11.5, 'center'); lab(ctx, 'CLIP ViT-L/14, 336 px', ex + ew / 2, eY + 36, C.Mu, nar ? 9 : 10, 'center');
    lab(ctx, 'frozen', ex + ew / 2, eY + eh - 6, C.Br, 9.5, 'center'); ctx.restore();
    if (!nar) PE.arrow(ctx, gx + s + 4, y1 + s / 2, ex - 3, y1 + s / 2, C.Mu, 1.2, 6, a2);
    else PE.arrow(ctx, gx + s / 2, y1 + s + 34, ex + ew / 2, eY - 3, C.Mu, 1.2, 6, a2);
    const cx = nar ? ex + ew + 22 : ex + ew + 28, cw = nar ? W - cx - 12 : W * .17, cY = eY;
    ctx.save(); ctx.globalAlpha = a3; rr(ctx, cx, cY, cw, eh, 7, tint(C.G, .2), C.G, 1.8);
    lab(ctx, 'connector', cx + cw / 2, cY + 22, C.G, nar ? 10.5 : 11.5, 'center'); lab(ctx, '2-layer MLP', cx + cw / 2, cY + 36, C.Mu, nar ? 9 : 10, 'center');
    lab(ctx, 'trainable', cx + cw / 2, cY + eh - 6, C.G, 9.5, 'center'); ctx.restore();
    PE.arrow(ctx, ex + ew + 3, eY + eh / 2, cx - 3, cY + eh / 2, C.Mu, 1.2, 6, a3);
    // Patch features in flight, then the 576-token strip and its 9-token toy stand-in.
    if (u < 1) for (let k = 0; k < 12; k++) { const t = PE.clamp(ease(u, .35, .8) * 1.3 - k * .025, 0, 1); if (t <= 0 || t >= 1) continue;
      PE.dot(ctx, PE.lerp(ex + 10, cx + cw - 10, t), eY + eh / 2 + (k % 3 - 1) * 8, 2.6, C.B, .9); }
    const sy = nar ? eY + eh + 30 : y1 + s + 64, sw = W - 24;
    ctx.save(); ctx.globalAlpha = a4;
    lab(ctx, '576 visual tokens, each as wide as a word embedding of the LLM', 12, sy, C.B, nar ? 10 : 11);
    const cwid = sw / 576; for (let i = 0; i < 576 * a4; i++) { ctx.fillStyle = i % 24 === 0 ? tint(C.B, .6) : C.B; ctx.fillRect(12 + i * cwid, sy + 8, Math.max(.6, cwid - .15), 14); }
    lab(ctx, 'the rest of this figure draws each image as 9 tokens so the rows fit', 12, sy + 42, C.Mu, nar ? 9.5 : 10.5);
    for (let i = 0; i < NV; i++) { ctx.fillStyle = C.B; ctx.fillRect(12 + i * 13, sy + 50, 11, 14); }
    ctx.restore();
    PE.arrow(ctx, cx + cw / 2, cY + eh + 3, cx + cw / 2, sy - 12, C.Mu, 1.2, 6, a4);
  };

  // Stage 3: the conversation, poured into LLaVA's template (Table 2), becomes a row of text tokens.
  DRAW.template = (ctx, W, H, u) => {
    const nar = W < 560, fs = nar ? 10 : 11, h = fs + 9;
    heading(ctx, W, '3. Text to tokens: the conversation template', 'LLaVA Table 2 format; SYS = system message, ### = <STOP>');
    const y0 = 56, rec = [['image', 'dog.jpg (becomes <image>)'], ['Q1', 'What is the dog doing?'], ['A1', 'Catching a frisbee.'], ['Q2', 'What color is it?'], ['A2', 'Red.']];
    rr(ctx, 12, y0, nar ? W - 24 : W * .4, rec.length * 16 + 14, 6, C.P, C.Rule, 1);
    rec.forEach(([k, v], i) => { lab(ctx, k + ':', 22, y0 + 18 + i * 16, C.Mu, 10.5); lab(ctx, v, 62, y0 + 18 + i * 16, k === 'image' ? C.B : C.F, 10.5); });
    thumb(ctx, nar ? W - 60 : W * .4 + 26, y0 + 4, 44, 'dog');
    const ty = y0 + rec.length * 16 + 36, P = flow(ctx, RAW.A, 12, ty, W - 24, fs, h, h + 7);
    lab(ctx, 'example A as the model reads it:', 12, ty - 9, C.F, 10.5);
    RAW.A.forEach((t, i) => { const a = ease(u, .1 + .8 * i / RAW.A.length, .15 + .8 * i / RAW.A.length); if (a > 0) chip(ctx, t, P[i].x, P[i].y + (1 - a) * -8, P[i].w, h, fs, a); });
    const by = P[P.length - 1].y + h + 22;
    ctx.save(); ctx.globalAlpha = ease(u, .85, 1);
    const y2 = para(ctx, 'Human text and the assistant\'s answers are both plain tokens: nothing yet marks which ones will be graded.', 12, by, W - 24, C.Mu, nar ? 9.5 : 10.5);
    para(ctx, 'Here the <image> comes after the first question; LLaVA puts it before or after at random.', 12, y2 + 16, W - 24, C.Mu, nar ? 9.5 : 10.5);
    ctx.restore();
  };

  // Stage 4: the <image> placeholder is replaced by the visual tokens.
  DRAW.splice = (ctx, W, H, u) => {
    const nar = W < 560, fs = nar ? 10 : 11, h = fs + 9;
    heading(ctx, W, '4. Splice: <image> becomes the visual tokens', 'one sequence per example, images and words side by side');
    const ty = 58, P0 = flow(ctx, RAW.A, 12, ty, W - 24, fs, h, h + 7), P1 = flow(ctx, SEQ[0], 12, ty, W - 24, fs, h, h + 7), e = ease(u, .1, .7);
    let j = 0;
    RAW.A.forEach((t, i) => {
      if (t.k === 'ph') { for (let k = 0; k < NV; k++) { const p1 = P1[j + k], x = PE.lerp(P0[i].x + k * P0[i].w / NV, p1.x, e), y = PE.lerp(P0[i].y, p1.y, e), w = PE.lerp(P0[i].w / NV - 1, p1.w, e);
          chip(ctx, { k: 'img', w: '' }, x, y, w, h, fs); }
        if (e < 1) chip(ctx, t, P0[i].x, P0[i].y, P0[i].w, h, fs, 1 - e); j += NV; }
      else { const p1 = P1[j]; chip(ctx, t, PE.lerp(P0[i].x, p1.x, e), PE.lerp(P0[i].y, p1.y, e), P0[i].w, h, fs); j++; }
    });
    // All three examples, as rows of cells: blue = visual, yellow = text. Lengths differ.
    const by = P1[P1.length - 1].y + h + 34, cw = (W - 110) / LMAX, rh = nar ? 14 : 18;
    ctx.save(); ctx.globalAlpha = ease(u, .6, 1);
    lab(ctx, 'all three examples after splicing (toy lengths):', 12, by - 10, C.F, 10.5);
    SEQ.forEach((s, r) => { lab(ctx, NAMES[r], 12, by + r * (rh + 6) + rh - 3, C.F, 11); cellRow(ctx, s, 30, by + r * (rh + 6), cw, rh);
      lab(ctx, LEN[r] + ' tokens', 34 + s.length * cw, by + r * (rh + 6) + rh - 3, C.Mu, 10); });
    ctx.restore();
  };

  // Stage 5: forming the batch. Pad to the longest, or pack with a block-diagonal mask.
  function batchGeom(W, H) { const nar = W < 560, cw = (W - 44) / LMAX, rh = nar ? 16 : 22; return { nar, x: 30, y: 66, cw, rh, gap: 12 }; }
  function drawGrid(ctx, g, e, showPad = true) {
    // e = 0: padded rows, e = 1: packed (examples run on across row ends). Cells tween between the two.
    let off = 0;
    SEQ.forEach((s, r) => { s.forEach((t, c) => { const gi = off + c, pr = Math.floor(gi / LMAX), pc = gi % LMAX;
        const x = g.x + PE.lerp(c, pc, e) * g.cw, y = g.y + PE.lerp(r, pr, e) * (g.rh + g.gap);
        ctx.fillStyle = cellColor(t.k); ctx.fillRect(x, y, Math.max(1, g.cw - (g.cw > 4 ? 1 : .4)), g.rh); });
      if (showPad) { ctx.save(); ctx.globalAlpha = 1 - e; for (let c = s.length; c < LMAX; c++) { ctx.fillStyle = tint(C.Br, .6); ctx.fillRect(g.x + c * g.cw, g.y + r * (g.rh + g.gap), Math.max(1, g.cw - (g.cw > 4 ? 1 : .4)), g.rh); } ctx.restore(); }
      off += s.length; });
    if (e > 0) { ctx.save(); ctx.globalAlpha = e; ctx.strokeStyle = C.Mu; ctx.setLineDash([3, 2]); ctx.lineWidth = 1;
      for (let gi = SUM; gi < 3 * LMAX; gi++) { const pr = Math.floor(gi / LMAX), pc = gi % LMAX; ctx.strokeRect(g.x + pc * g.cw + .5, g.y + pr * (g.rh + g.gap) + .5, g.cw - 1.5, g.rh - 1); }
      ctx.restore(); }
  }
  function legend(ctx, x, y, items, fs) { let xx = x; items.forEach(([c, t]) => { ctx.fillStyle = c; ctx.fillRect(xx, y - 8, 10, 10); lab(ctx, t, xx + 14, y + 1, C.Mu, fs); xx += 20 + textW(ctx, t, fs) + 8; }); }
  DRAW.pad = (ctx, W, H, u) => {
    const g = batchGeom(W, H), e = ease(u, 0, .6);
    heading(ctx, W, '5a. Batch by padding: every row as long as the longest', '3 rows x ' + LMAX + ' positions; brown cells are padding');
    SEQ.forEach((s, r) => lab(ctx, NAMES[r], 12, g.y + r * (g.rh + g.gap) + g.rh - 4, C.F, 11));
    let off = 0; SEQ.forEach((s, r) => { cellRow(ctx, s, g.x, g.y + r * (g.rh + g.gap), g.cw, g.rh); off += s.length;
      ctx.save(); ctx.globalAlpha = e; for (let c = s.length; c < LMAX; c++) { ctx.fillStyle = tint(C.Br, .6); ctx.fillRect(g.x + c * g.cw, g.y + r * (g.rh + g.gap), Math.max(1, g.cw - (g.cw > 4 ? 1 : .4)), g.rh); } ctx.restore(); });
    const ly = g.y + 3 * (g.rh + g.gap) + 20;
    legend(ctx, 12, ly, [[C.B, 'visual'], [tint(C.Y, .78), 'text'], [tint(C.Y, .42), 'role, SYS, ###'], [tint(C.Br, .6), 'pad']], g.nar ? 9.5 : 10.5);
    ctx.save(); ctx.globalAlpha = ease(u, .5, 1);
    const y2 = para(ctx, 'Pad positions are masked: real tokens never attend to them.', 12, ly + 26, W - 24, C.Mu, g.nar ? 10 : 11);
    para(ctx, 'Guess before you press 5b: what share of these ' + 3 * LMAX + ' cells is padding?', 12, y2 + 22, W - 24, C.F, g.nar ? 10.5 : 12);
    ctx.restore();
  };
  function maskMatrix(ctx, x, y, size, blocks, shared, a) {
    // blocks: [{s, n}] token ranges; shared: a leading image block every later block may read.
    const n = blocks.reduce((m, b) => Math.max(m, b.s + b.n), 0), k = size / n;
    ctx.save(); ctx.globalAlpha = a; ctx.fillStyle = C.BG; ctx.fillRect(x, y, size, size); ctx.strokeStyle = C.Rule; ctx.strokeRect(x - .5, y - .5, size + 1, size + 1);
    const tri = (s, m, col) => { ctx.fillStyle = col; ctx.beginPath(); ctx.moveTo(x + s * k, y + s * k); ctx.lineTo(x + s * k, y + (s + m) * k); ctx.lineTo(x + (s + m) * k, y + (s + m) * k); ctx.closePath(); ctx.fill(); };
    blocks.forEach(b => { if (shared && b !== shared) { ctx.fillStyle = tint(C.B, .75); ctx.fillRect(x + shared.s * k, y + b.s * k, shared.n * k, b.n * k); }
      tri(b.s, b.n, b === shared ? tint(C.B, .75) : tint(C.F, .55)); ctx.strokeStyle = C.F; ctx.lineWidth = 1; ctx.strokeRect(x + b.s * k, y + b.s * k, b.n * k, b.n * k); });
    ctx.restore(); return k;
  }
  DRAW.pack = (ctx, W, H, u) => {
    const g = batchGeom(W, H), e = ease(u, 0, .55), m = ease(u, .45, 1);
    heading(ctx, W, '5b. Batch by packing: concatenate, then mask', W < 560 ? 'one packed row of ' + 3 * LMAX + ', wrapped to fit' : 'the same ' + 3 * LMAX + ' cells as one long packed row, wrapped here to fit (common practice)');
    drawGrid(ctx, g, e);
    const ly = g.y + 3 * (g.rh + g.gap) + 14;
    let off = 0; ctx.save(); ctx.globalAlpha = e; SEQ.forEach((s, r) => { const gi = off, pr = Math.floor(gi / LMAX), pc = gi % LMAX;
      lab(ctx, NAMES[r], g.x + pc * g.cw + 2, g.y + pr * (g.rh + g.gap) - 3, C.F, 10); off += s.length; }); ctx.restore();
    ctx.save(); ctx.globalAlpha = 1 - e; SEQ.forEach((s, r) => lab(ctx, NAMES[r], 12, g.y + r * (g.rh + g.gap) + g.rh - 4, C.F, 11)); ctx.restore();
    // The attention mask of the packed sequence: each example sees only itself (causal triangle), never its neighbours.
    const size = g.nar ? Math.min(W * .42, H - ly - 128) : Math.min(H - ly - 22, W * .3), mx = 12, my = ly + 10;
    let s0 = 0; const blocks = SEQ.map(s => { const b = { s: s0, n: s.length }; s0 += s.length; return b; });
    maskMatrix(ctx, mx, my, size, blocks, null, m);
    ctx.save(); ctx.globalAlpha = m;
    const tx = g.nar ? 12 : mx + size + 14, fs = g.nar ? 9.5 : 11, ty = g.nar ? my + size + 4 : my;
    lab(ctx, 'attention mask of the packed row', tx, ty + 10, C.F, fs);
    lab(ctx, '(row = query token, column = key token)', tx, ty + 24, C.Mu, fs - .5);
    lab(ctx, 'grey triangles: A, B, C each see only', tx, ty + 44, C.Mu, fs);
    lab(ctx, 'their own earlier tokens', tx, ty + 57, C.Mu, fs);
    lab(ctx, 'padded: ' + PADS + ' of ' + 3 * LMAX + ' cells wasted (' + pct(PADS, 3 * LMAX) + ')', tx, ty + 80, C.Br, fs + .5);
    lab(ctx, 'packed: 0 wasted; ' + PADS + ' free cells', tx, ty + 96, C.G, fs + .5);
    lab(ctx, 'take the next example', tx, ty + 110, C.G, fs + .5);
    ctx.restore();
  };
  // Stage 5c: Molmo's variant. One image, all its questions in one sequence, masked apart.
  DRAW.share = (ctx, W, H, u) => {
    const nar = W < 560, e = ease(u, .1, .65), m = ease(u, .5, 1), cw = (W - 44) / 50, rh = nar ? 14 : 18, x = 30;
    heading(ctx, W, '5c. Molmo: one image, several questions, one sequence', 'three toy questions about the dog photo');
    const y0 = 62; lab(ctx, 'one sequence per question: the image is encoded 3 times', 12, y0 - 4, C.Mu, nar ? 9.5 : 10.5);
    const ys = y0 + 3 * (rh + 5) + 26; lab(ctx, 'Molmo: the image once, every question after it', 12, ys - 6, C.F, nar ? 10 : 11);
    const img = Array(NV).fill({ k: 'img', w: '' });
    let off = NV;
    QA.forEach((qa, r) => { const y = y0 + r * (rh + 5), dim = 1 - .45 * e;
      cellRow(ctx, img, x, y, cw, rh, dim); cellRow(ctx, qa, x + NV * cw, y, cw, rh, dim);
      lab(ctx, 'Q' + (r + 1), 12, y + rh - 3, C.F, 10);
      if (e > 0) { if (r === 0) cellRow(ctx, img, x, PE.lerp(y, ys, e), cw, rh, e);
        cellRow(ctx, qa, PE.lerp(x + NV * cw, x + off * cw, e), PE.lerp(y, ys, e), cw, rh, e); }
      off += qa.length; });
    const size = nar ? Math.min(W * .42, H - ys - rh - 116) : Math.min(H - ys - rh - 28, W * .28), my = ys + rh + 16;
    let s0 = NV; const img0 = { s: 0, n: NV }, blocks = [img0, ...QA.map(q => { const b = { s: s0, n: q.length }; s0 += q.length; return b; })];
    maskMatrix(ctx, 12, my, size, blocks, img0, m);
    ctx.save(); ctx.globalAlpha = m; const tx = nar ? 12 : 12 + size + 14, fs = nar ? 9.5 : 11, ty = nar ? my + size + 4 : my;
    lab(ctx, 'mask: every question sees the image', tx, ty + 12, C.F, fs); lab(ctx, '(blue columns) and itself, never', tx, ty + 26, C.F, fs);
    lab(ctx, 'the other questions', tx, ty + 40, C.F, fs);
    lab(ctx, 'toy: ' + NAIVE.reduce((a, b) => a + b) + ' tokens, 3 encodes', tx, ty + 64, C.Br, fs); lab(ctx, '-> ' + SHARED + ' tokens, 1 encode', tx, ty + 78, C.G, fs);
    ctx.restore();
  };

  // Stage 6: next-token targets, then the loss mask.
  function targetLayout(ctx, W, fs, h) {
    const toks = [...SEQ[0], { k: 'pad', w: 'pad \u00d7' + (LMAX - LEN[0]) }], next = toks.map((t, i) => toks[i + 1] || { k: 'pad', w: '' });
    next[toks.length - 2] = { k: 'pad', w: 'pad' }; next[toks.length - 1] = { k: 'pad', w: '-' };
    const widths = toks.map((t, i) => Math.max(chipW(ctx, t, fs, h), chipW(ctx, next[i], fs, h)));
    return { toks, next, P: flow(ctx, toks, 12, 70, W - 24, fs, h, 2 * h + 13, widths) };
  }
  function drawTargets(ctx, W, H, u, reveal) {
    const nar = W < 560, fs = nar ? 9.5 : 10.5, h = fs + 8, { toks, next, P } = targetLayout(ctx, W, fs, h);
    const gv = glow('visual'), ga = Math.max(glow('answer'), glow('loss')), gi = glow('instruct'), gp = glow('prev');
    lab(ctx, 'input (example A)', 12, 60, C.Mu, 10); lab(ctx, 'target below each: the next token', W - 12, 60, C.Mu, 10, 'right');
    const firstAns = toks.findIndex(t => t.k === 'a'), secondTurn = toks.findIndex((t, i) => i > firstAns && t.k === 'role' && t.w === 'Human:');
    toks.forEach((t, i) => {
      const p = P[i], w = p.w; chip(ctx, t, p.x, p.y, w, h, fs);
      const tg = next[i], isL = i < LEN[0] - 1 && isLoss(tg), r = reveal ? ease(u, .1 + .7 * i / toks.length, .2 + .7 * i / toks.length) : 0;
      let st = { f: tint(C.F, .06), s: C.Rule, t: C.F, d: null };
      if (r > 0) st = isL ? { f: tint(C.R, .15 + .35 * r), s: C.R, t: C.F } : { f: tint(C.Br, .12 * r), s: tint(C.Br, .5 * r + .2), t: tint(C.Mu, 1 - .45 * r) };
      chip(ctx, tg.k === 'img' ? { k: 'x', w: '' } : tg, p.x, p.y + h + 3, w, h, fs, 1, tg.k === 'img' && r === 0 ? { f: tint(C.B, .55), s: null, t: C.F } : st);
      if (reveal && !isL && r > .5) { ctx.save(); ctx.strokeStyle = tint(C.Br, .8); ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(p.x + 2, p.y + 2 * h + 1); ctx.lineTo(p.x + w - 2, p.y + h + 5); ctx.stroke(); ctx.restore(); }
      // Equation links: visual tokens, the human's turns, earlier answers, the graded answers.
      const ring = (g, col) => { if (g > 0) { ctx.save(); ctx.globalAlpha = g; rr(ctx, p.x - 2, p.y - 2, w + 4, h + 4, 4, null, col, 2.5); ctx.restore(); } };
      if (t.k === 'img') ring(gv, C.B);
      if (t.k === 'q' || (t.k === 'role' && t.w === 'Human:')) ring(gi, C.Y);
      if (t.k === 'a' && i < secondTurn) ring(gp, C.F);
      if (isL && ga > 0) { ctx.save(); ctx.globalAlpha = ga; rr(ctx, p.x - 2, p.y + h + 1, w + 4, h + 4, 4, null, C.R, 2.5); ctx.restore(); }
    });
    return P[P.length - 1].y + 2 * h + 3;
  }
  DRAW.targets = (ctx, W, H, u) => {
    heading(ctx, W, '6a. Forward: every position predicts the next token', W < 560 ? 'one pass; a causal mask hides the future' : 'the LLM reads the whole row at once; a causal mask hides the future');
    const yb = drawTargets(ctx, W, H, u, false);
    ctx.save(); ctx.globalAlpha = ease(u, .3, 1);
    para(ctx, 'Predict before you press 6b: which of these targets should the model be graded on?', 12, yb + 26, W - 24, C.F, W < 560 ? 10.5 : 12);
    ctx.restore();
  };
  DRAW.mask = (ctx, W, H, u) => {
    const nar = W < 560;
    heading(ctx, W, '6b. The loss mask: grade only what the assistant says', 'red targets carry loss; brown ones are ignored (LLaVA Table 2)');
    const yb = drawTargets(ctx, W, H, u, true);
    // The same rule over the whole padded batch: red = graded token, brown = ignored.
    const g = { x: 30, y: yb + 30, cw: (W - 44) / LMAX, rh: nar ? 9 : 12, gap: 3 }, a = ease(u, .7, 1);
    ctx.save(); ctx.globalAlpha = a;
    lab(ctx, 'whole padded batch: ' + LOSSN + ' of ' + 3 * LMAX + ' cells are graded target tokens (' + pct(LOSSN, 3 * LMAX) + ')', 12, g.y - 7, C.F, nar ? 10 : 11);
    SEQ.forEach((s, r) => { const row = [...s, ...Array(LMAX - s.length).fill({ k: 'pad' })];
      lab(ctx, NAMES[r], 12, g.y + r * (g.rh + g.gap) + g.rh - 1, C.F, nar ? 9 : 10);
      cellRow(ctx, row, g.x, g.y + r * (g.rh + g.gap), g.cw, g.rh, 1, t => isLoss(t) ? C.R : t.k === 'pad' ? tint(C.Br, .6) : tint(C.Br, .28)); });
    ctx.restore();
  };

  // Stage 7: backward. Gradients flow from the loss toward every weight; only trainable weights change.
  DRAW.stage1 = (ctx, W, H, u) => backward(ctx, W, H, u, 1);
  DRAW.stage2 = (ctx, W, H, u) => backward(ctx, W, H, u, 2);
  function backward(ctx, W, H, u, st) {
    const nar = W < 560, gt = glow('trainable');
    heading(ctx, W, st === 1 ? '7a. Backward, LLaVA stage 1: train the connector only' : '7b. Backward, LLaVA stage 2: connector and LLM',
      st === 1 ? (nar ? 'the gradient still passes through the frozen LLM' : 'the LLM is frozen, yet the gradient must pass through it to reach the connector') : 'the vision encoder stays frozen in both stages');
    // Boxes: encoder, connector, LLM, loss. Wide: left to right; narrow: bottom to top.
    let E, Cn, L, Ls, I, Tt;
    if (nar) { const x = 60, w = W - 72; Ls = { x: x + w * .25, y: 54, w: w * .5, h: 30 }; L = { x, y: 104, w, h: 86 }; Cn = { x: x + w * .2, y: 214, w: w * .6, h: 42 };
      E = { x: x + w * .2, y: 280, w: w * .6, h: 42 }; I = { x: x + w / 2 - 18, y: 340, s: 36 }; Tt = { x: 12, y: 132 }; }
    else { const y = H * .36; I = { x: 12, y: y - 22, s: 44 }; E = { x: 74, y: y - 34, w: W * .17, h: 68 }; Cn = { x: 74 + W * .17 + 34, y: y - 30, w: W * .14, h: 60 };
      L = { x: Cn.x + Cn.w + 34, y: y - 64, w: W * .3, h: 128 }; Ls = { x: L.x + L.w + 34, y: y - 20, w: W - (L.x + L.w + 34) - 12, h: 40 }; Tt = { x: L.x + 6, y: L.y + L.h + 26 }; }
    thumb(ctx, I.x, I.y, I.s, 'dog');
    const box = (b, fill, stroke, t1, t2, c1, dash) => { rr(ctx, b.x, b.y, b.w, b.h, 8, fill, stroke, 2, dash); lab(ctx, t1, b.x + b.w / 2, b.y + b.h / 2 - 2, c1, nar ? 11 : 12, 'center'); lab(ctx, t2, b.x + b.w / 2, b.y + b.h / 2 + 13, C.Mu, nar ? 9.5 : 10, 'center'); };
    box(E, tint(C.Br, .25), C.Br, 'vision encoder', 'frozen', C.B);
    const llmTrain = st === 2, pulse = ease(u, .75, 1);
    box(Cn, tint(C.G, .2 + .25 * pulse + .3 * gt), C.G, 'connector', 'updated', C.G);
    box(L, llmTrain ? tint(C.Y, .18 + .2 * pulse + .25 * gt) : tint(C.Br, .25), llmTrain ? C.Y : C.Br, 'language model', llmTrain ? 'updated' : 'frozen: gradient passes through', llmTrain ? C.Y : C.Br);
    box(Ls, tint(C.R, .2), C.R, 'loss', 'graded tokens', C.R);
    // Forward arrows (muted), then red gradient pulses running backward.
    if (nar) { PE.arrow(ctx, I.x + I.s / 2, I.y - 2, E.x + E.w / 2, E.y + E.h + 3, C.Mu, 1.2, 6); PE.arrow(ctx, E.x + E.w / 2, E.y - 2, Cn.x + Cn.w / 2, Cn.y + Cn.h + 3, C.Mu, 1.2, 6);
      PE.arrow(ctx, Cn.x + Cn.w / 2, Cn.y - 2, L.x + L.w / 2, L.y + L.h + 3, C.Mu, 1.2, 6); PE.arrow(ctx, L.x + L.w / 2, L.y - 2, Ls.x + Ls.w / 2, Ls.y + Ls.h + 3, C.Mu, 1.2, 6);
      for (let k = 0; k < 3; k++) chip(ctx, { k: 'q', w: ['Human:', 'What', '...'][k] }, 8, Tt.y + k * 22, 44, 18, 9.5); PE.arrow(ctx, 54, Tt.y + 22, L.x - 3, Tt.y + 22, C.Mu, 1.2, 6); }
    else { PE.arrow(ctx, I.x + I.s + 2, I.y + I.s / 2, E.x - 3, E.y + E.h / 2, C.Mu, 1.2, 6); PE.arrow(ctx, E.x + E.w + 2, E.y + E.h / 2, Cn.x - 3, Cn.y + Cn.h / 2, C.Mu, 1.2, 6);
      PE.arrow(ctx, Cn.x + Cn.w + 2, Cn.y + Cn.h / 2, L.x - 3, L.y + L.h / 2, C.Mu, 1.2, 6); PE.arrow(ctx, L.x + L.w + 2, L.y + L.h / 2, Ls.x - 3, Ls.y + Ls.h / 2, C.Mu, 1.2, 6);
      let xx = Tt.x; ['Human:', 'What', 'is', '...', '###'].forEach(w => { const t = { k: w === '###' ? 'stop' : w === 'Human:' ? 'role' : 'q', w }, cw = chipW(ctx, t, 10, 18); chip(ctx, t, xx, Tt.y, cw, 18, 10); xx += cw + 3; });
      PE.arrow(ctx, L.x + L.w / 2, Tt.y - 3, L.x + L.w / 2, L.y + L.h + 3, C.Mu, 1.2, 6); }
    // Gradient path: loss -> LLM -> connector. It stops there: nothing before the connector is trained.
    const path = nar ? [[Ls.x + Ls.w * .8, Ls.y + Ls.h], [L.x + L.w * .8, L.y], [L.x + L.w * .8, L.y + L.h], [Cn.x + Cn.w * .8, Cn.y]]
      : [[Ls.x, Ls.y + Ls.h * .8], [L.x + L.w, L.y + L.h * .8], [L.x, L.y + L.h * .8], [Cn.x + Cn.w, Cn.y + Cn.h * .8]];
    PE.polyline(ctx, path, C.R, 1.6, [5, 4], .8);
    const seg = [0, 1, 2].map(i => Math.hypot(path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1])), tot = seg.reduce((a, b) => a + b, 0);
    const at = d => { for (let i = 0; i < 3; i++) { if (d <= seg[i]) { const f = d / seg[i]; return [PE.lerp(path[i][0], path[i + 1][0], f), PE.lerp(path[i][1], path[i + 1][1], f)]; } d -= seg[i]; } return path[3]; };
    if (u < 1) for (let k = 0; k < 5; k++) { const d = (ease(u, 0, .85) * 1.25 - k * .06) * tot; if (d > 0 && d < tot) { const [x, y] = at(d); PE.dot(ctx, x, y, 4, C.R); } }
    const [ex, ey] = path[3]; PE.arrow(ctx, ...path[2], ex, ey, C.R, 2, 8, .9);
    lab(ctx, 'gradient', nar ? path[1][0] + 6 : (path[1][0] + path[2][0]) / 2, nar ? (path[1][1] + path[2][1]) / 2 : path[1][1] + 16, C.R, 10.5, nar ? 'left' : 'center');
    if (pulse > 0) { ctx.save(); ctx.globalAlpha = pulse; rr(ctx, Cn.x - 4, Cn.y - 4, Cn.w + 8, Cn.h + 8, 10, null, C.G, 2.5); if (llmTrain) rr(ctx, L.x - 4, L.y - 4, L.w + 8, L.h + 8, 10, null, C.Y, 2.5); ctx.restore(); }
    // Stage facts (LLaVA v2, Sec. 4.2 and 5).
    const fy = nar ? H - 34 : H - 46, fs = nar ? 9.5 : 11;
    const f1 = st === 1 ? ['data: 595K CC3M image-caption pairs as one-turn chats', 'LLaVA: 1 epoch, lr 2e-3, batch 128; trainable: W only'] : ['data: LLaVA-Instruct-158K (conversation, detail, reasoning)', 'LLaVA: 3 epochs, lr 2e-5, batch 32; trainable: W and the LLM'];
    lab(ctx, f1[0], 12, fy, C.F, fs); lab(ctx, f1[1], 12, fy + 16, C.Mu, fs);
  }

  const cv = PE.canvas(el, w => w < 560 ? 1.34 : .6, (ctx, W, H) => {
    C = { B: PE.css('--blue'), Y: PE.css('--yellow'), G: PE.css('--green'), R: PE.css('--red'), Br: PE.css('--brown'), F: PE.css('--fg'),
      Mu: PE.css('--muted'), Rule: PE.css('--rule'), BG: PE.css('--bg'), P: PE.css('--panel') };
    ctx.fillStyle = C.BG; ctx.fillRect(0, 0, W, H);
    if (W < 100) return;
    DRAW[S.scene](ctx, W, H, S.u);
  });

  const READ = {
    overview: () => 'The map of one generative training step: sample examples, turn each image into tokens, pour the text into a template, splice the two, form a batch, mask the loss, and send gradients back. Press a stage to zoom in.',
    contrastive: () => 'Mode A (CLIP): images and captions go through two separate encoders in parallel; the loss compares every image with every caption in the batch, so the batch itself supplies the negatives.',
    sample: () => 'Toy batch of 3: A and B come from LLaVA-Instruct, C from VQAv2. Below, Flamingo takes one batch from each of its 4 datasets and weights their losses 1.0 / 0.2 / 0.2 / 0.03 before one update.',
    encode: () => 'Image 3 x 336 x 336 pixels -> 24 x 24 = 576 patches of 14 x 14 -> frozen encoder -> 576 feature vectors -> MLP connector -> 576 tokens of the LLM\'s embedding width.',
    template: () => `Example A in LLaVA's template: ${RAW.A.length} tokens including one <image> placeholder (toy tokenization: one word, one token).`,
    splice: () => `Example A: the placeholder becomes ${NV} visual tokens (576 in the real model). Lengths now ${LEN.join(', ')} tokens for A, B, C.`,
    pad: () => `Padded batch: a 3 x ${LMAX} grid of embedding vectors (visual and word alike), plus a mask saying which cells are real.`,
    pack: () => `Padding wastes ${PADS} of ${3 * LMAX} cells (${pct(PADS, 3 * LMAX)}); packing uses ${SUM} cells with no padding and a block-diagonal mask. With real 576-token images the same three examples are ${REAL.join(', ')} tokens and padding wastes only ${pct1(3 * RLMAX - RSUM, 3 * RLMAX)}: the image dominates every row.`,
    share: () => `Toy: three questions about one image take ${NAIVE.reduce((a, b) => a + b)} tokens and 3 image encodings as separate sequences, or ${SHARED} tokens and 1 encoding in Molmo's shared sequence.`,
    targets: () => 'Each position of example A is paired with the next token as its target (none where the next token is visual). Nothing has been masked yet.',
    mask: () => `Graded: the assistant's answer tokens and every ### (${SEQ[0].filter(isLoss).length} targets in A). Ignored: SYS, the human's turns, the role labels, the visual tokens, padding. With real-size images (576 visual tokens) but the same toy text, only ${pct1(LOSSN, 3 * RLMAX)} of the padded batch carries loss.`,
    stage1: () => 'Stage 1: the loss is computed on the caption tokens and the <STOP>s; gradients flow back through the frozen LLM (it must pass them on) and update only the connector W.',
    stage2: () => 'Stage 2: the same backward pass, but now the LLM\'s weights update too. The encoder never receives gradients in either stage.',
  };

  function go(k) {
    stop(); S.scene = k; S.u = 0;
    chips.forEach((b, i) => b.setAttribute('aria-pressed', String(ORDER[i] === k)));
    texts.forEach(d => { d.hidden = d.dataset.stage !== k; });
    out.textContent = READ[k]();
    stop = PE.animate(k === 'overview' ? 1200 : k === 'stage1' || k === 'stage2' ? 1800 : 1500, v => { S.u = v; cv.redraw(); });
  }
  ORDER.forEach((k, i) => chips[i].addEventListener('click', () => go(k)));
  const idx = () => ORDER.indexOf(S.scene);
  document.getElementById('btn-pipeline-next').addEventListener('click', () => go(ORDER[Math.min(ORDER.length - 1, idx() + 1)]));
  document.getElementById('btn-pipeline-prev').addEventListener('click', () => { const i = idx(); go(i <= 0 ? 'overview' : ORDER[i - 1]); });
  document.getElementById('btn-pipeline-reset').addEventListener('click', () => {
    stop(); S.scene = 'overview'; S.u = 1; chips.forEach(b => b.setAttribute('aria-pressed', 'false'));
    texts.forEach(d => { d.hidden = d.dataset.stage !== 'overview'; }); out.textContent = READ.overview(); cv.redraw(); });
  texts.forEach(d => { d.hidden = d.dataset.stage !== 'overview'; });
  out.textContent = READ.overview();
})();
</script>
