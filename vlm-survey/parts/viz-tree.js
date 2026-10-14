<script>
(() => {
  const fig = document.getElementById('viz-tree'), el = document.getElementById('tree-canvas'), out = document.getElementById('tree-readout');
  const lanes = ['image encoders', 'bridging to LMs', 'instruction tuning', 'resolution + grounding', 'recipes + open data', 'early fusion'];
  // Dates: first arXiv submission (notes/claims/meta.json). ch = chapter anchor.
  const P = [
    ['vit', 'ViT', '2020-10-22', 0, 'patches', 'a plain transformer on image patches'],
    ['align', 'ALIGN', '2021-02-11', 0, 'shared-space', 'contrastive training on over a billion noisy pairs'],
    ['clip', 'CLIP', '2021-02-26', 0, 'shared-space', 'the image-caption matching game and zero-shot'],
    ['blip', 'BLIP', '2022-01-28', 1, 'bootstrap', 'one model that matches and writes; CapFilt'],
    ['flamingo', 'Flamingo', '2022-04-29', 1, 'flamingo', 'gated cross-attention into a frozen LM'],
    ['blip2', 'BLIP-2', '2023-01-30', 1, 'qformer', 'a small Q-Former between two frozen models'],
    ['siglip', 'SigLIP', '2023-03-27', 0, 'sigmoid', 'a per-pair sigmoid loss'],
    ['llava', 'LLaVA', '2023-04-17', 2, 'llava', 'visual instruction tuning'],
    ['qwenvl', 'Qwen-VL', '2023-08-24', 3, 'resolution', '448 px input and boxes as text'],
    ['llava15', 'LLaVA-1.5', '2023-10-05', 2, 'llava', 'MLP connector, 336 px, answer-format prompts'],
    ['florence2', 'Florence-2', '2023-11-10', 3, 'grounding', 'every vision task as text with location tokens'],
    ['idefics2', 'Idefics2', '2024-05-03', 4, 'recipes', 'controlled ablations of design choices'],
    ['chameleon', 'Chameleon', '2024-05-16', 5, 'early-fusion', 'images as discrete tokens in one vocabulary'],
    ['qwen2vl', 'Qwen2-VL', '2024-09-18', 3, 'resolution', 'native dynamic resolution and M-RoPE'],
    ['molmo', 'Molmo', '2024-09-25', 4, 'recipes', 'open data without VLM distillation; pointing'],
    ['qwen3vl', 'Qwen3-VL', '2025-11-26', 3, 'qwen3', 'DeepStack, interleaved M-RoPE, text timestamps'],
  ].map(([k, name, d, lane, ch, intro]) => ({ k, name, t: +new Date(d), d, lane, ch, intro }));
  const byK = Object.fromEntries(P.map(p => [p.k, p]));
  // [earlier, later, the limitation the later paper fixed]
  const A = [
    ['clip', 'siglip', 'the softmax needs every pair in the batch'],
    ['clip', 'blip', 'it can match but cannot write'],
    ['align', 'blip', 'noisy web captions'],
    ['clip', 'flamingo', 'it can match but cannot write, and each new task needs fine-tuning'],
    ['flamingo', 'blip2', 'billions of new trainable parameters'],
    ['blip2', 'llava', 'it describes but does not follow instructions'],
    ['llava', 'llava15', 'weak short-answer benchmark scores'],
    ['llava15', 'qwen2vl', 'a small fixed square cannot read documents'],
    ['qwenvl', 'qwen2vl', 'one fixed resolution for every image'],
    ['clip', 'florence2', 'image-level answers only, no locations'],
    ['llava', 'molmo', 'instruction data an LLM wrote from sparse, noisy image annotations'],
    ['llava15', 'idefics2', 'design choices that were never ablated'],
    ['llava', 'chameleon', 'late fusion can only output text'],
    ['qwen2vl', 'qwen3vl', 'vision enters the language model only at its input layer'],
  ];
  const spine = new Set(['clip>flamingo', 'flamingo>blip2', 'blip2>llava', 'llava>llava15', 'llava15>qwen2vl', 'qwen2vl>qwen3vl']);
  const S = { sel: null, mode: 'all', u: 1 };
  const glow = PE.linkPulses(fig, () => cv.redraw());
  let stop = () => {};
  const t0 = +new Date('2020-08-01'), t1 = +new Date('2026-01-15');

  function layout(W, H) {
    const narrow = W < 560;
    if (!narrow) { const x0 = 150, x1 = W - 40, y0 = 30, lh = (H - 70) / lanes.length, X = p => x0 + (p.t - t0) / (t1 - t0) * (x1 - x0);
      // Papers close in time on the same lane alternate above and below the lane line.
      const off = {}; lanes.forEach((_, li) => { let prev = null;
        P.filter(p => p.lane === li).forEach(p => { off[p.k] = prev && X(p) - X(prev) < 70 ? -(off[prev.k] || -1) : 0; prev = p; }); });
      return { narrow, off, pos: p => [X(p), y0 + (p.lane + .5) * lh + (off[p.k] || 0) * lh * .24], lh, x0, x1, y0 }; }
    const y0 = 56, y1 = H - 20, lw = (W - 16) / lanes.length;
    return { narrow, pos: p => [8 + (p.lane + .5) * lw, y0 + (p.t - t0) / (t1 - t0) * (y1 - y0)], lw, y0, y1 };
  }
  const cv = PE.canvas(el, w => w < 560 ? 1.9 : .62, (ctx, W, H) => {
    const G = PE.css('--green'), F = PE.css('--fg'), M = PE.css('--muted'), Rl = PE.css('--rule');
    ctx.fillStyle = PE.css('--bg'); ctx.fillRect(0, 0, W, H);
    const L = layout(W, H);
    // Lanes and year ticks.
    ctx.strokeStyle = Rl; ctx.lineWidth = 1;
    lanes.forEach((name, i) => {
      if (!L.narrow) { const y = L.y0 + (i + .5) * L.lh; ctx.beginPath(); ctx.moveTo(L.x0 - 6, y); ctx.lineTo(L.x1, y); ctx.stroke(); PE.label(ctx, name, L.x0 - 12, y + 4, M, 11, 'right'); }
      else { const x = 8 + (i + .5) * L.lw; ctx.beginPath(); ctx.moveTo(x, L.y0 - 4); ctx.lineTo(x, L.y1); ctx.stroke();
        ctx.save(); ctx.translate(x + 3, L.y0 - 8); ctx.rotate(-Math.PI / 5); PE.label(ctx, name.split(' ')[0], 0, 0, M, 9.5); ctx.restore(); }
    });
    for (let yr = 2021; yr <= 2025; yr++) { const t = +new Date(yr + '-01-01'), f = (t - t0) / (t1 - t0);
      if (!L.narrow) { const x = L.x0 + f * (L.x1 - L.x0); PE.label(ctx, String(yr), x, H - 14, M, 11, 'center'); ctx.globalAlpha = .4; ctx.beginPath(); ctx.moveTo(x, L.y0); ctx.lineTo(x, H - 30); ctx.stroke(); ctx.globalAlpha = 1; }
      else { const y = L.y0 + f * (L.y1 - L.y0); ctx.globalAlpha = .4; ctx.beginPath(); ctx.moveTo(4, y); ctx.lineTo(W - 4, y); ctx.stroke(); ctx.globalAlpha = 1; PE.label(ctx, String(yr), W - 6, y - 3, M, 10, 'right'); } }
    // Arcs: earlier -> later, bowed so parallel lanes stay readable.
    A.forEach(([a, b], i) => {
      const key = a + '>' + b, on = S.mode === 'spine' ? spine.has(key) : true;
      const hit = S.sel && (S.sel === a || S.sel === b);
      const [x1, y1] = L.pos(byK[a]), [x2, y2] = L.pos(byK[b]);
      const mx = (x1 + x2) / 2 + (L.narrow ? 26 : 0) * (i % 2 ? 1 : -1), my = (y1 + y2) / 2 - (L.narrow ? 0 : 22) * (i % 2 ? 1 : -1);
      const alpha = (S.sel ? (hit ? 1 : .12) : on ? .75 : .1) * S.u;
      ctx.save(); ctx.globalAlpha = alpha; ctx.strokeStyle = hit ? G : on && S.mode === 'spine' ? G : M; ctx.lineWidth = hit ? 2.5 : 1.5;
      ctx.beginPath(); ctx.moveTo(x1, y1); ctx.quadraticCurveTo(mx, my, x2, y2); ctx.stroke(); ctx.restore();
      if (alpha > .5) { const ux = .15 * x1 + .5 * mx + .35 * x2 - (.15 * x1 + .5 * mx + .35 * x2 - x2) * .1, uy = .15 * y1 + .5 * my + .35 * y2;
        const ex = x2 - (x2 - mx) * .12, ey = y2 - (y2 - my) * .12; PE.arrow(ctx, ux, uy, ex, ey, hit ? G : M, 0.01, 7, alpha); }
    });
    P.forEach(p => { const [x, y] = L.pos(p), on = S.sel === p.k, rel = S.sel && A.some(([a, b]) => (a === S.sel && b === p.k) || (b === S.sel && a === p.k));
      PE.dot(ctx, x, y, on ? 8 : 6, !S.sel || on || rel ? F : M, S.sel && !on && !rel ? .35 : 1);
      if (on) { ctx.strokeStyle = F; ctx.lineWidth = 1.5; ctx.beginPath(); ctx.arc(x, y, 11, 0, 7); ctx.stroke(); }
      const below = !L.narrow && (L.off[p.k] || 0) > 0;
      PE.label(ctx, p.name, x, y + (L.narrow ? -9 : below ? 19 : -10), S.sel && !on && !rel ? M : F, L.narrow ? 9.5 : 11, 'center'); });
  });
  function readout() {
    if (!S.sel) { out.textContent = S.mode === 'spine' ? 'The main line: CLIP to Flamingo to BLIP-2 to LLaVA to LLaVA-1.5 to Qwen2-VL to Qwen3-VL. Select any paper for details.' : 'Select a paper (click, or focus the tree and use the arrow keys).'; return; }
    const p = byK[S.sel], fixed = A.filter(a => a[1] === p.k).map(a => `${byK[a[0]].name}'s ${a[2]}`), by = A.filter(a => a[0] === p.k).map(a => `${byK[a[1]].name} (${a[2]})`);
    out.textContent = `${p.name}, first on arXiv ${p.d}: ${p.intro}. ` + (fixed.length ? `It fixed ${fixed.join('; ')}. ` : '') + (by.length ? `Its limitations were taken up by ${by.join('; ')}.` : 'Nothing on this page fixes it yet.');
  }
  function fade(mode) { stop(); S.mode = mode; S.sel = null; S.u = 0; stop = PE.animate(700, u => { S.u = u; cv.redraw(); readout(); }); }
  document.getElementById('btn-tree-all').addEventListener('click', () => fade('all'));
  document.getElementById('btn-tree-spine').addEventListener('click', () => fade('spine'));
  document.getElementById('btn-tree-reset').addEventListener('click', () => { stop(); S.mode = 'all'; S.sel = null; S.u = 1; cv.redraw(); readout(); });
  el.addEventListener('pointerdown', e => { const r = el.getBoundingClientRect(), L = layout(r.width, r.height), x = e.clientX - r.left, y = e.clientY - r.top;
    let best = null, bd = 22 * 22; P.forEach(p => { const [px, py] = L.pos(p), d = (px - x) ** 2 + (py - y) ** 2; if (d < bd) { bd = d; best = p.k; } });
    S.sel = best; S.u = 1; cv.redraw(); readout(); });
  el.addEventListener('keydown', e => { const d = { ArrowRight: 1, ArrowDown: 1, ArrowLeft: -1, ArrowUp: -1 }[e.key]; if (!d) return; e.preventDefault();
    const i = S.sel ? P.findIndex(p => p.k === S.sel) : -1; S.sel = P[(i + d + P.length) % P.length].k; S.u = 1; cv.redraw(); readout(); });
  readout();
})();
</script>
