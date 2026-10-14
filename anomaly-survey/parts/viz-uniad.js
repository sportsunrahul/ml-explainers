<script>
(() => {
  // ch-uniad toy: one attention layer rebuilds the 15 x 15 feature tokens of a scratched toy mesh. Checked in
  // build/toy-uniad2.mjs (tau 0.08, 7 x 7 window): full attention puts all weight on the token itself and the scratch
  // error stays at 0.12-0.33, like normal texture (up to 0.39); with the neighbour mask the scratch error is 1.42-1.48
  // against at most 0.73 elsewhere; the learned query behaves the same (1.44-1.52 vs 0.74). The cap was tried first and
  // dropped: with 3-d toy features its unique parts (the centre boss) have no look-alike outside their window either.
  const fig = document.getElementById('viz-uniad'), el = document.getElementById('cv-uniad'), out = fig.querySelector('.readout');
  const G = UNIAD_KIT.G, NP = G * G, img = TOY.defect(TOY.make('mesh', 200), 'scratch', 200);
  const x = UNIAD_KIT.feats(img);
  // learned queries: mean good-mesh feature per position
  const Q = new Float64Array(NP * 3); for (let s = 0; s < 20; s++) { const f = UNIAD_KIT.feats(TOY.make('mesh', 600 + s)); for (let i = 0; i < NP * 3; i++) Q[i] += f[i] / 20; }
  const MODES = ['full', 'mask', 'query'];
  const NAMES = { full: 'Full attention', mask: 'Neighbor mask', query: 'Learned query' };
  const tau = UNIAD_KIT.TAU, Rw = UNIAD_KIT.R;
  function attend(mode, i) {
    const q = mode === 'query' ? Q : x, w = new Float64Array(NP); let mx = -Infinity;
    for (let j = 0; j < NP; j++) {
      if (mode === 'mask' && UNIAD_KIT.masked(i, j)) { w[j] = -Infinity; continue; }
      let d = 0; for (let c = 0; c < 3; c++) d += (q[i * 3 + c] - x[j * 3 + c]) ** 2; w[j] = -d / tau; if (w[j] > mx) mx = w[j];
    }
    let z = 0; for (let j = 0; j < NP; j++) { w[j] = w[j] === -Infinity ? 0 : Math.exp(w[j] - mx); z += w[j]; } for (let j = 0; j < NP; j++) w[j] /= z; return w;
  }
  const R = {};
  MODES.forEach(m => { const err = new Float64Array(NP); for (let i = 0; i < NP; i++) { const w = attend(m, i); let e = 0;
    for (let c = 0; c < 3; c++) { let y = 0; for (let j = 0; j < NP; j++) y += w[j] * x[j * 3 + c]; e += (x[i * 3 + c] - y) ** 2; } err[i] = Math.sqrt(e); } R[m] = err; });
  const inScratch = new Uint8Array(NP); for (let q = 0; q < NP; q++) { const i = q % G, j = Math.floor(q / G); let n = 0; for (let y = j * 8; y < j * 8 + 16; y++) for (let xx = i * 8; xx < i * 8 + 16; xx++) n += img.mask[y * 128 + xx]; inScratch[q] = n > 20 ? 1 : 0; }
  let seed = 0; for (let q = 0; q < NP; q++) if (inScratch[q] && R.mask[q] > (inScratch[seed] ? R.mask[seed] : -1)) seed = q;
  const stats = m => { let a = 0, b = 0; for (let q = 0; q < NP; q++) inScratch[q] ? (a = Math.max(a, R[m][q])) : (b = Math.max(b, R[m][q])); return [a, b]; };
  const HI = 1.5, HB = {};
  const heat = (m, red) => HB[m + red] || (HB[m + red] = TOY.heatBitmap(TOY.toPixels(Float32Array.from(R[m])), .1, HI, red));
  const DEF = { mode: 'full', sel: seed };
  const S = { ...DEF, fade: 1, prev: 'full' };
  const glow = PE.linkPulses(fig, () => view.redraw());
  const view = PE.canvas(el, w => w < 560 ? (2 * (w - 40) + 110) / w : ((w - 56) / 2 + 74) / w, (ctx, W, H) => draw(ctx, W, H, S));
  function layout(W) {
    const narrow = W < 560, s = narrow ? W - 40 : (W - 56) / 2;
    return { narrow, s, a: { x: narrow ? 20 : 16, y: 40 }, b: narrow ? { x: 20, y: 40 + s + 48 } : { x: 16 + s + 24, y: 40 } };
  }
  let L = null;
  function draw(ctx, W, H, st) {
    if (!W) return; L = layout(W);
    const C = { blue: PE.css('--blue'), red: PE.css('--red'), yellow: PE.css('--yellow'), green: PE.css('--green'), brown: PE.css('--brown'), muted: PE.css('--muted'), rule: PE.css('--rule'), fg: PE.css('--fg'), bg: PE.css('--bg') };
    ctx.clearRect(0, 0, W, H);
    const { a, b, s } = L, ps = s / TOY.N;
    const tokXY = q => [a.x + (q % G) * TOY.S * ps, a.y + Math.floor(q / G) * TOY.S * ps];
    PE.label(ctx, `attention of the selected token (${NAMES[st.mode].toLowerCase()})`, a.x, a.y - 9, C.muted, 12);
    TOY.draw(ctx, img, a.x, a.y, s, {});
    // masked window
    if (st.mode === 'mask') { const i = st.sel % G, j = Math.floor(st.sel / G), gm = glow('mask');
      const x0 = Math.max(0, i - Rw), y0 = Math.max(0, j - Rw), x1 = Math.min(G - 1, i + Rw), y1 = Math.min(G - 1, j + Rw);
      const X0 = a.x + x0 * TOY.S * ps, Y0 = a.y + y0 * TOY.S * ps, X1 = a.x + (x1 * TOY.S + TOY.P) * ps, Y1 = a.y + (y1 * TOY.S + TOY.P) * ps;
      ctx.save(); ctx.fillStyle = C.bg; ctx.globalAlpha = .62 * st.fade; ctx.fillRect(X0, Y0, X1 - X0, Y1 - Y0); ctx.globalAlpha = 1;
      ctx.strokeStyle = C.brown; ctx.lineWidth = 1.5 + 2 * gm; ctx.setLineDash([5, 3]); ctx.strokeRect(X0, Y0, X1 - X0, Y1 - Y0); ctx.restore(); }
    // attention weights of the selected token: blue squares, alpha by weight
    const w = attend(st.mode, st.sel), wm = Math.max(...w) || 1, gq = Math.max(glow('query'), glow('copy'));
    ctx.save();
    for (let q = 0; q < NP; q++) { const v = w[q] / wm; if (v < .04) continue; const [tx, ty] = tokXY(q);
      ctx.globalAlpha = (.25 + .7 * v) * st.fade; ctx.fillStyle = C.blue; const side = TOY.S * ps * .8, cx = tx + TOY.P * ps / 2, cy = ty + TOY.P * ps / 2; ctx.fillRect(cx - side / 2, cy - side / 2, side, side); }
    ctx.restore();
    const [sx, sy] = tokXY(st.sel);
    ctx.save(); ctx.strokeStyle = C.yellow; ctx.lineWidth = 2 + 2 * gq; ctx.strokeRect(sx, sy, TOY.P * ps, TOY.P * ps); ctx.restore();
    // error map
    const ge = glow('error');
    PE.label(ctx, 'reconstruction error', b.x, b.y - 9, ge > .05 ? C.fg : C.muted, 12);
    ctx.save(); ctx.globalAlpha = .35; ctx.imageSmoothingEnabled = false; ctx.drawImage(TOY.bitmap(img), b.x, b.y, s, s); ctx.restore();
    if (st.fade < 1 && st.prev !== st.mode) { ctx.save(); ctx.globalAlpha = 1 - st.fade; ctx.drawImage(heat(st.prev, C.red), b.x, b.y, s, s); ctx.restore(); }
    ctx.save(); ctx.globalAlpha = st.fade; ctx.drawImage(heat(st.mode, C.red), b.x, b.y, s, s); ctx.restore();
    ctx.save(); ctx.strokeStyle = ge > .05 ? C.yellow : C.rule; ctx.lineWidth = 1 + 2 * ge; ctx.strokeRect(b.x - .5, b.y - .5, s + 1, s + 1); ctx.restore();
    const [ins, outs] = stats(st.mode);
    PE.label(ctx, `max error on the scratch ${ins.toFixed(2)}, elsewhere ${outs.toFixed(2)}`, b.x, b.y + s + 18, ins > 1.5 * outs ? C.red : C.muted, 12);
    PE.label(ctx, `self weight ${(w[st.sel] * 100).toFixed(0)}%`, a.x, a.y + s + 18, w[st.sel] > .5 ? C.yellow : C.muted, 12);
  }
  function say() {
    const w = attend(S.mode, S.sel), [ins, outs] = stats(S.mode), i = S.sel % G, j = Math.floor(S.sel / G);
    const why = S.mode === 'full' ? 'The rest goes to the overlapping tokens next to it, which look almost the same. So each token is rebuilt from itself, the scratch is copied, and the error map stays nearly blank: the identical shortcut.'
      : S.mode === 'mask' ? 'The token cannot see its own 7 by 7 window; normal texture finds look-alikes elsewhere, the scratch does not, so it lights up.'
      : 'Each position asks with the average good-mesh feature there, so the output looks like normal mesh; the scratch cannot be copied and lights up.';
    out.textContent = `${NAMES[S.mode]}, token at column ${i + 1}, row ${j + 1}${inScratch[S.sel] ? ' (on the scratch)' : ''}: ${(w[S.sel] * 100).toFixed(0)}% of its weight on itself. ${why} Largest error on the scratch ${ins.toFixed(2)}, elsewhere ${outs.toFixed(2)}.`;
  }
  function setMode(m) {
    const prev = S.mode; MODES.forEach(k => document.getElementById('btn-uniad-' + k).setAttribute('aria-pressed', k === m));
    if (prev === m) { view.redraw(); say(); return; }
    S.prev = prev; S.mode = m; PE.animate(600, u => { S.fade = u; view.redraw(); }, () => { S.fade = 1; S.prev = m; view.redraw(); }); say();
  }
  function select(q) { S.sel = q; view.redraw(); say(); }
  MODES.forEach(m => document.getElementById('btn-uniad-' + m).addEventListener('click', () => setMode(m)));
  document.getElementById('btn-uniad-reset').addEventListener('click', () => { select(DEF.sel); setMode(DEF.mode); });
  el.addEventListener('pointerdown', e => { if (!L) return; const r = el.getBoundingClientRect(), px = e.clientX - r.left, py = e.clientY - r.top, ps = L.s / TOY.N;
    const i = Math.floor((px - L.a.x) / (TOY.S * ps)), j = Math.floor((py - L.a.y) / (TOY.S * ps)); if (i >= 0 && j >= 0 && i < G && j < G) select(j * G + i); });
  el.addEventListener('keydown', e => { const d = { ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowUp: [0, -1], ArrowDown: [0, 1] }[e.key]; if (!d) return; e.preventDefault();
    const i = PE.clamp(S.sel % G + d[0], 0, G - 1), j = PE.clamp(Math.floor(S.sel / G) + d[1], 0, G - 1); select(j * G + i); });
  say();
})();
</script>
<script>
(() => {
  // UniAD architecture walkthrough. Shapes from Sec. 4.2 (224 input, 14 x 14, 272 -> 256 channels, 4 + 4 layers);
  // the token count is arithmetic; the per-stage split of the 272 channels is not stated, so no stage widths are shown.
  const x = '×';
  ArchFlow.mount({
    prefix: 'uniadflow', grid: { cols: 7, rows: 2 },
    boxes: {
      img: { c: 0, r: .5, lane: 'blue', title: ['image', 'any product'], glyph: 'photo', data: true },
      bb: { c: 1, r: .5, lane: 'mixed', title: ['EfficientNet-b4', 'stages 1-4'], glyph: 'layers' },
      cat: { c: 2, r: 0, lane: 'blue', title: ['resize to 14x14,', 'concatenate'], glyph: 'grid' },
      tok: { c: 3, r: .5, lane: 'green', title: ['tokens, project', '+ jitter (train)'], glyph: 'tokens' },
      enc: { c: 4, r: .5, lane: 'green', title: ['encoder', 'neighbor mask'], glyph: 'layers' },
      dec: { c: 5, r: .5, lane: 'green', title: ['decoder', 'learned queries'], glyph: 'layers' },
      loss: { c: 6, r: 0, lane: 'red', title: ['MSE vs', 'clean features'], glyph: 'loss' },
    },
    edges: [['img', 'bb'], ['bb', 'cat'], ['cat', 'tok'], ['tok', 'enc'], ['enc', 'dec'], ['dec', 'loss'], ['cat', 'loss']],
    stages: [{ key: 'image', show: ['img'] }, { key: 'backbone', show: ['bb'] }, { key: 'tokens', show: ['cat', 'tok'] }, { key: 'transformer', show: ['enc', 'dec'] },
      { key: 'loss', show: ['loss'] }, { key: 'backward', back: true }, { key: 'update', back: true, done: true }],
    variants: {}, variant: undefined,
    shape: id => ({ img: `64${x}3${x}224${x}224`, bb: 'four stage maps', cat: `64${x}272${x}14${x}14`, tok: `64${x}196${x}256`, enc: `64${x}196${x}256`, dec: `64${x}196${x}256|back to 272`, loss: 'one number' })[id],
    params: id => ({ tok: 'projection, positions', enc: '4 layers', dec: '4 layers, queries, projection' })[id] || (id === 'bb' ? 'all layers (frozen)' : null),
    frozen: id => id === 'bb',
    readout: s => ({
      overview: 'UniAD: a frozen CNN, then a transformer built so it cannot copy its input; one model for all 15 MVTec AD categories.',
      image: `Step 1: a batch of 64 images of any category, 224 ${x} 224, no label.`,
      backbone: 'Step 2: ImageNet EfficientNet-b4, frozen; stages 1 to 4 are kept.',
      tokens: `Step 3: all four stage maps resized to 14 ${x} 14 and concatenated: 272 channels. 196 tokens, projected to 256; Gaussian jitter (alpha 20, probability 1) in training only.`,
      transformer: `Step 4: 4 encoder layers with neighbor-masked attention (7 ${x} 7 window), 4 decoder layers each with its own learned query; projection back to 272 ${x} 14 ${x} 14.`,
      loss: 'Step 5: mean squared error between the reconstruction and the clean (un-jittered) backbone features.',
      backward: 'Backward: projections, encoder, decoder, learned queries and position embeddings get gradient (7.48 M weights); the frozen backbone does not.',
      update: 'Update: AdamW, learning rate 1e-4, weight decay 1e-4, cut tenfold after epoch 800; 1000 epochs at batch 64 on 8 V100 GPUs.',
    })[s],
  });
})();
</script>
