<script>
(() => {
  // ch-efficientad toy: camera view, local map (student vs teacher), global map (autoencoder vs student), combined map.
  // Checked in build/toy-efficientad.mjs (40 training caps, 3 neighbours, 8 components, 3x3 map smoothing): good caps
  // (seeds 500-509) image score 0.00-0.19, mostly under 0.1; scratches: local peak 0.38-0.54 and inside the mask 10/10,
  // global peak mostly under 0.2; extra hole: global peak 0.14-0.64 and inside the mask 10/10, local peak mostly under
  // 0.12. Displayed: good 500 (0.02), scratch 200 (local 0.43, global 0.09), extra hole on cap 201 (local 0.08, global
  // 0.40). A missing bolt was tried first and dropped: on the toy cap the empty spot's shading is also locally new, so the
  // local map fired more than the global one, which would misrepresent the paper's point.
  const fig = document.getElementById('viz-efficientad'), el = document.getElementById('cv-efficientad');
  const out = fig.querySelector('.readout'), slider = document.getElementById('rng-ead-phard'), sOut = document.getElementById('out-ead-phard');
  const PH = [0, .9, .99, .999, .9999, .99999], PHS = ['0', '0.9', '0.99', '0.999', '0.9999', '0.99999'];
  const CASES = {
    good: { label: 'good cap', make: () => TOY.make('cap', 500) },
    scratch: { label: 'scratched cap', make: () => TOY.defect(TOY.make('cap', 200), 'scratch', 200) },
    extra: { label: 'cap with an extra hole', make: () => EAD_KIT.extraHole(TOY.make('cap', 201), 1) },
  };
  const DEF = { c: 'good', ph: 3 };
  const S = { ...DEF, fade: 1 };
  let ready = false; const R = {}, IMG = {}, HB = {};
  function ensure() {
    if (ready) return; EAD_KIT.fit();
    for (const k in CASES) { IMG[k] = CASES[k].make(); R[k] = EAD_KIT.score(IMG[k]); }
    ready = true;
  }
  const HI = .3, THR = .1;
  function heat(k, which, red) {
    const key = k + which + red; if (HB[key]) return HB[key];
    return HB[key] = TOY.heatBitmap(TOY.toPixels(Float32Array.from(R[k][which])), 0, HI, red);
  }
  const glow = PE.linkPulses(fig, () => view.redraw());
  const view = PE.canvas(el, w => w < 560 ? (2 * ((w - 28) / 2 + 66) + 22) / w : ((w - 64) / 4 + 88) / w, (ctx, W, H) => draw(ctx, W, H, S));

  function layout(W) {
    const narrow = W < 560, g = narrow ? 12 : 16;
    const s = narrow ? (W - 16 - g) / 2 : (W - 16 - 3 * g) / 4;
    const at = i => narrow ? { x: 8 + (i % 2) * (s + g), y: 40 + Math.floor(i / 2) * (s + 74) } : { x: 8 + i * (s + g), y: 40 };
    return { narrow, s, at };
  }
  function draw(ctx, W, H, st) {
    if (!W) return;
    const C = { blue: PE.css('--blue'), red: PE.css('--red'), yellow: PE.css('--yellow'), green: PE.css('--green'), brown: PE.css('--brown'), muted: PE.css('--muted'), rule: PE.css('--rule'), fg: PE.css('--fg') };
    ctx.clearRect(0, 0, W, H);
    const L = layout(W);
    if (!ready) { PE.label(ctx, 'Computing the toy model...', 12, 40, C.muted, 13); return; }
    const r = R[st.c], img = IMG[st.c];
    const panels = [
      { title: 'camera view', word: null },
      { title: 'student vs teacher', word: 'local', map: 'nst', peak: r.maxSt },
      { title: 'autoencoder vs student', word: 'global', map: 'nae', peak: r.maxAe },
      { title: 'combined map', word: 'combine', map: 'comb', peak: r.image },
    ];
    const wordGlow = p => p.word === 'local' ? Math.max(glow('local'), glow('teacher'), glow('student')) : p.word === 'global' ? Math.max(glow('global'), glow('ae')) : p.word === 'combine' ? glow('combine') : 0;
    panels.forEach((p, i) => {
      const a = L.at(i), gw = wordGlow(p);
      PE.label(ctx, p.title, a.x, a.y - 9, gw > .05 ? C.fg : C.muted, 12);
      if (!p.map) { TOY.draw(ctx, img, a.x, a.y, L.s, {}); }
      else {
        ctx.save(); ctx.globalAlpha = .35; ctx.imageSmoothingEnabled = false; ctx.drawImage(TOY.bitmap(img), a.x, a.y, L.s, L.s); ctx.restore();
        ctx.save(); ctx.globalAlpha = st.fade; ctx.drawImage(heat(st.c, p.map, C.red), a.x, a.y, L.s, L.s); ctx.restore();
        ctx.save(); ctx.strokeStyle = gw > .05 ? C.yellow : C.rule; ctx.lineWidth = 1 + 2 * gw; ctx.strokeRect(a.x - .5, a.y - .5, L.s + 1, L.s + 1); ctx.restore();
        PE.label(ctx, `peak ${p.peak.toFixed(2)}`, a.x, a.y + L.s + 18, p.peak > THR ? C.red : C.muted, 12);
      }
    });
    // hard-feature-loss mask on the camera view
    const m = EAD_KIT.hardMask(r, PH[st.ph]), a0 = L.at(0), ps = L.s / TOY.N, gh = glow('hard');
    ctx.save(); ctx.strokeStyle = C.yellow; ctx.lineWidth = 1.2 + 1.8 * gh; ctx.globalAlpha = .9;
    for (let q = 0; q < 225; q++) if (m.keep[q]) ctx.strokeRect(a0.x + (q % 15) * 8 * ps, a0.y + Math.floor(q / 15) * 8 * ps, 16 * ps, 16 * ps);
    ctx.restore();
    PE.label(ctx, `${m.n} of 675 elements train`, a0.x, a0.y + L.s + 18, C.yellow, 12);
    // verdict under the combined map
    const a3 = L.at(3), rej = r.image > THR, word = rej ? 'reject' : 'pass';
    ctx.save(); ctx.font = '600 12px system-ui, sans-serif'; const tw = ctx.measureText(word).width + 16, tx = a3.x, ty = a3.y + L.s + 28;
    ctx.strokeStyle = rej ? C.red : C.blue; ctx.lineWidth = 1.2 + 1.5 * glow('combine'); ctx.beginPath(); ctx.roundRect(tx, ty, tw, 20, 10); ctx.stroke();
    ctx.fillStyle = rej ? C.red : C.blue; ctx.fillText(word, tx + 8, ty + 14); ctx.restore();
  }
  function say() {
    if (!ready) return; const r = R[S.c], m = EAD_KIT.hardMask(r, PH[S.ph]);
    const lead = S.c === 'good' ? 'Good cap: both maps stay quiet' : S.c === 'scratch' ? 'Scratch: the student-versus-teacher map finds it; the autoencoder map barely moves' : 'Extra hole: the autoencoder map finds it; the student-versus-teacher map sees an ordinary hole';
    out.textContent = `${lead}. Peaks ${r.maxSt.toFixed(2)} (local) and ${r.maxAe.toFixed(2)} (global); image score ${r.image.toFixed(2)}, ${r.image > THR ? 'over' : 'under'} the 0.1 level: ${r.image > THR ? 'reject' : 'pass'}. At p_hard ${PHS[S.ph]} the hard feature loss would train on ${m.n} of this image's 675 toy feature elements (yellow).`;
  }
  function markCol() { fig.querySelectorAll('#tbl-ead-phard td[data-col]').forEach(td => { td.style.color = +td.dataset.col === S.ph ? PE.css('--yellow') : ''; td.style.fontWeight = +td.dataset.col === S.ph ? '700' : ''; }); }
  function setCase(c) {
    ensure(); const changed = c !== S.c; S.c = c;
    for (const k in CASES) document.getElementById('btn-ead-' + k).setAttribute('aria-pressed', k === c);
    if (changed) PE.animate(500, u => { S.fade = u; view.redraw(); }, () => { S.fade = 1; view.redraw(); }); else view.redraw();
    say();
  }
  function setPh(i) { ensure(); S.ph = i; slider.value = i; sOut.textContent = PHS[i]; markCol(); view.redraw(); say(); }
  for (const k in CASES) document.getElementById('btn-ead-' + k).addEventListener('click', () => setCase(k));
  slider.addEventListener('input', () => setPh(+slider.value));
  document.getElementById('btn-ead-reset').addEventListener('click', () => { setPh(DEF.ph); setCase(DEF.c); });
  el.addEventListener('keydown', e => { const d = { ArrowLeft: -1, ArrowRight: 1 }[e.key]; if (!d) return; e.preventDefault(); const ks = Object.keys(CASES); setCase(ks[(ks.indexOf(S.c) + d + ks.length) % ks.length]); });
  // Fit the toy only when the figure comes near the viewport (about a second of work), or on first use.
  const io = new IntersectionObserver(es => { if (es.some(e => e.isIntersecting)) { io.disconnect(); setTimeout(() => { ensure(); markCol(); view.redraw(); say(); }, 30); } }, { rootMargin: '600px' });
  io.observe(fig);
  markCol();
})();
</script>
<script>
(() => {
  // Stage 1 (distillation). Shapes for a 256 input with the listed padding (Tables 6-7; our computation); parameter counts
  // our computation from the layer tables (they sum, with the student and autoencoder, to the paper's printed 8 M and 21 M).
  const x = '×';
  const V = { s: { name: 'EfficientAD-S', tp: '2,694,144' }, m: { name: 'EfficientAD-M', tp: '8,026,624' } };
  ArchFlow.mount({
    prefix: 'eadistill', grid: { cols: 3, rows: 2 },
    boxes: {
      big: { c: 0, r: 0, lane: 'blue', title: ['ImageNet image', '512 px'], glyph: 'photo', data: true },
      small: { c: 0, r: 1, lane: 'blue', title: ['same image', '256 px'], glyph: 'photo', data: true },
      psi: { c: 1, r: 0, lane: 'mixed', title: ['WideResNet-101', '+ PatchCore pooling'], glyph: 'layers' },
      pdn: { c: 1, r: 1, lane: 'green', title: ['teacher PDN'], glyph: 'layers' },
      loss: { c: 2, r: .5, lane: 'red', title: ['MSE loss'], glyph: 'loss' },
    },
    edges: [['big', 'psi'], ['small', 'pdn'], ['psi', 'loss'], ['pdn', 'loss']],
    stages: [{ key: 'images', show: ['big', 'small'] }, { key: 'features', show: ['psi', 'pdn'] }, { key: 'loss', show: ['loss'] },
      { key: 'backward', back: true }, { key: 'update', back: true, done: true }],
    variants: V, variant: 's',
    shape: id => ({ big: `16${x}3${x}512${x}512`, small: `16${x}3${x}256${x}256`, psi: `16${x}384${x}64${x}64`, pdn: `16${x}384${x}64${x}64`, loss: 'one number' })[id],
    sub: (id, v) => id === 'pdn' ? `${v.tp} weights` : null,
    params: id => id === 'pdn' ? 'all conv layers' : id === 'psi' ? 'all layers (frozen)' : null,
    frozen: id => id === 'psi',
    readout: (s, v) => ({
      overview: `${v.name}: distill a frozen WideResNet-101's 384-channel features into a small PDN teacher, once, on ImageNet.`,
      images: `Step 1: a batch of 16 ImageNet images, each at 512 ${x} 512 for the big network and 256 ${x} 256 for the PDN.`,
      features: `Step 2: both produce 384 ${x} 64 ${x} 64 maps; the big network's channels are normalized with statistics from 10,000 ImageNet images.`,
      loss: 'Step 3: mean squared difference over all elements, averaged over the batch.',
      backward: `Backward: the gradient reaches only the ${v.name} teacher PDN (${v.tp} weights, our count). The big network is frozen.`,
      update: 'Update: Adam, learning rate 1e-4, weight decay 1e-5, 60,000 steps at batch 16. Then the teacher is frozen.',
    })[s],
  });
})();
</script>
<script>
(() => {
  // Stage 2 (per product). Shapes for a 256 input with the listed padding (our computation, Tables 6-8).
  const x = '×';
  const V = { s: { name: 'EfficientAD-S', sp: '4,267,392' }, m: { name: 'EfficientAD-M', sp: '11,615,488' } };
  ArchFlow.mount({
    prefix: 'eaflow', grid: { cols: 3, rows: 4 },
    boxes: {
      img: { c: 0, r: 1, lane: 'blue', title: ['good image I'], glyph: 'photo', data: true },
      pimg: { c: 0, r: 3, lane: 'yellow', title: ['ImageNet image P'], glyph: 'photo', data: true },
      teach: { c: 1, r: 0, lane: 'mixed', title: ['teacher PDN'], glyph: 'layers' },
      ae: { c: 1, r: 1, lane: 'green', title: ['autoencoder'], glyph: 'latents' },
      stud: { c: 1, r: 2, lane: 'green', title: ['student PDN'], glyph: 'layers' },
      studp: { c: 1, r: 3, lane: 'green', title: ['student on P', '(same weights)'], glyph: 'layers' },
      lae: { c: 2, r: .5, lane: 'red', title: ['L_AE'], glyph: 'loss' },
      lstae: { c: 2, r: 1.5, lane: 'red', title: ['L_STAE'], glyph: 'loss' },
      lst: { c: 2, r: 2.6, lane: 'red', title: ['L_ST', 'hard + penalty'], glyph: 'loss' },
    },
    edges: [['img', 'teach'], ['img', 'ae'], ['img', 'stud'], ['pimg', 'studp'], ['teach', 'lae'], ['ae', 'lae'], ['ae', 'lstae'], ['stud', 'lstae'], ['teach', 'lst'], ['stud', 'lst'], ['studp', 'lst']],
    stages: [{ key: 'inputs', show: ['img', 'pimg'] }, { key: 'networks', show: ['teach', 'ae', 'stud', 'studp'] }, { key: 'losses', show: ['lae', 'lstae', 'lst'] },
      { key: 'backward', back: true }, { key: 'update', back: true, done: true }],
    variants: V, variant: 's',
    shape: id => ({ img: `1${x}3${x}256${x}256`, pimg: `1${x}3${x}256${x}256`, teach: `1${x}384${x}64${x}64`, ae: `1${x}384${x}64${x}64|code: 64`, stud: `1${x}768${x}64${x}64`, studp: `1${x}768${x}64${x}64`, lae: 'one number', lstae: 'one number', lst: 'one number' })[id],
    sub: (id, v) => id === 'stud' ? `${v.sp} weights` : id === 'ae' ? '1,096,320 weights' : null,
    params: id => ({ ae: 'encoder and decoder convs', stud: 'all conv layers', studp: 'shared with the student' })[id] || null,
    frozen: id => id === 'teach',
    readout: (s, v) => ({
      overview: `${v.name}, per product: the frozen teacher sets the targets; the student and autoencoder learn from good images only.`,
      inputs: `Step 1: one good image (batch size 1, 256 ${x} 256) and one random ImageNet image, resized to 512, gray with probability 0.3, center-cropped to 256.`,
      networks: `Step 2: teacher 384 ${x} 64 ${x} 64 (normalized per channel); autoencoder: 256 px down to a 64-number code and back up to 384 ${x} 64 ${x} 64; student 768 ${x} 64 ${x} 64, split into two halves of 384.`,
      losses: 'Step 3: L_ST = hard feature loss (p_hard 0.999) on the first half + ImageNet penalty; L_AE = autoencoder vs teacher; L_STAE = second half vs autoencoder.',
      backward: `Backward: the student (${v.sp} weights, our count) gets gradient from L_ST and L_STAE; the autoencoder from L_AE and L_STAE; the teacher none.`,
      update: 'Update: Adam, learning rate 1e-4 (decayed to 1e-5 after step 66,500), 70,000 steps. Then stage 3 stores the 0.9 and 0.995 quantiles of each map.',
    })[s],
  });
})();
</script>
