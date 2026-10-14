<script>
(() => {
  // ch-patchcore: greedy k-center coreset on toy cap features (20 caps = 4,500 patches, all three toy features).
  // Checked in build/toy-patchcore.mjs and build/toy-patchcore-pair.mjs: coverage radius 0.993 at 1%, 0.306 at 10%,
  // 0.174 at 25% (random subsampling: 1.750 at 1%, 0.566 at 10%); test caps good 104 / scratched 200 score 0.34 / 0.86
  // at 10% (separated) and 1.01 / 0.98 at 1% (not separated); at 0.5% the scratched cap's worst patch is off the scratch.
  const fig = document.getElementById('viz-patchcore'), el3 = document.getElementById('cv-patchcore'), elS = document.getElementById('cv-patchcore-side');
  const out = fig.querySelector('.readout'), slider = document.getElementById('rng-patchcore-pct'), sOut = document.getElementById('out-patchcore-pct');
  const PCTS = [.5, 1, 2, 5, 10, 25, 50, 100], BTN = { 100: 7, 25: 5, 10: 4, 1: 1 };
  const B = TOY.bank('cap', 20, 1), n = B.length / 3;
  const d2 = (A, i, C, j) => (A[i * 3] - C[j * 3]) ** 2 + (A[i * 3 + 1] - C[j * 3 + 1]) ** 2 + (A[i * 3 + 2] - C[j * 3 + 2]) ** 2;
  // Greedy farthest-point order over the whole bank; rad[k] = coverage radius of the first k picks (k >= 1).
  const order = new Int32Array(n), rank = new Int32Array(n), rad = new Float32Array(n + 1);
  (() => {
    const r = TOY.rng(7), anchors = Array.from({ length: 10 }, () => Math.floor(r() * n)), md = new Float64Array(n);
    for (let i = 0; i < n; i++) { let s = 0; for (const a of anchors) s += Math.sqrt(d2(B, i, B, a)); md[i] = s / 10; }
    for (let k = 0; k < n; k++) {
      let bi = 0; for (let i = 1; i < n; i++) if (md[i] > md[bi]) bi = i;
      rad[k] = md[bi]; order[k] = bi; rank[bi] = k;
      for (let i = 0; i < n; i++) { const d = Math.sqrt(d2(B, i, B, bi)); if (d < md[i]) md[i] = d; }
    }
    rad[n] = 0;
  })();
  const lOf = idx => Math.max(1, Math.floor(n * PCTS[idx] / 100));
  const tests = { good: TOY.make('cap', 104), scratch: TOY.defect(TOY.make('cap', 200), 'scratch', 200) };
  const tf = { good: TOY.features(tests.good), scratch: TOY.features(tests.scratch) };
  const cache = {};
  function scored(test, c) {
    const key = test + c; if (cache[key]) return cache[key];
    const f = tf[test], s = new Float32Array(TOY.G * TOY.G), nnI = new Int32Array(TOY.G * TOY.G);
    for (let q = 0; q < s.length; q++) { let b = Infinity, bi = 0; for (let k = 0; k < c; k++) { const d = d2(f, q, B, order[k]); if (d < b) { b = d; bi = order[k]; } } s[q] = Math.sqrt(b); nnI[q] = bi; }
    let w = 0; for (let q = 1; q < s.length; q++) if (s[q] > s[w]) w = q;
    let fb = Infinity, fi = 0; for (let i = 0; i < n; i++) { const d = d2(f, w, B, i); if (d < fb) { fb = d; fi = i; } }
    let onDefect = false; const m = tests[test].mask;
    if (m) { const gi = w % TOY.G, gj = Math.floor(w / TOY.G); for (let y = gj * TOY.S; y < gj * TOY.S + TOY.P && !onDefect; y++) for (let x = gi * TOY.S; x < gi * TOY.S + TOY.P; x++) if (m[y * TOY.N + x]) { onDefect = true; break; } }
    return cache[key] = { s, worst: w, score: s[w], coreNN: nnI[w], fullNN: fi, fullD: Math.sqrt(fb), onDefect, map: TOY.toPixels(s) };
  }
  const scratchRef = scored('scratch', n).score;   // the scratch patch's distance to the full bank
  const SC = 1.3, Wp = (A, i) => [A[i * 3] / SC, A[i * 3 + 2] / SC, A[i * 3 + 1] / SC];
  const WB = Array.from({ length: n }, (_, i) => Wp(B, i));
  const DEF = { idx: 4, test: 'scratch' };
  const S = { idx: DEF.idx, c: lOf(DEF.idx), test: DEF.test, anim: false };
  let stop = () => {};

  const s3 = PE3.scene(el3, { aspect: w => w < 560 ? .95 : .56, yaw: .75, pitch: .42, dist: 7.4, items: [] });
  function items() {
    const kept = [], rest = [], k = PE.clamp((el3.clientWidth || 800) / 820, .5, 1);
    for (let i = 0; i < n; i++) (rank[i] < S.c ? kept : rest).push(WB[i]);
    const it = [
      { type: 'axes', len: 2.3, color: '--muted', labels: ['h vs v edges', 'brightness', 'diagonal edges'] },
      { type: 'points', pts: rest, color: '--blue', size: 1.4 * k, alpha: .16, link: 'bank' },
      { type: 'points', pts: kept, color: '--blue', size: 3.1 * k, alpha: .95, link: 'core' },
    ];
    if (S.c >= 1 && S.c < n) {
      const w = order[S.c]; let b = Infinity, bi = 0; for (let k = 0; k < S.c; k++) { const d = d2(B, w, B, order[k]); if (d < b) { b = d; bi = order[k]; } }
      it.push({ type: 'line', pts: [WB[w], WB[bi]], color: '--red', width: 2.6, link: 'radius' },
        { type: 'points', pts: [WB[w]], color: '--fg', size: 5 * k, link: 'pick' },
        { type: 'label', at: WB[w], text: `coverage radius ${rad[S.c].toFixed(2)}`, color: '--red', dx: 8, dy: 18 });
    }
    if (!S.anim) {
      const R = scored(S.test, S.c), t = Wp(tf[S.test], R.worst);
      it.push({ type: 'line', pts: [t, WB[R.fullNN]], color: '--blue', width: 1.6, dash: [4, 3], link: 'nn' },
        { type: 'line', pts: [t, WB[R.coreNN]], color: '--red', width: 2.6, link: 'nn' },
        { type: 'points', pts: [t], color: '--yellow', size: Math.max(4.5, 6.5 * k), link: 'test' },
        { type: 'label', at: t, text: 'worst test patch', color: '--yellow', dx: 10, dy: -12 });
    }
    return it;
  }
  function render3() { s3.items = items(); s3.redraw(); }
  let lastW = 0; new ResizeObserver(() => { if (el3.clientWidth !== lastW) { lastW = el3.clientWidth; render3(); } }).observe(el3);

  let heat = null, heatKey = '';
  const glow = PE.linkPulses(fig, () => side.redraw());
  const side = PE.canvas(elS, w => w < 560 ? 1.05 : .3, (ctx, W, H) => drawSide(ctx, W, H));
  function drawSide(ctx, W, H) {
    if (!W) return;
    const C = { blue: PE.css('--blue'), red: PE.css('--red'), yellow: PE.css('--yellow'), muted: PE.css('--muted'), fg: PE.css('--fg') };
    ctx.clearRect(0, 0, W, H);
    const narrow = W < 560, isz = narrow ? Math.min(W * .5, H * .4) : Math.min(H - 54, W * .22), ix = narrow ? (W - isz) / 2 : 18, iy = 28;
    const R = S.anim ? null : scored(S.test, S.c);
    const hk = S.test + S.c + C.red;
    if (R && hk !== heatKey) { heat = TOY.heatBitmap(R.map, .3, 1.1, C.red); heatKey = hk; }
    TOY.draw(ctx, tests[S.test], ix, iy, isz, { heat: R ? heat : null });
    PE.label(ctx, 'camera view', ix, iy - 10, C.muted, 12);
    if (R) {
      const sz = isz / TOY.N, w = R.worst, wx = ix + (w % TOY.G) * TOY.S * sz, wy = iy + Math.floor(w / TOY.G) * TOY.S * sz;
      ctx.save(); ctx.strokeStyle = C.yellow; ctx.lineWidth = 2 + 2 * glow('test'); ctx.strokeRect(wx, wy, TOY.P * sz, TOY.P * sz); ctx.restore();
      PE.label(ctx, `image score ${R.score.toFixed(2)}`, ix, iy + isz + 20, C.fg, 12);
    }
    // coverage radius against % kept
    const bx = narrow ? { x: 52, y: iy + isz + 46, w: W - 72, h: H - (iy + isz + 46) - 46 } : { x: ix + isz + 74, y: 18, w: W - (ix + isz + 74) - 18, h: H - 64 };
    const ymax = 1.6, A = PE.axes(ctx, bx, [.4, 100], [0, ymax], { logx: true, xt: [.5, 1, 2, 5, 10, 25, 50, 100], yt: [0, .5, 1, 1.5], xf: v => v + '%', yf: v => v.toFixed(1), xl: 'share of the bank kept (log scale)', yl: 'coverage radius', ylo: 36 });
    const pts = []; for (let k = 0; k <= 160; k++) { const pct = .4 * Math.pow(250, k / 160), l = Math.min(n - 1, Math.max(1, Math.floor(n * pct / 100))); pts.push([A.X(pct), A.Y(Math.min(ymax, rad[l]))]); }
    PE.polyline(ctx, pts, C.blue, 2);
    ctx.save(); ctx.setLineDash([5, 4]); ctx.strokeStyle = C.yellow; ctx.lineWidth = 1.4; ctx.beginPath(); ctx.moveTo(bx.x, A.Y(scratchRef)); ctx.lineTo(bx.x + bx.w, A.Y(scratchRef)); ctx.stroke(); ctx.restore();
    PE.label(ctx, "scratch patch's distance to the full bank", bx.x + bx.w - 4, A.Y(scratchRef) - 6, C.yellow, 11, 'right');
    const pc = Math.max(.4, 100 * S.c / n), rv = S.c >= n ? 0 : rad[S.c];
    PE.dot(ctx, A.X(pc), A.Y(Math.min(ymax, rv)), 5 + 3 * glow('radius'), C.red);
  }

  function say() {
    const R = scored(S.test, S.c), other = scored(S.test === 'good' ? 'scratch' : 'good', S.c), rv = S.c >= n ? 0 : rad[S.c];
    const g = S.test === 'good' ? R : other, sc = S.test === 'good' ? other : R, sep = sc.score > g.score;
    let t = `Keeping ${PCTS[S.idx]}% of the bank: ${S.c.toLocaleString('en-US')} of ${n.toLocaleString('en-US')} toy patches. Coverage radius ${rv.toFixed(2)}: every normal patch is within that distance of a kept one. `;
    t += `The ${S.test === 'good' ? 'good' : 'scratched'} cap's worst patch is ${R.score.toFixed(2)} from its nearest kept patch (${R.fullD.toFixed(2)} from the full bank)`;
    t += S.test === 'scratch' ? (R.onDefect ? ', and it lies on the scratch. ' : ', but it is not on the scratch: a coverage gap scores higher than the defect. ') : '. ';
    t += `Good cap ${g.score.toFixed(2)} against scratched cap ${sc.score.toFixed(2)}: ${sep ? 'separated' : 'not separated, the gaps are as large as the defect'}.`;
    out.textContent = t;
  }
  function settle() { S.anim = false; render3(); side.redraw(); say(); }
  function pressed() {
    Object.entries(BTN).forEach(([p, i]) => document.getElementById('btn-patchcore-pct-' + p).setAttribute('aria-pressed', String(S.idx === i)));
    document.getElementById('btn-patchcore-good').setAttribute('aria-pressed', String(S.test === 'good'));
    document.getElementById('btn-patchcore-scratch').setAttribute('aria-pressed', String(S.test === 'scratch'));
    slider.value = S.idx; sOut.textContent = PCTS[S.idx] + '%';
  }
  function setPct(idx) {
    stop(); const from = S.c, to = lOf(idx); S.idx = idx; pressed();
    if (from === to) { settle(); return; }
    S.anim = true;
    stop = PE.animate(600, u => { S.c = Math.round(from + (to - from) * u); render3(); side.redraw(); }, settle);
  }
  function run() {
    stop(); const to = lOf(S.idx); S.anim = true; S.c = 0;
    stop = PE.animate(2600, u => { S.c = Math.max(1, Math.round(to * u)); render3(); side.redraw(); }, () => { S.c = to; settle(); });
  }
  Object.entries(BTN).forEach(([p, i]) => document.getElementById('btn-patchcore-pct-' + p).addEventListener('click', () => setPct(i)));
  slider.addEventListener('input', () => setPct(+slider.value));
  document.getElementById('btn-patchcore-run').addEventListener('click', run);
  document.getElementById('btn-patchcore-good').addEventListener('click', () => { S.test = 'good'; pressed(); settle(); });
  document.getElementById('btn-patchcore-scratch').addEventListener('click', () => { S.test = 'scratch'; pressed(); settle(); });
  document.getElementById('btn-patchcore-reset').addEventListener('click', () => { stop(); S.test = DEF.test; S.idx = DEF.idx; S.c = lOf(DEF.idx); pressed(); s3.setView({ yaw: .75, pitch: .42, dist: 7.4 }, 600); settle(); });
  pressed(); settle();
})();
</script>
<script>
(() => {
  // ch-patchcore walkthrough. Bank sizes: bottle's 209 training images x 784 patches, floor rule of the official code.
  const V = { p25: { pct: '25%', n: '40,964' }, p10: { pct: '10%', n: '16,385' }, p1: { pct: '1%', n: '1,638' } };
  const SH = {
    img: () => '209 x 3 x 224 x 224', test: () => '3 x 224 x 224', bb: () => 'layer2 512 x 28 x 28|layer3 1024 x 14 x 14',
    feat: () => '784 x 1024 per image', bank: () => '163,856 x 1024', core: v => `${v.n} x 1024`, nn: () => '784 distances', score: () => '1 number + map' };
  ArchFlow.mount({
    prefix: 'pcflow', grid: { cols: 7, rows: 2 },
    boxes: {
      img: { c: 0, r: 0, lane: 'blue', title: ['normal', 'images'], glyph: 'photos', data: true },
      test: { c: 0, r: 1, lane: 'yellow', title: ['test', 'image'], glyph: 'photo', data: true },
      bb: { c: 1, r: .5, lane: 'mixed', title: ['WideResNet50', '(ImageNet)'], glyph: 'layers' },
      feat: { c: 2, r: .5, lane: 'blue', title: ['patch', 'features'], glyph: 'grid', cls: false },
      bank: { c: 3, r: 0, lane: 'blue', title: ['memory', 'bank'], glyph: 'layers' },
      core: { c: 4, r: 0, lane: 'blue', title: ['coreset'], glyph: 'codebook' },
      nn: { c: 5, r: 1, lane: 'yellow', title: ['nearest', 'neighbor'], glyph: 'vector' },
      score: { c: 6, r: .5, lane: 'red', title: ['anomaly', 'score + map'], glyph: 'unit' },
    },
    edges: [['img', 'bb'], ['test', 'bb'], ['bb', 'feat'], ['feat', 'bank'], ['bank', 'core'], ['core', 'nn'], ['feat', 'nn'], ['nn', 'score']],
    stages: [
      { key: 'images', show: ['img'] }, { key: 'backbone', show: ['bb'] }, { key: 'features', show: ['feat'] }, { key: 'bank', show: ['bank'] },
      { key: 'coreset', show: ['core'] }, { key: 'test', show: ['test'] }, { key: 'nn', show: ['nn'] }, { key: 'score', show: ['score'] },
      { key: 'backward', back: true }],
    variants: V, variant: 'p10',
    shape: (id, v) => SH[id](v),
    params: () => null,
    frozen: id => id === 'bb' || id === 'bank' || id === 'core',
    readout: (s, v) => ({
      overview: `Bottle category, ${v.pct} coreset: 209 normal images give 163,856 patch features of 1024 numbers; greedy selection keeps ${v.n}. A test image's 784 patches each look up their nearest kept vector.`,
      images: 'Step 1: 209 normal images, resized to 256 and center-cropped to 224 x 224.',
      backbone: 'Step 2: frozen WideResNet50, no gradients. Read block 2 (512 x 28 x 28) and block 3 (1024 x 14 x 14) per image.',
      features: 'Step 3: 3 x 3 neighborhood averaging, layer 3 upsampled to 28 x 28, each layer pooled to 1024 numbers, the two joined and pooled again to 1024 (official code): 784 x 1024 per image.',
      bank: 'Step 4: 209 x 784 = 163,856 vectors of 1024 numbers, about 671 MB as 32-bit floats.',
      coreset: `Step 5: greedy farthest-point selection in a 128-dimensional random projection; keep ${v.pct} = ${v.n} full-size vectors.`,
      test: 'Step 6: the test image takes the same frozen path: 784 x 1024.',
      nn: `Step 7: each of the 784 test patches finds its nearest of the ${v.n} kept vectors; the distances are the patch scores.`,
      score: 'Step 8: image score = the largest of the 784 distances; map = the 28 x 28 grid, upsampled to 224 x 224, Gaussian smoothing width 4.',
      backward: 'Backward: there is no loss, so there is no gradient. The backbone is frozen and the bank and coreset are stores filled by a forward pass and a search.',
    })[s],
  });
})();
</script>
