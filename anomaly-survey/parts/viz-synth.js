<script>
(() => {
  // ch-synthetic: CutPaste and DRAEM training samples made from TOY images with parts/lib-synthetic.js.
  // Checked in build/synth-sim.mjs: Perlin threshold 0.70 keeps 9-21% of the image (seeds 1-6), 0.60 keeps 22-41%;
  // a DRAEM fake on the mesh scores 1.70 under the chapter-1 nearest-neighbour score vs 0.08 for a good mesh.
  const K = SYNTH_KIT, fig = document.getElementById('viz-synth'), el = document.getElementById('cv-synth'), out = fig.querySelector('.readout');
  const $ = id => document.getElementById(id), N = K.N;
  const DEF = { mode: 'cutpaste', product: 'mesh', src: [24, 30], dst: [70, 66], area: .06, aspect: 1.6, scar: false, seed: 3, thr: .72, beta: .6 };
  const S = { ...DEF, fade: 1, rw: 0, rh: 0, ang: 0 };
  const base = { mesh: TOY.make('mesh', 41), cap: TOY.make('cap', 41) };
  const glow = PE.linkPulses(fig, () => view.redraw());
  const rect = () => S.scar ? { w: 4, h: 12, ang: .5 } : { w: Math.min(N - 4, Math.sqrt(S.area * N * N * S.aspect)), h: Math.min(N - 4, Math.sqrt(S.area * N * N / S.aspect)), ang: 0 };
  const cl = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
  let cache = {}, L = null;
  function data() {
    const key = [S.mode, S.product, S.src, S.dst, S.rw.toFixed(2), S.rh.toFixed(2), S.ang.toFixed(2), S.seed, S.thr, S.beta].join('|');
    if (cache.key === key) return cache;
    const I = base[S.product].px;
    if (S.mode === 'cutpaste') {
      const r = K.cutpaste(I, S.src, S.dst, S.rw, S.rh, S.ang, .08);
      cache = { key, I: { px: I }, Ia: { px: r.img }, mask: r.mask };
    } else {
      const m = K.perlinMask(S.seed, S.thr), A = K.texture(S.seed);
      cache = { key, I: { px: I }, A: { px: A }, mask: m, Ia: { px: K.draem(I, m, A, S.beta) } };
    }
    return cache;
  }
  const view = PE.canvas(el, w => w < 560 ? 1.2 : .37, (ctx, W, H) => draw(ctx, W, H));
  function layout(W, H) {
    const n = S.mode === 'draem' ? 4 : 3, narrow = W < 560, cols = narrow ? 2 : n, rows = Math.ceil(n / cols);
    // top clears the corner badge; panels are centred vertically when the row is shorter than the canvas.
    const gap = 18, top = 40, s = Math.min((W - gap * (cols + 1)) / cols, (H - top * rows - 30) / rows);
    const used = top + rows * s + (rows - 1) * (top + 10), oy = Math.max(0, (H - 10 - used) / 2);
    const P = []; for (let i = 0; i < n; i++) { const c = i % cols, r = Math.floor(i / cols); P.push({ x: gap + c * (s + gap), y: oy + top + r * (s + top + 10), s }); }
    return { P, s, narrow };
  }
  function outline(ctx, p, color, w) { ctx.save(); ctx.strokeStyle = color; ctx.lineWidth = w; ctx.strokeRect(p.x - 5, p.y - 5, p.s + 10, p.s + 10); ctx.restore(); }
  function rectPath(ctx, p, c, w, h, ang) {
    const k = p.s / N, cx = p.x + (c[0] + w / 2) * k, cy = p.y + (c[1] + h / 2) * k, co = Math.cos(ang), si = Math.sin(ang);
    ctx.beginPath(); [[-1, -1], [1, -1], [1, 1], [-1, 1]].forEach(([a, b], i) => { const u = a * w / 2 * k, v = b * h / 2 * k, X = cx + co * u - si * v, Y = cy + si * u + co * v; i ? ctx.lineTo(X, Y) : ctx.moveTo(X, Y); }); ctx.closePath();
  }
  function draw(ctx, W, H) {
    if (!W) return;
    L = layout(W, H); const D = data(), C = { blue: PE.css('--blue'), red: PE.css('--red'), yellow: PE.css('--yellow'), muted: PE.css('--muted'), fg: PE.css('--fg'), brown: PE.css('--brown'), rule: PE.css('--rule') };
    ctx.clearRect(0, 0, W, H); ctx.globalAlpha = S.fade;
    const P = L.P, lab = (i, t, c) => PE.label(ctx, t, P[i].x, P[i].y - 9, c || C.muted, 12);
    const maskImg = m => { const f = new Float32Array(N * N); for (let k = 0; k < N * N; k++) f[k] = m[k]; return TOY.heatBitmap(f, 0, 1, C.red); };
    if (S.mode === 'cutpaste') {
      TOY.draw(ctx, D.I, P[0].x, P[0].y, P[0].s); lab(0, 'good image x');
      ctx.save(); ctx.setLineDash([5, 4]); ctx.strokeStyle = C.blue; ctx.lineWidth = 2; rectPath(ctx, P[0], S.src, S.rw, S.rh, S.ang); ctx.stroke(); ctx.restore();
      TOY.draw(ctx, D.Ia, P[1].x, P[1].y, P[1].s); lab(1, 'CutPaste(x)');
      ctx.save(); ctx.strokeStyle = C.yellow; ctx.lineWidth = 2; rectPath(ctx, P[1], S.dst, S.rw, S.rh, S.ang); ctx.stroke(); ctx.restore();
      const p = P[2]; ctx.save(); ctx.strokeStyle = C.rule; ctx.strokeRect(p.x, p.y, p.s, p.s); ctx.restore(); lab(2, 'training label');
      const fs = Math.max(11, Math.min(15, p.s / 11)), lx = p.x + 12; let ly = p.y + fs * 2;
      PE.label(ctx, 'x', lx, ly, C.blue, fs); PE.label(ctx, 'class 0: normal', lx + fs * 2.6, ly, C.fg, fs); ly += fs * 2;
      PE.label(ctx, 'CutPaste(x)', lx, ly, C.yellow, fs); ly += fs * 1.4; PE.label(ctx, 'class 1: pasted', lx + fs * 2.6, ly, C.fg, fs); ly += fs * 2.2;
      ctx.save(); ctx.font = `${fs - 2}px system-ui, sans-serif`; ctx.fillStyle = C.muted;
      ['No mask: the classifier', 'only learns which image', 'was edited.'].forEach((t, i) => ctx.fillText(t, lx, ly + i * (fs + 2))); ctx.restore();
    } else {
      TOY.draw(ctx, D.I, P[0].x, P[0].y, P[0].s); lab(0, 'good image I');
      TOY.draw(ctx, D.A, P[1].x, P[1].y, P[1].s, { vf: false }); lab(1, 'foreign texture A', C.brown);
      ctx.save(); ctx.fillStyle = PE.css('--bg'); ctx.fillRect(P[2].x, P[2].y, P[2].s, P[2].s); ctx.drawImage(maskImg(D.mask), P[2].x, P[2].y, P[2].s, P[2].s); ctx.strokeStyle = C.rule; ctx.strokeRect(P[2].x, P[2].y, P[2].s, P[2].s); ctx.restore(); lab(2, 'mask M from Perlin noise', C.red);
      TOY.draw(ctx, D.Ia, P[3].x, P[3].y, P[3].s); lab(3, 'fake anomaly I_a', C.yellow);
      [['image', 0, C.blue], ['texture', 1, C.brown], ['mask', 2, C.red], ['beta', 3, C.yellow]].forEach(([w, i, c]) => { const g = glow(w); if (g > 0) outline(ctx, P[i], c, 1 + 3 * g); });
    }
    if (S.mode === 'cutpaste') { const g = glow('mask'); if (g > 0) outline(ctx, P[1], C.yellow, 1 + 3 * g); }
    ctx.globalAlpha = 1;
  }
  function say() {
    const D = data(), pct = (K.frac(D.mask) * 100).toFixed(0);
    out.textContent = S.mode === 'cutpaste'
      ? `CutPaste${S.scar ? ' (scar)' : ''}: a ${Math.round(S.rw)} by ${Math.round(S.rh)} pixel ${S.scar ? 'strip, rotated,' : 'patch'} (${pct}% of the image) copied from one place in the same image to another, slightly brightened. The classifier learns class 0 for the original and class 1 for the edited copy.`
      : `DRAEM: the threshold keeps ${pct}% of the Perlin noise as the mask; inside it the image is ${Math.round(S.beta * 100)}% foreign texture and ${Math.round((1 - S.beta) * 100)}% original (beta ${S.beta.toFixed(2)}). Targets: the clean image I for the reconstruction, the mask M for the segmentation.`;
  }
  function sync() {
    $('btn-synth-cutpaste').setAttribute('aria-pressed', S.mode === 'cutpaste'); $('btn-synth-draem').setAttribute('aria-pressed', S.mode === 'draem');
    $('btn-synth-mesh').setAttribute('aria-pressed', S.product === 'mesh'); $('btn-synth-cap').setAttribute('aria-pressed', S.product === 'cap');
    $('btn-synth-large').setAttribute('aria-pressed', !S.scar); $('btn-synth-scar').setAttribute('aria-pressed', S.scar);
    $('rng-synth-area').value = S.area * 100; $('out-synth-area').textContent = Math.round(S.area * 100) + '%';
    $('rng-synth-aspect').value = S.aspect; $('out-synth-aspect').textContent = S.aspect.toFixed(1);
    $('rng-synth-thr').value = S.thr; $('out-synth-thr').textContent = S.thr.toFixed(2);
    $('rng-synth-beta').value = S.beta; $('out-synth-beta').textContent = S.beta.toFixed(2);
  }
  function morphRect(done) {
    const a = { w: S.rw, h: S.rh, ang: S.ang }, b = rect();
    PE.animate(450, u => { S.rw = PE.lerp(a.w, b.w, u); S.rh = PE.lerp(a.h, b.h, u); S.ang = PE.lerp(a.ang, b.ang, u); clampRects(); view.redraw(); }, () => { Object.assign(S, { rw: b.w, rh: b.h, ang: b.ang }); clampRects(); view.redraw(); say(); done && done(); });
  }
  function clampRects() { const m = S.ang ? 6 : 0; S.src = [cl(S.src[0], m, N - S.rw - m), cl(S.src[1], m, N - S.rh - m)]; S.dst = [cl(S.dst[0], m, N - S.rw - m), cl(S.dst[1], m, N - S.rh - m)]; }
  function fadeIn() { PE.animate(400, u => { S.fade = u; view.redraw(); }, () => { S.fade = 1; view.redraw(); }); }
  function setMode(m) { if (S.mode === m) return; S.mode = m; sync(); fadeIn(); say(); }
  $('btn-synth-cutpaste').addEventListener('click', () => { setMode('cutpaste'); });
  $('btn-synth-draem').addEventListener('click', () => { setMode('draem'); });
  $('btn-synth-mesh').addEventListener('click', () => { S.product = 'mesh'; sync(); fadeIn(); say(); });
  $('btn-synth-cap').addEventListener('click', () => { S.product = 'cap'; sync(); fadeIn(); say(); });
  $('btn-synth-large').addEventListener('click', () => { setMode('cutpaste'); S.scar = false; sync(); morphRect(); });
  $('btn-synth-scar').addEventListener('click', () => { setMode('cutpaste'); S.scar = true; sync(); morphRect(); });
  $('rng-synth-area').addEventListener('input', e => { setMode('cutpaste'); S.scar = false; S.area = e.target.value / 100; const r = rect(); Object.assign(S, { rw: r.w, rh: r.h, ang: 0 }); clampRects(); sync(); view.redraw(); say(); });
  $('rng-synth-aspect').addEventListener('input', e => { setMode('cutpaste'); S.scar = false; S.aspect = +e.target.value; const r = rect(); Object.assign(S, { rw: r.w, rh: r.h, ang: 0 }); clampRects(); sync(); view.redraw(); say(); });
  $('rng-synth-thr').addEventListener('input', e => { setMode('draem'); S.thr = +e.target.value; sync(); view.redraw(); say(); });
  $('rng-synth-beta').addEventListener('input', e => { setMode('draem'); S.beta = +e.target.value; sync(); view.redraw(); say(); });
  // Cue-exempt: "New noise" samples a new Perlin field and texture on purpose; Reset restores the seed.
  $('btn-synth-noise').addEventListener('click', () => { setMode('draem'); S.seed++; fadeIn(); say(); });
  $('btn-synth-reset').addEventListener('click', () => { Object.assign(S, DEF); const r = rect(); Object.assign(S, { rw: r.w, rh: r.h, ang: 0 }); sync(); fadeIn(); say(); });
  // Drag: the dashed source rectangle in the first panel, the yellow target in the second. Arrow keys move the target.
  let drag = null;
  const hit = e => { if (!L || S.mode !== 'cutpaste') return null; const r = el.getBoundingClientRect(), x = e.clientX - r.left, y = e.clientY - r.top;
    for (const i of [0, 1]) { const p = L.P[i]; if (x >= p.x && x <= p.x + p.s && y >= p.y && y <= p.y + p.s) return { which: i ? 'dst' : 'src', u: (x - p.x) / p.s * N, v: (y - p.y) / p.s * N }; } return null; };
  el.addEventListener('pointerdown', e => { const h = hit(e); if (!h) return; drag = h.which; el.setPointerCapture(e.pointerId); S[drag] = [h.u - S.rw / 2, h.v - S.rh / 2]; clampRects(); view.redraw(); });
  el.addEventListener('pointermove', e => { if (!drag) return; const h = hit(e); if (!h || h.which !== drag) return; S[drag] = [h.u - S.rw / 2, h.v - S.rh / 2]; clampRects(); view.redraw(); });
  el.addEventListener('pointerup', () => { if (drag) { drag = null; say(); } });
  el.addEventListener('keydown', e => { const d = { ArrowLeft: [-3, 0], ArrowRight: [3, 0], ArrowUp: [0, -3], ArrowDown: [0, 3] }[e.key]; if (!d || S.mode !== 'cutpaste') return;
    e.preventDefault(); const k = e.shiftKey ? 'src' : 'dst'; S[k] = [S[k][0] + d[0], S[k][1] + d[1]]; clampRects(); view.redraw(); say(); });
  { const r = rect(); Object.assign(S, { rw: r.w, rh: r.h, ang: 0 }); }
  // DRAEM's Eq. 4 terms light DRAEM panels, so hovering one switches the figure to DRAEM.
  fig.addEventListener('pe-link', e => { if (['beta', 'image', 'texture'].includes(e.detail) && S.mode !== 'draem') setMode('draem'); });
  sync(); say();
})();
</script>
