<script>
(() => {
  // ch-padim: one 2-d Gaussian per patch position of the toy cap (features: diagonal edges, brightness; eps = 0.01 as in the
  // paper). Checked in build/toy-padim2d.mjs and build/toy-padim-shape.mjs: hole AUROC 0.573 (position-free kNN) -> 0.950
  // (per position); at the hole position (79) the hole is Euclidean 2.63 / Mahalanobis 19.1 from the mean (normal max 0.26 /
  // 1.91); at the rim position (71) the Gaussian is long along edge direction and the probe one long sigma across it has
  // Euclidean 1.38 (normal max 1.91) but Mahalanobis 13.4 (normal max 1.39).
  const fig = document.getElementById('viz-padim'), el = document.getElementById('cv-padim'), pick = document.getElementById('cv-padim-pick');
  const stage = fig.querySelector('.padim-stage'), out = fig.querySelector('.readout');
  const G = TOY.G, G2 = G * G, NTR = 20, EPS = .01, H = 1.3, R = 3, POS = { hole: 79, rim: 71 };
  const f2 = f => { const o = []; for (let k = 0; k < G2; k++) o.push([f[k * 3 + 1], f[k * 3 + 2]]); return o; };
  const train = Array.from({ length: NTR }, (_, i) => f2(TOY.features(TOY.make('cap', 1 + i))));
  const gs = [];
  for (let q = 0; q < G2; q++) {
    const m = [0, 0]; train.forEach(f => { m[0] += f[q][0] / NTR; m[1] += f[q][1] / NTR; });
    let a = EPS, b = 0, d = EPS; train.forEach(f => { const u = f[q][0] - m[0], v = f[q][1] - m[1]; a += u * u / (NTR - 1); b += u * v / (NTR - 1); d += v * v / (NTR - 1); });
    const tr = a + d, det = a * d - b * b, disc = Math.sqrt(Math.max(0, tr * tr / 4 - det)), l1 = tr / 2 + disc, l2 = tr / 2 - disc;
    const e = Math.abs(b) > 1e-12 ? [l1 - d, b] : (a >= d ? [1, 0] : [0, 1]), n = Math.hypot(...e), e1 = [e[0] / n, e[1] / n];
    const L11 = Math.sqrt(a), L21 = b / L11, L22 = Math.sqrt(d - L21 * L21);
    gs.push({ m, a, b, d, det, l1, l2, e1, e2: [-e1[1], e1[0]], s: Math.sqrt(l1), chol: [L11, L21, L22] });
  }
  const maha = (q, p) => { const { m, a, b, d, det } = gs[q], u = p[0] - m[0], v = p[1] - m[1]; return Math.sqrt(Math.max(0, (d * u * u - 2 * b * u * v + a * v * v) / det)); };
  const eu = (q, p) => Math.hypot(p[0] - gs[q].m[0], p[1] - gs[q].m[1]);
  const tests = { hole: TOY.defect(TOY.make('cap', 207), 'hole', 207), good: TOY.make('cap', 104) };
  const tf = { hole: f2(TOY.features(tests.hole)), good: f2(TOY.features(tests.good)) };
  const maps = Object.fromEntries(Object.keys(tf).map(k => [k, TOY.toPixels(Float32Array.from(tf[k].map((p, q) => maha(q, p))))]));
  const normals = Array.from({ length: 20 }, (_, i) => f2(TOY.features(TOY.make('cap', 100 + i))));
  const nMax = gs.map((_, q) => Math.max(...normals.map(f => maha(q, f[q])))), nEu = gs.map((_, q) => Math.max(...normals.map(f => eu(q, f[q]))));
  const bank = train.flat();
  const knn = p => Math.min(...bank.map(m => Math.hypot(m[0] - p[0], m[1] - p[1])));
  const DEF = { test: 'hole', q: POS.hole, mode: 'maha', probe: null };
  const S = { ...DEF, h: 1, ring: 1, tp: null };
  const toWorld = (q, p) => [(p[0] - gs[q].m[0]) / gs[q].s, (p[1] - gs[q].m[1]) / gs[q].s];
  const toFeat = (q, w) => [gs[q].m[0] + w[0] * gs[q].s, gs[q].m[1] + w[1] * gs[q].s];
  const testFeat = () => S.probe || tf[S.test][S.q];
  const clampR = w => { const r = Math.hypot(w[0], w[1]); return r > R - .1 ? [w[0] * (R - .1) / r, w[1] * (R - .1) / r, true] : [w[0], w[1], false]; };
  const height = (q, X, Z) => { const g = gs[q], v = [X * g.s, Z * g.s], m2 = (g.d * v[0] * v[0] - 2 * g.b * v[0] * v[1] + g.a * v[1] * v[1]) / g.det; return H * Math.exp(-m2 / 2); };

  const s3 = PE3.scene(el, { aspect: w => w < 560 ? .95 : .7, fov: 36, dist: 10.5, yaw: .75, pitch: .62, items: [] });
  function build() {
    const q = S.q, g = gs[q], bg = PE.css('--bg'), blue = PE.css('--blue'), items = [];
    items.push({ type: 'grid', y: 0, size: 2 * R, step: 1, color: '--rule' });
    // flat quads (height ~0) are dropped so the floor grid stays visible around the bell
    items.push(...PE3.surface((X, Z) => S.h * height(q, X, Z), { x0: -R, x1: R, z0: -R, z1: R, nx: 30, nz: 30,
      fill: y => PE3.mix(bg, blue, .18 + .55 * y / H), link: 'cov', alpha: .9 }).filter(it => Math.max(...it.pts.map(p => p[1])) > .02 * H * Math.max(S.h, .05)));
    items.push({ type: 'points', pts: train.map(f => { const w = toWorld(q, f[q]); return [w[0], .01, w[1]]; }), color: '--blue', size: 4.5, link: 'normals' });
    items.push({ type: 'points', pts: [[0, .01, 0]], color: '--blue', size: 8, link: 'mean' });
    // rings: Mahalanobis ellipses or Euclidean circles at 1, 2, 3
    for (let k = 1; k <= 3; k++) {
      const pts = [];
      for (let t = 0; t <= 64; t++) { const c = Math.cos(t / 32 * Math.PI), sn = Math.sin(t / 32 * Math.PI);
        if (S.mode === 'maha') { const [L11, L21, L22] = g.chol, v = [k * L11 * c, k * (L21 * c + L22 * sn)]; const w = [v[0] / g.s, v[1] / g.s]; if (Math.hypot(...w) <= R) pts.push([w[0], .012, w[1]]); else pts.push(null); }
        else pts.push([k * c, .012, k * sn]); }
      // split at clipped points
      let run = []; pts.forEach(p => { if (p) run.push(p); else { if (run.length > 1) items.push({ type: 'line', pts: run, color: S.mode === 'maha' ? '--blue' : '--muted', width: 1.4, alpha: .8 * S.ring, link: 'eps' }); run = []; } });
      if (run.length > 1) items.push({ type: 'line', pts: run, color: S.mode === 'maha' ? '--blue' : '--muted', width: 1.4, alpha: .8 * S.ring, link: 'eps' });
    }
    const [tx, tz, off] = S.tp || clampR(toWorld(q, testFeat())), th = S.h * height(q, tx, tz);
    items.push({ type: 'line', pts: [[0, .015, 0], [tx, .015, tz]], color: '--red', width: 2.6, link: 'maha' });
    items.push({ type: 'line', pts: [[tx, 0, tz], [tx, Math.max(th, .02), tz]], color: '--yellow', width: 1.6, dash: [4, 3], link: 'test' });
    items.push({ type: 'points', pts: [[tx, .02, tz]], color: '--yellow', size: 10, link: 'test' });
    items.push({ type: 'label', at: [tx, .02, tz], text: off ? 'test patch (off the floor)' : (S.probe ? 'probe' : 'test patch'), color: '--yellow', dy: -14, align: 'center' });
    items.push({ type: 'label', at: [0, .02, 0], text: 'mean', color: '--blue', dy: 18, align: 'center' });
    items.push({ type: 'label', at: [R, 0, 0], text: 'diagonal edges', color: '--muted', align: 'right', dy: 16 });
    items.push({ type: 'label', at: [0, 0, R + .3], text: 'brightness', color: '--muted', align: 'center', dy: 14 });
    s3.items = items; s3.redraw();
  }
  // position picker: camera view with the Mahalanobis heat map
  let heats = {}, heatColor = '';
  const pv = PE.canvas(pick, 1.08, (ctx, W, Hh) => {
    if (!W) return; const red = PE.css('--red'); if (red !== heatColor) { heats = {}; heatColor = red; }
    heats[S.test] = heats[S.test] || TOY.heatBitmap(maps[S.test], 3, 16, red);
    ctx.clearRect(0, 0, W, Hh); const sz = W - 16, x0 = 8, y0 = 22;
    PE.label(ctx, 'click a patch position', x0, 13, PE.css('--muted'), 11);
    TOY.draw(ctx, tests[S.test], x0, y0, sz, { heat: heats[S.test] });
    const s = sz / TOY.N, q = S.q; ctx.save(); ctx.strokeStyle = PE.css('--yellow'); ctx.lineWidth = 2;
    ctx.strokeRect(x0 + (q % G) * TOY.S * s, y0 + Math.floor(q / G) * TOY.S * s, TOY.P * s, TOY.P * s); ctx.restore();
  });
  function say() {
    const q = S.q, p = testFeat(), e = eu(q, p), mh = maha(q, p), r = Math.floor(q / G) + 1, c = q % G + 1;
    let t = `Position row ${r}, column ${c}: a Gaussian fitted to ${NTR} good caps. `;
    t += S.probe ? `Probe patch: Euclidean distance ${e.toFixed(2)}, Mahalanobis distance ${mh.toFixed(1)}. `
                 : `Test patch (${S.test === 'hole' ? 'cap with a hole' : 'good cap'}): Euclidean distance ${e.toFixed(2)}, Mahalanobis distance ${mh.toFixed(1)}. `;
    t += `Good caps at this position reach at most ${nEu[q].toFixed(2)} and ${nMax[q].toFixed(1)}.`;
    if (!S.probe) t += ` Position-free nearest neighbor (chapter 4): ${knn(p).toFixed(2)}.`;
    out.textContent = t;
  }
  function pressed() {
    document.getElementById('btn-padim-hole').setAttribute('aria-pressed', !S.probe && S.test === 'hole');
    document.getElementById('btn-padim-good').setAttribute('aria-pressed', !S.probe && S.test === 'good');
    document.getElementById('btn-padim-poshole').setAttribute('aria-pressed', S.q === POS.hole && !S.probe);
    document.getElementById('btn-padim-posrim').setAttribute('aria-pressed', S.q === POS.rim && !S.probe);
    document.getElementById('btn-padim-fool').setAttribute('aria-pressed', !!S.probe);
    document.getElementById('btn-padim-maha').setAttribute('aria-pressed', S.mode === 'maha');
    document.getElementById('btn-padim-euclid').setAttribute('aria-pressed', S.mode === 'euclid');
  }
  let cancel = () => {};
  function setPos(q) { cancel(); const same = q === S.q; S.q = q; S.tp = null; pressed(); pv.redraw(); say();
    if (same) { build(); return; } cancel = s3.tween(650, u => { S.h = u; build(); }); }
  function moveTest(fn) { cancel(); const from = clampR(toWorld(S.q, testFeat())); fn(); const to = clampR(toWorld(S.q, testFeat())); pressed(); pv.redraw(); say();
    cancel = s3.tween(550, u => { S.tp = [from[0] + (to[0] - from[0]) * u, from[1] + (to[1] - from[1]) * u, to[2]]; build(); }, () => { S.tp = null; build(); }); }
  function setMode(m) { if (m === S.mode) return; cancel(); S.mode = m; pressed(); cancel = s3.tween(450, u => { S.ring = u; build(); }); }
  document.getElementById('btn-padim-hole').addEventListener('click', () => moveTest(() => { S.probe = null; S.test = 'hole'; }));
  document.getElementById('btn-padim-good').addEventListener('click', () => moveTest(() => { S.probe = null; S.test = 'good'; }));
  document.getElementById('btn-padim-poshole').addEventListener('click', () => { S.probe = null; setPos(POS.hole); });
  document.getElementById('btn-padim-posrim').addEventListener('click', () => { S.probe = null; setPos(POS.rim); });
  document.getElementById('btn-padim-fool').addEventListener('click', () => { const g = gs[POS.rim], pr = [g.m[0] + g.s * g.e2[0], g.m[1] + g.s * g.e2[1]];
    if (S.q === POS.rim) moveTest(() => { S.probe = pr; }); else { S.probe = pr; setPos(POS.rim); } });
  document.getElementById('btn-padim-maha').addEventListener('click', () => setMode('maha'));
  document.getElementById('btn-padim-euclid').addEventListener('click', () => setMode('euclid'));
  document.getElementById('btn-padim-reset').addEventListener('click', () => { S.test = DEF.test; S.probe = null; S.mode = DEF.mode; S.ring = 1; setPos(DEF.q); s3.setView({ yaw: .75, pitch: .62, dist: 10.5 }, 500); });
  // picker: click or arrow keys choose the position
  pick.addEventListener('pointerdown', e => { const r = pick.getBoundingClientRect(), sz = r.width - 16, s = sz / TOY.N, x = (e.clientX - r.left - 8) / s, y = (e.clientY - r.top - 22) / s;
    if (x < 0 || y < 0 || x > TOY.N || y > TOY.N) return; S.probe = null;
    setPos(PE.clamp(Math.round((y - TOY.P / 2) / TOY.S), 0, G - 1) * G + PE.clamp(Math.round((x - TOY.P / 2) / TOY.S), 0, G - 1)); });
  pick.addEventListener('keydown', e => { const d = { ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowUp: [0, -1], ArrowDown: [0, 1] }[e.key]; if (!d) return;
    e.preventDefault(); S.probe = null; setPos(PE.clamp(Math.floor(S.q / G) + d[1], 0, G - 1) * G + PE.clamp(S.q % G + d[0], 0, G - 1)); });
  // drag the yellow point on the floor (captured before pe3 starts an orbit)
  const unproject = (sx, sy, w0) => { let w = [...w0];
    for (let it = 0; it < 6; it++) { const p = s3.project([w[0], 0, w[1]]), px = s3.project([w[0] + .01, 0, w[1]]), pz = s3.project([w[0], 0, w[1] + .01]);
      const J = [[(px.x - p.x) / .01, (pz.x - p.x) / .01], [(px.y - p.y) / .01, (pz.y - p.y) / .01]], det = J[0][0] * J[1][1] - J[0][1] * J[1][0]; if (Math.abs(det) < 1e-6) break;
      const rx = sx - p.x, ry = sy - p.y; w = [w[0] + (J[1][1] * rx - J[0][1] * ry) / det, w[1] + (-J[1][0] * rx + J[0][0] * ry) / det]; }
    return clampR(w); };
  let drag = false;
  stage.addEventListener('pointerdown', e => { const r = el.getBoundingClientRect(), w = clampR(toWorld(S.q, testFeat())), p = s3.project([w[0], .02, w[1]]);
    if (Math.hypot(e.clientX - r.left - p.x, e.clientY - r.top - p.y) > 18) return;
    e.stopPropagation(); e.preventDefault(); drag = true; }, { capture: true });
  window.addEventListener('pointermove', e => { if (!drag) return; const r = el.getBoundingClientRect(), w0 = clampR(toWorld(S.q, testFeat())), w = unproject(e.clientX - r.left, e.clientY - r.top, w0);
    S.probe = toFeat(S.q, w); S.tp = null; build(); say(); pressed(); });
  window.addEventListener('pointerup', () => { drag = false; });
  matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => build());
  pressed(); build(); say();
})();
</script>
