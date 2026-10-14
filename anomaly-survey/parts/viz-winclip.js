<script>
(() => {
  // ch-winclip toy. A window's p(damaged) = logistic((mean |diagonal-edge feature| over its patches - M0) / SC): a fixed stand-in
  // for WinCLIP's two-class softmax, using no normal image of the product. Checked in build/toy-winclip.mjs (M0 0.6, SC 0.12):
  // mesh scratch 200 -> max small-window p 0.998, mid 0.976, whole image 0.036; good mesh max 0.030; pixel AUROC small 0.990,
  // whole image 0.500, all three 0.990; flagged area at p > 0.5: harmonic small 64 px vs arithmetic 640 px (mask 163 px).
  // On the cap the cue also fires on the rim (designed edges), which the chapter uses as the failure case.
  const fig = document.getElementById('viz-winclip'), el = document.getElementById('cv-winclip'), out = fig.querySelector('.readout');
  const { G, N, S: ST, P } = TOY, M0 = .6, SC = .12, SCALES = { small: 2, mid: 3, img: G };
  const sig = z => 1 / (1 + Math.exp(-z));
  const parts = { mesh: TOY.defect(TOY.make('mesh', 200), 'scratch', 200), cap: TOY.defect(TOY.make('cap', 200), 'scratch', 200) };
  const model = {};
  function build(part) {
    if (model[part]) return model[part];
    const f = TOY.features(parts[part]), d = []; for (let k = 0; k < G * G; k++) d.push(Math.abs(f[k * 3 + 1]));
    const W = {};
    for (const [name, k] of Object.entries(SCALES)) { W[name] = [];
      for (let j = 0; j + k <= G; j++) for (let i = 0; i + k <= G; i++) { let m = 0; for (let b = 0; b < k; b++) for (let a = 0; a < k; a++) m += d[(j + b) * G + i + a];
        m /= k * k; W[name].push({ i, j, k, m, p: sig((m - M0) / SC) }); } }
    return model[part] = { W, maps: {} };
  }
  const box = w => [w.i * ST, w.j * ST, (w.i + w.k - 1) * ST + P, (w.j + w.k - 1) * ST + P];
  function mapOf(ws, harm) {
    const acc = new Float64Array(N * N), cnt = new Float64Array(N * N);
    for (const w of ws) { const [x0, y0, x1, y1] = box(w);
      for (let y = y0; y < y1; y++) for (let x = x0; x < x1; x++) { acc[y * N + x] += harm ? 1 / Math.max(w.p, 1e-6) : w.p; cnt[y * N + x]++; } }
    const o = new Float32Array(N * N); for (let q = 0; q < N * N; q++) o[q] = cnt[q] ? (harm ? cnt[q] / acc[q] : acc[q] / cnt[q]) : 0; return o;
  }
  const order = scale => scale === 'all' ? ['small', 'mid', 'img'] : [scale];
  function fullMap(part, scale, harm) {
    const M = build(part), key = scale + harm; if (M.maps[key]) return M.maps[key];
    const ms = order(scale).map(s => mapOf(M.W[s], harm));
    if (ms.length === 1) return M.maps[key] = ms[0];
    const o = new Float32Array(N * N);
    for (let q = 0; q < N * N; q++) o[q] = harm ? 3 / ms.reduce((s, m) => s + 1 / Math.max(m[q], 1e-6), 0) : (ms[0][q] + ms[1][q] + ms[2][q]) / 3;
    return M.maps[key] = o;
  }
  function auroc(map, mask) {
    const pos = [], neg = []; for (let q = 0; q < N * N; q++) (mask[q] ? pos : neg).push(map[q]); neg.sort((a, b) => a - b);
    const lb = (v, le) => { let a = 0, b = neg.length; while (a < b) { const m = (a + b) >> 1; if (le ? neg[m] <= v : neg[m] < v) a = m + 1; else b = m; } return a; };
    let s = 0; for (const v of pos) { const l = lb(v, false), h = lb(v, true); s += l + .5 * (h - l); } return s / (pos.length * neg.length);
  }
  const DEF = { part: 'mesh', scale: 'small', harm: true };
  const S = { ...DEF, t: 1, sel: null };
  let stop = () => {}, L = null;
  const glow = PE.linkPulses(fig, () => view.redraw());
  const view = PE.canvas(el, w => w < 620 ? 1.32 : .44, (ctx, W, H) => draw(ctx, W, H));
  const seq = () => order(S.scale).flatMap(s => build(S.part).W[s]);
  const shown = () => { const all = seq(); return all.slice(0, Math.max(0, Math.round(S.t * all.length))); };
  const best = () => { const all = seq().filter(w => w.k !== G); const pool = all.length ? all : seq(); return pool.reduce((a, b) => b.p > a.p ? b : a); };
  const selected = () => { if (S.t < 1) { const s = shown(); return s[s.length - 1] || seq()[0]; } return S.sel || best(); };
  function layout(W, H) {
    const narrow = W < 620;
    if (narrow) { const s = Math.min(W * .58, H * .4); return { narrow, img: { x: (W - s) / 2, y: 30, s }, grid: { x: 24, y: 30 + s + 44, w: W * .5 - 24, h: H - s - 104 }, bars: { x: W * .58, y: 30 + s + 44, w: W * .38, h: H - s - 104 } }; }
    const s = Math.min(W * .34, H - 70);
    return { narrow, img: { x: 22, y: 40, s }, grid: { x: 22 + s + 46, y: 40, w: W * .26, h: s }, bars: { x: 22 + s + 46 + W * .26 + 40, y: 40, w: W - (22 + s + 46 + W * .26 + 40) - 22, h: s } };
  }
  function draw(ctx, W, H) {
    if (!W) return;
    L = layout(W, H);
    const C = { blue: PE.css('--blue'), red: PE.css('--red'), yellow: PE.css('--yellow'), muted: PE.css('--muted'), rule: PE.css('--rule'), fg: PE.css('--fg'), panel: PE.css('--panel2') };
    ctx.clearRect(0, 0, W, H);
    const img = parts[S.part], ws = shown();
    const map = S.t >= 1 ? fullMap(S.part, S.scale, S.harm) : (() => { const per = order(S.scale).map(s => mapOf(ws.filter(w => w.k === SCALES[s]), S.harm)).filter((m, i) => ws.some(w => w.k === SCALES[order(S.scale)[i]]));
      if (!per.length) return new Float32Array(N * N); if (per.length === 1) return per[0]; const o = new Float32Array(N * N);
      for (let q = 0; q < N * N; q++) o[q] = S.harm ? per.length / per.reduce((s, m) => s + 1 / Math.max(m[q], 1e-6), 0) : per.reduce((s, m) => s + m[q], 0) / per.length; return o; })();
    const gh = glow('harm');
    TOY.draw(ctx, img, L.img.x, L.img.y, L.img.s, { heat: TOY.heatBitmap(map, .02, .55, C.red), heatAlpha: Math.min(1, .85 + gh) });
    PE.label(ctx, L.narrow ? 'camera view' : 'camera view, window scores laid over it', L.img.x, L.img.y - 12, C.muted, 12);
    const w = selected(), sc = L.img.s / N, [x0, y0, x1, y1] = box(w), gw = glow('win');
    ctx.save(); ctx.strokeStyle = C.yellow; ctx.lineWidth = 2 + 2.5 * gw; ctx.strokeRect(L.img.x + x0 * sc, L.img.y + y0 * sc, (x1 - x0) * sc, (y1 - y0) * sc); ctx.restore();
    // window grid(s): one cell per window, red alpha = p(damaged)
    const grids = order(S.scale), gap = 14, gwid = (L.grid.w - gap * (grids.length - 1)) / grids.length;
    PE.label(ctx, 'one cell per window', L.grid.x, L.grid.y - 12, C.muted, 12);
    grids.forEach((s, gi) => {
      const n = G - SCALES[s] + 1, side = Math.min(gwid, L.grid.h - 22), cell = side / n, gx = L.grid.x + gi * (gwid + gap), gy = L.grid.y;
      ctx.save(); ctx.fillStyle = C.panel; ctx.fillRect(gx, gy, side, side); ctx.restore();
      const done = new Set(ws.filter(v => v.k === SCALES[s]).map(v => v.j * n + v.i));
      build(S.part).W[s].forEach(v => { if (!done.has(v.j * n + v.i)) return; ctx.save(); ctx.globalAlpha = .1 + .9 * v.p; ctx.fillStyle = C.red; ctx.fillRect(gx + v.i * cell + .5, gy + v.j * cell + .5, Math.max(1, cell - 1), Math.max(1, cell - 1)); ctx.restore(); });
      if (w.k === SCALES[s]) { ctx.save(); ctx.strokeStyle = C.yellow; ctx.lineWidth = 1.5 + 2 * gw; ctx.strokeRect(gx + w.i * cell, gy + w.j * cell, cell, cell); ctx.restore(); }
      PE.label(ctx, s === 'small' ? '2 x 2' : s === 'mid' ? '3 x 3' : 'whole', gx, gy + side + 15, C.muted, 11);
      if (gi === 0) L.cells = { gx, gy, cell, n, s };
    });
    // bars: flawless vs damaged for the selected window, and for the whole image
    const gt = glow('text'), gp = glow('prob');
    const barSet = (y, label, p) => {
      PE.label(ctx, label, L.bars.x, y, C.fg, 12);
      [['flawless', 1 - p, C.blue], ['damaged', p, C.red]].forEach(([nm, v, col], r) => {
        const by = y + 10 + r * 24, bw = Math.max(30, L.bars.w - 104);
        ctx.save(); ctx.fillStyle = C.panel; ctx.fillRect(L.bars.x + 64, by, bw, 14); ctx.globalAlpha = .55 + .45 * (r ? Math.max(gp, gt) : gt); ctx.fillStyle = col; ctx.fillRect(L.bars.x + 64, by, bw * v, 14); ctx.restore();
        PE.label(ctx, nm, L.bars.x, by + 11, col, 11); PE.label(ctx, v.toFixed(2), L.bars.x + 64 + bw + 8, by + 11, C.muted, 11);
      });
    };
    barSet(L.bars.y + 4, w.k === G ? 'whole image' : `selected window (${w.k} x ${w.k})`, w.p);
    const whole = build(S.part).W.img[0];
    barSet(L.bars.y + 92, 'whole image', whole.p);
    PE.label(ctx, `toy cue: edge energy ${w.m.toFixed(2)}`, L.bars.x, L.bars.y + 176, C.muted, 11);
  }
  function say() {
    const M = build(S.part), map = fullMap(S.part, S.scale, S.harm), au = auroc(map, parts[S.part].mask);
    const nWin = order(S.scale).reduce((s, k) => s + M.W[k].length, 0), top = Math.max(...order(S.scale).flatMap(k => M.W[k]).map(v => v.p)), whole = M.W.img[0].p;
    let flagged = 0, inMask = 0; for (let q = 0; q < N * N; q++) if (map[q] > .5) { flagged++; inMask += parts[S.part].mask[q]; }
    const scaleTxt = { small: 'Small windows (2 x 2 patches)', mid: 'Mid windows (3 x 3 patches)', img: 'The whole image as one window', all: 'All three scales' }[S.scale];
    let t = `${scaleTxt}: ${nWin} ${nWin === 1 ? 'window' : 'windows'}, highest p(damaged) ${top.toFixed(3)}. Whole-image p(damaged) ${whole.toFixed(3)}. ${S.harm ? 'Harmonic' : 'Arithmetic'} map: ${flagged} pixels above 0.5 (${inMask} inside the scratch), toy pixel AUROC ${au.toFixed(3)}.`;
    if (S.scale === 'img') t += ' One score for every pixel: no map at all.';
    if (S.part === 'cap') t += ' On the cap the cue also fires on the designed rim.';
    out.textContent = t;
  }
  const press = (ids, on) => ids.forEach(k => document.getElementById('btn-winclip-' + k).setAttribute('aria-pressed', k === on));
  function setPart(p) { S.part = p; S.sel = null; press(['mesh', 'cap'], p); view.redraw(); say(); }
  function setScale(s) { S.scale = s; S.sel = null; press(['small', 'mid', 'img', 'all'], s); view.redraw(); say(); }
  function setHarm(h) { S.harm = h; document.getElementById('btn-winclip-harm').setAttribute('aria-pressed', h); document.getElementById('btn-winclip-arith').setAttribute('aria-pressed', !h); view.redraw(); say(); }
  function slide() { stop(); S.sel = null; S.t = 0; stop = PE.animate(2400, u => { S.t = u; view.redraw(); }, () => { S.t = 1; view.redraw(); say(); }); }
  ['mesh', 'cap'].forEach(k => document.getElementById('btn-winclip-' + k).addEventListener('click', () => setPart(k)));
  ['small', 'mid', 'img', 'all'].forEach(k => document.getElementById('btn-winclip-' + k).addEventListener('click', () => setScale(k)));
  document.getElementById('btn-winclip-harm').addEventListener('click', () => setHarm(true));
  document.getElementById('btn-winclip-arith').addEventListener('click', () => setHarm(false));
  document.getElementById('btn-winclip-slide').addEventListener('click', slide);
  document.getElementById('btn-winclip-reset').addEventListener('click', () => { stop(); S.t = 1; S.sel = null; setPart(DEF.part); setScale(DEF.scale); setHarm(DEF.harm); });
  // Select a window: click the camera view (window whose top-left patch is nearest) or a grid cell; arrows move it.
  const pick = (i, j) => { const k = SCALES[order(S.scale)[0]], n = G - k + 1; i = PE.clamp(i, 0, n - 1); j = PE.clamp(j, 0, n - 1);
    S.sel = build(S.part).W[order(S.scale)[0]][j * n + i]; S.t = 1; view.redraw(); };
  el.addEventListener('click', e => { if (!L) return; const r = el.getBoundingClientRect(), x = e.clientX - r.left, y = e.clientY - r.top, k = SCALES[order(S.scale)[0]];
    if (x >= L.img.x && x <= L.img.x + L.img.s && y >= L.img.y && y <= L.img.y + L.img.s) { const px = (x - L.img.x) / L.img.s * N, py = (y - L.img.y) / L.img.s * N;
      pick(Math.round((px - (k * ST + ST) / 2) / ST), Math.round((py - (k * ST + ST) / 2) / ST)); }
    else if (L.cells && x >= L.cells.gx && y >= L.cells.gy && x < L.cells.gx + L.cells.cell * L.cells.n && y < L.cells.gy + L.cells.cell * L.cells.n) pick(Math.floor((x - L.cells.gx) / L.cells.cell), Math.floor((y - L.cells.gy) / L.cells.cell)); });
  el.addEventListener('keydown', e => { const k = { ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowUp: [0, -1], ArrowDown: [0, 1] }[e.key]; if (!k) return;
    e.preventDefault(); const w = selected(); if (w.k === G) return; pick(w.i + k[0], w.j + k[1]); });
  say();
})();
</script>
