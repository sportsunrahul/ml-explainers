/* 1. Overview: the hero motion. Patch embeddings rise, a block is hidden, the predictor's guess lands, the miss shrinks. */
SCENES.overview = ctx => {
  const el = mkCanvas(ctx.stage), G = 10, pts0 = [];
  const r = PE.rng(17), gs = PE.gauss(r);
  for (let i = 0; i < G; i++) for (let j = 0; j < G; j++) { const hid = i >= 3 && i < 6 && j >= 5 && j < 8; const p = [(j - 4.5) * .3 + gs() * .05, -1.6, (i - 4.5) * .3 + gs() * .05]; const e = [p[0] * 1.6 + gs() * .25, .4 + gs() * .5 + (hid ? .3 : 0), p[2] * 1.6 + gs() * .25]; pts0.push({ p, e, hid }); }
  const cen = mid(pts0.filter(q => q.hid).map(q => q.e)), guess0 = [cen[0] - 1.1, cen[1] - .5, cen[2] + .9];
  const s3 = PE3.scene(el, { aspect: S3ASPECT(ctx.stage), fov: 34, dist: 7.4, yaw: .5, pitch: .32, autoRotate: .08 });
  const sm = (a, b, u) => PE.clamp((u - a) / (b - a), 0, 1);
  let cancel = () => {};
  function frame(u) {
    const rise = sm(0, .3, u), hide = sm(.3, .5, u), land = sm(.5, .75, u), fix = sm(.75, 1, u);
    const items = [{ type: 'grid', y: -1.6, size: 3.2, step: .3, color: '--rule' }], vis = [], yel = [];
    pts0.forEach(q => { const p = q.p.map((v, k) => v + (q.e[k] - v) * rise); (q.hid ? yel : vis).push(p); });
    items.push({ type: 'points', pts: vis, color: '--blue', size: 5, alpha: .9 });
    items.push({ type: 'points', pts: yel, color: hide > 0 ? '--yellow' : '--blue', size: 5, alpha: hide > 0 ? .35 + .55 * hide : .9 });
    if (rise < 1) items.push({ type: 'label', at: [0, -1.75, 1.7], text: 'image patches', color: '--muted', dy: 14 });
    if (hide > 0) items.push({ type: 'label', at: cen, text: 'hidden targets', color: '--yellow', dy: -14, alpha: hide });
    if (rise >= 1) items.push({ type: 'label', at: mid(vis), text: 'context embeddings', color: '--blue', dy: 16 });
    if (land > 0) { const g = guess0.map((v, k) => v + (cen[k] - v) * fix), start = [-1.2, .3, -.6], tip = start.map((v, k) => v + (g[k] - v) * land);
      items.push({ type: 'line', pts: [start, tip], color: '--green', width: 3, arrow: true }, { type: 'points', pts: [tip], color: '--green', size: 8 });
      if (land >= 1) { items.push({ type: 'line', pts: [g, cen], color: '--red', width: 2.5, dash: [4, 3] }, { type: 'label', at: mid([g, cen]), text: fix < 1 ? 'the miss' : 'the miss, learned away', color: '--red', dy: -10, dx: 8 }, { type: 'label', at: tip, text: 'predicted', color: '--green', dy: 16 }); } }
    s3.items = items; s3.redraw();
    ctx.hud(badge('toy', u < .3 ? 'Each patch of the image rises to its embedding.' : u < .5 ? 'A block of patches is hidden: these are the targets.' : u < .75 ? 'The predictor guesses the target embeddings from the context embeddings.' : 'The loss is the distance between guess and target. Training shrinks it.'));
  }
  function play() { cancel(); if (PE3.instant()) { frame(1); return; } cancel = s3.tween(6000, frame); }
  play();
  return { replay: play, destroy: () => { cancel(); s3.destroy(); } };
};

/* 2. Where the loss lives: pixels vs two views vs embedding prediction, with a noise slider. */
SCENES.loss = ctx => {
  const el = mkCanvas(ctx.stage), slider = ctx.q('#loss-noise'), sv = ctx.q('#loss-noise-value');
  const S = { step: 0, noise: +slider.value, u: 1 };
  const N = 10, r0 = PE.rng(5), base = [];
  for (let i = 0; i < N; i++) { base.push([]); for (let j = 0; j < N; j++) { const sky = i < 4, tree = Math.hypot(i - 5.5, j - 6) < 2.4; base[i].push(sky ? [120, 170, 230] : tree ? [60, 130, 70] : [150, 120, 80]); } }
  const hid = { i0: 4, j0: 5, n: 3 };
  const noise = []; { const r = PE.rng(9), g = PE.gauss(r); for (let k = 0; k < 144; k++) noise.push(g()); }
  const cv = PE.canvas(el, S3ASPECT(ctx.stage), draw);
  function draw(c, W, H) {
    const C = { bg: PE.css('--bg'), fg: PE.css('--fg'), mu: PE.css('--muted'), rule: PE.css('--rule'), blue: PE.css('--blue'), yel: PE.css('--yellow'), green: PE.css('--green'), red: PE.css('--red'), brown: PE.css('--brown') };
    c.fillStyle = C.bg; c.fillRect(0, 0, W, H);
    // wide: image | boxes | result side by side. narrow: image and boxes on top, the result below.
    const narrow = W < 620, pad = 20, side = narrow ? Math.min(W * .38, H * .3) : Math.min(W * .34, H * .56), cs = side / N, ox = pad, oy = narrow ? pad + 14 : H / 2 - side / 2;
    for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) { const inH = i >= hid.i0 && i < hid.i0 + hid.n && j >= hid.j0 && j < hid.j0 + hid.n; const [R, G, B] = base[i][j];
      if (inH) { for (let a = 0; a < 2; a++) for (let b = 0; b < 2; b++) { const nz = noise[((i * N + j) * 4 + a * 2 + b) % 144] * 70 * S.noise; c.fillStyle = `rgb(${PE.clamp(R + nz, 0, 255)},${PE.clamp(G + nz, 0, 255)},${PE.clamp(B + nz, 0, 255)})`; c.fillRect(ox + j * cs + b * cs / 2, oy + i * cs + a * cs / 2, cs / 2 + .5, cs / 2 + .5); } }
      else { c.fillStyle = `rgb(${R},${G},${B})`; c.fillRect(ox + j * cs, oy + i * cs, cs + .5, cs + .5); } }
    c.save(); c.strokeStyle = C.rule; c.lineWidth = 1; c.globalAlpha = .5; for (let k = 0; k <= N; k++) { c.beginPath(); c.moveTo(ox + k * cs, oy); c.lineTo(ox + k * cs, oy + side); c.stroke(); c.beginPath(); c.moveTo(ox, oy + k * cs); c.lineTo(ox + side, oy + k * cs); c.stroke(); } c.restore();
    c.save(); c.strokeStyle = C.yel; c.lineWidth = 3; c.strokeRect(ox + hid.j0 * cs, oy + hid.i0 * cs, hid.n * cs, hid.n * cs); c.restore();
    PE.label(c, narrow ? 'input' : 'input, with a hidden block', ox, oy - 8, C.mu, 12);
    PE.label(c, narrow ? 'context' : 'visible = context', ox, oy + side + 16, C.blue, 11); PE.label(c, narrow ? 'target' : 'hidden = target', ox + side, oy + side + 16, C.yel, 11, 'right');
    const bw = narrow ? 76 : 88, bx = ox + side + (narrow ? 34 : 48) + bw / 2, by = oy + side / 2;
    const R = narrow ? { x: pad, y: oy + side + 44, w: W - 2 * pad, h: Math.max(60, H - (oy + side + 44) - 62) } : { x: bx + bw / 2 + 46, y: oy - 10, w: W - (bx + bw / 2 + 46) - pad, h: side + 20 };
    const barY = narrow ? H - 30 : oy + side + 40, barX = narrow ? pad : bx - bw / 2, barW = narrow ? W - 2 * pad : W - pad - (bx - bw / 2);
    const bar = (v, color, text) => { c.fillStyle = C.rule; c.fillRect(barX, barY, barW, 10); c.fillStyle = color; c.fillRect(barX, barY, barW * PE.clamp(v, 0, 1), 10); PE.label(c, text, barX, barY - 6, color, 11); };
    const pixLoss = .12 + .8 * S.noise, embLoss = .14, u = S.u;
    const toR = (x1, y1, tx, ty, color, lw = 1.5, a = .5) => PE.arrow(c, x1, y1, tx, ty, color, lw, 7, a);
    const boxOut = (yoff) => narrow ? [bx, by + yoff + 18] : [bx + bw / 2, by + yoff];
    if (S.step === 0) {
      PE.arrow(c, ox + side + 4, oy + side / 2, bx - bw / 2 - 2, by, C.blue, 2);
      PE.box(c, bx - bw / 2, by - 22, bw, 44, C.brown, 'encoder +\ndecoder');
      const pside = Math.min(R.h - 24, R.w * .5, side * .9), px = narrow ? R.x + R.w / 2 - pside / 2 : R.x, py = R.y + R.h / 2 - pside / 2 + 8, pcs = pside / hid.n;
      const o = boxOut(0); toR(o[0], o[1], narrow ? px + pside / 2 : px - 6, narrow ? py - 14 : by, C.brown, 2, 1);
      for (let i = 0; i < hid.n; i++) for (let j = 0; j < hid.n; j++) { const [Rr, G, B] = base[hid.i0 + i][hid.j0 + j]; c.fillStyle = `rgb(${Rr},${G},${B})`; c.fillRect(px + j * pcs, py + i * pcs, pcs + .5, pcs + .5);
        for (let a = 0; a < 2; a++) for (let b = 0; b < 2; b++) { const e = Math.abs(noise[(((hid.i0 + i) * N + hid.j0 + j) * 4 + a * 2 + b) % 144]) * S.noise; PE.dot(c, px + j * pcs + (b + .5) * pcs / 2, py + i * pcs + (a + .5) * pcs / 2, 2 + 4 * e, C.red, .35 + .5 * Math.min(1, e)); } }
      PE.label(c, 'painted block: error on every pixel', narrow ? R.x : px, py - 8, C.red, 11);
      bar(pixLoss, C.red, `pixel loss ${pixLoss.toFixed(2)}: grows with detail the model cannot know`);
    } else {
      const two = S.step === 1;
      PE.arrow(c, ox + side + 4, oy + side * .3, bx - bw / 2 - 2, by - 28, C.blue, 2); PE.arrow(c, ox + side + 4, oy + side * .7, bx - bw / 2 - 2, by + 28, two ? C.blue : C.yel, 2);
      PE.box(c, bx - bw / 2, by - 46, bw, 36, C.blue, two ? 'encoder' : 'context\nencoder'); PE.box(c, bx - bw / 2, by + 10, bw, 36, two ? C.blue : C.brown, two ? 'encoder' : 'target enc.\n(EMA)');
      if (!narrow) { PE.label(c, two ? 'view 1 (crop, color jitter)' : 'visible', ox + side + 6, oy + side * .3 - 8, two ? C.mu : C.blue, 10); PE.label(c, two ? 'view 2' : 'hidden', ox + side + 6, oy + side * .7 + 16, two ? C.mu : C.yel, 10); }
      c.save(); c.strokeStyle = C.rule; c.strokeRect(R.x, R.y, R.w, R.h); c.restore(); PE.label(c, 'embedding space', R.x + 6, R.y + 14, C.mu, 11);
      if (two) {
        const p1 = [R.x + R.w * (.3 + .15 * u), R.y + R.h * (.35 + .1 * u)], p2 = [R.x + R.w * (.7 - .15 * u), R.y + R.h * (.65 - .1 * u)];
        const a1 = boxOut(-28), a2 = boxOut(28); toR(a1[0], a1[1], p1[0] - 6, p1[1], C.blue); toR(a2[0], a2[1], p2[0] - 6, p2[1], C.blue);
        PE.polyline(c, [p1, p2], C.red, 2, [4, 3]); PE.dot(c, p1[0], p1[1], 7, C.blue); PE.dot(c, p2[0], p2[1], 7, C.blue);
        PE.label(c, 'pull together', (p1[0] + p2[0]) / 2 + 8, (p1[1] + p2[1]) / 2 - 6, C.red, 11);
        bar(.2 * (1 - u) + .05, C.red, narrow ? 'agreement loss: the trivial solution also reaches 0' : 'agreement loss: independent of the hidden detail, but the trivial solution also reaches 0');
      } else {
        const pc = [R.x + R.w * .28, R.y + R.h * .4], pt = [R.x + R.w * .72, R.y + R.h * .62], gpt = [pt[0] - 18 * (1 - u) - 8, pt[1] - 14 * (1 - u) - 6];
        const a1 = boxOut(-28), a2 = boxOut(28); toR(a1[0], a1[1], pc[0] - 8, pc[1], C.blue); toR(a2[0], a2[1], pt[0] - 8, pt[1], C.brown);
        PE.arrow(c, pc[0], pc[1], gpt[0], gpt[1], C.green, 3, 10); PE.polyline(c, [gpt, pt], C.red, 2, [4, 3]);
        PE.dot(c, pc[0], pc[1], 7, C.blue); PE.dot(c, pt[0], pt[1], 7, C.yel); PE.dot(c, gpt[0], gpt[1], 6, C.green);
        PE.label(c, 'predictor', (pc[0] + gpt[0]) / 2, (pc[1] + gpt[1]) / 2 - 10, C.green, 11, 'center'); PE.label(c, 'target', pt[0] + 10, pt[1] + 4, C.yel, 11); PE.label(c, 'loss', gpt[0] + 4, gpt[1] - 10, C.red, 11);
        bar(embLoss, C.red, narrow ? `embedding loss ${embLoss.toFixed(2)}: flat as noise grows` : `embedding loss ${embLoss.toFixed(2)}: flat, the target encoder drops what cannot be predicted`);
      }
    }
    if (S.step !== 1 && !narrow) PE.label(c, `at noise ${S.noise.toFixed(2)}:  pixel loss ${pixLoss.toFixed(2)}   embedding loss ${embLoss.toFixed(2)}`, barX, barY + 26, C.fg, 12);
  }
  const say = () => ctx.hud(badge('toy', ['Pixel reconstruction: the loss counts every pixel, so unpredictable detail costs as much as meaning.', 'Two views: embeddings of two augmentations are pulled together. No prediction target, and the constant embedding is a perfect solution too.', 'JEPA: predict the target embedding from the context embedding. The loss ignores detail the target encoder leaves out.'][S.step]));
  const offs = [listen(slider, 'input', () => { S.noise = +slider.value; sv.textContent = S.noise.toFixed(2); cv.redraw(); })];
  let cancel = () => {};
  function step(n) { S.step = n; cancel(); S.u = 0; cancel = PE.animate(900, u => { S.u = u; cv.redraw(); }); say(); }
  step(0);
  return { flat: true, step, replay: () => step(S.step), resize: cv.resize, destroy: () => { cancel(); offs.forEach(f => f()); cv.destroy(); } };
};

/* 3. I-JEPA: the paper's multi-block sampler on a 14 x 14 grid, then encode, predict, loss. */
SCENES.ijepa = ctx => {
  const el = mkCanvas(ctx.stage), G = 14, S = { strat: 'multi', seed: 1, grid: null, step: 0, u: 1 };
  const CFG = { multi: { nt: 4, ts: [.15, .2], ta: [.75, 1.5], cs: [.85, 1] }, tiny: { nt: 4, ts: [.0075, .01], ta: [1, 1], cs: [.85, 1] } };
  function block(r, scale, ar) { const n = Math.floor(G * G * (scale[0] + r() * (scale[1] - scale[0]))), a = ar[0] + r() * (ar[1] - ar[0]);
    const h = Math.min(G, Math.max(1, Math.round(Math.sqrt(n * a)))), w = Math.min(G, Math.max(1, Math.round(Math.sqrt(n / a))));
    return { top: Math.floor(r() * (G - h + 1)), left: Math.floor(r() * (G - w + 1)), h, w }; }
  function sample() { const r = PE.rng(S.seed * 7919 + G), g = []; for (let i = 0; i < G; i++) g.push(new Array(G).fill(0));  // 0 unused, 1 context, 2 target
    const paint = (b, v) => { for (let i = b.top; i < b.top + b.h; i++) for (let j = b.left; j < b.left + b.w; j++) g[i][j] = v; };
    const st = S.strat;
    if (st === 'multi' || st === 'tiny') { const c = CFG[st]; paint(block(r, c.cs, [1, 1]), 1); for (let k = 0; k < c.nt; k++) paint(block(r, c.ts, c.ta), 2); }
    else if (st === 'raster') { const q = Math.floor(r() * 4), h = G / 2; for (let i = 0; i < G; i++) for (let j = 0; j < G; j++) g[i][j] = ((i < h ? 0 : 2) + (j < h ? 0 : 1)) === q ? 1 : 2; }
    else { for (let i = 0; i < G; i++) for (let j = 0; j < G; j++) g[i][j] = r() < .6 ? 2 : 1; }
    S.grid = g; const jit = PE.rng(S.seed * 31), gs = PE.gauss(jit); S.jit = []; for (let k = 0; k < G * G; k++) S.jit.push([gs(), gs()]); }
  const cv = PE.canvas(el, S3ASPECT(ctx.stage), draw);
  function draw(c, W, H) {
    const C = { bg: PE.css('--bg'), fg: PE.css('--fg'), mu: PE.css('--muted'), rule: PE.css('--rule'), blue: PE.css('--blue'), yel: PE.css('--yellow'), green: PE.css('--green'), red: PE.css('--red'), brown: PE.css('--brown') };
    c.fillStyle = C.bg; c.fillRect(0, 0, W, H);
    const narrow = W < 640, pad = 22, side = Math.min(narrow ? (W - 3 * pad) / 2 : (W - 4 * pad) / 2.3, H - 90), cs = side / G, ox = pad, oy = Math.max(30, H / 2 - side / 2);
    let nc = 0, nt = 0; for (let i = 0; i < G; i++) for (let j = 0; j < G; j++) { const v = S.grid[i][j]; if (v === 1) nc++; if (v === 2) nt++; }
    const st = S.step, u = S.u;
    for (let i = 0; i < G; i++) for (let j = 0; j < G; j++) { const v = S.grid[i][j]; c.globalAlpha = v === 1 ? (st >= 1 ? .35 : .85) : v === 2 ? (st >= 1 ? .35 : .8) : .15; c.fillStyle = v === 1 ? C.blue : v === 2 ? C.yel : C.mu; c.fillRect(ox + j * cs + 1, oy + i * cs + 1, cs - 2, cs - 2); }
    c.globalAlpha = 1; PE.label(c, `image as ${G} x ${G} patches`, ox, oy - 8, C.mu, 12);
    PE.label(c, `${nc} context`, ox, oy + side + 16, C.blue, 11); PE.label(c, `${nt} targets`, ox + side, oy + side + 16, C.yel, 11, 'right');
    if (st === 0) { S.last = { nc, nt }; return; }
    // the embedding grid on the right: one vector per patch, drawn at the patch position with a little jitter
    const ex = ox + side + (narrow ? pad : pad * 2.2), ey = oy, es = side, ecs = es / G;
    c.save(); c.strokeStyle = C.rule; c.setLineDash([3, 3]); c.strokeRect(ex, ey, es, es); c.restore(); PE.label(c, 'embedding per patch', ex, ey - 8, C.mu, 12);
    const P = (i, j, k) => [ex + (j + .5) * ecs + S.jit[k][0] * ecs * .18, ey + (i + .5) * ecs + S.jit[k][1] * ecs * .18];
    const ctxPts = [], tgtPts = [];
    for (let i = 0; i < G; i++) for (let j = 0; j < G; j++) { const v = S.grid[i][j], k = i * G + j; if (!v) continue; const [px, py] = P(i, j, k); const sx = ox + (j + .5) * cs, sy = oy + (i + .5) * cs; const fl = st === 1 ? u : 1; const x = sx + (px - sx) * fl, y = sy + (py - sy) * fl;
      if (v === 1) { ctxPts.push([x, y]); PE.dot(c, x, y, Math.max(1.6, ecs * .22), C.blue, .9); } else { tgtPts.push([x, y, i, j, k]); PE.dot(c, x, y, Math.max(1.6, ecs * .22), C.yel, st >= 1 ? .9 : .3); } }
    if (st === 1) { PE.label(c, narrow ? 'context enc.' : 'context encoder (trained)', ex, ey + es + 16, C.blue, 11); PE.label(c, narrow ? 'target enc. (EMA)' : 'target encoder (EMA, sees all)', ex + es, ey + es + 16, C.brown, 11, 'right'); }
    if (st >= 2) {
      const cm = ctxPts.reduce((s, p) => [s[0] + p[0] / ctxPts.length, s[1] + p[1] / ctxPts.length], [0, 0]);
      const gpts = tgtPts.map(t => { const off = [S.jit[t[4]][1] * ecs * .9, S.jit[t[4]][0] * ecs * .9]; const sh = st === 3 ? 1 - .75 * u : 1; return [t[0] + off[0] * sh, t[1] + off[1] * sh, t]; });
      const show = st === 2 ? Math.floor(u * gpts.length) : gpts.length;
      gpts.slice(0, show).forEach((g, k) => { if (k % Math.max(1, Math.floor(gpts.length / 40)) === 0 || st === 3) PE.arrow(c, cm[0], cm[1], g[0], g[1], C.green, 1, 5, st === 3 ? .12 : .3); PE.ring(c, g[0], g[1], Math.max(2, ecs * .28), C.green, 1.5, .9);
        if (st === 3) PE.polyline(c, [[g[0], g[1]], [g[2][0], g[2][1]]], C.red, 1.5, null, .9); });
      PE.label(c, st === 2 ? (narrow ? 'predictor: one guess per target' : 'predictor: one guess per target patch, told where to predict') : (narrow ? `loss: squared miss, ${nt} targets` : `loss: mean squared miss over the ${nt} targets, shrinking`), ex, ey + es + 16, st === 2 ? C.green : C.red, 11);
    }
    S.last = { nc, nt };
  }
  const say = () => { const { nc, nt } = S.last, f = (nc / (G * G)).toFixed(2); const txt = [
    { multi: `Multi-block: ${nt} target patches in four blocks, ${nc} context patches after removing the overlap (context fraction ${f}; the paper measures 0.25 on average).`, tiny: 'Tiny targets: each block is about one patch. The predictor can fill it by interpolating neighbours, and accuracy collapses (Table 8).', raster: 'Rasterised: one quadrant predicts three. Much to predict, little to predict from: 15.5 (Table 6).', random: 'Random 60%: every hole has visible neighbours, so copying works: 17.6 (Table 6).' }[S.strat],
    'Encode: context patches go through the trained encoder (blue). The EMA target encoder sees the whole image and its outputs at the target patches are the targets (yellow).',
    'Predict: the predictor reads the context embeddings and one mask token per target position, and outputs a guess (green ring) for each target.',
    'Loss: the squared distance between each guess and its target, averaged. Gradients reach the context encoder and the predictor only.'][S.step];
    ctx.hud(badge('computed', txt)); };
  let cancel = () => {};
  function go(anim) { sample(); cancel(); if (anim) { S.u = 0; cancel = PE.animate(700, u => { S.u = u; cv.redraw(); }); } else { S.u = 1; cv.redraw(); } say(); }
  function step(n) { S.step = n; cancel(); S.u = 0; cancel = PE.animate(1100, u => { S.u = u; cv.redraw(); }); say(); }
  const sb = ctx.qa('[data-strat]'), offs = sb.map(b => listen(b, 'click', () => { S.strat = b.dataset.strat; press(sb, x => x === b); go(true); }));
  offs.push(listen(ctx.q('[data-act="sample"]'), 'click', () => { S.seed++; go(true); }));
  go(false);
  return { flat: true, step, replay: () => step(S.step), resize: cv.resize, destroy: () => { cancel(); offs.forEach(f => f()); cv.destroy(); } };
};

/* 4. Collapse: the deep linear self-distillation toy (Littwin et al. 2024), trained live. */
SCENES.collapse = ctx => {
  const el = mkCanvas(ctx.stage), d = 3, n = 256, eta = .1, eps = .3, STEPS = 600;
  const S = { sg: 1, pred: 1, ema: 0, L: 2, step: 0 };
  let W, V, Wt, X, Y, stopRun = () => {};
  const axes = [[.99, .1], [0, 1], [.6, .6]], rho = () => axes.map(([g, s]) => g / (1 + s * s));
  function data(seed) { const r = PE.rng(seed), gs = PE.gauss(r); X = []; Y = [];
    for (let k = 0; k < n; k++) { const x = [], y = []; for (let i = 0; i < d; i++) { const [g, s] = axes[i], u = gs(), u2 = g * u + Math.sqrt(1 - g * g) * gs(); x.push(u + s * gs()); y.push(u2 + s * gs()); } X.push(x); Y.push(y); } }
  const eye = () => [[1, 0, 0], [0, 1, 0], [0, 0, 1]], mul = (A, B) => A.map(r => B[0].map((_, j) => r.reduce((s, v, k) => s + v * B[k][j], 0))), mv = (A, v) => A.map(r => r.reduce((s, a, k) => s + a * v[k], 0)), T = A => A[0].map((_, j) => A.map(r => r[j]));
  function init(seed) { const r = PE.rng(seed + 11), orth = () => { const a = r() * 6.28, c = Math.cos(a), s = Math.sin(a); return [[c, -s, 0], [s, c, 0], [0, 0, 1]]; };
    W = []; for (let a = 0; a < S.L; a++) W.push(orth().map(row => row.map(v => v * Math.pow(eps, 1 / S.L) + (r() - .5) * .01)));
    V = orth().map(row => row.map(v => v * Math.pow(eps, 1 / S.L))); Wt = W.map(m => m.map(r => r.slice())); S.step = 0; }
  const prod = Ws => Ws.reduce((P, m) => mul(m, P), eye());
  function stepOnce() { const Wb = prod(W), Wtb = S.ema ? prod(Wt) : Wb, Vm = S.pred ? V : eye(); const GV = [[0, 0, 0], [0, 0, 0], [0, 0, 0]], GW = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
    for (let k = 0; k < n; k++) { const h = mv(Wb, X[k]), p = mv(Vm, h), t = mv(Wtb, Y[k]), e = p.map((v, i) => v - t[i]), Vte = mv(T(Vm), e);
      for (let i = 0; i < d; i++) for (let j = 0; j < d; j++) { GV[i][j] += e[i] * h[j] / n; GW[i][j] += (Vte[i] * X[k][j] - (S.sg ? 0 : e[i] * Y[k][j])) / n; } }
    for (let a = 0; a < S.L; a++) { const left = prod(W.slice(a + 1)), right = prod(W.slice(0, a)), Gm = mul(mul(T(left), GW), T(right)); W[a] = W[a].map((row, i) => row.map((v, j) => v - eta * Gm[i][j])); }
    if (S.pred) V = V.map((row, i) => row.map((v, j) => v - eta * GV[i][j]));
    if (S.ema) Wt = Wt.map((m, a) => m.map((row, i) => row.map((v, j) => .99 * v + .01 * W[a][i][j])));
    S.step++; }
  const s3 = PE3.scene(el, { aspect: S3ASPECT(ctx.stage), fov: 36, dist: 6, yaw: .7, pitch: .4 });
  function scene() { const Wb = prod(W), pts = X.map(x => mv(Wb, x)), wn = [0, 1, 2].map(i => Math.hypot(...Wb.map(r => r[i])));
    s3.items = [{ type: 'axes', len: 1.3, color: '--muted', labels: ['slow', 'noisy', 'mixed'] }, { type: 'points', pts: pts.map(p => [p[0], p[2], p[1]]), color: '--blue', size: 4.5, alpha: .85 },
      { type: 'line', pts: [[0, 0, 0], [wn[0] * 1.2, 0, 0]], color: '--yellow', width: 3 }, { type: 'line', pts: [[0, 0, 0], [0, 0, wn[1] * 1.2]], color: '--red', width: 3 }]; s3.redraw();
    const pred = rho().map(v => Math.pow(v, S.L)), collapsed = S.step && wn.every(v => v < .05);
    ctx.hud(badge('toy', `Step ${S.step}. Encoder weight per axis: slow ${wn[0].toFixed(2)} (theory ${pred[0].toFixed(2)}), noisy ${wn[1].toFixed(2)} (theory ${pred[1].toFixed(2)}), mixed ${wn[2].toFixed(2)} (theory ${pred[2].toFixed(2)}).` + (collapsed ? ' <b>Collapsed: every input maps to the same point.</b>' : '') + (!S.sg || !S.pred ? ' Theory values assume stop-gradient and a predictor.' : ''))); }
  function reset() { stopRun(); data(3); init(3); scene(); }
  function run() { stopRun(); if (PE.instant()) { while (S.step < STEPS) stepOnce(); scene(); return; } let id = 0; const tick = () => { for (let i = 0; i < 12 && S.step < STEPS; i++) stepOnce(); scene(); if (S.step < STEPS) id = requestAnimationFrame(tick); }; id = requestAnimationFrame(tick); stopRun = () => cancelAnimationFrame(id); }
  const kb = ctx.qa('[data-key]'), offs = kb.map(b => listen(b, 'click', () => { const k = b.dataset.key; S[k] = +b.dataset.val; press(kb.filter(x => x.dataset.key === k), x => x === b); reset(); }));
  offs.push(listen(ctx.q('[data-act="run"]'), 'click', run), listen(ctx.q('[data-act="reset"]'), 'click', () => { Object.assign(S, { sg: 1, pred: 1, ema: 0, L: 2 }); press(kb, x => +x.dataset.val === S[x.dataset.key]); reset(); }));
  reset();
  return { replay: run, destroy: () => { stopRun(); offs.forEach(f => f()); s3.destroy(); } };
};

/* 5. Video: the V-JEPA token block with tube masks sampled at the paper's sizes. */
SCENES.video = ctx => {
  const el = mkCanvas(ctx.stage), T = 8, G = 14, S = { mask: 'short', seed: 1 };
  function sample() { const r = PE.rng(S.seed * 131 + 7), m = []; for (let i = 0; i < G; i++) m.push(new Array(G).fill(0));
    const cfg = S.mask === 'long' ? [2, .7] : S.mask === 'frame' ? [1, .7] : [8, .15];
    for (let k = 0; k < cfg[0]; k++) { const nn = G * G * cfg[1], a = S.mask === 'frame' ? 1 : .75 + r() * .75, h = Math.min(G, Math.max(1, Math.round(Math.sqrt(nn * a)))), w = Math.min(G, Math.max(1, Math.round(Math.sqrt(nn / a)))), top = Math.floor(r() * (G - h + 1)), left = Math.floor(r() * (G - w + 1));
      for (let i = top; i < top + h; i++) for (let j = left; j < left + w; j++) m[i][j] = 1; }
    S.m = m; }
  const s3 = PE3.scene(el, { aspect: S3ASPECT(ctx.stage), fov: 34, dist: 7.2, yaw: .75, pitch: .42 });
  function build() { const items = [], sp = .17, cell = .15; let masked = 0; const frameOnly = S.mask === 'frame';
    for (let t = 0; t < T; t++) for (let i = 0; i < G; i++) for (let j = 0; j < G; j++) { const hidden = S.m[i][j] && (!frameOnly || t === 3); if (hidden) masked++;
      const x = (j - (G - 1) / 2) * sp, y = ((G - 1) / 2 - i) * sp, z = (t - (T - 1) / 2) * sp * 1.6;
      items.push({ type: 'quad', pts: [[x - cell / 2, y - cell / 2, z], [x + cell / 2, y - cell / 2, z], [x + cell / 2, y + cell / 2, z], [x - cell / 2, y + cell / 2, z]], fill: hidden ? '--yellow' : '--blue', alpha: hidden ? .9 : .22 }); }
    items.push({ type: 'label', at: [(G / 2) * sp + .3, -(G / 2) * sp - .1, 0], text: 'time ->', color: '--muted', dx: 6, dy: 14 }, { type: 'label', at: [0, (G / 2) * sp + .55, -(T / 2) * sp * 1.6], text: '14 x 14 tokens per time slot, 8 slots', color: '--muted', dy: -6 });
    s3.items = items; s3.redraw(); const tot = T * G * G, f = (100 * masked / tot).toFixed(0);
    ctx.hud(badge('computed', frameOnly ? `Ablation, not the paper's setting: one time slot hidden, ${masked} of ${tot} tokens (${f}%). Every hidden token has a visible copy one slot away, so the predictor can copy instead of modelling motion.` : `${S.mask === 'short' ? 'Short-range' : 'Long-range'} mask: ${masked} of ${tot} tokens hidden (${f}% of the clip), the same footprint in all 8 time slots. The paper's average over both mask types is about 90%.`)); }
  const mb = ctx.qa('[data-mask]'), offs = mb.map(b => listen(b, 'click', () => { S.mask = b.dataset.mask; press(mb, x => x === b); sample(); build(); }));
  offs.push(listen(ctx.q('[data-act="sample"]'), 'click', () => { S.seed++; sample(); build(); }));
  sample(); build();
  return { replay: () => { S.seed++; sample(); build(); }, destroy: () => { offs.forEach(f => f()); s3.destroy(); } };
};

/* 6. Planning: CEM over a toy latent dynamics, the algorithm of V-JEPA 2-AC on a hand-written world. */
SCENES.plan = ctx => {
  const el = mkCanvas(ctx.stage), hz = ctx.q('#plan-h'), hv = ctx.q('#plan-h-value');
  const GOAL = [1.3, .55, -.9], S = { z: [-1.3, .1, .9], seed: 1, mu: null, sig: null, cands: [], best: null, iter: 0 };
  const dyn = (z, a) => [z[0] + .32 * a[0] - .06 * z[2] * a[1], z[1] + .3 * a[1] + .04 * (z[0] * a[2]), z[2] + .32 * a[2] + .05 * z[1] * a[0]];
  const roll = (z, acts) => { const p = [z]; let c = z; for (const a of acts) { c = dyn(c, a); p.push(c); } return p; };
  const dist = (p, q) => Math.hypot(p[0] - q[0], p[1] - q[1], p[2] - q[2]), H = () => +hz.value;
  function initDist() { S.mu = Array.from({ length: H() }, () => [0, 0, 0]); S.sig = Array.from({ length: H() }, () => [1, 1, 1]); S.iter = 0; }
  function sample() { const r = PE.rng(S.seed++), g = PE.gauss(r); S.cands = [];
    for (let k = 0; k < 60; k++) { const acts = S.mu.map((m, t) => m.map((v, i) => PE.clamp(v + S.sig[t][i] * g(), -1.5, 1.5))), path = roll(S.z, acts); S.cands.push({ acts, path, e: dist(path[path.length - 1], GOAL) }); }
    S.cands.sort((a, b) => a.e - b.e); S.best = S.cands[0]; }
  function refine() { if (!S.cands.length) sample(); const el6 = S.cands.slice(0, 6);
    S.mu = S.mu.map((m, t) => m.map((_, i) => el6.reduce((s, c) => s + c.acts[t][i], 0) / 6));
    S.sig = S.sig.map((s, t) => s.map((_, i) => Math.max(.08, Math.sqrt(el6.reduce((q, c) => q + (c.acts[t][i] - S.mu[t][i]) ** 2, 0) / 6)))); S.iter++; sample(); }
  const s3 = PE3.scene(el, { aspect: S3ASPECT(ctx.stage), fov: 36, dist: 6.2, yaw: .55, pitch: .38 });
  function scene() { const items = [{ type: 'grid', y: -1.4, size: 3.6, step: .6, color: '--rule' }];
    S.cands.slice(1).forEach(c => items.push({ type: 'line', pts: c.path, color: '--green', width: 1, dash: [3, 4], alpha: .22 }));
    if (S.best) items.push({ type: 'line', pts: S.best.path, color: '--green', width: 3, arrow: true });
    items.push({ type: 'points', pts: [S.z], color: '--blue', size: 9 }, { type: 'label', at: S.z, text: 'current embedding', color: '--blue', dy: 18 }, { type: 'points', pts: [GOAL], color: '--yellow', size: 9 }, { type: 'label', at: GOAL, text: 'goal embedding', color: '--yellow', dy: -10 });
    if (S.best) items.push({ type: 'line', pts: [S.best.path[S.best.path.length - 1], GOAL], color: '--red', width: 1.5, dash: [4, 3] });
    s3.items = items; s3.redraw();
    ctx.hud(badge('toy', S.best ? `Horizon ${H()}, refinement ${S.iter}: the best of ${S.cands.length} imagined futures ends ${S.best.e.toFixed(2)} from the goal (red). Current distance to the goal: ${dist(S.z, GOAL).toFixed(2)}. The dynamics are hand-written; the search is the paper's CEM.` : 'Press "Imagine futures" to sample candidate action sequences and roll them through the world model.')); }
  const ensure = () => { if (!S.mu || S.mu.length !== H()) initDist(); };
  let busy = false;
  const offs = [listen(ctx.q('[data-act="sample"]'), 'click', () => { ensure(); sample(); scene(); }), listen(ctx.q('[data-act="refine"]'), 'click', () => { ensure(); refine(); scene(); }),
    listen(ctx.q('[data-act="step"]'), 'click', () => { if (busy) return; ensure(); if (!S.best) { for (let i = 0; i < 4; i++) refine(); } const from = S.z.slice(), to = dyn(S.z, S.best.acts[0]); initDist(); busy = true;
      s3.tween(500, u => { S.z = from.map((v, i) => v + (to[i] - v) * u); scene(); }, () => { S.z = to; for (let i = 0; i < 3; i++) refine(); scene(); busy = false; }); }),
    listen(hz, 'input', () => { hv.textContent = hz.value; initDist(); S.cands = []; S.best = null; scene(); }),
    listen(ctx.q('[data-act="reset"]'), 'click', () => { S.z = [-1.3, .1, .9]; S.seed = 1; hz.value = 3; hv.textContent = '3'; initDist(); S.cands = []; S.best = null; scene(); })];
  initDist(); scene();
  return { replay: () => { ensure(); sample(); scene(); }, destroy: () => { offs.forEach(f => f()); s3.destroy(); } };
};

/* 7. LeJEPA: SIGReg's Epps-Pulley test on one direction of a toy cloud (Algorithm 1 of the paper). */
SCENES.lejepa = ctx => {
  const el = mkCanvas(ctx.stage), N = 600, S = { cloud: 'stretched', dir: [1, 0, 0], dirName: 'x', seed: 5 };
  const base = PE3.gaussian(N, { scale: [1, 1, 1], seed: 11 });
  const shape = { stretched: p => [p[0] * 2.2, p[1] * .9, p[2] * .25], iso: p => p.slice(), clumpy: (p, k) => { const c = [[1.1, .3, .2], [-1, -.4, .5], [.1, .9, -1]][k % 3]; return [c[0] + p[0] * .35, c[1] + p[1] * .35, c[2] + p[2] * .35]; } };
  let pts = base.map((p, k) => shape.stretched(p, k));
  function ep(dir) { const nn = Math.hypot(...dir), a = dir.map(v => v / nn), u = pts.map(p => a[0] * p[0] + a[1] * p[1] + a[2] * p[2]);
    const t = Array.from({ length: 17 }, (_, k) => -5 + 10 * k / 16), dt = 10 / 16; const err = t.map(tk => { let re = 0, im = 0; for (const x of u) { re += Math.cos(tk * x); im += Math.sin(tk * x); } re /= u.length; im /= u.length; const phi = Math.exp(-tk * tk / 2); return ((re - phi) ** 2 + im * im) * phi; });
    let I = 0; for (let k = 0; k < 16; k++) I += (err[k] + err[k + 1]) / 2 * dt; return { T: I * u.length, Tn: I, u }; }
  const s3 = PE3.scene(el, { aspect: S3ASPECT(ctx.stage), fov: 36, dist: 7.4, yaw: .5, pitch: .3 });
  const hist = document.createElement('canvas'); hist.style.cssText = 'position:absolute;left:16px;top:16px;width:220px;height:130px;border-radius:8px;border:1px solid var(--line)'; hist.setAttribute('aria-hidden', 'true'); ctx.stage.appendChild(hist);
  function drawHist(u) { const r = devicePixelRatio || 1; hist.width = 220 * r; hist.height = 130 * r; const c = hist.getContext('2d'); c.setTransform(r, 0, 0, r, 0, 0);
    c.fillStyle = PE.css('--panel2'); c.fillRect(0, 0, 220, 130); const bins = new Array(30).fill(0); for (const x of u) { const b = Math.floor((x + 4) / 8 * 30); if (b >= 0 && b < 30) bins[b]++; }
    const mx = Math.max(...bins, 1); c.fillStyle = PE.css('--yellow'); bins.forEach((v, i) => { const h = 95 * v / mx; c.globalAlpha = .7; c.fillRect(14 + i * 6.4, 115 - h, 5.4, h); }); c.globalAlpha = 1;
    c.strokeStyle = PE.css('--fg'); c.lineWidth = 1.5; c.beginPath(); for (let i = 0; i <= 60; i++) { const x = -4 + 8 * i / 60, y = Math.exp(-x * x / 2) / Math.sqrt(2 * Math.PI) * u.length * (8 / 30), h = 95 * y / mx; i ? c.lineTo(14 + i * 3.2, 115 - h) : c.moveTo(14 + i * 3.2, 115 - h); } c.stroke();
    PE.label(c, 'projection (bars) vs N(0,1) (line)', 12, 14, PE.css('--muted'), 10); }
  function scene() { const r = ep(S.dir), a = S.dir.map(v => v / Math.hypot(...S.dir));
    s3.items = [{ type: 'grid', y: -2.2, size: 5, step: 1, color: '--rule' }, { type: 'points', pts, color: '--blue', size: 4, alpha: .8 }, { type: 'line', pts: [a.map(v => -2.6 * v), a.map(v => 2.6 * v)], color: '--yellow', width: 3, arrow: true }, { type: 'label', at: a.map(v => 2.7 * v), text: 'test direction', color: '--yellow', dy: -8 }]; s3.redraw();
    ctx.hud(badge('computed', `Direction ${S.dirName}: Epps-Pulley statistic ${r.Tn.toFixed(3)} per sample. ${r.Tn < .004 ? '<b>Passes:</b> the 1D projection looks standard Gaussian.' : '<b>Fails:</b> the projection is not standard Gaussian.'} ${S.cloud === 'iso' ? 'An isotropic Gaussian cloud passes in every direction.' : S.cloud === 'stretched' ? 'The stretched cloud passes only along its unit-variance axis.' : 'A clumpy cloud fails on most directions even though its overall variance is modest.'}`)); drawHist(r.u); }
  let cancel = () => {};
  const cb = ctx.qa('[data-cloud]'), db = ctx.qa('[data-dir]');
  function setCloud(k) { S.cloud = k; press(cb, b => b.dataset.cloud === k); const from = pts.map(p => p.slice()), to = base.map((p, i) => shape[k](p, i)); cancel(); cancel = s3.tween(700, u => { pts = from.map((p, i) => p.map((v, j) => v + (to[i][j] - v) * u)); scene(); }); }
  function setDir(k) { S.dirName = k; press(db, b => b.dataset.dir === k); if (k === 'rand') { const r = PE.rng(S.seed++), g = PE.gauss(r); S.dir = [g(), g(), g()]; } else S.dir = k === 'x' ? [1, 0, 0] : [0, 0, 1]; scene(); }
  const offs = [...cb.map(b => listen(b, 'click', () => setCloud(b.dataset.cloud))), ...db.map(b => listen(b, 'click', () => setDir(b.dataset.dir))), PE.onTheme(scene)];
  scene();
  return { replay: () => setDir('rand'), destroy: () => { cancel(); offs.forEach(f => f()); s3.destroy(); hist.remove(); } };
};

/* 8. The family: the arXiv scan (2026-10-08) and the lineage of the papers in this tour. */
SCENES.family = ctx => {
  const el = mkCanvas(ctx.stage), S = { view: 'months' };
  const SCAN = { months: ["2024-02", "2024-03", "2024-04", "2024-05", "2024-06", "2024-07", "2024-08", "2024-09", "2024-10", "2024-11", "2024-12", "2025-01", "2025-02", "2025-03", "2025-04", "2025-05", "2025-06", "2025-07", "2025-08", "2025-09", "2025-10", "2025-11", "2025-12", "2026-01", "2026-02", "2026-03", "2026-04", "2026-05", "2026-06", "2026-07", "2026-08", "2026-09", "2026-10"],
    counts: [1, 4, 1, 5, 3, 2, 3, 5, 7, 4, 3, 3, 6, 2, 6, 3, 8, 7, 2, 10, 8, 11, 10, 17, 26, 27, 20, 46, 53, 61, 69, 97, 31], before: 8,
    domains: [["world models, planning, robotics", 196], ["other", 73], ["video", 73], ["theory, analysis, recipes", 46], ["graphs, molecules, science", 32], ["audio, speech, music", 31], ["medical, bio-signals", 28], ["language, reasoning, LLMs", 24], ["time series, tabular, finance", 20], ["remote sensing, wireless", 16], ["images, general vision", 14], ["3D and point clouds", 11], ["generation", 5]] };
  // lineage: x = year (fractional), lane = 0 image, 1 video, 2 world model, 3 theory
  const NODES = [['BYOL / SimSiam', 2020.5, 0, 'blue'], ['data2vec', 2022.1, 0, 'blue'], ['LeCun position paper', 2022.5, 2, 'green'], ['I-JEPA', 2023.1, 0, 'blue'], ['IWM', 2024.2, 2, 'green'], ['V-JEPA', 2024.3, 1, 'yellow'], ['DINO-WM', 2024.85, 2, 'green'], ['Intuitive physics', 2025.1, 1, 'yellow'], ['V-JEPA 2 (+AC)', 2025.45, 1, 'yellow'], ['LeJEPA', 2025.85, 0, 'blue'], ['V-JEPA 2.1', 2026.2, 1, 'yellow'], ['LeWorldModel', 2026.25, 2, 'green'], ['H-JEPA', 2026.75, 2, 'green'], ['slow features', 2022.85, 3, 'red'], ['uniform prior', 2022.75, 3, 'red'], ['deep linear SD', 2024.5, 3, 'red'], ['recon. uninformative', 2024.15, 3, 'red']];
  const EDGES = [[0, 1], [1, 3], [2, 3], [3, 4], [3, 5], [5, 7], [5, 8], [8, 10], [3, 6], [6, 11], [11, 12], [8, 12], [3, 9], [13, 3], [14, 3], [15, 9], [16, 9], [2, 12]];
  const cv = PE.canvas(el, S3ASPECT(ctx.stage), draw);
  function draw(c, W, H) {
    const C = { bg: PE.css('--bg'), fg: PE.css('--fg'), mu: PE.css('--muted'), rule: PE.css('--rule'), blue: PE.css('--blue'), yel: PE.css('--yellow'), green: PE.css('--green'), red: PE.css('--red'), acc: PE.css('--accent') };
    c.fillStyle = C.bg; c.fillRect(0, 0, W, H); const narrow = W < 640;
    if (S.view === 'months') { const b = { x: 50, y: 40, w: W - 70, h: H - 190 }, m = SCAN.counts.length, A = PE.axes(c, b, [0, m], [0, 100], { yt: [0, 25, 50, 75, 100], yl: 'papers per month' });
      SCAN.counts.forEach((v, i) => { const x = A.X(i) + 2, w = Math.max(2, b.w / m - 4), last = i === m - 1; c.fillStyle = last ? C.mu : C.acc; c.globalAlpha = last ? .6 : .9; c.fillRect(x, A.Y(v), w, b.y + b.h - A.Y(v)); c.globalAlpha = 1;
        if (SCAN.months[i].endsWith('-01') || i === 0) PE.label(c, SCAN.months[i], x + w / 2, b.y + b.h + 14, C.mu, 11, 'center', 400); if (v >= 40 || last) PE.label(c, last ? v + ' (partial)' : v, x + w / 2, A.Y(v) - 5, C.fg, 10, 'center'); });
      PE.label(c, narrow ? 'papers per month of first arXiv version' : '569 unique papers by month of first arXiv version (8 before 2024-02 not shown); scanned 2026-10-08', b.x, 24, C.mu, 12, 'left', 400);
    } else if (S.view === 'domains') { const rows = SCAN.domains, lw = narrow ? 150 : 230, b = { x: lw + 10, y: 36, w: W - lw - 60, h: H - 170 }, rh = b.h / rows.length, mx = 200;
      PE.label(c, 'papers by domain (hand-assigned from titles and abstracts)', 16, 24, C.mu, 12, 'left', 400);
      rows.forEach(([name, v], i) => { const y = b.y + i * rh + rh * .2, h = rh * .6, w = b.w * v / mx; c.fillStyle = i === 0 ? C.green : C.acc; c.globalAlpha = .9; c.fillRect(b.x, y, w, h); c.globalAlpha = 1; PE.label(c, name, b.x - 8, y + h / 2 + 4, C.fg, narrow ? 10 : 12, 'right', 400); PE.label(c, v, b.x + w + 6, y + h / 2 + 4, C.fg, 11); });
    } else { const yrs = [2020, 2027], lanes = ['images', 'video', 'world models', 'theory'], b = { x: 30, y: 44, w: W - 60, h: H - 190 }, X = y => b.x + (y - yrs[0]) / (yrs[1] - yrs[0]) * b.w, Y = l => b.y + (l + .5) * b.h / 4;
      lanes.forEach((l, i) => { c.strokeStyle = C.rule; c.beginPath(); c.moveTo(b.x, Y(i)); c.lineTo(b.x + b.w, Y(i)); c.stroke(); PE.label(c, l, b.x, Y(i) - b.h / 8 + 12, C.mu, 11); });
      for (let y = 2020; y <= 2026; y++) PE.label(c, y, X(y + .5), b.y + b.h + 16, C.mu, 11, 'center', 400);
      const P = n => [X(NODES[n][1]), Y(NODES[n][2])];
      EDGES.forEach(([a, bb]) => { const p = P(a), q = P(bb); c.save(); c.strokeStyle = C.rule; c.lineWidth = 1.5; c.beginPath(); c.moveTo(p[0], p[1]); c.bezierCurveTo((p[0] + q[0]) / 2, p[1], (p[0] + q[0]) / 2, q[1], q[0], q[1]); c.stroke(); c.restore(); });
      NODES.forEach((nd, i) => { const [x, y] = P(i), col = { blue: C.blue, yellow: C.yel, green: C.green, red: C.red }[nd[3]]; PE.dot(c, x, y, 6, col); const up = i % 2 === 0; PE.label(c, nd[0], x, y + (up ? -11 : 20), C.fg, narrow ? 9 : 11, 'center'); });
      PE.label(c, 'lineage of the papers in this tour (position = first arXiv version)', b.x, 24, C.mu, 12, 'left', 400); }
  }
  const say = () => ctx.hud(badge('sourced', { months: 'Monthly counts from the arXiv scan. The last bar covers 2026-10-01 to 2026-10-08 only. The curve bends up in early 2026, after V-JEPA 2 and LeJEPA.', domains: 'One third of the field is world models, planning and robotics. "Other" collects papers whose abstract names no domain, mostly general method papers.', tree: 'Blue: the image line from BYOL through I-JEPA to LeJEPA. Yellow: the video line. Green: world models and planning. Red: the theory papers that explain why the recipe works.' }[S.view]));
  const vb = ctx.qa('[data-view]'), offs = vb.map(b => listen(b, 'click', () => { S.view = b.dataset.view; press(vb, x => x === b); cv.redraw(); say(); }));
  say();
  return { flat: true, resize: cv.resize, destroy: () => { offs.forEach(f => f()); cv.destroy(); } };
};
