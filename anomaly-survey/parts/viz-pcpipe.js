<script>
(() => {
  // Interlude: PatchCore fit and test, one stage at a time. Drawings use the toy cap; readout shapes are PatchCore's own
  // (224 crop, WRN50 blocks 2 and 3, p = 3, d = 1024, 784 patches per image, bank = images x 784, floor rule for coresets).
  const fig = document.getElementById('viz-pcpipe'), el = document.getElementById('cv-pcpipe'), out = fig.querySelector('.readout');
  const STAGES = ['images', 'backbone', 'aggregate', 'merge', 'features', 'bank', 'coreset', 'test', 'nn', 'score', 'map'];
  const CATS = { bottle: 209, screw: 320, hazelnut: 391, toothbrush: 60 };
  const fmt = v => v.toLocaleString('en-US');
  const caps = [1, 2, 3, 4, 5].map(s => TOY.make('cap', s));
  const testImg = TOY.defect(TOY.make('cap', 200), 'scratch', 200);
  const scores = TOY.nnScores(TOY.features(testImg), TOY.bank('cap', 20, 1));
  let top = 0; scores.forEach((v, q) => { if (v > scores[top]) top = q; });
  const pix = TOY.toPixels(scores);
  // toy greedy picks in 2D (two of the three toy features) for the coreset drawing
  const f2 = (() => { const f = TOY.features(caps[0]), o = []; for (let k = 0; k < f.length / 3; k++) o.push([f[k * 3 + 1], f[k * 3 + 2]]); return o; })();
  const picks = (() => { const n = f2.length, md = new Float64Array(n).fill(Infinity), ord = []; let cur = 0;
    for (let k = 0; k < 24; k++) { ord.push(cur); for (let i = 0; i < n; i++) md[i] = Math.min(md[i], Math.hypot(f2[i][0] - f2[cur][0], f2[i][1] - f2[cur][1]));
      let b = 0; for (let i = 1; i < n; i++) if (md[i] > md[b]) b = i; cur = b; } return ord; })();
  const S = { st: -1, cat: 'bottle', u: 1 };
  let stop = () => {}, heatS = null, heatK = '';
  const view = PE.canvas(el, w => w < 560 ? 1.15 : .44, (ctx, W, H) => draw(ctx, W, H));

  const ease = u => u < 0 ? 0 : u > 1 ? 1 : u * u * (3 - 2 * u);
  function grid(ctx, x, y, s, cells, col, a = 1, lw = .6) {
    ctx.save(); ctx.globalAlpha = a; ctx.strokeStyle = col; ctx.lineWidth = lw; ctx.beginPath();
    for (let k = 0; k <= cells; k++) { ctx.moveTo(x + k * s / cells, y); ctx.lineTo(x + k * s / cells, y + s); ctx.moveTo(x, y + k * s / cells); ctx.lineTo(x + s, y + k * s / cells); }
    ctx.stroke(); ctx.restore();
  }
  function planes(ctx, x, y, s, cells, depth, col, label, C) {
    for (let k = depth - 1; k >= 0; k--) { const o = k * 5; ctx.save(); ctx.fillStyle = PE.mix(C.bgh, C.panel2h, .6); ctx.globalAlpha = .9; ctx.fillRect(x + o, y - o, s, s); ctx.restore(); grid(ctx, x + o, y - o, s, cells, col, k ? .35 : .9); }
    PE.label(ctx, label, x + s / 2, y + s + 16, C.fg, 11, 'center');
  }
  function slots(W, H, k) {  // k positions along the main axis, with a common size
    const narrow = W < 560;
    if (narrow) { const s = Math.min(W * .5, (H - 40) / k - 34); return Array.from({ length: k }, (_, i) => ({ x: (W - s) / 2, y: 18 + i * (H - 20) / k, s })); }
    const s = Math.min(H - 70, (W - 40) / k - 60); return Array.from({ length: k }, (_, i) => ({ x: 20 + i * (W - 40) / k + ((W - 40) / k - s) / 2, y: (H - s) / 2 - 8, s }));
  }
  function link(ctx, a, b, W, C) { const narrow = W < 560;
    if (narrow) PE.arrow(ctx, a.x + a.s / 2, a.y + a.s + 22, b.x + b.s / 2, b.y - 6, C.muted, 1.5, 7);
    else PE.arrow(ctx, a.x + a.s + 10, a.y + a.s / 2, b.x - 10, b.y + b.s / 2, C.muted, 1.5, 7); }

  function draw(ctx, W, H) {
    if (!W) return;
    const C = { blue: PE.css('--blue'), red: PE.css('--red'), yellow: PE.css('--yellow'), brown: PE.css('--brown'), muted: PE.css('--muted'), fg: PE.css('--fg'), green: PE.css('--green'),
      bgh: PE.css('--bg'), panel2h: PE.css('--panel2') };
    ctx.clearRect(0, 0, W, H);
    const st = STAGES[S.st] || 'overview', u = ease(S.u), n = CATS[S.cat], bank = n * 784;
    if (st === 'overview') {
      const sl = slots(W, H, 4);
      TOY.draw(ctx, caps[0], sl[0].x, sl[0].y, sl[0].s); PE.label(ctx, 'normal images', sl[0].x + sl[0].s / 2, sl[0].y + sl[0].s + 18, C.blue, 12, 'center');
      planes(ctx, sl[1].x, sl[1].y + sl[1].s * .15, sl[1].s * .7, 28, 3, C.brown, 'frozen features', C);
      stackDraw(ctx, sl[2], 1, C); PE.label(ctx, 'memory bank', sl[2].x + sl[2].s / 2, sl[2].y + sl[2].s + 18, C.brown, 12, 'center');
      TOY.draw(ctx, testImg, sl[3].x, sl[3].y, sl[3].s); PE.label(ctx, 'test image', sl[3].x + sl[3].s / 2, sl[3].y + sl[3].s + 18, C.yellow, 12, 'center');
      for (let i = 0; i < 3; i++) link(ctx, sl[i], sl[i + 1], W, C);
      return;
    }
    if (st === 'images') {
      const sl = slots(W, H, 3), a = sl[0], s = a.s * .8;
      caps.forEach((c, i) => { const o = (4 - i) * 10 * u; TOY.draw(ctx, c, a.x + o, a.y + a.s * .1 - o * .6, s, { vf: i === 4 }); });
      PE.label(ctx, `${n} normal images`, a.x + a.s / 2, a.y + a.s + 18, C.blue, 12, 'center');
      link(ctx, a, sl[1], W, C);
      const b = sl[1]; ctx.save(); ctx.strokeStyle = C.muted; ctx.setLineDash([4, 3]); ctx.strokeRect(b.x + b.s * .12, b.y + b.s * .12, b.s * .76, b.s * .76); ctx.restore();
      TOY.draw(ctx, caps[4], b.x + b.s * .12, b.y + b.s * .12, b.s * .76 * (1 - .2 * u) + b.s * .76 * .2 * u, { vf: false });
      PE.label(ctx, 'resize 256, crop 224', b.x + b.s / 2, b.y + b.s + 18, C.muted, 12, 'center');
      link(ctx, b, sl[2], W, C);
      const c = sl[2]; TOY.draw(ctx, caps[4], c.x + c.s * .1, c.y + c.s * .1, c.s * .8); PE.label(ctx, '3 x 224 x 224', c.x + c.s / 2, c.y + c.s + 18, C.fg, 12, 'center');
    } else if (st === 'backbone') {
      const sl = slots(W, H, 3);
      TOY.draw(ctx, caps[4], sl[0].x + sl[0].s * .1, sl[0].y + sl[0].s * .1, sl[0].s * .8); PE.label(ctx, '3 x 224 x 224', sl[0].x + sl[0].s / 2, sl[0].y + sl[0].s + 18, C.fg, 12, 'center');
      link(ctx, sl[0], sl[1], W, C);
      const b = sl[1]; ctx.save(); ctx.fillStyle = PE.mix(C.bgh, C.brown, .2); ctx.strokeStyle = C.brown; ctx.setLineDash([4, 3]); ctx.lineWidth = 1.4;
      ctx.beginPath(); ctx.moveTo(b.x, b.y + b.s * .1); ctx.lineTo(b.x + b.s, b.y + b.s * .3); ctx.lineTo(b.x + b.s, b.y + b.s * .7); ctx.lineTo(b.x, b.y + b.s * .9); ctx.closePath(); ctx.fill(); ctx.stroke(); ctx.restore();
      PE.label(ctx, 'WideResNet50', b.x + b.s / 2, b.y + b.s / 2 - 4, C.brown, 12, 'center'); PE.label(ctx, 'frozen, no gradient', b.x + b.s / 2, b.y + b.s / 2 + 12, C.brown, 11, 'center');
      link(ctx, sl[1], sl[2], W, C);
      const c = sl[2], s2 = c.s * .5;
      ctx.save(); ctx.globalAlpha = u; planes(ctx, c.x, c.y + 6, s2, 28, 3, C.blue, '512 x 28 x 28', C); planes(ctx, c.x + c.s * .55, c.y + c.s * .3, s2 * .8, 14, 4, C.blue, '1024 x 14 x 14', C); ctx.restore();
      PE.label(ctx, 'block 2', c.x + s2 / 2, c.y - 10, C.muted, 11, 'center'); PE.label(ctx, 'block 3', c.x + c.s * .55 + s2 * .4, c.y + c.s * .3 - 24, C.muted, 11, 'center');
    } else if (st === 'aggregate') {
      const sl = slots(W, H, 2), a = sl[0], cells = 14, cs = a.s / cells;
      grid(ctx, a.x, a.y, a.s, cells, C.blue, .8);
      const path = Math.floor(u * 30), gx = 2 + path % 10, gy = 3 + Math.floor(path / 10) * 3;
      ctx.save(); ctx.fillStyle = C.yellow; ctx.globalAlpha = .22; ctx.fillRect(a.x + (gx - 1) * cs, a.y + (gy - 1) * cs, 3 * cs, 3 * cs); ctx.restore();
      ctx.save(); ctx.strokeStyle = C.yellow; ctx.lineWidth = 2; ctx.strokeRect(a.x + (gx - 1) * cs, a.y + (gy - 1) * cs, 3 * cs, 3 * cs); ctx.fillStyle = C.yellow; ctx.fillRect(a.x + gx * cs + 2, a.y + gy * cs + 2, cs - 4, cs - 4); ctx.restore();
      PE.label(ctx, 'one feature map (drawn 14 x 14)', a.x + a.s / 2, a.y + a.s + 18, C.muted, 12, 'center');
      link(ctx, a, sl[1], W, C);
      const b = sl[1]; for (let k = 0; k < 9; k++) { const x = b.x + (k % 3) * b.s * .2, y = b.y + Math.floor(k / 3) * b.s * .2; ctx.save(); ctx.strokeStyle = C.yellow; ctx.globalAlpha = .5; ctx.strokeRect(x, y, b.s * .18, b.s * .18); ctx.restore(); }
      PE.arrow(ctx, b.x + b.s * .62, b.y + b.s * .3, b.x + b.s * .78, b.y + b.s * .3, C.muted, 1.4, 6);
      ctx.save(); ctx.fillStyle = C.yellow; ctx.fillRect(b.x + b.s * .8, b.y + b.s * .21, b.s * .18, b.s * .18); ctx.restore();
      PE.label(ctx, '3 x 3 neighbors', b.x + b.s * .3, b.y + b.s * .75, C.fg, 12, 'center'); PE.label(ctx, 'averaged', b.x + b.s * .89, b.y + b.s * .52, C.fg, 12, 'center');
      PE.label(ctx, 'stride 1: every cell', b.x + b.s / 2, b.y + b.s + 18, C.muted, 12, 'center');
    } else if (st === 'merge') {
      const sl = slots(W, H, 3), a = sl[0], b = sl[1], s3 = a.s * .55 + a.s * .45 * u;
      grid(ctx, a.x, a.y + (a.s - s3) / 2, s3, 14, C.blue, .8); PE.label(ctx, 'block 3: 14 x 14, upsampled', a.x + a.s / 2, a.y + a.s + 18, C.muted, 12, 'center');
      grid(ctx, b.x, b.y, b.s, 28, C.blue, .7, .4); PE.label(ctx, 'block 2: 28 x 28', b.x + b.s / 2, b.y + b.s + 18, C.muted, 12, 'center');
      link(ctx, b, sl[2], W, C);
      const c = sl[2], bar = (x, y, w, col, lab) => { ctx.save(); ctx.fillStyle = col; ctx.globalAlpha = .85; ctx.fillRect(x, y, w, 12); ctx.restore(); PE.label(ctx, lab, x, y - 5, C.fg, 11); };
      // bar widths follow vector lengths (9,216 = full width), pooled down to 1024 (drawn at .28 so it stays readable)
      bar(c.x, c.y + c.s * .12, c.s * PE.lerp(.5, .28, u), C.blue, u < 1 ? 'block 2: 512 x 9 = 4,608' : 'block 2: pooled to 1024');
      bar(c.x, c.y + c.s * .38, c.s * PE.lerp(1, .28, u), C.blue, u < 1 ? 'block 3: 1024 x 9 = 9,216' : 'block 3: pooled to 1024');
      ctx.save(); ctx.globalAlpha = u; bar(c.x, c.y + c.s * .72, c.s * .28, C.blue, 'joined, pooled: 1024'); ctx.restore();
      PE.label(ctx, 'no learned weights', c.x + c.s / 2, c.y + c.s + 18, C.muted, 12, 'center');
    } else if (st === 'features') {
      const sl = slots(W, H, 2), a = sl[0];
      TOY.draw(ctx, caps[4], a.x, a.y, a.s); grid(ctx, a.x, a.y, a.s, 28, C.fg, .25, .4);
      PE.label(ctx, '28 x 28 patches', a.x + a.s / 2, a.y + a.s + 18, C.muted, 12, 'center');
      link(ctx, a, sl[1], W, C);
      const b = sl[1], rows = Math.round(18 * u) + 2;
      for (let k = 0; k < rows; k++) { ctx.save(); ctx.fillStyle = C.blue; ctx.globalAlpha = .25 + .5 * ((k * 37) % 7) / 7; ctx.fillRect(b.x, b.y + k * b.s / 20, b.s, b.s / 20 - 2); ctx.restore(); }
      PE.label(ctx, '784 x 1024', b.x + b.s / 2, b.y + b.s + 18, C.fg, 12, 'center');
    } else if (st === 'bank') {
      const sl = slots(W, H, 2);
      caps.slice(0, 3).forEach((c, i) => TOY.draw(ctx, c, sl[0].x + i * 14, sl[0].y + i * 10, sl[0].s * .75, { vf: false }));
      PE.label(ctx, `${n} images x 784 patches`, sl[0].x + sl[0].s / 2, sl[0].y + sl[0].s + 18, C.blue, 12, 'center');
      link(ctx, sl[0], sl[1], W, C);
      stackDraw(ctx, sl[1], u, C); PE.label(ctx, `${fmt(bank)} x 1024`, sl[1].x + sl[1].s / 2, sl[1].y + sl[1].s + 18, C.brown, 12, 'center');
    } else if (st === 'coreset') {
      const sl = slots(W, H, 2), a = sl[0];
      const lo = [Math.min(...f2.map(p => p[0])), Math.min(...f2.map(p => p[1]))], hi = [Math.max(...f2.map(p => p[0])), Math.max(...f2.map(p => p[1]))];
      const X = p => a.x + (p[0] - lo[0]) / (hi[0] - lo[0] || 1) * a.s, Y = p => a.y + a.s - (p[1] - lo[1]) / (hi[1] - lo[1] || 1) * a.s;
      f2.forEach(p => PE.dot(ctx, X(p), Y(p), 1.6, C.blue, .35));
      const k = Math.max(1, Math.round(u * picks.length));
      picks.slice(0, k).forEach((i, j) => PE.dot(ctx, X(f2[i]), Y(f2[i]), 3.6, j === k - 1 && u < 1 ? C.fg : C.blue));
      PE.label(ctx, 'farthest point first (toy)', a.x + a.s / 2, a.y + a.s + 18, C.muted, 12, 'center');
      link(ctx, a, sl[1], W, C);
      stackDraw(ctx, sl[1], 1 - .9 * u, C); PE.label(ctx, `10%: ${fmt(Math.floor(bank * .1))} x 1024`, sl[1].x + sl[1].s / 2, sl[1].y + sl[1].s + 18, C.brown, 12, 'center');
    } else if (st === 'test') {
      const sl = slots(W, H, 3);
      TOY.draw(ctx, testImg, sl[0].x, sl[0].y, sl[0].s); PE.label(ctx, 'test image', sl[0].x + sl[0].s / 2, sl[0].y + sl[0].s + 18, C.yellow, 12, 'center');
      link(ctx, sl[0], sl[1], W, C);
      const b = sl[1]; ctx.save(); ctx.strokeStyle = C.brown; ctx.setLineDash([4, 3]); ctx.strokeRect(b.x + b.s * .1, b.y + b.s * .25, b.s * .8, b.s * .5); ctx.restore();
      PE.label(ctx, 'same frozen path', b.x + b.s / 2, b.y + b.s / 2 + 4, C.brown, 12, 'center');
      link(ctx, sl[1], sl[2], W, C);
      const c = sl[2]; ctx.save(); ctx.globalAlpha = u; grid(ctx, c.x, c.y, c.s, 28, C.yellow, .6, .4); ctx.restore(); PE.label(ctx, '784 x 1024', c.x + c.s / 2, c.y + c.s + 18, C.fg, 12, 'center');
    } else if (st === 'nn' || st === 'score') {
      const sl = slots(W, H, 2), a = sl[0], g = TOY.G, cs = a.s / g, mx = scores[top], shown = st === 'nn' ? Math.floor(u * g * g) : g * g;
      for (let q = 0; q < shown; q++) { const v = PE.clamp((scores[q] - .05) / (mx - .05), 0, 1);
        ctx.save(); ctx.fillStyle = PE.mix(C.bgh, C.red, .08 + .92 * v * v); ctx.fillRect(a.x + (q % g) * cs, a.y + Math.floor(q / g) * cs, cs - 1, cs - 1); ctx.restore(); }
      PE.label(ctx, 'patch scores (toy 15 x 15)', a.x + a.s / 2, a.y + a.s + 18, C.muted, 12, 'center');
      link(ctx, a, sl[1], W, C);
      const b = sl[1];
      if (st === 'nn') { for (let k = 0; k < 14; k++) { ctx.save(); ctx.fillStyle = C.brown; ctx.globalAlpha = .55; ctx.fillRect(b.x + b.s * .2, b.y + k * b.s / 15, b.s * .6, b.s / 15 - 2); ctx.restore(); }
        PE.label(ctx, 'coreset: nearest vector', b.x + b.s / 2, b.y + b.s + 18, C.brown, 12, 'center'); }
      else { const tx = a.x + (top % g + .5) * cs, ty = a.y + (Math.floor(top / g) + .5) * cs;
        ctx.save(); ctx.strokeStyle = C.yellow; ctx.lineWidth = 2.2; ctx.strokeRect(a.x + (top % g) * cs - 1, a.y + Math.floor(top / g) * cs - 1, cs + 1, cs + 1); ctx.restore();
        ctx.save(); ctx.globalAlpha = u; PE.label(ctx, 'max', b.x + b.s / 2, b.y + b.s * .38, C.muted, 13, 'center'); PE.label(ctx, `toy image score ${mx.toFixed(2)}`, b.x + b.s / 2, b.y + b.s * .55, C.red, 15, 'center');
        PE.label(ctx, '(paper: then Eq. 7)', b.x + b.s / 2, b.y + b.s * .7, C.muted, 11, 'center'); ctx.restore();
        PE.dot(ctx, tx, ty, 3, C.yellow); }
    } else if (st === 'map') {
      const sl = slots(W, H, 2);
      const g = TOY.G, a = sl[0], cs = a.s / g, mx = scores[top];
      for (let q = 0; q < g * g; q++) { const v = PE.clamp((scores[q] - .05) / (mx - .05), 0, 1); ctx.save(); ctx.fillStyle = PE.mix(C.bgh, C.red, .08 + .92 * v * v); ctx.fillRect(a.x + (q % g) * cs, a.y + Math.floor(q / g) * cs, cs - 1, cs - 1); ctx.restore(); }
      PE.label(ctx, 'score grid', a.x + a.s / 2, a.y + a.s + 18, C.muted, 12, 'center');
      link(ctx, a, sl[1], W, C);
      if (heatK !== C.red) { heatS = TOY.heatBitmap(pix, .05, mx, C.red); heatK = C.red; }
      TOY.draw(ctx, testImg, sl[1].x, sl[1].y, sl[1].s, { heat: heatS, heatAlpha: u });
      PE.label(ctx, 'upsampled, Gaussian width 4', sl[1].x + sl[1].s / 2, sl[1].y + sl[1].s + 18, C.red, 12, 'center');
    }
  }
  function stackDraw(ctx, b, frac, C) {
    const total = 18, k = Math.max(1, Math.round(total * frac));
    for (let i = 0; i < total; i++) { ctx.save(); ctx.fillStyle = C.brown; ctx.globalAlpha = i < k ? .7 : .08; ctx.fillRect(b.x + b.s * .15, b.y + i * b.s / total, b.s * .7, b.s / total - 2); ctx.restore(); }
  }
  function say() {
    const n = CATS[S.cat], bank = n * 784, p = pc => fmt(Math.floor(bank * pc)), mb = v => (v * 4096 / 1e6).toFixed(1);
    const st = STAGES[S.st] || 'overview';
    out.textContent = ({
      overview: `${S.cat}: ${n} training images. Fit stores ${fmt(bank)} patch vectors of 1024 numbers; a 10% coreset keeps ${p(.1)}.`,
      images: `Stage 1: ${n} images, each 3 x 224 x 224 after resize to 256 and center crop.`,
      backbone: 'Stage 2: frozen WideResNet50; block 2 gives 512 x 28 x 28, block 3 gives 1024 x 14 x 14 per image.',
      aggregate: 'Stage 3: each cell averaged over its 3 x 3 neighborhood, stride 1. Unfolded lengths: 512 x 9 = 4,608 and 1024 x 9 = 9,216.',
      merge: 'Stage 4: block 3 upsampled 14 x 14 to 28 x 28; each layer pooled to 1024, then the pair pooled to 1024 (code-only).',
      features: 'Stage 5: 28 x 28 = 784 patch vectors per image: 784 x 1024.',
      bank: `Stage 6: ${n} x 784 = ${fmt(bank)} vectors, ${mb(bank)} MB as 32-bit floats.`,
      coreset: `Stage 7: greedy selection in a 128-dimensional projection. 25%: ${p(.25)}, 10%: ${p(.1)} (${mb(Math.floor(bank * .1))} MB), 1%: ${p(.01)} vectors.`,
      test: 'Stage 8: the test image takes the same path: 784 x 1024.',
      nn: `Stage 9: 784 nearest-neighbor searches against the coreset (${p(.1)} vectors at 10%); 784 distances.`,
      score: 'Stage 10: image score = the largest of the 784 distances (official code); the paper adds the Eq. 7 reweighting.',
      map: 'Stage 11: 28 x 28 scores, bilinear to 224 x 224, Gaussian smoothing of width 4 pixels.',
    })[st];
  }
  function show() {
    fig.querySelectorAll('.cf-text > div').forEach(d => { d.hidden = d.dataset.stage !== (STAGES[S.st] || 'overview'); });
    STAGES.forEach((k, i) => document.getElementById('btn-pcpipe-' + k).setAttribute('aria-pressed', String(i === S.st)));
    Object.keys(CATS).forEach(k => document.getElementById('btn-pcpipe-' + k).setAttribute('aria-pressed', String(k === S.cat)));
    say();
  }
  function go(i) { stop(); S.st = i; S.u = 0; show(); stop = PE.animate(900, u => { S.u = u; view.redraw(); }, () => { S.u = 1; view.redraw(); }); }
  STAGES.forEach((k, i) => document.getElementById('btn-pcpipe-' + k).addEventListener('click', () => go(i)));
  Object.keys(CATS).forEach(k => document.getElementById('btn-pcpipe-' + k).addEventListener('click', () => { S.cat = k; show(); view.redraw(); }));
  document.getElementById('btn-pcpipe-next').addEventListener('click', () => go(Math.min(STAGES.length - 1, S.st + 1)));
  document.getElementById('btn-pcpipe-prev').addEventListener('click', () => { if (S.st <= 0) { stop(); S.st = -1; S.u = 1; show(); view.redraw(); } else go(S.st - 1); });
  document.getElementById('btn-pcpipe-reset').addEventListener('click', () => { stop(); S.st = -1; S.u = 1; S.cat = 'bottle'; show(); view.redraw(); });
  show();
})();
</script>
