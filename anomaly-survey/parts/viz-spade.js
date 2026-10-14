<script>
(() => {
  // ch-spade: SPADE's two stages on toy cap features (2-d view: diagonal edges, brightness). Checked in build/toy-spade-fig.mjs:
  // stage 1 (K = 3 of 12 by pooled features) image scores: scratched 0.0222, hole 0.0014, good 0.0003; stage 2 squared pixel
  // scores peak at 0.837 on the scratch and stay near 0.045 on the hole and good caps. The hole patch (79) matches a patch on
  // a bolt hole of cap 2 at squared distance 0.0003 (build/toy-padim-pick.mjs).
  const fig = document.getElementById('viz-spade'), el = document.getElementById('cv-spade'), out = fig.querySelector('.readout');
  const G = TOY.G, G2 = G * G, K = 3, LO = .06, HI = .6, HOLE_PATCH = 79;
  const f2 = f => { const o = []; for (let k = 0; k < G2; k++) o.push([f[k * 3 + 1], f[k * 3 + 2]]); return o; };
  const pooled = f => { const m = [0, 0, 0, 0]; for (let k = 0; k < G2; k++) for (let d = 0; d < 3; d++) m[d] += f[k * 3 + d] / G2;
    let v = 0; for (let k = 0; k < G2; k++) v += (f[k * 3 + 2] - m[2]) ** 2 / G2; m[3] = Math.sqrt(v); return m; };
  const train = Array.from({ length: 12 }, (_, i) => { const img = TOY.make('cap', 1 + i), F = TOY.features(img); return { img, f: f2(F), p: pooled(F) }; });
  const mk = img => { const F = TOY.features(img), f = f2(F), p = pooled(F);
    const d = train.map((t, i) => [t.p.reduce((a, x, j) => a + (x - p[j]) ** 2, 0), i]);
    const nearK = [...d].sort((a, b) => a[0] - b[0]).slice(0, K).map(x => x[1]);
    const gal = nearK.flatMap(i => train[i].f.map((g, k) => ({ i, k, g })));
    const best = f.map(q => { let b = Infinity, bj = 0; gal.forEach((e, j) => { const s = (e.g[0] - q[0]) ** 2 + (e.g[1] - q[1]) ** 2; if (s < b) { b = s; bj = j; } }); return [b, gal[bj]]; });
    let worst = 0; best.forEach((v, q) => { if (v[0] > best[worst][0]) worst = q; });
    return { img, f, d: d.map(x => x[0]), nearK, gal, best, worst, score: nearK.reduce((a, i) => a + d[i][0], 0) / K, map: TOY.toPixels(Float32Array.from(best.map(v => v[0]))) };
  };
  const T = { scratch: mk(TOY.defect(TOY.make('cap', 200), 'scratch', 200)), hole: mk(TOY.defect(TOY.make('cap', 207), 'hole', 207)), good: mk(TOY.make('cap', 104)) };
  const allF = train.flatMap(t => t.f).concat(T.scratch.f);
  const lo = [Math.min(...allF.map(p => p[0])) - .2, Math.min(...allF.map(p => p[1])) - .2], hi = [Math.max(...allF.map(p => p[0])) + .2, Math.max(...allF.map(p => p[1])) + .2];
  const DEF = { test: 'scratch', mode: 'pixel' };
  const S = { ...DEF, q: T.scratch.worst, mix: 1, line: 1 };
  const heats = {}; let heatColor = '';
  const glow = PE.linkPulses(fig, () => view.redraw());
  const view = PE.canvas(el, w => w < 620 ? 1.55 : .45, (ctx, W, H) => draw(ctx, W, H));
  let L = null;

  function layout(W, H) {
    const narrow = W < 620;
    if (narrow) {
      const si = Math.min(W * .5, 190), tw = (W - 24 - 11 * 4) / 12;
      const img = { x: 12, y: 30, s: si }, thumbs = { x: 12, y: img.y + si + 34, w: tw, gap: 4 };
      const lowY = thumbs.y + tw + 46, sm = Math.min((W - 36) / 2, H - lowY - 40);
      return { narrow, img, thumbs, match: { x: 12, y: lowY, s: sm }, fs: { x: 24 + sm, y: lowY, w: W - 36 - sm, h: sm } };
    }
    const si = Math.min(W * .36, H - 92), x0 = 20 + si + 40, rw = W - x0 - 16, tw = Math.min(40, (rw - 11 * 6) / 12);
    const img = { x: 20, y: 36, s: si }, thumbs = { x: x0, y: 36, w: tw, gap: 6 };
    const lowY = thumbs.y + tw + 52, sm = Math.min(rw * .42, H - lowY - 44);
    return { narrow, img, thumbs, match: { x: x0, y: lowY, s: sm }, fs: { x: x0 + sm + 18, y: lowY, w: rw - sm - 18, h: sm } };
  }
  const thumbX = i => L.thumbs.x + i * (L.thumbs.w + L.thumbs.gap);
  const patchCenter = (box, k) => { const s = box.s / TOY.N; return [box.x + ((k % G) * TOY.S + TOY.P / 2) * s, box.y + (Math.floor(k / G) * TOY.S + TOY.P / 2) * s]; };
  const patchRect = (box, k) => { const s = box.s / TOY.N; return [box.x + (k % G) * TOY.S * s, box.y + Math.floor(k / G) * TOY.S * s, TOY.P * s]; };
  const onBolt = (img, k) => { const cx = (k % G) * TOY.S + TOY.P / 2, cy = Math.floor(k / G) * TOY.S + TOY.P / 2; return Math.min(...img.bolts.map(([x, y]) => Math.hypot(x - cx, y - cy))) < 12; };

  function draw(ctx, W, H) {
    if (!W) return;
    L = layout(W, H);
    const C = { blue: PE.css('--blue'), red: PE.css('--red'), yellow: PE.css('--yellow'), muted: PE.css('--muted'), rule: PE.css('--rule'), fg: PE.css('--fg') };
    if (heatColor !== C.red) { for (const k in heats) delete heats[k]; heatColor = C.red; }
    const t = T[S.test], pm = S.mix;   // pm: 1 = pixel stage, 0 = image stage
    ctx.clearRect(0, 0, W, H);
    // test image, heat in the pixel stage
    heats[S.test] = heats[S.test] || TOY.heatBitmap(t.map, LO, HI, C.red);
    TOY.draw(ctx, t.img, L.img.x, L.img.y, L.img.s, { heat: heats[S.test], heatAlpha: pm });
    PE.label(ctx, 'cap under inspection', L.img.x, L.img.y - 12, C.muted, 12);
    if (glow('img') > 0) { ctx.save(); ctx.strokeStyle = C.yellow; ctx.globalAlpha = glow('img'); ctx.lineWidth = 3; ctx.strokeRect(L.img.x - 6, L.img.y - 6, L.img.s + 12, L.img.s + 12); ctx.restore(); }
    // remembered caps; the K nearest outlined, pooled distance bars in the image stage
    PE.label(ctx, '12 good caps in memory', L.thumbs.x, L.thumbs.y - 12, C.muted, 12);
    const dmax = Math.max(...Object.values(T).flatMap(x => x.d));
    train.forEach((tr, i) => {
      const x = thumbX(i), near = t.nearK.indexOf(i);
      ctx.save(); ctx.globalAlpha = near >= 0 ? 1 : .45; TOY.draw(ctx, tr.img, x, L.thumbs.y, L.thumbs.w, { vf: false }); ctx.restore();
      if (near >= 0) { ctx.save(); ctx.strokeStyle = C.blue; ctx.lineWidth = 2 + 2 * Math.max(glow('knn'), glow('gallery')); ctx.strokeRect(x - 2, L.thumbs.y - 2, L.thumbs.w + 4, L.thumbs.w + 4); ctx.restore(); }
      const bh = Math.max(1.5, 26 * Math.sqrt(t.d[i] / dmax)) * (1 - pm);
      if (bh > 1) { ctx.save(); ctx.globalAlpha = 1 - pm; ctx.fillStyle = near >= 0 ? C.blue : C.muted; if (glow('dist') > 0) ctx.fillStyle = PE.mix(ctx.fillStyle.length === 7 ? ctx.fillStyle : '#888888', '#ffffff', .4 * glow('dist'));
        ctx.fillRect(x + L.thumbs.w * .25, L.thumbs.y + L.thumbs.w + 6, L.thumbs.w * .5, bh); ctx.restore(); }
    });
    // lower right
    const sel = t.best[S.q], g = sel[1], mimg = train[g.i].img, flagged = sel[0] > LO;
    if (pm > 0) {
      ctx.save(); ctx.globalAlpha = pm;
      TOY.draw(ctx, mimg, L.match.x, L.match.y, L.match.s);
      PE.label(ctx, `best match: cap ${g.i + 1}`, L.match.x, L.match.y - 12, C.muted, 12);
      const [mx, my, ms] = patchRect(L.match, g.k);
      ctx.strokeStyle = C.blue; ctx.lineWidth = 2 + 2 * glow('match'); ctx.strokeRect(mx, my, ms, ms);
      // feature space with the gallery
      ctx.strokeStyle = C.rule; ctx.lineWidth = 1; ctx.strokeRect(L.fs.x, L.fs.y, L.fs.w, L.fs.h);
      const X = p => L.fs.x + (p[0] - lo[0]) / (hi[0] - lo[0]) * L.fs.w, Y = p => L.fs.y + L.fs.h - (p[1] - lo[1]) / (hi[1] - lo[1]) * L.fs.h;
      ctx.save(); ctx.beginPath(); ctx.rect(L.fs.x, L.fs.y, L.fs.w, L.fs.h); ctx.clip();
      const gg = glow('gallery'); t.gal.forEach(e => PE.dot(ctx, X(e.g), Y(e.g), 1.4 + gg, C.blue, .45 + .3 * gg));
      const qf = t.f[S.q];
      ctx.strokeStyle = C.red; ctx.lineWidth = 2 + 2 * glow('match'); ctx.beginPath(); ctx.moveTo(X(g.g), Y(g.g)); ctx.lineTo(X(g.g) + (X(qf) - X(g.g)) * S.line, Y(g.g) + (Y(qf) - Y(g.g)) * S.line); ctx.stroke();
      PE.dot(ctx, X(g.g), Y(g.g), 3.2, C.blue); PE.dot(ctx, X(qf), Y(qf), 4 + 2 * glow('pix'), C.yellow);
      ctx.restore();
      PE.label(ctx, 'gallery features', L.fs.x + 5, L.fs.y + 14, C.muted, 11);
      PE.label(ctx, 'diagonal edges', L.fs.x + L.fs.w / 2, L.fs.y + L.fs.h + 15, C.muted, 10, 'center');
      // correspondence: the selected patch to its match
      const [ax, ay] = patchCenter(L.img, S.q), [bx, by] = patchCenter(L.match, g.k), u = S.line;
      ctx.strokeStyle = flagged ? C.red : C.blue; ctx.lineWidth = 1.6 + 1.5 * glow('match'); ctx.setLineDash([5, 4]); ctx.beginPath();
      const cx1 = ax + (bx - ax) * .5, cy1 = Math.min(ay, by) - 60;
      for (let s = 0; s <= 1.0001 * u; s += .02) { const x = (1 - s) ** 2 * ax + 2 * s * (1 - s) * cx1 + s * s * bx, y = (1 - s) ** 2 * ay + 2 * s * (1 - s) * cy1 + s * s * by; s ? ctx.lineTo(x, y) : ctx.moveTo(x, y); }
      ctx.stroke(); ctx.setLineDash([]);
      ctx.restore();
    }
    if (pm < 1) {
      ctx.save(); ctx.globalAlpha = 1 - pm;
      const x0 = L.match.x, y0 = L.match.y, w = L.match.s + 18 + L.fs.w, rows = [['scratched cap', T.scratch], ['cap with a hole', T.hole], ['good cap', T.good]];
      PE.label(ctx, 'image score: mean squared distance to the 3 nearest caps', x0, y0 - 12, C.muted, 12);
      const smax = Math.max(...rows.map(r => r[1].score));
      rows.forEach(([name, r], j) => { const y = y0 + 16 + j * 34, bw = Math.max(2, (w - 150) * Math.sqrt(r.score / smax)), cur = r === t;
        PE.label(ctx, name, x0, y + 13, cur ? C.yellow : C.muted, 12);
        ctx.fillStyle = cur ? C.yellow : C.muted; ctx.globalAlpha = (1 - pm) * (cur ? 1 : .5); ctx.fillRect(x0 + 110, y, bw, 16); ctx.globalAlpha = 1 - pm;
        PE.label(ctx, r.score.toFixed(4), x0 + 116 + bw, y + 13, cur ? C.fg : C.muted, 11); });
      PE.label(ctx, '(square-root scale)', x0, y0 + 16 + 3 * 34 + 6, C.muted, 10);
      ctx.restore();
    }
    // selected patch on the test image
    const [px, py, ps] = patchRect(L.img, S.q);
    ctx.save(); ctx.globalAlpha = .35 + .65 * pm; ctx.strokeStyle = C.yellow; ctx.lineWidth = 2 + 2 * glow('pix'); ctx.strokeRect(px, py, ps, ps); ctx.restore();
  }

  function say() {
    const t = T[S.test], name = { scratch: 'scratched cap', hole: 'cap with a hole', good: 'good cap' }[S.test];
    if (S.mode === 'image') {
      out.textContent = `Stage 1: the 3 remembered caps nearest to the ${name} by pooled feature are caps ${t.nearK.map(i => i + 1).join(', ')}. Its image score is ${t.score.toFixed(4)}; the scratched cap scores ${T.scratch.score.toFixed(4)}, the cap with a hole ${T.hole.score.toFixed(4)}, the good cap ${T.good.score.toFixed(4)}.`;
      return;
    }
    const [s, g] = t.best[S.q], r = Math.floor(S.q / G) + 1, c = S.q % G + 1;
    let msg = `Patch at row ${r}, column ${c}: its nearest gallery feature is in cap ${g.i + 1}, at squared distance ${s.toFixed(4)}`;
    msg += s > LO ? ', so it lights up.' : (onBolt(train[g.i].img, g.k) && S.test === 'hole' && S.q === HOLE_PATCH ? '. The match is a patch on a bolt hole, so SPADE sees nothing wrong.' : ', so it looks normal.');
    out.textContent = msg;
  }
  function select(q, animate = true) {
    S.q = q; if (!animate) { S.line = 1; view.redraw(); say(); return; }
    PE.animate(450, u => { S.line = u; view.redraw(); }); say();
  }
  function setTest(k) {
    S.test = k; ['scratch', 'hole', 'good'].forEach(x => document.getElementById('btn-spade-' + x).setAttribute('aria-pressed', x === k));
    select(k === 'hole' ? HOLE_PATCH : T[k].worst);
  }
  function setMode(m) {
    S.mode = m; ['image', 'pixel'].forEach(x => document.getElementById('btn-spade-' + x).setAttribute('aria-pressed', x === m));
    const a = S.mix, b = m === 'pixel' ? 1 : 0; PE.animate(500, u => { S.mix = a + (b - a) * u; view.redraw(); }); say();
  }
  ['scratch', 'hole', 'good'].forEach(k => document.getElementById('btn-spade-' + k).addEventListener('click', () => setTest(k)));
  document.getElementById('btn-spade-image').addEventListener('click', () => setMode('image'));
  document.getElementById('btn-spade-pixel').addEventListener('click', () => setMode('pixel'));
  document.getElementById('btn-spade-reset').addEventListener('click', () => { setTest(DEF.test); setMode(DEF.mode); });
  el.addEventListener('pointerdown', e => { if (!L) return; const r = el.getBoundingClientRect(), x = e.clientX - r.left - L.img.x, y = e.clientY - r.top - L.img.y;
    if (x < 0 || y < 0 || x > L.img.s || y > L.img.s) return; const s = L.img.s / TOY.N;
    const i = PE.clamp(Math.floor((x / s - TOY.P / 2) / TOY.S + .5), 0, G - 1), j = PE.clamp(Math.floor((y / s - TOY.P / 2) / TOY.S + .5), 0, G - 1);
    if (S.mode !== 'pixel') setMode('pixel'); select(j * G + i); });
  el.addEventListener('keydown', e => { const d = { ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowUp: [0, -1], ArrowDown: [0, 1] }[e.key]; if (!d) return;
    e.preventDefault(); const i = PE.clamp(S.q % G + d[0], 0, G - 1), j = PE.clamp(Math.floor(S.q / G) + d[1], 0, G - 1); select(j * G + i, false); });
  say();
})();
</script>
