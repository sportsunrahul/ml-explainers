<script>
(() => {
  // Hero: mesh panels pass an inspection ring; good panels drop patch features into a blue cloud; the scratched panel's worst patch
  // lands outside it, a red segment measures the gap, and the scratch lights up on the part. Features are the toy kit's,
  // shown on two of its three axes (diagonal edges, brightness); distances here are measured in that same 2D view.
  const el = document.getElementById('hero-canvas'), btn = document.getElementById('btn-hero-replay');
  const NORMAL = [101, 102, 103], BAD = 200, T_END = 6.6;
  const f2 = f => { const o = []; for (let k = 0; k < f.length / 3; k++) o.push([f[k * 3 + 1], f[k * 3 + 2]]); return o; };
  const caps = NORMAL.map(s => TOY.make('mesh', s)).concat([TOY.defect(TOY.make('mesh', BAD), 'scratch', BAD)]);
  const feats = caps.map(c => f2(TOY.features(c)));
  const bank = feats.slice(0, 3).flat();
  const nn = p => { let best = Infinity, bi = 0; bank.forEach((b, i) => { const d = Math.hypot(b[0] - p[0], b[1] - p[1]); if (d < best) { best = d; bi = i; } }); return [best, bi]; };
  const scores = feats[3].map(p => nn(p)[0]);
  let top = 0; scores.forEach((v, q) => { if (v > scores[top]) top = q; });
  const [gap, nearest] = nn(feats[3][top]);
  const heatMap = TOY.toPixels(Float32Array.from(scores));
  let heat = null, heatTheme = '';
  const all = bank.concat([feats[3][top]]);
  const span = Math.max(...all.map(p => p[0])) - Math.min(...all.map(p => p[0])), pad = .18 * span;
  const lo = [Math.min(...all.map(p => p[0])) - pad, Math.min(...all.map(p => p[1])) - pad], hi = [Math.max(...all.map(p => p[0])) + pad, Math.max(...all.map(p => p[1])) + pad];
  const cloud = [[Math.min(...bank.map(p => p[0])), Math.min(...bank.map(p => p[1]))], [Math.max(...bank.map(p => p[0])), Math.max(...bank.map(p => p[1]))]];
  const FLY = [...Array(225).keys()].filter(k => k % 9 === 4);   // patches whose flight is drawn; the rest fade in
  const ease = u => u < 0 ? 0 : u > 1 ? 1 : u < .5 ? 2 * u * u : 1 - Math.pow(-2 * u + 2, 2) / 2;
  const seg = (t, a, b) => ease((t - a) / (b - a));
  const arrive = i => .5 + 1.1 * i;
  let t = T_END, raf = 0;

  const view = PE.canvas(el, w => w < 620 ? 1.08 : .4, (ctx, W, H) => draw(ctx, W, H));
  function layout(W, H) {
    const narrow = W < 620;
    const belt = narrow ? { x: 0, y: 0, w: W, h: H * .5 } : { x: 0, y: 0, w: W * .6, h: H };
    const cs = Math.min(belt.w * .2, belt.h * .5), cy = belt.y + belt.h * .52, xi = belt.x + belt.w * .5;
    const fp = narrow ? { x: W * .14, y: H * .56, w: W * .78, h: H * .36 } : { x: W * .66, y: H * .1, w: W * .31, h: H * .72 };
    return { narrow, belt, cs, cy, xi, fp, sp: cs * 1.45 };
  }
  function draw(ctx, W, H) {
    if (!W) return;
    const L = layout(W, H), C = { blue: PE.css('--blue'), red: PE.css('--red'), yellow: PE.css('--yellow'), muted: PE.css('--muted'), rule: PE.css('--rule'), fg: PE.css('--fg') };
    if (heatTheme !== C.red) { heat = TOY.heatBitmap(heatMap, .05, gap, C.red); heatTheme = C.red; }
    ctx.clearRect(0, 0, W, H);
    const px = p => L.fp.x + (p[0] - lo[0]) / (hi[0] - lo[0]) * L.fp.w, py = p => L.fp.y + L.fp.h - (p[1] - lo[1]) / (hi[1] - lo[1]) * L.fp.h;
    // belt
    const v = L.sp / 1.1, shift = (Math.min(t, arrive(3)) * v) % 18;
    ctx.save(); ctx.strokeStyle = C.rule; ctx.lineWidth = 1;
    const by0 = L.cy - L.cs * .62, by1 = L.cy + L.cs * .62;
    ctx.beginPath(); ctx.moveTo(L.belt.x, by0); ctx.lineTo(L.belt.x + L.belt.w, by0); ctx.moveTo(L.belt.x, by1); ctx.lineTo(L.belt.x + L.belt.w, by1); ctx.stroke();
    for (let x = L.belt.x - 18 + shift; x < L.belt.x + L.belt.w; x += 18) { ctx.beginPath(); ctx.moveTo(x, by1 + 3); ctx.lineTo(x + 6, by1 + 3); ctx.stroke(); }
    ctx.restore();
    // caps
    ctx.save(); ctx.beginPath(); ctx.rect(L.belt.x, by0 - 4, L.belt.w, by1 - by0 + 30); ctx.clip();
    const capX = i => L.xi + v * (i === 3 ? Math.min(0, t - arrive(3)) : t - arrive(i)) - L.cs / 2;
    caps.forEach((c, i) => { const x = capX(i); if (x > L.belt.x + L.belt.w || x + L.cs < L.belt.x) return;
      TOY.draw(ctx, c, x, L.cy - L.cs / 2, L.cs, { vf: false, heat: i === 3 ? heat : null, heatAlpha: i === 3 ? seg(t, arrive(3) + 1.9, arrive(3) + 2.5) : 0 }); });
    ctx.restore();
    // inspection ring: the camera's spot, yellow = under inspection
    ctx.save(); ctx.strokeStyle = C.yellow; ctx.lineWidth = 2; const rr = L.cs * .78;
    for (let k = 0; k < 16; k++) { const a = k * Math.PI / 8; ctx.beginPath(); ctx.arc(L.xi, L.cy, rr, a + .06, a + Math.PI / 8 - .06); ctx.stroke(); }
    ctx.restore();
    // feature panel
    ctx.save(); ctx.strokeStyle = C.rule; ctx.lineWidth = 1; ctx.strokeRect(L.fp.x - 8, L.fp.y - 8, L.fp.w + 16, L.fp.h + 16); ctx.restore();
    PE.label(ctx, 'diagonal edges', L.fp.x + L.fp.w / 2, L.fp.y + L.fp.h + 24, C.muted, 11, 'center');
    ctx.save(); ctx.translate(L.fp.x - 16, L.fp.y + L.fp.h / 2); ctx.rotate(-Math.PI / 2); PE.label(ctx, 'brightness', 0, 0, C.muted, 11, 'center'); ctx.restore();
    const patchXY = (i, k) => { const g = TOY.G, s = L.cs / TOY.N; return [capX(i) + ((k % g) * TOY.S + TOY.P / 2) * s, L.cy - L.cs / 2 + (Math.floor(k / g) * TOY.S + TOY.P / 2) * s]; };
    for (let i = 0; i < 3; i++) {
      const a = arrive(i);
      feats[i].forEach((p, k) => { const fi = FLY.indexOf(k);
        if (fi < 0) { const u = seg(t, a + .2, a + .8); if (u > 0) PE.dot(ctx, px(p), py(p), 1.6, C.blue, .55 * u); return; }
        const e = a - .2 + .5 * fi / FLY.length, u = seg(t, e, e + .7); if (u <= 0) return;
        const [sx, sy] = patchXY(i, k), tx = px(p), ty = py(p), mx = (sx + tx) / 2, my = Math.min(sy, ty) - 40;
        const x = (1 - u) * (1 - u) * sx + 2 * u * (1 - u) * mx + u * u * tx, y = (1 - u) * (1 - u) * sy + 2 * u * (1 - u) * my + u * u * ty;
        PE.dot(ctx, x, y, u < 1 ? 2.4 : 1.6, C.blue, u < 1 ? .95 : .55); });
    }
    // the scratched cap's worst patch
    const a3 = arrive(3), hl = seg(t, a3 + .1, a3 + .3);
    if (hl > 0) { const s = L.cs / TOY.N, k = top, x = capX(3) + (k % TOY.G) * TOY.S * s, y = L.cy - L.cs / 2 + Math.floor(k / TOY.G) * TOY.S * s;
      ctx.save(); ctx.globalAlpha = hl; ctx.strokeStyle = C.yellow; ctx.lineWidth = 2; ctx.strokeRect(x, y, TOY.P * s, TOY.P * s); ctx.restore(); }
    const fu = seg(t, a3 + .3, a3 + 1.1), q = feats[3][top];
    if (fu > 0) { const [sx, sy] = patchXY(3, top), tx = px(q), ty = py(q), mx = (sx + tx) / 2, my = Math.min(sy, ty) - 50;
      const x = (1 - fu) ** 2 * sx + 2 * fu * (1 - fu) * mx + fu * fu * tx, y = (1 - fu) ** 2 * sy + 2 * fu * (1 - fu) * my + fu * fu * ty;
      PE.dot(ctx, x, y, 4, C.yellow); }
    const ru = seg(t, a3 + 1.2, a3 + 1.8);
    if (ru > 0) { const b = bank[nearest], x0 = px(b), y0 = py(b), x1 = px(q), y1 = py(q);
      ctx.save(); ctx.strokeStyle = C.red; ctx.lineWidth = 2.5; ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x0 + (x1 - x0) * ru, y0 + (y1 - y0) * ru); ctx.stroke(); ctx.restore();
      PE.dot(ctx, x0, y0, 3.2, C.blue); }
    // labels on the final frame
    const lu = seg(t, a3 + 1.8, a3 + 2.3);
    if (lu > 0) { ctx.save(); ctx.globalAlpha = lu;
      const b = bank[nearest], mxl = (px(b) + px(q)) / 2, myl = (py(b) + py(q)) / 2;
      PE.label(ctx, 'normal features', (px(cloud[0]) + px(cloud[1])) / 2, py(cloud[0]) + 18, C.blue, 12, 'center');
      PE.label(ctx, 'test patch', px(q), py(q) - 12, C.yellow, 12, 'center');
      PE.label(ctx, 'distance =', mxl - 6, myl - 30, C.red, 12, 'right'); PE.label(ctx, 'anomaly score', mxl - 6, myl - 15, C.red, 12, 'right');
      PE.label(ctx, 'anomaly map', L.xi, L.cy + L.cs / 2 + 30, C.red, 12, 'center');
      ctx.restore(); }
    const tu = seg(t, a3 + 2.3, a3 + 2.6);
    if (tu > 0) { ctx.save(); ctx.globalAlpha = tu; ctx.font = '600 12px system-ui, sans-serif'; const tw = ctx.measureText('reject').width + 16, tx = L.xi + L.cs * .55, ty = L.cy - L.cs * .78;
      ctx.strokeStyle = C.red; ctx.lineWidth = 1.2; ctx.beginPath(); ctx.roundRect(tx, ty - 11, tw, 20, 10); ctx.stroke(); ctx.fillStyle = C.red; ctx.fillText('reject', tx + 8, ty + 3); ctx.restore(); }
  }
  function play() {
    cancelAnimationFrame(raf);
    if (PE.instant()) { t = T_END; view.redraw(); return; }
    const t0 = performance.now(); t = 0;
    const tick = now => { t = Math.min(T_END, (now - t0) / 1000); view.redraw(); if (t < T_END) raf = requestAnimationFrame(tick); };
    raf = requestAnimationFrame(tick);
  }
  btn.addEventListener('click', play);
  play();
})();
</script>
