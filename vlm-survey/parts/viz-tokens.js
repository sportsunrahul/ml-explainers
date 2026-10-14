<script>
(() => {
  const fig = document.getElementById('viz-tokens'), el = document.getElementById('tokens-canvas');
  const out = document.getElementById('tokens-readout');
  const rngRes = document.getElementById('rng-tokens-res'), rngP = document.getElementById('rng-tokens-patch');
  const resVal = document.getElementById('tokens-res-val'), pVal = document.getElementById('tokens-patch-val');

  // ---- Procedural test images. The same two functions are pasted into viz-resolution.js. ----
  // Scene colors are fixed "paper" colors: they are the pixels of a photo, not page theme colors.
  function renderScene(kind, W, H) {
    const c = document.createElement('canvas'); c.width = W; c.height = H;
    const g = c.getContext('2d'), r = PE.rng(7), meta = { c, W, H, zoom: [0, 0, W, H], small: 0, anchor: [W / 2, H / 2] };
    const font = (px, w, mono) => `${w || 400} ${px}px ${mono ? 'Menlo, Consolas, "DejaVu Sans Mono", monospace' : 'Helvetica, Arial, "DejaVu Sans", sans-serif'}`;
    const T = (s, x, y, px, o = {}) => { g.font = font(px, o.w, o.mono); g.fillStyle = o.c || '#1d1d1f'; g.textAlign = o.a || 'left';
      g.textBaseline = 'alphabetic'; g.fillText(s, x, y); return g.measureText(s).width; };
    const rule = (x1, y, x2, col, lw, dash) => { g.save(); g.strokeStyle = col; g.lineWidth = lw; if (dash) g.setLineDash(dash);
      g.beginPath(); g.moveTo(x1, y); g.lineTo(x2, y); g.stroke(); g.restore(); };
    const wrap = (text, px, width) => { g.font = font(px); const out = []; let line = '';
      text.split(' ').forEach(w => { const t = line ? line + ' ' + w : w; if (g.measureText(t).width > width && line) { out.push(line); line = w; } else line = t; });
      if (line) out.push(line); return out; };
    if (kind === 'receipt') {
      g.fillStyle = '#3c4046'; g.fillRect(0, 0, W, H);
      for (let i = 0; i < W * H / 260; i++) { g.fillStyle = `rgba(255,255,255,${.02 + .04 * r()})`; g.fillRect(r() * W, r() * H, 1 + 3 * r(), 1 + 2 * r()); }
      const rh = .9 * H, rw = Math.min(.56 * rh, .9 * W), x0 = (W - rw) / 2, y0 = (H - rh) / 2, u = rh / 100;
      g.fillStyle = 'rgba(0,0,0,.35)'; g.fillRect(x0 + u, y0 + 1.2 * u, rw, rh);
      g.fillStyle = '#f5f2ea'; g.fillRect(x0, y0, rw, rh);
      const L = x0 + .09 * rw, R = x0 + .91 * rw, cx = x0 + rw / 2;
      T('CORNER CAFE', cx, y0 + 9 * u, 6.4 * u, { w: 800, a: 'center' });
      T('12 Harbour Street', cx, y0 + 13.6 * u, 3 * u, { a: 'center', c: '#4a4a4a' });
      rule(L, y0 + 17 * u, R, '#777', .35 * u, [u, .8 * u]);
      [['Latte', '4.50'], ['Bagel', '3.25'], ['Orange juice', '2.75'], ['Tax', '0.75']].forEach(([a, b], i) => {
        const y = y0 + (23.5 + 6 * i) * u; T(a, L, y, 3.9 * u); T(b, R, y, 3.9 * u, { a: 'right' }); });
      rule(L, y0 + 45.5 * u, R, '#222', .3 * u);
      T('TOTAL', L, y0 + 52.5 * u, 5.2 * u, { w: 800 }); T('11.25', R, y0 + 52.5 * u, 5.2 * u, { w: 800, a: 'right' });
      T('Card **** 4821   Auth 093117', L, y0 + 61 * u, 2.3 * u, { c: '#333' });
      T('Thank you, see you again soon', L, y0 + 65 * u, 2.3 * u, { c: '#333' });
      let bx = L; while (bx < R - u) { const bw = (.25 + .9 * r()) * u; g.fillStyle = '#222'; g.fillRect(bx, y0 + 71 * u, bw, 12 * u); bx += bw + (.3 + .8 * r()) * u; }
      T('0042 1187 3309', cx, y0 + 87.5 * u, 2.6 * u, { a: 'center', c: '#333', mono: true });
      meta.zoom = [L - .5 * u, y0 + 19.5 * u, R - L + u, 17.5 * u]; meta.small = 3.9 * u; meta.anchor = [x0 + .17 * rw, y0 + 7 * u];
    } else if (kind === 'doc') {
      g.fillStyle = '#fbfbf8'; g.fillRect(0, 0, W, H);
      const m = 84, tw = W - 2 * m;
      T('Quarterly Shipping Report', m, 150, 46, { w: 800 });
      T('Northern region, third quarter', m, 196, 22, { c: '#555' });
      rule(m, 222, W - m, '#999', 2);
      T('1  Summary', m, 282, 28, { w: 700 });
      let y = 326;
      const para = (s) => { wrap(s, 16, tw).forEach(l => { T(l, m, y, 16, { c: '#222' }); y += 25; }); y += 14; };
      para('In the third quarter the northern region delivered 1,284 loads, up 6.2 percent on the second quarter. On-time arrival held at 94.1 percent, while the average delay on late loads fell from 3.4 to 2.9 hours. Two routes account for most of the remaining delay, and both cross the river at the same bridge, which was under repair for five weeks.');
      para('Fuel cost per load rose slightly because trucks on the two slow routes idled longer at the bridge. Driver hours stayed inside the limits on every route. The depot at Lakeshore opened a second loading bay in August, which cut the average wait at the gate from 41 to 26 minutes.');
      y += 6; const cols = [m, m + 300, m + 460, m + 640];
      ['Route', 'Loads', 'On time', 'Avg delay (h)'].forEach((s, i) => T(s, cols[i], y, 17, { w: 700 })); rule(m, y + 10, W - m, '#444', 1.5); y += 38;
      [['North-7', '312', '88.5%', '4.8'], ['Lakeshore', '274', '91.2%', '3.9'], ['Ridge Road', '221', '97.3%', '1.6'], ['Harbour Loop', '198', '98.0%', '1.2'], ['East Mill', '167', '96.4%', '2.1'], ['Valley Run', '112', '99.1%', '0.7']]
        .forEach(row => { row.forEach((s, i) => T(s, cols[i], y, 16, { c: '#222' })); y += 28; });
      rule(m, y - 14, W - m, '#bbb', 1); y += 40;
      T('2  Routes to watch', m, y, 28, { w: 700 }); y += 44;
      para('North-7 and Lakeshore should return to normal once the bridge reopens in October. Until then, dispatch will send loads for the east depots along Ridge Road, which adds eleven minutes per trip but avoids the queue at the bridge entirely.');
      para('We will report the effect of the new loading bay separately, once a full quarter of data is available. The numbers in this report are counted at the destination depot and may change slightly after the audit.');
      T('* Loads are counted at the destination depot. Figures are provisional until the audit closes.', m, H - 90, 12, { c: '#555' });
      T('3', W / 2, H - 50, 13, { a: 'center', c: '#555' });
      meta.zoom = [m - 4, 308, 184, 76]; meta.small = 16; meta.anchor = [m + 120, 320];
    } else if (kind === 'screen') {
      g.fillStyle = '#ffffff'; g.fillRect(0, 0, W, H);
      g.fillStyle = '#e4e4e7'; g.fillRect(0, 0, W, 38);
      ['#ff5f57', '#febc2e', '#28c840'].forEach((col, i) => { g.fillStyle = col; g.beginPath(); g.arc(22 + 22 * i, 19, 7, 0, 7); g.fill(); });
      T('shipping_report.py - Editor', W / 2, 25, 15, { a: 'center', c: '#333' });
      g.fillStyle = '#f2f2f4'; g.fillRect(0, 38, 300, H - 66);
      T('EXPLORER', 20, 72, 12, { w: 700, c: '#666' });
      ['data/', '   loads_q3.csv', '   routes.csv', 'report/', '   shipping_report.py', '   charts.py', 'tests/', '   test_report.py', 'README.md']
        .forEach((s, i) => T(s, 20, 106 + 26 * i, 15, { c: i === 4 ? '#000' : '#444', w: i === 4 ? 700 : 400 }));
      const code = ['import csv', 'from dataclasses import dataclass', '', '@dataclass', 'class Route:', '    name: str', '    loads: int', '    late: int',
        '    delay_h: float', '', 'def load_routes(path):', '    with open(path) as f:', '        rows = list(csv.DictReader(f))',
        '    return [Route(r["name"], int(r["loads"]), int(r["late"]), float(r["delay_h"])) for r in rows]', '', 'def on_time_rate(routes):',
        '    total = sum(r.loads for r in routes)', '    late = sum(r.late for r in routes)', '    return 1 - late / total', '', 'def worst_routes(routes, k=2):',
        '    return sorted(routes, key=lambda r: r.delay_h, reverse=True)[:k]', '', 'if __name__ == "__main__":', '    routes = load_routes("data/routes.csv")',
        '    print(f"on time: {on_time_rate(routes):.1%}")', '    for r in worst_routes(routes):', '        print(r.name, r.delay_h)'];
      code.forEach((s, i) => { const y = 72 + 24 * i; T(String(i + 1), 350, y, 14, { a: 'right', c: '#9a9a9a', mono: true });
        T(s, 370, y, 15, { mono: true, c: /^\s*(def|class|import|from|if|return|@)/.test(s) ? '#6f2da8' : '#1e1e1e' }); });
      g.fillStyle = '#1f2227'; g.fillRect(300, 760, W - 300, 292);
      T('TERMINAL', 320, 786, 12, { w: 700, c: '#9aa0a6' });
      ['$ python report/shipping_report.py', 'on time: 94.1%', 'North-7 4.8', 'Lakeshore 3.9', '$ _'].forEach((s, i) => T(s, 320, 820 + 26 * i, 15, { mono: true, c: '#d4d4d4' }));
      g.fillStyle = '#5b5f66'; g.fillRect(0, H - 28, W, 28);
      T('Ln 17, Col 22    Spaces: 4    UTF-8    Python', 20, H - 9, 12, { c: '#ffffff' });
      g.font = font(15, 400, true); const cw = g.measureText('m').width;
      meta.zoom = [366, 72 + 24 * 15 - 17, 24 * cw + 8, 72]; meta.small = 15; meta.anchor = [700, 72 + 24 * 16];
    }
    return meta;
  }
  // Area-average downscale in halving steps (one big drawImage step would alias the text).
  function downsample(src, w, h) {
    w = Math.max(1, Math.round(w)); h = Math.max(1, Math.round(h));
    let cur = src;
    for (;;) {
      const nw = Math.max(w, Math.ceil(cur.width / 2)), nh = Math.max(h, Math.ceil(cur.height / 2)), last = nw === w && nh === h;
      const c = document.createElement('canvas'); c.width = nw; c.height = nh;
      const g = c.getContext('2d'); g.imageSmoothingEnabled = true; g.imageSmoothingQuality = 'high'; g.drawImage(cur, 0, 0, nw, nh);
      cur = c; if (last) return cur;
    }
  }
  // ---- end of shared scene code ----

  const SRC = renderScene('receipt', 896, 896), PS = [14, 16, 32];
  const snap = (R, P) => Math.max(2 * P, Math.round(R / P) * P);
  const S = { R: 224, P: 16, sel: null, fl: 1 };
  const glow = PE.linkPulses(fig, () => cv.redraw());
  let stop = () => {}, cache = { R: 0 }, Ecache = { P: 0 };
  const fmt = n => Math.round(n).toLocaleString('en-US');

  function model(R) {
    if (cache.R !== R) { const c = downsample(SRC.c, R, R); cache = { R, c, px: c.getContext('2d').getImageData(0, 0, R, R).data }; }
    return cache;
  }
  const grid = () => { const R = snap(S.R, S.P); return { R, g: R / S.P }; };
  function defaultSel() { const { R } = grid(); return [Math.floor(SRC.anchor[1] / 896 * R / S.P), Math.floor(SRC.anchor[0] / 896 * R / S.P)]; }
  function clampSel() { const { g } = grid(); if (!S.sel) S.sel = defaultSel(); S.sel = [PE.clamp(S.sel[0], 0, g - 1), PE.clamp(S.sel[1], 0, g - 1)]; }
  // Toy projection E: a fixed random matrix (not trained), 16 output dimensions drawn out of D.
  function projE(P) {
    if (Ecache.P !== P) { const r = PE.rng(11), n = PE.gauss(r), M = []; for (let i = 0; i < P * P * 3 * 16; i++) M.push(n()); Ecache = { P, M }; }
    return Ecache.M;
  }
  function patchPixels() {
    const { R } = grid(), m = model(R), [pr, pc] = S.sel, P = S.P, v = [];
    for (let y = 0; y < P; y++) for (let x = 0; x < P; x++) { const i = ((pr * P + y) * R + pc * P + x) * 4; v.push([m.px[i], m.px[i + 1], m.px[i + 2]]); }
    return v;
  }
  function embed(v) {
    const M = projE(S.P), n = v.length * 3, o = new Array(16).fill(0);
    v.forEach((p, k) => p.forEach((ch, j) => { const x = ch / 255 - .5, row = (k * 3 + j) * 16; for (let d = 0; d < 16; d++) o[d] += x * M[row + d]; }));
    return o.map(x => x / Math.sqrt(n) * 2.2);
  }
  const avgColor = (pr, pc) => { const { R } = grid(), m = model(R), P = S.P; let a = [0, 0, 0], k = 0;
    for (let y = 0; y < P; y += 2) for (let x = 0; x < P; x += 2) { const i = ((pr * P + y) * R + pc * P + x) * 4; a[0] += m.px[i]; a[1] += m.px[i + 1]; a[2] += m.px[i + 2]; k++; }
    return `rgb(${a.map(v => Math.round(v / k)).join(',')})`; };

  const layout = W => W < 560
    ? (() => { const side = W - 24; return { st: true, ix: 12, iy: 28, side, px: 12, py: 28 + side + 26, pw: W - 24 }; })()
    : (() => { const side = Math.round(W * .6 - 50); return { st: false, ix: 12, iy: 30, side, px: side + 40, py: 30, pw: W - side - 52 }; })();

  const cv = PE.canvas(el, w => w < 360 ? 2.3 : w < 560 ? 2.12 : w < 720 ? .74 : .6, (ctx, W, H) => {
    const B = PE.css('--blue'), Rd = PE.css('--red'), F = PE.css('--fg'), M = PE.css('--muted'), Rule = PE.css('--rule'), BG = PE.css('--bg');
    ctx.fillStyle = BG; ctx.fillRect(0, 0, W, H);
    clampSel();
    const Lh = layout(W), { R, g } = grid(), m = model(R), dpr = devicePixelRatio || 1;
    const { ix, iy, side } = Lh, cell = side / g, gp = glow('patch'), gE = glow('proj'), gc = glow('cls'), gpos = glow('pos');
    // The image exactly as the encoder receives it: R x R pixels, blown up without smoothing so pixels stay visible.
    ctx.imageSmoothingEnabled = side * dpr < R; ctx.drawImage(m.c, ix, iy, side, side); ctx.imageSmoothingEnabled = true;
    PE.label(ctx, Lh.st ? `${R} x ${R} px: ${g} x ${g} patches of ${S.P} px` : `image: ${R} x ${R} pixels, cut into ${g} x ${g} patches of ${S.P} x ${S.P}`, ix, iy - 10, F, Lh.st ? 11 : 12);
    ctx.save(); ctx.strokeStyle = B; ctx.globalAlpha = cell < 9 ? .55 : .8; ctx.lineWidth = 1; ctx.beginPath();
    for (let k = 0; k <= g; k++) { const t = Math.round(ix + k * cell) + .5; ctx.moveTo(t, iy); ctx.lineTo(t, iy + side); const u = Math.round(iy + k * cell) + .5; ctx.moveTo(ix, u); ctx.lineTo(ix + side, u); }
    ctx.stroke(); ctx.restore();
    const [sr, sc] = S.sel, sx = ix + sc * cell, sy = iy + sr * cell;
    // Halo in the page background under a foreground line: visible on the light receipt and the dark table.
    ctx.strokeStyle = BG; ctx.lineWidth = 5 + 3 * gp; ctx.strokeRect(sx - 1, sy - 1, cell + 2, cell + 2);
    ctx.strokeStyle = F; ctx.lineWidth = 2 + 2 * gp; ctx.strokeRect(sx - 1, sy - 1, cell + 2, cell + 2);

    // Right (or below): one patch becomes one token.
    const { px, py, pw } = Lh, ZB = 60, fl = S.fl, stage = i => PE.clamp(fl * 5 - i, 0, 1);
    const pix = patchPixels(), P = S.P, zc = ZB / P;
    pix.forEach((p, k) => { ctx.fillStyle = `rgb(${p.join(',')})`; ctx.fillRect(px + (k % P) * zc, py + Math.floor(k / P) * zc, zc + .6, zc + .6); });
    ctx.strokeStyle = B; ctx.lineWidth = 1.5 + 3 * gp; ctx.strokeRect(px, py, ZB, ZB);
    if (fl < .2) { const u = fl / .2, gx = PE.lerp(sx, px, u), gy = PE.lerp(sy, py, u), gs = PE.lerp(cell, ZB, u);
      ctx.strokeStyle = F; ctx.lineWidth = 2; ctx.strokeRect(gx, gy, gs, gs); }
    PE.label(ctx, `patch at row ${sr + 1}, column ${sc + 1}`, px + ZB + 12, py + 20, F, 12);
    PE.label(ctx, `${P} x ${P} pixels x 3 colors`, px + ZB + 12, py + 40, M, 11.5);
    ctx.save();
    ctx.globalAlpha = stage(1);
    PE.label(ctx, `flatten: ${P} x ${P} x 3 = ${fmt(P * P * 3)} numbers`, px, py + 86, F, 12);
    const stripW = pw - 4, sw = stripW / pix.length;
    pix.forEach((p, k) => { ctx.fillStyle = `rgb(${p.join(',')})`; ctx.fillRect(px + k * sw, py + 93, sw + .5, 11); });
    ctx.strokeStyle = B; ctx.lineWidth = 1 + 2 * gp; ctx.strokeRect(px, py + 93, stripW, 11);
    ctx.globalAlpha = stage(2);
    ctx.fillStyle = PE.mix(BG.length === 7 ? BG : '#16181d', B.length === 7 ? B : '#58c4dd', .15 + .5 * gE); ctx.fillRect(px, py + 116, 30, 24);
    ctx.strokeStyle = B; ctx.lineWidth = 1.5 + 2.5 * gE; ctx.strokeRect(px, py + 116, 30, 24);
    PE.label(ctx, 'E', px + 15, py + 133, B, 14, 'center');
    PE.label(ctx, `x E: ${fmt(P * P * 3)} numbers in, D numbers out`, px + 38, py + 126, gE > 0 ? F : M, 11.5);
    PE.label(ctx, 'the same E for every patch', px + 38, py + 141, M, 10.5);
    ctx.globalAlpha = stage(3);
    const vec = embed(pix), bw = Math.min(14, (pw - 8) / 16), base = py + 180;
    PE.label(ctx, 'patch embedding: D numbers (16 drawn)', px, py + 162, F, 11.5);
    vec.forEach((x, d) => { const h = PE.clamp(x, -1, 1) * 12; ctx.fillStyle = B; ctx.fillRect(px + d * bw + 1, h < 0 ? base : base - h, bw - 2, Math.abs(h) + .5); });
    ctx.strokeStyle = Rule; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(px, base + .5); ctx.lineTo(px + 16 * bw, base + .5); ctx.stroke();
    ctx.globalAlpha = stage(4);
    const k = sr * g + sc + 1;
    PE.label(ctx, `+ E_pos[${fmt(k)}]: slot ${fmt(k)} has its own learned vector`, px, py + 212, gpos > 0 ? F : M, (Lh.st || pw < 300 ? 10.5 : 11.5) + gpos);
    ctx.restore();

    // The sequence the transformer reads: [class], then the patches in reading order.
    const N = g * g, s = 18, gap = 5, ys = py + 244;
    PE.label(ctx, `sequence: [class] + ${fmt(N)} patch tokens`, px, ys - 10, F, 12);
    const nums = [...new Set([0, 1, 2, 3, k, N])].filter(v => v <= N).sort((a, b) => a - b), items = [];
    nums.forEach((v, i) => { if (i && v > nums[i - 1] + 1) items.push('...'); items.push(v); });
    let x = px;
    items.forEach(it => {
      if (it === '...') { PE.label(ctx, '...', x + 2, ys + 13, M, 12); x += 18; return; }
      if (it === 0) { ctx.fillStyle = B; ctx.globalAlpha = .35 + .4 * gc; ctx.fillRect(x, ys, s, s); ctx.globalAlpha = 1;
        ctx.strokeStyle = B; ctx.lineWidth = 1.5 + 2.5 * gc; ctx.strokeRect(x, ys, s, s); PE.label(ctx, 'cls', x + s / 2, ys + 13, F, 9, 'center'); }
      else { const pr = Math.floor((it - 1) / g), pc = (it - 1) % g; ctx.fillStyle = avgColor(pr, pc); ctx.fillRect(x, ys, s, s);
        ctx.strokeStyle = it === k ? F : B; ctx.lineWidth = it === k ? 2.5 : 1.2; ctx.strokeRect(x, ys, s, s); }
      PE.label(ctx, String(it), x + s / 2, ys + s + 13, gpos > 0 ? F : M, 9.5 + 1.5 * gpos, 'center');
      x += s + gap + (String(it).length > 2 ? 6 : 0);
    });
    // Attention compares every token with every token: the cost grows with N squared (log scale bar).
    const yb = ys + 58, frac = PE.clamp(Math.log10(N * N) / 7, 0, 1);
    PE.label(ctx, `attention pairs N x N = ${fmt(N * N)}`, px, yb, Rd, 12);
    ctx.fillStyle = Rule; ctx.fillRect(px, yb + 7, pw - 4, 10); ctx.fillStyle = Rd; ctx.fillRect(px, yb + 7, (pw - 4) * frac, 10);
    PE.label(ctx, 'log scale, 10 to 10,000,000', px, yb + 32, M, 10.5);
  });

  function readout() {
    const { R, g } = grid(), N = g * g;
    out.textContent = `${R} x ${R} pixel image, ${S.P} x ${S.P} patches: ${g} x ${g} = ${fmt(N)} patch tokens, plus one [class] token. ` +
      `Self-attention scores every token against every other: about ${fmt(N * N)} pairs. Selected patch: row ${S.sel[0] + 1}, column ${S.sel[1] + 1}, token ${fmt(S.sel[0] * g + S.sel[1] + 1)}.`;
  }
  function syncControls() { rngRes.value = snap(S.R, S.P); resVal.textContent = snap(S.R, S.P) + ' px'; rngP.value = PS.indexOf(S.P); pVal.textContent = S.P + ' px'; }
  function preset(R, P) {
    stop(); const r0 = snap(S.R, S.P), fr = S.sel ? [S.sel[0] / grid().g, S.sel[1] / grid().g] : null; S.P = P;
    stop = PE.animate(700, u => { S.R = PE.lerp(r0, R, u); const { g } = grid();
      if (fr) S.sel = [Math.floor(fr[0] * g), Math.floor(fr[1] * g)]; syncControls(); cv.redraw(); readout(); }, () => { S.R = R; syncControls(); cv.redraw(); readout(); });
  }
  const bind = (id, fn) => document.getElementById(id).addEventListener('click', fn);
  bind('btn-tokens-b32', () => preset(224, 32));
  bind('btn-tokens-b16', () => preset(224, 16));
  bind('btn-tokens-clip336', () => preset(336, 14));
  bind('btn-tokens-h14ft', () => preset(518, 14));
  bind('btn-tokens-flatten', () => { stop(); S.fl = 0; stop = PE.animate(1800, u => { S.fl = u; cv.redraw(); }); });
  bind('btn-tokens-reset', () => { stop(); S.R = 224; S.P = 16; S.sel = null; S.fl = 1; syncControls(); cv.redraw(); readout(); });
  rngRes.addEventListener('input', () => { stop(); S.R = +rngRes.value; resVal.textContent = snap(S.R, S.P) + ' px'; cv.redraw(); readout(); });
  rngRes.addEventListener('change', syncControls);
  rngP.addEventListener('input', () => { stop(); const fr = [S.sel[0] / grid().g, S.sel[1] / grid().g]; S.P = PS[+rngP.value];
    const { g } = grid(); S.sel = [Math.floor(fr[0] * g), Math.floor(fr[1] * g)]; syncControls(); cv.redraw(); readout(); });

  // Click a patch to select it, or Tab to the canvas and move the selection with the arrow keys.
  el.addEventListener('pointerdown', e => { const b = el.getBoundingClientRect(), x = e.clientX - b.left, y = e.clientY - b.top, L = layout(b.width), { g } = grid();
    if (x < L.ix || y < L.iy || x > L.ix + L.side || y > L.iy + L.side) return;
    S.sel = [Math.floor((y - L.iy) / L.side * g), Math.floor((x - L.ix) / L.side * g)]; cv.redraw(); readout(); });
  el.addEventListener('keydown', e => { const d = { ArrowUp: [-1, 0], ArrowDown: [1, 0], ArrowLeft: [0, -1], ArrowRight: [0, 1] }[e.key]; if (!d) return;
    e.preventDefault(); S.sel = [S.sel[0] + d[0], S.sel[1] + d[1]]; clampSel(); cv.redraw(); readout(); });
  clampSel(); syncControls(); readout();
})();
</script>
