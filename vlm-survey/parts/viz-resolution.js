<script>
(() => {
  const fig = document.getElementById('viz-resolution'), el = document.getElementById('resolution-canvas'), out = document.getElementById('resolution-readout');

  // ---- Procedural test images. The same two functions are in viz-tokens.js (chapter 3). ----
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

  const PRE = {
    doc: { name: 'Document page', kind: 'doc', W: 992, H: 1404, txt: 'body text' },
    wide: { name: 'Wide screenshot', kind: 'screen', W: 1920, H: 1080, txt: 'code' },
    photo: { name: 'Small photo', kind: 'receipt', W: 400, H: 300, txt: 'item text' } };
  const STR = ['fixed224', 'fixed336', 'fixed448', 'tile', 'native'];
  const SNAME = { fixed224: '224 square', fixed336: '336 square', fixed448: '448 square', tile: 'tiles + thumbnail', native: 'native size' };
  const scenes = {}, comps = {};
  const scene = k => scenes[k] || (scenes[k] = renderScene(PRE[k].kind, PRE[k].W, PRE[k].H));
  const fmt = n => Math.round(n).toLocaleString('en-US');
  // Tiling grid, Molmo's rule: the least up-scaling among grids that cover the image (ties: fewer crops),
  // otherwise the least down-scaling. Toy settings: 224-px crops, at most 6 of them.
  function tileGrid(W, H) {
    let best = null;
    for (let c = 1; c <= 6; c++) for (let r = 1; c * r <= 6; r++) {
      const s = Math.min(c * 224 / W, r * 224 / H), n = c * r, cover = s >= 1, eq = best && Math.abs(s - best.s) < 1e-9;
      const better = !best || (cover !== best.cover ? cover : cover ? s < best.s - 1e-9 || (eq && n < best.n) : s > best.s + 1e-9 || (eq && n < best.n));
      if (better) best = { c, r, s, n, cover };
    }
    return best;
  }
  function comp(pk, st) {
    const key = pk + ':' + st; if (comps[key]) return comps[key];
    const P = PRE[pk], sc = scene(pk), W = P.W, H = P.H; let o;
    if (st.startsWith('fixed')) {
      const Q = +st.slice(5);
      // Squash into the square (aspect ratio lost). One token per 14-px patch, except Qwen-VL's adapter: 256 queries
      // plus its <img> and </img> markers (Sec. 2.2). The LLaVA-style 224 and 336 counts have no markers (neither paper describes any).
      o = { img: downsample(sc.c, Q, Q), mw: Q, mh: Q, sx: Q / W, sy: Q / H, vit: (Q / 14) ** 2, llm: Q === 448 ? 256 + 2 : (Q / 14) ** 2 };
    } else if (st === 'tile') {
      const g = tileGrid(W, H), w = Math.round(W * g.s), h = Math.round(H * g.s), gw = g.c * 224, gh = g.r * 224;
      const c = document.createElement('canvas'); c.width = gw; c.height = gh; const x = c.getContext('2d');
      x.fillStyle = '#000'; x.fillRect(0, 0, gw, gh); x.drawImage(downsample(sc.c, w, h), 0, 0);
      const t = Math.min(224 / W, 224 / H), th = document.createElement('canvas'); th.width = 224; th.height = 224; const y = th.getContext('2d');
      y.fillStyle = '#000'; y.fillRect(0, 0, 224, 224); y.drawImage(downsample(sc.c, W * t, H * t), Math.round((224 - W * t) / 2), Math.round((224 - H * t) / 2));
      const cols = Math.ceil(w / 14), rows = Math.ceil(h / 14);
      // LLaVA-1.5-HD counting: padding-only features dropped, one row-end token per row, plus the 224-px thumbnail's 256.
      o = { img: c, mw: gw, mh: gh, cw: w, ch: h, sx: g.s, sy: g.s, g, thumb: th, vit: (g.n + 1) * 256, grid: cols * rows, rows, llm: cols * rows + rows + 256 };
    } else {
      // Qwen2-VL: close to the image's own size. The toy snaps each side to the nearest multiple of 28 (the paper does not say how).
      const w = Math.max(28, Math.round(W / 28) * 28), h = Math.max(28, Math.round(H / 28) * 28);
      o = { img: downsample(sc.c, w, h), mw: w, mh: h, sx: w / W, sy: h / H, vit: (w / 14) * (h / 14), llm: (w / 28) * (h / 28) + 2 };
    }
    o.small = sc.small * o.sy; o.zoom = sc.zoom; o.pk = pk; o.st = st;
    return (comps[key] = o);
  }

  const S = { pk: 'doc', st: 'fixed336', u: 1, prev: null, from: null };
  const glow = PE.linkPulses(fig, () => cv.redraw());
  let stop = () => {};
  const layout = (W, H) => {
    if (W < 560) { const bw = W - 24, bh = Math.round(bw * .92), zy = 30 + bh + 84;
      return { st: true, bx: 12, by: 30, bw, bh, sy: 30 + bh + 34, zx: 12, zy, zw: bw, zh: 96, ty: zy + 96 + 20, qy: zy + 96 + 62, qx: 12, qw: bw, rh: 30 }; }
    const bw = Math.round(W * .56) - 24, bh = H - 84, rx = bw + 40, rw = W - rx - 12;
    return { st: false, bx: 12, by: 30, bw, bh, sy: H - 30, zx: rx, zy: 48, zw: rw, zh: 118, ty: 48 + 118 + 22, qy: 48 + 118 + 62, qx: rx, qw: rw, rh: 38 };
  };

  function drawMain(ctx, o, L, a, full) {
    const dpr = devicePixelRatio || 1, B = PE.css('--blue'), F = PE.css('--fg'), Br = PE.css('--brown'), BG = PE.css('--bg');
    const boxW = o.thumb ? L.bw * .72 : L.bw;
    const k = Math.min(boxW / o.mw, L.bh / o.mh), dw = o.mw * k, dh = o.mh * k, dx = L.bx + (boxW - dw) / 2, dy = L.by + (L.bh - dh) / 2;
    ctx.save(); ctx.globalAlpha = a;
    ctx.imageSmoothingEnabled = dw * dpr < o.mw; ctx.drawImage(o.img, dx, dy, dw, dh); ctx.imageSmoothingEnabled = true;
    if (full) {
      const gp = glow('patch'), gm = glow('merge'), gs = glow('size');
      const lines = (step, alpha, lw, x1, y1) => { if (alpha <= 0) return; ctx.save(); ctx.globalAlpha = a * alpha; ctx.strokeStyle = B; ctx.lineWidth = lw; ctx.beginPath();
        for (let v = step; v < x1 - .01; v += step) { const t = dx + v * k; ctx.moveTo(t, dy); ctx.lineTo(t, dy + y1 * k); }
        for (let v = step; v < y1 - .01; v += step) { const t = dy + v * k; ctx.moveTo(dx, t); ctx.lineTo(dx + x1 * k, t); }
        ctx.stroke(); ctx.restore(); };
      const sp = 14 * k, pa = PE.clamp((sp - 2.5) / 8, 0, .45) + .5 * gp;
      if (o.st === 'native') { lines(14, pa * .6, 1, o.mw, o.mh); lines(28, PE.clamp((28 * k - 3) / 10, .12, .75) + .4 * gm, 1 + 2 * gm, o.mw, o.mh); }
      else if (o.st === 'tile') { lines(14, pa, 1, o.cw, o.ch);
        ctx.save(); ctx.globalAlpha = a * .5; ctx.fillStyle = Br; if (o.cw < o.mw) ctx.fillRect(dx + o.cw * k, dy, (o.mw - o.cw) * k, dh);
        if (o.ch < o.mh) ctx.fillRect(dx, dy + o.ch * k, o.cw * k, (o.mh - o.ch) * k); ctx.restore();
        // Crop boundaries: a background halo under a dashed foreground line, visible on any image.
        ctx.save(); ctx.globalAlpha = a; ctx.beginPath();
        for (let c = 1; c < o.g.c; c++) { ctx.moveTo(dx + c * 224 * k, dy); ctx.lineTo(dx + c * 224 * k, dy + dh); }
        for (let r = 1; r < o.g.r; r++) { ctx.moveTo(dx, dy + r * 224 * k); ctx.lineTo(dx + dw, dy + r * 224 * k); }
        ctx.strokeStyle = BG; ctx.lineWidth = 5; ctx.stroke(); ctx.strokeStyle = F; ctx.lineWidth = 2; ctx.setLineDash([7, 4]); ctx.stroke(); ctx.restore();
        const ts = Math.min(L.bw - boxW - 12, L.bh * .5), tx = L.bx + boxW + 10, ty = L.by + 18;
        ctx.imageSmoothingEnabled = ts * dpr < 224; ctx.drawImage(o.thumb, tx, ty, ts, ts); ctx.imageSmoothingEnabled = true;
        ctx.strokeStyle = B; ctx.lineWidth = 1.5; ctx.strokeRect(tx, ty, ts, ts);
        PE.label(ctx, 'thumbnail', tx, ty - 6, F, 11); PE.label(ctx, '224 x 224', tx, ty + ts + 14, PE.css('--muted'), 10.5);
      } else lines(14, pa, 1, o.mw, o.mh);
      ctx.strokeStyle = B; ctx.lineWidth = 1 + 3 * gs; ctx.strokeRect(dx - .5, dy - .5, dw + 1, dh + 1);
      // Where the zoom panel looks.
      const z = o.zoom, zx = dx + z[0] * o.sx * k, zy = dy + z[1] * o.sy * k, zw = Math.max(4, z[2] * o.sx * k), zh = Math.max(4, z[3] * o.sy * k);
      ctx.strokeStyle = BG; ctx.lineWidth = 4; ctx.strokeRect(zx - 2, zy - 2, zw + 4, zh + 4); ctx.strokeStyle = F; ctx.lineWidth = 1.5; ctx.strokeRect(zx - 2, zy - 2, zw + 4, zh + 4);
    }
    ctx.restore();
    return { dx, dy, dw, dh };
  }
  function drawZoom(ctx, o, L, a) {
    const z = o.zoom, x0 = Math.floor(z[0] * o.sx), y0 = Math.floor(z[1] * o.sy), x1 = Math.ceil((z[0] + z[2]) * o.sx), y1 = Math.ceil((z[1] + z[3]) * o.sy);
    const mw = Math.max(1, x1 - x0), mh = Math.max(1, y1 - y0), k = Math.min(L.zw / mw, L.zh / mh), dw = mw * k, dh = mh * k;
    const dx = L.zx + (L.zw - dw) / 2, dy = L.zy + (L.zh - dh) / 2;
    ctx.save(); ctx.globalAlpha = a; ctx.imageSmoothingEnabled = false; ctx.drawImage(o.img, x0, y0, mw, mh, dx, dy, dw, dh); ctx.restore();
  }
  function stripItems(o) {
    const y = 'y', b = 'b';
    if (o.st === 'fixed448') return [['<img>', y], [fmt(256) + ' query tokens', b], ['</img>', y]];
    if (o.st === 'tile') return [['thumbnail 256', b], ['grid ' + fmt(o.grid), b], [fmt(o.rows) + ' row ends', y]];
    if (o.st === 'native') return [['<|vision_start|>', y], [fmt(o.llm - 2) + ' image tokens', b], ['<|vision_end|>', y]];
    return [[fmt(o.llm) + ' image tokens', b]];
  }

  const cv = PE.canvas(el, w => w < 560 ? 2.14 : .74, (ctx, W, H) => {
    const B = PE.css('--blue'), Yl = PE.css('--yellow'), Rd = PE.css('--red'), F = PE.css('--fg'), M = PE.css('--muted'), Rule = PE.css('--rule'), BG = PE.css('--bg');
    ctx.fillStyle = BG; ctx.fillRect(0, 0, W, H);
    const L = layout(W, H), o = comp(S.pk, S.st), u = S.u, pv = S.prev;
    PE.label(ctx, L.st ? `what the encoder gets: ${o.mw} x ${o.mh} px` : `what the encoder gets: ${o.mw} x ${o.mh} pixels (${PRE[S.pk].W} x ${PRE[S.pk].H} originally)`, L.bx, 18, F, L.st ? 11 : 12);
    if (pv && u < 1) drawMain(ctx, pv, L, 1 - u, false);
    drawMain(ctx, o, L, pv ? u : 1, true);

    // What the language model reads: image tokens (blue) and marker tokens (yellow, they are text-vocabulary tokens).
    const gsp = glow('special'); let x = L.bx; let fs = L.st ? 9.5 : 11;
    // Shrink the row's font until it fits the canvas (narrow screens, long marker tokens).
    const rowW = f => { ctx.font = `600 ${f}px system-ui, sans-serif`;
      return (L.st ? 0 : ctx.measureText('LLM reads:').width + 8) + stripItems(o).reduce((s, [t]) => s + ctx.measureText(t).width + 17, 0); };
    while (fs > 7.5 && L.bx + rowW(fs) > W - 4) fs -= .5;
    ctx.save(); ctx.globalAlpha = pv ? u : 1; ctx.font = `600 ${fs}px system-ui, sans-serif`;
    if (!L.st) { PE.label(ctx, 'LLM reads:', x, L.sy + 4, M, fs); x += ctx.measureText('LLM reads:').width + 8; }
    else PE.label(ctx, 'what the LLM reads:', x, L.sy - 16, M, 10.5);
    stripItems(o).forEach(([t, c]) => { const w = ctx.measureText(t).width + 12, col = c === 'y' ? Yl : B, gl = c === 'y' ? gsp : 0;
      ctx.strokeStyle = col; ctx.lineWidth = 1.3 + 2.5 * gl; ctx.globalAlpha = (pv ? u : 1); ctx.strokeRect(x, L.sy - 10, w, 19);
      ctx.globalAlpha = (pv ? u : 1) * (.14 + .3 * gl); ctx.fillStyle = col; ctx.fillRect(x, L.sy - 10, w, 19); ctx.globalAlpha = pv ? u : 1;
      PE.label(ctx, t, x + 6, L.sy + 4, col, fs); x += w + 5; });
    ctx.restore();

    // Zoom: one line of small text, drawn with the model's own pixels (one square per pixel).
    PE.label(ctx, 'zoom: the boxed text, in the model\'s own pixels', L.zx, L.zy - 8, F, L.st ? 11 : 11.5);
    ctx.fillStyle = PE.css('--panel'); ctx.fillRect(L.zx, L.zy, L.zw, L.zh);
    if (pv && u < 1) drawZoom(ctx, pv, L, 1 - u);
    drawZoom(ctx, o, L, pv ? u : 1);
    ctx.strokeStyle = Rule; ctx.lineWidth = 1; ctx.strokeRect(L.zx + .5, L.zy + .5, L.zw - 1, L.zh - 1);
    const sc = scene(S.pk), small = pv && u < 1 ? PE.lerp(pv.pk === S.pk ? pv.small : o.small, o.small, u) : o.small;
    PE.label(ctx, `${fmt(sc.small)}-px ${PRE[S.pk].txt}: ${small.toFixed(1)} px tall at the encoder`, L.zx, L.ty, small < 6 ? Rd : F, L.st ? 11 : 12);

    // Tokens the language model reads for this image, per strategy (red = cost). Bars count up on a change.
    PE.label(ctx, 'tokens the LLM reads, per strategy', L.qx, L.qy - 12, F, L.st ? 11 : 12);
    const vals = STR.map(s => comp(S.pk, s).llm), shown = vals.map((v, i) => S.from && u < 1 ? PE.lerp(S.from[i], v, u) : v);
    const mx = Math.max(...vals, ...(S.from && u < 1 ? S.from : [])), lw = L.st ? 96 : 104, bx = L.qx + lw, bwid = L.qw - lw - 50;
    STR.forEach((s, i) => { const y = L.qy + i * L.rh, cur = s === S.st, h = L.st ? 14 : 16;
      PE.label(ctx, SNAME[s], L.qx, y + h - 3, cur ? F : M, L.st ? 10.5 : 11.5);
      ctx.fillStyle = Rule; ctx.fillRect(bx, y, bwid, h);
      ctx.fillStyle = cur ? Rd : M; ctx.globalAlpha = cur ? 1 : .55; ctx.fillRect(bx, y, bwid * shown[i] / mx, h); ctx.globalAlpha = 1;
      PE.label(ctx, fmt(shown[i]), bx + bwid + 6, y + h - 3, cur ? F : M, 11); });
  });

  function readout() {
    const o = comp(S.pk, S.st), P = PRE[S.pk], sc = scene(S.pk);
    const how = ({
      fixed224: () => `resized into a 224 x 224 square (aspect ratio lost); ${fmt(o.vit)} patches of 14 px, one token each`,
      fixed336: () => `resized into a 336 x 336 square (aspect ratio lost); ${fmt(o.vit)} patches of 14 px, one token each`,
      fixed448: () => `resized into a 448 x 448 square (aspect ratio lost); its ${fmt(o.vit)} patches are compressed by one cross-attention layer to ${fmt(o.llm - 2)} tokens, plus 2 marker tokens`,
      tile: () => `resized to ${o.cw} x ${o.ch} and cut into a ${o.g.c} x ${o.g.r} grid of 224-px crops (${fmt(o.grid)} patch tokens, ${o.rows} row-end tokens) plus a 224-px thumbnail (256 tokens)`,
      native: () => `resized only to ${o.mw} x ${o.mh}, the nearest multiples of 28; ${fmt(o.vit)} patches of 14 px, merged 2 x 2 into ${fmt(o.llm - 2)} tokens, plus 2 marker tokens`,
    })[S.st]();
    out.textContent = `${P.name} (${P.W} x ${P.H} px), ${SNAME[S.st]}: ${how}. The language model reads ${fmt(o.llm)} tokens for this image, and the ${fmt(sc.small)}-px ${P.txt} reaches the encoder ${o.small.toFixed(1)} px tall.`;
  }
  function go(pk, st) {
    if (pk === S.pk && st === S.st) return;
    stop(); const prev = comp(S.pk, S.st), from = STR.map(s => comp(S.pk, s).llm);
    S.pk = pk; S.st = st; S.prev = prev; S.from = from; S.u = 0; press(); readout();
    stop = PE.animate(800, v => { S.u = v; cv.redraw(); }, () => { S.prev = null; S.from = null; S.u = 1; cv.redraw(); });
  }
  const btns = {};
  [...Object.keys(PRE), ...STR].forEach(k => { btns[k] = document.getElementById('btn-resolution-' + k); });
  function press() { Object.keys(btns).forEach(k => btns[k].setAttribute('aria-pressed', String(k === S.pk || k === S.st))); }
  Object.keys(PRE).forEach(k => btns[k].addEventListener('click', () => go(k, S.st)));
  STR.forEach(k => btns[k].addEventListener('click', () => go(S.pk, k)));
  document.getElementById('btn-resolution-reset').addEventListener('click', () => { stop(); S.pk = 'doc'; S.st = 'fixed336'; S.prev = null; S.from = null; S.u = 1; press(); cv.redraw(); readout(); });
  el.addEventListener('keydown', e => { const d = { ArrowLeft: -1, ArrowRight: 1 }[e.key]; if (!d) return; e.preventDefault();
    go(S.pk, STR[(STR.indexOf(S.st) + d + STR.length) % STR.length]); });
  press(); readout();
})();
</script>
