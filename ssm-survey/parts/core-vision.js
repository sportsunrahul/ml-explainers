/* CoreVision: how far a patch can be reached by others when a 14 x 14 patch grid is flattened and scanned.
   Shared by the long page (viz-vision) and atlas stage 9. Toy: one layer, one constant per-step decay a,
   routes combined by taking the strongest one (the real models differ: Vim sums its two gated outputs,
   VMamba's merge rule is not stated in its text). The token counts and MambaOut's L vs 6D rule are exact. */
const CoreVision = (() => {
  const G = 14, NP = G * G;
  const rows = [...Array(NP).keys()];
  const cols = []; for (let c = 0; c < G; c++) for (let r = 0; r < G; r++) cols.push(r * G + c);
  const R = (name, ord) => ({ name, ord, pos: ord.reduce((p, q, i) => { p[q] = i; return p; }, new Array(NP)) });
  const ROW = R('rows', rows), ROWR = R('rows, reversed', rows.slice().reverse()), COL = R('columns', cols), COLR = R('columns, reversed', cols.slice().reverse());
  const SCANS = {
    raster: { label: 'Causal raster scan', short: 'raster', routes: [ROW] },
    vim: { label: 'Vim: forward + backward', short: 'Vim', routes: [ROW, ROWR] },
    vmamba: { label: 'VMamba: four routes', short: 'VMamba', routes: [ROW, COL, ROWR, COLR] },
  };
  // Influence of every patch q on target p: on route r, q is seen only if it comes no later than p, with weight
  // a^(steps between them); a selective SSM multiplies a different factor per step, the toy uses one a for all.
  function influence(kind, p, a) {
    const out = new Float64Array(NP);
    for (const r of SCANS[kind].routes) for (let q = 0; q < NP; q++) {
      const d = r.pos[p] - r.pos[q];
      if (d >= 0) out[q] = Math.max(out[q], Math.pow(a, d));
    }
    return out;
  }
  const blend = (A, B, u) => A.map((v, i) => v + (B[i] - v) * u);
  const tokens = (res, patch) => Math.pow(Math.floor(res / patch), 2);
  const verdict = (L, D) => ({ tau: 6 * D, r: L / (6 * D), long: L > 6 * D });
  const nf = x => Math.round(x).toLocaleString('en-US');

  // Geometry shared by drawing and hit-testing. Side by side when there is room, stacked when the canvas is tall.
  function layout(W, H) {
    if (H > W * 1.12) {
      const side = Math.min(W - 32, H * .56), gx = (W - side) / 2, gy = 40;
      return { mode: 'stacked', gx, gy, side, cs: side / G, px: 16, py: gy + side + 26, pw: W - 32, ph: H - (gy + side + 26) - 8 };
    }
    const side = Math.min(H - 64, W * .5), gx = 18, gy = Math.max(34, (H - side) / 2 + 10), px = gx + side + Math.max(24, W * .04);
    if (W - px - 18 >= 190) return { mode: 'wide', gx, gy, side, cs: side / G, px, py: 46, pw: W - px - 18, ph: H - 54 };
    // Short, narrow stage (the atlas on a phone): the grid and a column of route glyphs; the readout carries the rest.
    const s2 = Math.min(H - 52, W - 96), gy2 = Math.max(30, (H - s2) / 2 + 8);
    return { mode: 'compact', gx: 12, gy: gy2, side: s2, cs: s2 / G, px: 12 + s2 + 16, py: gy2, pw: W - s2 - 36, ph: s2 };
  }
  function hit(W, H, x, y) {
    const g = layout(W, H), c = Math.floor((x - g.gx) / g.cs), r = Math.floor((y - g.gy) / g.cs);
    return r >= 0 && r < G && c >= 0 && c < G ? r * G + c : -1;
  }
  const move = (p, key) => {
    let r = Math.floor(p / G), c = p % G;
    if (key === 'ArrowUp') r--; else if (key === 'ArrowDown') r++; else if (key === 'ArrowLeft') c--; else if (key === 'ArrowRight') c++; else return p;
    return PE.clamp(r, 0, G - 1) * G + PE.clamp(c, 0, G - 1);
  };

  // A small 3 x 3 grid with its scan order drawn as a path: VMamba's Fig. 2 picture of each route.
  function glyph(ctx, x, y, s, route, col, rule, alpha = 1) {
    const k = s / 3;
    ctx.save(); ctx.globalAlpha = alpha; ctx.strokeStyle = rule; ctx.lineWidth = 1;
    for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) ctx.strokeRect(x + j * k + .5, y + i * k + .5, k - 1, k - 1);
    ctx.restore();
    // Same route rule on a 3 x 3 grid: rows, columns, or either reversed.
    const small = [];
    const byCol = route === COL || route === COLR, rev = route === ROWR || route === COLR;
    for (let a = 0; a < 3; a++) for (let b = 0; b < 3; b++) small.push(byCol ? [b, a] : [a, b]);
    if (rev) small.reverse();
    const pts = small.map(([i, j]) => [x + (j + .5) * k, y + (i + .5) * k]);
    PE.polyline(ctx, pts, col, 1.6, null, alpha);
    PE.arrow(ctx, pts[7][0], pts[7][1], pts[8][0], pts[8][1], col, 1.6, 6, alpha);
    PE.dot(ctx, pts[0][0], pts[0][1], 2.6, col, alpha);
  }

  // S: { kind, heat (Float64Array), tr, tc (target row/col, may be fractional mid-tween), a, res, patch, D }.
  function draw(ctx, W, H, S, glow = () => 0) {
    const C = { bg: PE.css('--bg'), fg: PE.css('--fg'), mu: PE.css('--muted'), rule: PE.css('--rule'), pan: PE.css('--panel2'),
      blue: PE.css('--blue'), green: PE.css('--green'), red: PE.css('--red'), yel: PE.css('--yellow'), brown: PE.css('--brown') };
    ctx.clearRect(0, 0, W, H); ctx.fillStyle = C.bg; ctx.fillRect(0, 0, W, H);
    const g = layout(W, H), { gx, gy, cs } = g, small = W < 520;
    const gB = glow('before'), gD = glow('decay'), gP = glow('patches'), gR = glow('routes'), gT = glow('target');
    PE.label(ctx, `${G} x ${G} patches = ${NP} tokens, one layer`, gx, gy - 10, C.mu, small ? 11 : 12);
    // Heat: blue = how strongly that input patch can reach the green patch's output. Square root so a^14 still shows.
    for (let q = 0; q < NP; q++) {
      const r = Math.floor(q / G), c = q % G, v = S.heat[q], x = gx + c * cs, y = gy + r * cs;
      if (v <= 1e-4) { ctx.save(); ctx.strokeStyle = C.rule; ctx.lineWidth = 1; ctx.strokeRect(x + 1.5, y + 1.5, cs - 3, cs - 3); ctx.restore(); }
      else { ctx.fillStyle = C.pan; ctx.fillRect(x + 1, y + 1, cs - 2, cs - 2); }
      if (v > 1e-4) { ctx.globalAlpha = Math.min(1, (.06 + .94 * Math.sqrt(v)) * (1 + .5 * gD)); ctx.fillStyle = C.blue; ctx.fillRect(x + 1, y + 1, cs - 2, cs - 2); ctx.globalAlpha = 1; }
      if (gB > .01 && v > 1e-4) { ctx.save(); ctx.globalAlpha = gB; ctx.strokeStyle = C.fg; ctx.lineWidth = 1.5; ctx.strokeRect(x + 1.5, y + 1.5, cs - 3, cs - 3); ctx.restore(); }
      if (gP > .01) { ctx.save(); ctx.globalAlpha = gP; ctx.strokeStyle = C.blue; ctx.lineWidth = 1.5; ctx.strokeRect(x + 2, y + 2, cs - 4, cs - 4); ctx.restore(); }
    }
    // While "routes" glows, trace the first route through the grid.
    const routes = SCANS[S.kind].routes;
    if (gR > .01) PE.polyline(ctx, routes[0].ord.map(q => [gx + (q % G + .5) * cs, gy + (Math.floor(q / G) + .5) * cs]), C.fg, 1.2, null, .7 * gR);
    // The target patch: its output y is what we ask about.
    const tx = gx + S.tc * cs, ty = gy + S.tr * cs;
    ctx.save(); ctx.strokeStyle = C.green; ctx.lineWidth = 3 + 3 * gT; ctx.strokeRect(tx + 1, ty + 1, cs - 2, cs - 2); ctx.restore();
    PE.dot(ctx, tx + cs / 2, ty + cs / 2, Math.max(2.5, cs * .16), C.green);

    // Panel: routes, the heat key, and the sequence-length gauge.
    const px = g.px, pw = g.pw, label2 = (t, x, yy) => t.split(', ').forEach((w, i) => PE.label(ctx, w, x, yy + 11 * i, C.mu, 10, 'center'));
    const rc = gR > .01 ? C.fg : C.mu;
    if (g.mode === 'compact') {
      const gs = Math.min(30, (g.ph - 10) / 4 - 26);
      routes.forEach((r, i) => { const yy = g.py + i * (gs + 30); glyph(ctx, px + (pw - gs) / 2, yy, gs * (1 + .12 * gR), r, rc, C.rule); label2(r.name, px + pw / 2, yy + gs + 12); });
      return;
    }
    const gs = Math.min(small ? 34 : 52, (pw - 30) / 4 - 14), need = gs + 54 + 52 + 130;
    let y = g.mode === 'wide' ? g.py + Math.max(0, (g.ph - need) / 2 - 20) : g.py;
    PE.label(ctx, routes.length === 1 ? 'one route, causal' : `${routes.length} routes, each its own SSM`, px, y, C.fg, small ? 11 : 12);
    y += 10;
    routes.forEach((r, i) => { const x = px + i * (gs + (small ? 30 : 38)); glyph(ctx, x, y, gs * (1 + .12 * gR), r, rc, C.rule); label2(r.name, x + gs / 2, y + gs + 13); });
    y += gs + 40;
    // Heat key.
    const kw = Math.min(pw - 60, 230);
    for (let i = 0; i < 40; i++) { const v = Math.pow(1 - i / 39, 2); ctx.globalAlpha = .06 + .94 * Math.sqrt(v); ctx.fillStyle = C.blue; ctx.fillRect(px + i * kw / 40, y, kw / 40 + .5, 9); }
    ctx.globalAlpha = 1; ctx.save(); ctx.strokeStyle = C.rule; ctx.strokeRect(px + kw + 8.5, y + .5, 11, 9); ctx.restore();
    PE.label(ctx, 'reaches the green patch: strongly', px, y + 22, C.mu, 10);
    PE.label(ctx, 'weakly', px + kw, y + 22, C.mu, 10, 'right');
    PE.label(ctx, 'never', px + kw + 24, y + 9, C.mu, 10);
    y += small ? 40 : 52;
    gauge(ctx, px, y, pw, Math.max(70, Math.min(130, g.py + g.ph - y)), S, C, glow, small);
  }

  // Log axis of sequence length with MambaOut's threshold 6D for the chosen width.
  function gauge(ctx, x, y, w, h, S, C, glow, small) {
    const L = tokens(S.res, S.patch), v = verdict(L, S.D), lo = 40, hi = 30000;
    const X = t => x + (Math.log10(t) - Math.log10(lo)) / (Math.log10(hi) - Math.log10(lo)) * w;
    const ay = y + h - 22, gQ = glow('quad'), gL = glow('linear'), gT = glow('threshold');
    PE.label(ctx, 'sequence length L (log scale)', x, y + 4, C.fg, small ? 11 : 12);
    // Short side and long side of the threshold.
    ctx.save(); ctx.globalAlpha = .1 + .25 * gL; ctx.fillStyle = C.mu; ctx.fillRect(x, ay - 24, X(v.tau) - x, 24);
    ctx.globalAlpha = .14 + .3 * gQ; ctx.fillStyle = C.red; ctx.fillRect(X(v.tau), ay - 24, x + w - X(v.tau), 24); ctx.restore();
    PE.polyline(ctx, [[X(v.tau), ay - 30], [X(v.tau), ay + 2]], C.red, 2 + 2 * gT);
    const tl = `6D = ${nf(v.tau)}`;
    PE.label(ctx, tl, X(v.tau) + (X(v.tau) > x + w - 70 ? -4 : 4), ay - 32, C.red, 11, X(v.tau) > x + w - 70 ? 'right' : 'left');
    PE.label(ctx, 'short', x + 4, ay - 8, C.mu, 10);
    PE.label(ctx, small ? 'long' : "long: attention's L\u00b2 term wins", x + w - 4, ay - 8, C.red, 10, 'right');
    // Axis and ticks.
    PE.polyline(ctx, [[x, ay], [x + w, ay]], C.mu, 1);
    [[100, '100'], [1000, '1K'], [10000, '10K']].forEach(([t, s]) => { PE.polyline(ctx, [[X(t), ay], [X(t), ay + 4]], C.mu, 1); PE.label(ctx, s, X(t), ay + 15, C.mu, 10, 'center'); });
    // Reference lengths from the papers.
    [[196, 'ImageNet'], [4096, small ? '~4K' : 'COCO, ADE20K ~4K'], [6084, '1248 px']].forEach(([t, s], i) => {
      ctx.save(); ctx.fillStyle = C.mu; ctx.beginPath(); ctx.moveTo(X(t), ay + 1); ctx.lineTo(X(t) - 3, ay + 7); ctx.lineTo(X(t) + 3, ay + 7); ctx.fill(); ctx.restore();
      if (!small || i !== 1) PE.label(ctx, s, X(t), ay + (i === 2 ? 27 : 27), C.mu, 9.5, i === 2 ? 'left' : i === 1 ? 'right' : 'center');
    });
    PE.polyline(ctx, [[X(PE.clamp(L, lo, hi)), ay - 26], [X(PE.clamp(L, lo, hi)), ay]], C.blue, 2);
    PE.dot(ctx, X(PE.clamp(L, lo, hi)), ay, 5.5, C.blue);
    PE.label(ctx, `L = ${nf(L)}`, PE.clamp(X(L), x + 24, x + w - 24), ay - 40 - (Math.abs(X(L) - X(v.tau)) < 60 ? 12 : 0), C.blue, 11, 'center');
  }

  const SCAN_TXT = {
    raster: 'one causal raster scan',
    vim: "Vim's two scans (row order, forward and backward)",
    vmamba: "VMamba's four routes",
  };
  function readout(S) {
    const p = Math.round(S.tr) * G + Math.round(S.tc), h = influence(S.kind, p, S.a);
    const n = h.reduce((s, v) => s + (v > 0 ? 1 : 0), 0), r = Math.floor(p / G), c = p % G;
    const at = (dr, dc) => (r + dr < 0 || r + dr >= G || c + dc < 0 || c + dc >= G) ? null : h[(r + dr) * G + c + dc];
    const f = v => v === null ? 'none (edge)' : v === 0 ? '0' : v >= .01 ? v.toFixed(2) : v.toExponential(0);
    const L = tokens(S.res, S.patch), v = verdict(L, S.D);
    let s = `Toy, one layer, decay ${S.a.toFixed(2)} per step: with ${SCAN_TXT[S.kind]}, ${n} of ${NP} patches can reach the green one; the patch above gives ${f(at(-1, 0))}, below ${f(at(1, 0))}, right ${f(at(0, 1))}. `;
    s += `At ${S.res} px with ${S.patch} px patches L = ${nf(L)} tokens; with width D = ${S.D}, L/6D = ${v.r < .1 ? v.r.toFixed(2) : v.r.toFixed(1)}, ${v.long ? 'long' : 'short'} by MambaOut's rule. Attention would score ${nf(L * L)} token pairs per head per layer; a scan keeps the same state at any L.`;
    if (L === 6084) s += ' Vim measured this size: Vim-Ti 2.8x faster than DeiT-Ti, 86.8% less GPU memory (backbone only).';
    return s;
  }
  return { G, NP, SCANS, influence, blend, tokens, verdict, layout, hit, move, draw, readout };
})();
