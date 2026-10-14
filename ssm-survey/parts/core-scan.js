/* CoreScan: a linear recurrence h_t = a_t h_(t-1) + b_t computed by a work-efficient parallel scan (up-sweep, then
   down-sweep, then one combine per token), drawn as a tree you can step through; and where the expanded state lives
   on a GPU. Shared by the long page (viz-scan, viz-scanmem) and atlas stage 7.
   A step is the pair (a, b). Applying p then q is one pair again: combine(p, q) = (a_p a_q, a_q b_p + b_q). */
const CoreScan = (() => {
  const combine = (p, q) => [p[0] * q[0], q[0] * p[1] + q[1]];
  const ID = [1, 0];
  // Toy inputs: a_t = A_bar_t in (.45, .95) (a decay set by token t), b_t = B_bar_t x_t with x_t in [-1, 1].
  function data(L, seed) {
    const r = PE.rng(seed), a = [], b = [];
    for (let t = 0; t < L; t++) { const at = .45 + .5 * r(), x = 2 * r() - 1; a.push(at); b.push((1 - at) * x); }
    return { a, b };
  }
  function sequential(a, b) { let h = 0; return a.map((at, t) => (h = at * h + b[t])); }
  // U[l][i]: the pair for tokens i*2^l .. (i+1)*2^l - 1 (up-sweep). D[l][i]: the pair for every token before that span
  // (down-sweep), so its b is the state just before the span. H[t] = combine(D[0][t], U[0][t]) b-part = h_t.
  function tree(a, b) {
    const L = a.length, m = Math.round(Math.log2(L)), U = [a.map((at, t) => [at, b[t]])];
    for (let l = 1; l <= m; l++) U.push(U[l - 1].filter((_, i) => i % 2 === 0).map((p, i) => combine(p, U[l - 1][2 * i + 1])));
    const D = []; D[m] = [ID];
    for (let l = m; l >= 1; l--) { D[l - 1] = []; D[l].forEach((p, i) => { D[l - 1][2 * i] = p; D[l - 1][2 * i + 1] = combine(p, U[l - 1][2 * i]); }); }
    const H = U[0].map((p, t) => combine(D[0][t], p)[1]);
    return { m, U, D, H };
  }
  // Rounds: 0 = inputs; 1..m = up-sweep level r; m+1..2m = down-sweep to level 2m - r; 2m+1 = final combine.
  const lastRound = L => 2 * Math.round(Math.log2(L)) + 1;
  function combinesIn(L, r) {
    const m = Math.round(Math.log2(L));
    if (r <= 0) return 0; if (r <= m) return L >> r; if (r <= 2 * m) return 1 << (r - m - 1); return L;
  }
  const combinesTo = (L, r) => { let s = 0; for (let k = 1; k <= r; k++) s += combinesIn(L, k); return s; };
  function phase(L, r) { const m = Math.round(Math.log2(L)); return r <= 0 ? 'inputs' : r <= m ? 'up' : r <= 2 * m ? 'down' : 'final'; }
  // Level whose values are computed in round r (up: r; down: 2m - r; final: the leaves' states).
  function levelOf(L, r) { const m = Math.round(Math.log2(L)); return r <= m ? r : r <= 2 * m ? 2 * m - r : -1; }

  // ---------- drawing the tree ----------
  function rr(ctx, x, y, w, h, r, fill, stroke, lw = 1.4, alpha = 1, dash = null) {
    ctx.save(); ctx.globalAlpha = alpha; ctx.beginPath(); ctx.moveTo(x + r, y); ctx.arcTo(x + w, y, x + w, y + h, r); ctx.arcTo(x + w, y + h, x, y + h, r);
    ctx.arcTo(x, y + h, x, y, r); ctx.arcTo(x, y, x + w, y, r); ctx.closePath();
    if (fill) { ctx.fillStyle = fill; ctx.fill(); }
    if (stroke) { ctx.strokeStyle = stroke; ctx.lineWidth = lw; if (dash) ctx.setLineDash(dash); ctx.stroke(); }
    ctx.restore();
  }
  const span = (l, i) => { const s = i << l, e = ((i + 1) << l) - 1; return s === e ? `${s + 1}` : `${s + 1}-${e + 1}`; };
  // S = { L, seed, round, u } ; u in [0, 1] is the tween into the current round.
  function draw(ctx, W, H, S, glow = () => 0) {
    const C = { bg: PE.css('--bg'), fg: PE.css('--fg'), mu: PE.css('--muted'), rule: PE.css('--rule'), B: PE.css('--blue'), Y: PE.css('--yellow'), G: PE.css('--green'), R: PE.css('--red'), panel: PE.css('--panel') };
    ctx.clearRect(0, 0, W, H);
    const L = S.L, { a, b } = data(L, S.seed), T = tree(a, b), m = T.m, r = S.round, u = S.u == null ? 1 : S.u, ph = phase(L, r), lv = levelOf(L, r);
    const narrow = W < 520, fs = narrow ? 9.5 : 11;
    const top = narrow ? 44 : 40, bottom = S.bottom || 0;
    const loopH = narrow ? 34 : 38, stateH = narrow ? 46 : 52;
    const treeBot = H - bottom - loopH - stateH - 10;
    const rowY = l => treeBot - 30 - l * (treeBot - 30 - top) / m;  // y of level l's box top
    const x0 = narrow ? 8 : 16, x1 = W - (narrow ? 8 : 16), slot = (x1 - x0) / L;
    const cx = (l, i) => x0 + slot * ((i << l) + ((1 << l) - 1) / 2 + .5);
    const bw = l => Math.min(slot * (1 << l) - (narrow ? 3 : 6), narrow ? 64 : 120), bh = narrow ? 20 : 24;
    const chipH = narrow ? 13 : 15;
    // Visibility of each node's up value and down value at round r.
    const upShown = l => l === 0 || r >= l, upNew = l => r === l && l > 0;
    const dnShown = l => r > m && 2 * m - r <= l || r > 2 * m, dnNew = l => r === 2 * m - l && r > m;
    const dnRoot = r > m;
    const gComb = glow('combine'), gPair = glow('pair'), gState = glow('state'), gAssoc = glow('assoc'), gLoop = glow('loop');
    // Phase title.
    const title = { inputs: `${L} tokens, each a pair (a, b)`, up: `up-sweep: level ${lv} of ${m}`, down: `down-sweep: level ${lv}`, final: 'one last combine per token' }[ph];
    PE.label(ctx, title, x0, 18, C.fg, narrow ? 11 : 12.5);
    PE.label(ctx, narrow ? `round ${r} of ${lastRound(L)}` : `round ${r} of ${lastRound(L)}: ${combinesIn(L, r)} combine${combinesIn(L, r) === 1 ? '' : 's'} at once`, x0, 33, C.mu, narrow ? 10 : 11);
    // Tree edges (child -> parent), drawn faint for the whole tree.
    for (let l = 1; l <= m; l++) for (let i = 0; i < L >> l; i++) for (const c of [2 * i, 2 * i + 1]) {
      const on = upShown(l), al = on ? (upNew(l) ? .25 + .55 * u : .55) : .18;
      PE.polyline(ctx, [[cx(l - 1, c), rowY(l - 1)], [cx(l, i), rowY(l) + bh]], gAssoc > 0 ? C.fg : C.rule, 1.2 + 1.6 * gAssoc, null, Math.min(1, al + .4 * gAssoc));
      if (upNew(l) && u < 1) PE.dot(ctx, PE.lerp(cx(l - 1, c), cx(l, i), u), PE.lerp(rowY(l - 1), rowY(l) + bh, u), 3.2, C.B);
    }
    // Down-sweep arrows for the level computed this round: parent's state chip -> both children; left sibling's span -> right child.
    if (ph === 'down' && lv >= 0) for (let i = 0; i < L >> (lv + 1); i++) {
      const px = cx(lv + 1, i), py = rowY(lv + 1) - chipH, lx = cx(lv, 2 * i), rx = cx(lv, 2 * i + 1), cy = rowY(lv) - chipH;
      PE.arrow(ctx, px - 6, py + chipH * .5, lx, cy, C.Y, 1.4, 6, .55);
      PE.arrow(ctx, px + 6, py + chipH * .5, rx, cy, C.Y, 1.8, 7, .8);
      PE.arrow(ctx, lx + bw(lv) / 2 - 2, rowY(lv) + bh / 2, rx - bw(lv) / 2 + 2, cy + chipH * .4, C.B, 1.4, 6, .7);
      if (u < 1) PE.dot(ctx, PE.lerp(px + 6, rx, u), PE.lerp(py + chipH * .5, cy, u), 3.2, C.Y);
    }
    // Nodes.
    for (let l = 0; l <= m; l++) for (let i = 0; i < L >> l; i++) {
      const x = cx(l, i) - bw(l) / 2, y = rowY(l), w = bw(l), shown = upShown(l), al = !shown ? .22 : upNew(l) ? .25 + .75 * u : 1;
      const isLeaf = l === 0, lit = (isLeaf ? gPair : gComb) > 0 ? (isLeaf ? gPair : gComb) : 0;
      rr(ctx, x, y, w, bh, 5, PE.mix(C.bg, C.B, shown ? (upNew(l) ? .1 + .18 * u : .14) : .04), C.B, 1.2 + (upNew(l) ? 1.2 : 0) + 2 * lit, al);
      if (shown || !isLeaf) {
        const txt = isLeaf ? (narrow && L > 8 ? span(0, i) : `x${i + 1}`) : span(l, i);
        ctx.save(); ctx.globalAlpha = Math.max(al, .35); PE.label(ctx, txt, cx(l, i), y + bh / 2 + 4, shown ? C.fg : C.mu, fs, 'center'); ctx.restore();
      }
      // The state chip: the state just before this span, delivered by the down-sweep.
      const dsh = l === m ? dnRoot : dnShown(l);
      if (dsh) {
        const da = dnNew(l) ? u : 1, cw = Math.min(w, narrow ? 30 : 46), s0 = i << l;
        rr(ctx, cx(l, i) - cw / 2, y - chipH - 1, cw, chipH, 4, PE.mix(C.bg, C.Y, .22), C.Y, 1.2 + 2 * gState, da);
        ctx.save(); ctx.globalAlpha = da; PE.label(ctx, `h${s0}`, cx(l, i), y - 4, C.Y, narrow ? 8.5 : 10, 'center'); ctx.restore();
      }
    }
    // Leaf values (a, b) on desktop for 8 tokens.
    if (!narrow && L <= 8) for (let t = 0; t < L; t++) PE.label(ctx, `a ${a[t].toFixed(2)}  b ${b[t] >= 0 ? ' ' : ''}${b[t].toFixed(2)}`, cx(0, t), rowY(0) + bh + 13, C.mu, 10, 'center');
    // States row: h_t as bars around a zero line; the final round fills them in.
    const sy = treeBot + (narrow ? 8 : 26), mid = sy + stateH / 2 - 2, hmax = Math.max(...T.H.map(Math.abs), 1e-9);
    PE.polyline(ctx, [[x0, mid], [x1, mid]], C.rule, 1);
    PE.label(ctx, narrow ? 'states' : 'states h_t', x0, sy - 1, C.mu, narrow ? 9 : 10);
    const fin = ph === 'final' ? u : 0;
    for (let t = 0; t < L; t++) {
      const x = x0 + slot * (t + .5), bwd = Math.max(3, Math.min(slot * .5, 22)), hh = (stateH / 2 - 6) * T.H[t] / hmax * fin;
      if (fin > 0) { ctx.save(); ctx.globalAlpha = .85; ctx.fillStyle = C.Y; ctx.fillRect(x - bwd / 2, Math.min(mid, mid - hh), bwd, Math.abs(hh) + .5); ctx.restore(); }
      if (gState > 0 && fin > 0) rr(ctx, x - bwd / 2 - 3, mid - stateH / 2 + 3, bwd + 6, stateH - 6, 3, null, C.Y, 2 * gState);
      if (ph === 'final' && u < 1) PE.dot(ctx, x, PE.lerp(rowY(0) + bh, mid, u), 3, C.Y);
    }
    // The sequential loop, for comparison: one state per round.
    const ly = H - bottom - loopH / 2 + 2, done = Math.min(L, Math.max(0, r - 1 + u));
    PE.label(ctx, narrow ? `loop: ${Math.min(L, r)} of ${L}` : `sequential loop after ${r} round${r === 1 ? '' : 's'}: ${Math.min(L, r)} of ${L} states`, x0, ly - 11, C.mu, narrow ? 9 : 10);
    for (let t = 0; t < L; t++) {
      const x = x0 + slot * (t + .5), f = PE.clamp(done - t, 0, 1);
      if (t > 0) PE.polyline(ctx, [[x - slot + 4, ly + 4], [x - 4, ly + 4]], C.rule, 1);
      ctx.save(); ctx.globalAlpha = .9; ctx.strokeStyle = gLoop > 0 ? C.fg : C.Y; ctx.lineWidth = 1.2 + 1.5 * gLoop; ctx.beginPath(); ctx.arc(x, ly + 4, narrow ? 3 : 4, 0, 7); ctx.stroke(); ctx.restore();
      if (f > 0) PE.dot(ctx, x, ly + 4, (narrow ? 3 : 4) * f, C.Y, .95);
    }
  }
  function readout(S) {
    const L = S.L, r = S.round, m = Math.round(Math.log2(L)), { a, b } = data(L, S.seed), T = tree(a, b), ph = phase(L, r), lv = levelOf(L, r), N = lastRound(L), done = combinesTo(L, r);
    const f = v => (v < 0 ? '-' : '') + Math.abs(v).toFixed(2);
    if (ph === 'inputs') return `${L} tokens. Token t is the pair (a_t, b_t): keep a_t of the old state, then add b_t. The loop would need ${L} rounds, one after another. The scan will need ${N} rounds and ${3 * L - 2} combines.`;
    if (ph === 'up') { const p = T.U[lv][0]; return `Round ${r} of ${N}, up-sweep: ${combinesIn(L, r)} combine${combinesIn(L, r) > 1 ? 's' : ''} at once (${done} so far). Each new node holds the effect of ${1 << lv} tokens as one pair; tokens ${span(lv, 0)} combine to (${f(p[0])}, ${f(p[1])}).`; }
    if (ph === 'down') { const i = (L >> lv) - 1, s0 = i << lv; return `Round ${r} of ${N}, down-sweep: ${combinesIn(L, r)} combine${combinesIn(L, r) > 1 ? 's' : ''} (${done} so far). Each node receives the state just before its span: a left child copies its parent's, a right child combines it with its left sibling's pair. Node ${span(lv, i)} now knows h${s0} = ${f(T.D[lv][i][1])}.`; }
    const seq = sequential(a, b), err = Math.max(...seq.map((v, t) => Math.abs(v - T.H[t])));
    return `Round ${N} of ${N}: each token combines its incoming state with its own pair. All ${L} states in ${N} rounds and ${done} combines; the loop takes ${L} rounds and ${L} steps. Largest difference from the loop: ${err < 1e-15 ? 'below 1e-15' : err.toExponential(1)}. At L = 2,048 the scan needs 23 rounds instead of 2,048, for about 3 times the combines.`;
  }

  // ---------- where the state lives: HBM vs SRAM ----------
  // One selective-scan layer, one sequence: Mamba 2.8B's inner width (E x d_model = 2 x 2560), N = 16, L = 2048, 16-bit values.
  const CFG = { B: 1, L: 2048, D: 5120, N: 16, bytes: 2 };
  const SZ = (() => { const { B, L, D, N } = CFG; return { BLD: B * L * D, BLN: B * L * N, DN: D * N, BLDN: B * L * D * N }; })();
  const inRead = 2 * SZ.BLD + 2 * SZ.BLN + SZ.DN;   // x and Delta (B, L, D); B and C (B, L, N); A (D, N)
  // HBM values moved in each stage [read, written]; stage 5 is the backward pass.
  const TRAFFIC = {
    fused: [null, [inRead, 0], [0, 0], [0, 0], [0, SZ.BLD], [inRead + SZ.BLD, inRead]],
    standard: [null, [inRead, 0], [0, 2 * SZ.BLDN], [2 * SZ.BLDN, SZ.BLDN], [SZ.BLDN, SZ.BLD], [3 * SZ.BLDN + SZ.BLD, inRead]],
  };
  const moved = (mode, k) => { let s = 0; for (let i = 1; i <= Math.min(k, 4); i++) s += TRAFFIC[mode][i][0] + TRAFFIC[mode][i][1]; return s * CFG.bytes; };
  const fwdTotal = mode => moved(mode, 4);
  function fmtB(v) { const u = ['B', 'KiB', 'MiB', 'GiB']; let i = 0; while (v >= 1024 && i < 3) { v /= 1024; i++; } return (v >= 100 ? v.toFixed(0) : v >= 10 ? v.toFixed(1) : v.toFixed(2)).replace(/\.0+$/, '') + ' ' + u[i]; }
  const STAGES = ['overview', 'read', 'expand', 'scan', 'out', 'back'];
  // M = { mode: 'fused' | 'standard', stage: 0..5, u }
  function drawMem(ctx, W, H, M, glow = () => 0) {
    const C = { bg: PE.css('--bg'), fg: PE.css('--fg'), mu: PE.css('--muted'), rule: PE.css('--rule'), B: PE.css('--blue'), Y: PE.css('--yellow'), G: PE.css('--green'), R: PE.css('--red'), Br: PE.css('--brown') };
    ctx.clearRect(0, 0, W, H);
    const narrow = W < 520, k = M.stage, u = M.u == null ? 1 : M.u, fused = M.mode === 'fused', bottom = M.bottom || 0;
    const gH = glow('hbm'), gS = glow('sram'), gR = glow('recompute');
    // Layout: HBM left (or below on narrow), SRAM right (or above).
    const pad = narrow ? 8 : 16;
    const hb = narrow ? { x: pad, y: 168, w: W - 2 * pad, h: H - bottom - 168 - 54 } : { x: pad, y: 40, w: W * .55, h: H - bottom - 40 - 62 };
    const sb = narrow ? { x: W * .22, y: 38, w: W * .56, h: 96 } : { x: W * .55 + pad + W * .1, y: 40 + (hb.h - 150) / 2, w: W - (W * .55 + pad + W * .1) - pad, h: 150 };
    rr(ctx, hb.x, hb.y, hb.w, hb.h, 9, PE.mix(C.bg, C.mu, .06), gH > 0 ? C.fg : C.rule, 1.3 + 2 * gH);
    rr(ctx, sb.x, sb.y, sb.w, sb.h, 9, PE.mix(C.bg, C.Y, .06), gS > 0 ? C.fg : C.mu, 1.3 + 2 * gS);
    PE.label(ctx, narrow ? 'HBM: large, slow GPU memory' : 'HBM: the GPU\'s large, slow memory', hb.x + 10, hb.y + 18, C.fg, narrow ? 11 : 12);
    PE.label(ctx, narrow ? 'SRAM: small, fast, on chip' : 'SRAM: small, fast, on chip', sb.x + 10, sb.y + 18, C.fg, narrow ? 11 : 12);
    // Tensors in HBM as bars whose length is proportional to their bytes.
    const unit = (hb.w - 24) / 32, bar = narrow ? 13 : 15, gap = narrow ? 7 : 9;
    let y = hb.y + 34;
    const tensor = (name, shape, n, color, alpha, grid) => {
      const len = Math.max(4, unit * n / SZ.BLD);
      rr(ctx, hb.x + 10, y, len, bar, 3, PE.mix(C.bg, color, .3), color, 1.2, alpha);
      if (grid && alpha > .05) { ctx.save(); ctx.globalAlpha = alpha * .5; ctx.strokeStyle = color; ctx.lineWidth = .8; for (let x = hb.x + 10 + 8; x < hb.x + 10 + len; x += 8) { ctx.beginPath(); ctx.moveTo(x, y); ctx.lineTo(x, y + bar); ctx.stroke(); } ctx.restore(); }
      const ghost = alpha < .05, inside = !ghost && len > hb.w * .55; ctx.save(); ctx.globalAlpha = Math.max(alpha, .3); PE.label(ctx, ghost ? `(${name}: not in HBM)` : `${name} ${shape}`, inside || ghost ? hb.x + 18 : hb.x + 16 + len, y + bar - 3, ghost ? C.mu : inside ? C.fg : alpha > .3 ? color : C.mu, narrow ? 9 : 10.5); ctx.restore();
      const yy = y; y += bar + gap; return yy;
    };
    const yX = tensor('x, \u0394', '(B, L, D) x 2', 2 * SZ.BLD, C.B, 1);
    tensor('B, C, A', 'small', 2 * SZ.BLN + SZ.DN, C.B, 1);
    const showStd = !fused;
    const aStd = (s) => showStd ? (k > s ? 1 : k === s ? u : 0) : 0;
    const yAB = tensor('\u0100, B\u0304x', '(B, L, D, N) x 2', 2 * SZ.BLDN, C.Y, aStd(2) * .9, false);
    const yH = tensor('states h', '(B, L, D, N)', SZ.BLDN, C.Y, aStd(3), true);
    const yY = tensor('output y', '(B, L, D)', SZ.BLD, C.G, k > 4 || k === 4 ? (k === 4 ? u : 1) : 0);
    // Inside SRAM: the expanded state, one chunk at a time.
    const inS = fused ? (k >= 2 && k <= 4 || k === 5) : (k === 2 || k === 3 || k === 5 && false);
    const g = { x: sb.x + 14, y: sb.y + 30, n: 4, c: narrow ? 9 : 12 };
    if (inS) {
      const al = (k === 2 || (k === 5 && fused)) ? u : 1;
      for (let i = 0; i < g.n; i++) for (let j = 0; j < g.n * 2; j++) rr(ctx, g.x + j * (g.c + 2), g.y + i * (g.c + 2), g.c, g.c, 2, PE.mix(C.bg, C.Y, .25 + .5 * (((i * 7 + j * 3 + k) % 5) / 5)), C.Y, 1 + 2 * gR * (k === 5 ? 1 : 0), al);
      PE.label(ctx, k === 5 ? 'states recomputed here' : 'states: one chunk at a time', g.x, g.y + g.n * (g.c + 2) + 14, C.Y, narrow ? 9.5 : 10.5);
    } else if (k >= 1) PE.label(ctx, k === 1 ? 'inputs arrive' : fused ? 'done' : 'computes, then sends it all back', g.x, g.y + 14, C.mu, narrow ? 9.5 : 10.5);
    // Arrows for this stage: reads HBM -> SRAM, writes SRAM -> HBM, red with width ~ bytes.
    if (k >= 1) {
      const [rd, wr] = TRAFFIC[M.mode][k], w = v => 1.2 + 9 * Math.sqrt(v / (3 * SZ.BLDN));
      const hx = narrow ? null : hb.x + hb.w, sx = narrow ? null : sb.x;
      const path = (yA, toS, v, lab) => {
        if (!v) return;
        const a = narrow ? [W / 2 + (toS ? -40 : 40), hb.y] : [hx, yA + bar / 2], c = narrow ? [W / 2 + (toS ? -40 : 40), sb.y + sb.h] : [sx, sb.y + sb.h / 2 + (toS ? -18 : 18)];
        const [p, q] = toS ? [a, c] : [c, a];
        PE.arrow(ctx, p[0], p[1], PE.lerp(p[0], q[0], u), PE.lerp(p[1], q[1], u), C.R, w(v), 9, .85);
        const lx = (p[0] + q[0]) / 2 + (narrow ? (toS ? -8 : 8) : 0), ly = (p[1] + q[1]) / 2 + (narrow ? 0 : toS ? -10 : 18);
        PE.label(ctx, `${toS ? 'read' : 'write'} ${fmtB(v * CFG.bytes)}`, lx, ly, C.R, narrow ? 9.5 : 10.5, narrow ? (toS ? 'right' : 'left') : 'center');
      };
      const readY = k === 3 || (k === 5 && !fused) ? yAB : k === 4 && !fused ? yH : yX;
      const writeY = k === 2 ? yAB : k === 3 ? yH : k === 4 ? yY : yX;
      path(readY, true, rd); path(writeY, false, wr);
    }
    // Traffic meter: forward-pass HBM traffic so far, against the standard total.
    const my = H - bottom - (narrow ? 34 : 38), mx = pad, mw = W - 2 * pad, tot = fwdTotal('standard');
    const sofar = k === 5 ? fwdTotal(M.mode) : moved(M.mode, k - 1) + (k >= 1 && k <= 4 ? (TRAFFIC[M.mode][k][0] + TRAFFIC[M.mode][k][1]) * CFG.bytes * u : 0);
    rr(ctx, mx, my, mw, 10, 4, PE.mix(C.bg, C.mu, .12), null);
    rr(ctx, mx, my, Math.max(2, mw * sofar / tot), 10, 4, C.R, null);
    const fx = mx + mw * fwdTotal('fused') / tot;
    PE.polyline(ctx, [[fx, my - 3], [fx, my + 13]], C.fg, 1.2, [2, 2], .7);
    PE.label(ctx, `forward-pass HBM traffic so far: ${fmtB(sofar)}` + (narrow ? '' : `   (standard total ${fmtB(tot)}, fused total ${fmtB(fwdTotal('fused'))})`), mx, my + 26, C.fg, narrow ? 9.5 : 10.5);
  }
  function readoutMem(M) {
    const fused = M.mode === 'fused', k = M.stage, mode = fused ? 'Fused (Mamba)' : 'Standard';
    const T = TRAFFIC[M.mode][k] || [0, 0];
    const totals = `Forward traffic: ${fmtB(fwdTotal(M.mode))} ${fused ? `(the standard way moves ${fmtB(fwdTotal('standard'))}, ${(fwdTotal('standard') / fwdTotal('fused')).toFixed(0)} times more)` : `(the fused kernel moves ${fmtB(fwdTotal('fused'))})`}.`;
    return [
      `One selective-scan layer, one sequence of 2,048 tokens, inner width 5,120, state size 16, 16 bits per value. The inputs and outputs are (B, L, D): ${fmtB(SZ.BLD * CFG.bytes)} each. The expanded state is (B, L, D, N): ${fmtB(SZ.BLDN * CFG.bytes)}. Pick a stage.`,
      `${mode}, read: x and \u0394 (B, L, D), B and C (B, L, N) and A (D, N) come in from HBM: ${fmtB(T[0] * CFG.bytes)}.`,
      fused ? `${mode}, expand: \u0100 = exp(\u0394A) and B\u0304x are computed in SRAM, one chunk at a time. Nothing of size (B, L, D, N) goes to HBM.` : `${mode}, expand: \u0100 and B\u0304x are computed and written to HBM: ${fmtB(T[1] * CFG.bytes)}, ${SZ.BLDN / SZ.BLD} times the size of the input.`,
      fused ? `${mode}, scan: the parallel scan runs on the chunk in SRAM. Still no traffic.` : `${mode}, scan: \u0100 and B\u0304x are read back (${fmtB(T[0] * CFG.bytes)}) and every state h is written out (${fmtB(T[1] * CFG.bytes)}).`,
      fused ? `${mode}, read out: y = C h is formed in SRAM and only y goes back to HBM: ${fmtB(T[1] * CFG.bytes)}. ${totals}` : `${mode}, read out: the states are read again (${fmtB(T[0] * CFG.bytes)}) to form y = C h, and y is written (${fmtB(T[1] * CFG.bytes)}). ${totals}`,
      fused ? `${mode}, backward: no state was saved. The kernel reloads the inputs and the output gradient (${fmtB(T[0] * CFG.bytes)}), recomputes the states in SRAM, and writes the input gradients (${fmtB(T[1] * CFG.bytes)}).` : `${mode}, backward: the states (and the expanded inputs) must be kept from the forward pass, at least ${fmtB(SZ.BLDN * CFG.bytes)} per layer per sequence, and read back: about ${fmtB(T[0] * CFG.bytes)}.`,
    ][k];
  }
  return { combine, ID, data, sequential, tree, lastRound, combinesIn, combinesTo, phase, levelOf, draw, readout, CFG, SZ, TRAFFIC, moved, fwdTotal, fmtB, STAGES, drawMem, readoutMem };
})();
