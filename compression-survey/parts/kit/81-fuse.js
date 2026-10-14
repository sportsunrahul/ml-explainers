/* Inference-compiler toy: a residual block with explicit Q/DQ, rewritten by fusions that the TensorRT 11.4 docs list
   (BN folded into conv weights, Conv+ReLU, Q/DQ fused into quantized layers, Conv+ElementWise with a trailing
   activation). Kernel and byte counts are computed on the toy graph; tactic names and times are made up. */
KIT.FUSE_NODES = [
  { id: 'qx', kind: 'q', label: 'Q', slot: 0, out: 1 }, { id: 'dqx', kind: 'dq', label: 'DQ', slot: 1, out: 2 },
  { id: 'c1', kind: 'conv', label: 'Conv', slot: 2, out: 2 }, { id: 'bn1', kind: 'bn', label: 'BN', slot: 3, out: 2 },
  { id: 'r1', kind: 'act', label: 'ReLU', slot: 4, out: 2 }, { id: 'qa', kind: 'q', label: 'Q', slot: 5, out: 1 },
  { id: 'dqa', kind: 'dq', label: 'DQ', slot: 6, out: 2 }, { id: 'c2', kind: 'conv', label: 'Conv', slot: 7, out: 2 },
  { id: 'bn2', kind: 'bn', label: 'BN', slot: 8, out: 2 }, { id: 'add', kind: 'add', label: 'Add', slot: 9, out: 2 },
  { id: 'r2', kind: 'act', label: 'ReLU', slot: 10, out: 2 }, { id: 'dqs', kind: 'dq', label: 'DQ', slot: 8, out: 2, upper: true },
];
KIT.FUSE_EDGES = [['qx', 'dqx'], ['dqx', 'c1'], ['c1', 'bn1'], ['bn1', 'r1'], ['r1', 'qa'], ['qa', 'dqa'], ['dqa', 'c2'], ['c2', 'bn2'], ['bn2', 'add'], ['add', 'r2'], ['dqs', 'add']];
KIT.FUSE_K1 = ['dqx', 'c1', 'bn1', 'r1', 'qa']; KIT.FUSE_K2 = ['dqa', 'c2', 'bn2', 'dqs', 'add', 'r2'];
KIT.FUSE_STAGES = [
  { name: 'As exported: every node runs as its own kernel', groups: [] },
  { name: 'BN folded into the conv weights and bias', groups: [['c1', 'bn1'], ['c2', 'bn2']] },
  { name: 'Conv, bias and ReLU fused into one kernel', groups: [['c1', 'bn1', 'r1'], ['c2', 'bn2']] },
  { name: 'Q/DQ absorbed: the convolutions now run in INT8', groups: [KIT.FUSE_K1, ['dqa', 'c2', 'bn2']] },
  { name: 'Residual add and ReLU fused into the second conv', groups: [KIT.FUSE_K1, KIT.FUSE_K2] },
  { name: 'Tactics timed on the GPU; the fastest kept', groups: [KIT.FUSE_K1, KIT.FUSE_K2] },
];
KIT.FUSE_LABELS = { 'c1,bn1': ['Conv+bias', 'Conv+b'], 'c2,bn2': ['Conv+bias', 'Conv+b'], 'c1,bn1,r1': ['Conv+bias+ReLU', 'Conv+b+ReLU'],
  [KIT.FUSE_K1.join()]: ['INT8 Conv+bias+ReLU', 'INT8 Conv+b+ReLU'], 'dqa,c2,bn2': ['INT8 Conv+bias', 'INT8 Conv+b'], [KIT.FUSE_K2.join()]: ['INT8 Conv+bias+Add+ReLU', 'INT8 Conv+b+Add+ReLU'] };
KIT.FUSE_ELEMS = 64 * 80 * 80;
KIT.FUSE_TACTICS = [
  { kernel: 'Q (input)', cands: [['4 values per thread', 9.0], ['8 values per thread', 7.5]] },
  { kernel: 'INT8 Conv+bias+ReLU', cands: [['tile 64x64', 52], ['tile 128x64', 38], ['tile 128x128', 44], ['tile 256x64', 61]] },
  { kernel: 'INT8 Conv+bias+Add+ReLU', cands: [['tile 64x64', 57], ['tile 128x64', 49], ['tile 128x128', 41], ['tile 256x64', 66]] },
];
// Kernels of a stage: each group plus every node not in a group. A kernel writes the output of its last node.
KIT.fuseKernels = st => {
  const G = KIT.FUSE_STAGES[st].groups, inG = new Set(G.flat()), N = KIT.FUSE_NODES;
  const order = id => N.findIndex(n => n.id === id);
  const ks = G.map(g => g.slice()).concat(N.filter(n => !inG.has(n.id)).map(n => [n.id]));
  return ks.map(m => { const last = m.reduce((a, b) => (b === 'dqs' ? a : order(b) > order(a) ? b : a), m[0]); return { members: m, bytes: N[order(last)].out * KIT.FUSE_ELEMS }; });
};
KIT.fuseInfo = st => {
  const ks = KIT.fuseKernels(st), bytes = ks.reduce((s, k) => s + k.bytes, 0), k0 = KIT.fuseKernels(0);
  const win = KIT.FUSE_TACTICS.map(t => t.cands.reduce((a, b) => (b[1] < a[1] ? b : a)));
  return { stage: st, name: KIT.FUSE_STAGES[st].name, kernels: ks.length, bytes, kernels0: k0.length, bytes0: k0.reduce((s, k) => s + k.bytes, 0), win, total: win.reduce((s, w) => s + w[1], 0) };
};
KIT.fuse = (el, opts = {}) => {
  const S = { stage: 0, u: 1, prev: 0, race: 0 };
  const glow = opts.fig ? PE.linkPulses(opts.fig, () => cv.redraw()) : () => 0;
  const N = KIT.FUSE_NODES, byId = Object.fromEntries(N.map(n => [n.id, n]));
  const groupOf = (st, id) => KIT.FUSE_STAGES[st].groups.find(g => g.includes(id)) || [id];
  function geom(w, h) {
    const narrow = KIT.narrow(w), H = h - (opts.bottomPad || 0);
    if (!narrow) {
      const pad = 26, sp = (w - 2 * pad) / 11, bh = 34, yM = 30 + 64, yU = 30 + 14;
      const slotRect = (a, b, upper) => ({ x0: pad + a * sp + 3, x1: pad + (b + 1) * sp - 3, y0: (upper ? yU : yM) - bh / 2, y1: (upper ? yU : yM) + bh / 2 });
      return { narrow, H, slotRect, graphBottom: yM + bh / 2 + 18, sp };
    }
    const top = 34, sp = Math.min(32, (H * .58 - top) / 11), bh = sp - 6, xL = 14, xR = w * .66;
    const slotRect = (a, b, upper) => upper ? { x0: xR + 14, x1: w - 10, y0: top + a * sp + 3, y1: top + (b + 1) * sp - 3 }
      : { x0: xL, x1: xR, y0: top + a * sp + 3, y1: top + (b + 1) * sp - 3 };
    return { narrow, H, slotRect, graphBottom: top + 11 * sp + 16, sp, bh };
  }
  function rectAt(G, st, id) {
    const g = groupOf(st, id), main = g.filter(m => !byId[m].upper);
    if (!main.length) { const n = byId[id]; return G.slotRect(n.slot, n.slot, true); }
    const sl = main.map(m => byId[m].slot); return G.slotRect(Math.min(...sl), Math.max(...sl), false);
  }
  const lerpR = (a, b, u) => ({ x0: PE.lerp(a.x0, b.x0, u), x1: PE.lerp(a.x1, b.x1, u), y0: PE.lerp(a.y0, b.y0, u), y1: PE.lerp(a.y1, b.y1, u) });
  const colorOf = (g, C) => g.length > 1 ? C('--green') : ({ q: C('--yellow'), dq: C('--yellow'), bn: C('--blue'), conv: C('--blue'), act: C('--blue'), add: C('--blue') })[byId[g[0]].kind];
  function rr(ctx, r, stroke, lw, alpha, dash) {
    ctx.save(); ctx.globalAlpha = alpha; ctx.fillStyle = PE.css('--panel'); ctx.strokeStyle = stroke; ctx.lineWidth = lw; if (dash) ctx.setLineDash(dash);
    ctx.beginPath(); ctx.roundRect(r.x0, r.y0, r.x1 - r.x0, r.y1 - r.y0, 7); ctx.fill(); ctx.stroke(); ctx.restore();
  }
  function draw(ctx, w, h) {
    ctx.clearRect(0, 0, w, h);
    const C = n => PE.css(n), G = geom(w, h), u = S.u, narrow = G.narrow;
    const R = id => lerpR(rectAt(G, S.prev, id), rectAt(G, S.stage, id), u);
    const cx = r => (r.x0 + r.x1) / 2, cy = r => (r.y0 + r.y1) / 2;
    // input and output arrows
    const rq = R('qx'), rout = R('r2');
    if (!narrow) { PE.arrow(ctx, 4, cy(rq), rq.x0 - 1, cy(rq), C('--muted'), 1.4, 7); }
    else { PE.arrow(ctx, cx(rq), 14, cx(rq), rq.y0 - 1, C('--muted'), 1.4, 7); PE.label(ctx, 'x (FP16)', cx(rq) + 8, 22, C('--muted'), 10.5, 'left', 400); }
    if (!narrow) PE.label(ctx, 'x (FP16)', 2, cy(rq) + 30, C('--muted'), 10.5, 'left', 400);
    // edges that cross kernel boundaries
    const same = (st, a, b) => groupOf(st, a).includes(b);
    for (const [a, b] of KIT.FUSE_EDGES) {
      const alpha = same(S.stage, a, b) ? (same(S.prev, a, b) ? 0 : 1 - u) : (same(S.prev, a, b) ? u : 1);
      if (alpha <= .02) continue;
      const ra = R(a), rb = R(b);
      if (a === 'dqs') {
        if (!narrow) PE.polyline(ctx, [[ra.x1, cy(ra)], [cx(rb), cy(ra)], [cx(rb), rb.y0 - 1]], C('--muted'), 1.4, null, alpha);
        else PE.polyline(ctx, [[cx(ra), ra.y1], [cx(ra), cy(rb)], [rb.x1 + 1, cy(rb)]], C('--muted'), 1.4, null, alpha);
        continue;
      }
      if (!narrow) PE.arrow(ctx, ra.x1, cy(ra), rb.x0 - 1, cy(rb), C('--muted'), 1.4, 7, alpha);
      else PE.arrow(ctx, cx(ra), ra.y1, cx(rb), rb.y0 - 1, C('--muted'), 1.4, 6, alpha);
    }
    // skip connection: from the input Q to the skip DQ (or to the fused kernel that absorbed it)
    { const rs = R('dqs'), upper = !narrow && cy(rs) < cy(rq) - 4;
      if (!narrow) { const yU = rq.y0 - 30;
        PE.polyline(ctx, upper ? [[cx(rq), rq.y0], [cx(rq), cy(rs)], [rs.x0, cy(rs)]] : [[cx(rq), rq.y0], [cx(rq), yU], [cx(rs), yU], [cx(rs), rs.y0]], C('--muted'), 1.4);
        PE.label(ctx, 'skip', cx(rq) + 6, Math.min(cy(rs), yU) - 4, C('--muted'), 10.5, 'left', 400); }
      else { const col = G.slotRect(0, 0, true), xc = (col.x0 + col.x1) / 2, side = cx(rs) > col.x0;
        PE.polyline(ctx, side ? [[rq.x1, cy(rq)], [xc, cy(rq)], [xc, rs.y0]] : [[rq.x1, cy(rq)], [xc, cy(rq)], [xc, cy(rs)], [rs.x1, cy(rs)]], C('--muted'), 1.4); } }
    if (!narrow) PE.arrow(ctx, rout.x1, cy(rout), w - 4, cy(rout), C('--muted'), 1.4, 7);
    else PE.arrow(ctx, cx(rout), rout.y1, cx(rout), rout.y1 + 12, C('--muted'), 1.4, 6);
    // nodes: every node draws its interpolated rectangle; members of one kernel converge on the same rectangle
    const gl = k => glow(k), gf = glow('fuse');
    const drawn = new Set();
    for (const n of N) {
      const g = groupOf(S.stage, n.id), key = g.join();
      if (u >= 1 && drawn.has(key)) continue; drawn.add(key);
      const r = R(n.id), col = colorOf(g, C), lit = Math.max(...g.map(m => gl(byId[m].kind))), fl = g.length > 1 ? gf : 0;
      rr(ctx, r, col, 1.5 + 2.5 * Math.max(lit, fl), 1, byId[g[0]].kind === 'dq' && g.length === 1 ? [4, 3] : null);
    }
    // labels: old groups fade out, new groups fade in
    const labelSet = (st, alpha) => {
      const seen = new Set();
      for (const n of N) {
        const g = groupOf(st, n.id), key = g.join(); if (seen.has(key)) continue; seen.add(key);
        const r = lerpR(rectAt(G, S.prev, n.id), rectAt(G, S.stage, n.id), u), col = colorOf(g, C);
        const lab = g.length > 1 ? (KIT.FUSE_LABELS[key] || ['fused', 'fused'])[narrow || r.x1 - r.x0 < 150 ? 1 : 0] : byId[g[0]].label;
        const fs = narrow ? 11 : (r.x1 - r.x0 < 40 ? 10.5 : 12);
        ctx.save(); ctx.globalAlpha = alpha; PE.label(ctx, lab, cx(r), cy(r) + 4, col, fs, 'center', 600); ctx.restore();
      }
    };
    if (u < 1 && S.prev !== S.stage) labelSet(S.prev, 1 - u);
    labelSet(S.stage, u < 1 && S.prev !== S.stage ? u : 1);
    // lower panel
    const top = G.graphBottom + 14, H = G.H, bw = w - 24;
    if (S.stage < 5) history(ctx, 12, top, bw, H - top - 6, C, narrow);
    else race(ctx, 12, top, bw, H - top - 6, C, narrow);
  }
  function history(ctx, x, y, w, h, C, narrow) {
    const infos = [0, 1, 2, 3, 4].map(KIT.fuseInfo), cols = narrow ? 1 : 2, pw = (w - (cols - 1) * 28) / cols;
    const panels = [['kernels launched', i => i.kernels, v => String(v)], ['memory written per image (toy)', i => i.bytes, v => KIT.bytes(v)]];
    const ph = narrow ? (h - 16) / 2 : h;
    panels.forEach(([title, f, fmt], p) => {
      const px = x + (narrow ? 0 : p * (pw + 28)), py = y + (narrow ? p * (ph + 16) : 0);
      PE.label(ctx, title, px, py + 12, C('--fg'), 12, 'left', 600);
      const mx = f(infos[0]), bh0 = ph - 62, sw = pw / 5;
      infos.forEach((i, s) => {
        const v = f(i), bh = Math.max(2, v / mx * bh0), bx = px + s * sw + sw * .2, by = py + 38 + bh0 - bh, bwid = sw * .6;
        ctx.save();
        if (s <= S.stage) { ctx.fillStyle = s === S.stage ? C('--green') : C('--muted'); ctx.globalAlpha = s === S.stage ? .85 : .45; ctx.fillRect(bx, by, bwid, bh); }
        else { ctx.strokeStyle = C('--rule'); ctx.setLineDash([3, 3]); ctx.strokeRect(bx, by, bwid, bh); }
        ctx.restore();
        if (s <= S.stage) PE.label(ctx, fmt(v), bx + bwid / 2, by - 4, s === S.stage ? C('--green') : C('--muted'), 10.5, 'center', 600);
        PE.label(ctx, 'step ' + s, bx + bwid / 2, py + 52 + bh0, C('--muted'), 10.5, 'center', 400);
      });
    });
  }
  function race(ctx, x, y, w, h, C, narrow) {
    const gt = glow('tactic');
    PE.label(ctx, narrow ? 'Tactics timed on this GPU (toy, \u00b5s)' : 'Tactic selection: each candidate kernel is timed on this GPU (toy names and times, \u00b5s)', x, y + 12, gt > 0 ? C('--green') : C('--fg'), narrow ? 10.5 : 12, 'left', 600);
    const rows = KIT.FUSE_TACTICS.reduce((s, t) => s + t.cands.length + 1, 0), rh = Math.min(22, (h - 24) / rows);
    const tmax = Math.max(...KIT.FUSE_TACTICS.flatMap(t => t.cands.map(c => c[1])));
    const lw = narrow ? 112 : 150, bx = x + lw, bwid = w - lw - (narrow ? 34 : 60);
    let yy = y + 24;
    KIT.FUSE_TACTICS.forEach(t => {
      const best = Math.min(...t.cands.map(c => c[1]));
      PE.label(ctx, t.kernel, x, yy + rh * .7, C('--muted'), 10.5, 'left', 600); yy += rh;
      t.cands.forEach(([name, ms]) => {
        const done = S.race * tmax >= ms, len = Math.min(ms, S.race * tmax) / tmax * bwid, win = ms === best;
        const col = S.race >= 1 ? (win ? C('--green') : C('--brown')) : C('--blue');
        PE.label(ctx, name, x + 10, yy + rh * .68, C('--fg'), 10.5, 'left', 400);
        ctx.save(); ctx.fillStyle = col; ctx.globalAlpha = S.race >= 1 && !win ? .5 : .85; ctx.fillRect(bx, yy + rh * .2, Math.max(1, len), rh * .6); ctx.restore();
        if (done) PE.label(ctx, ms.toFixed(1) + (S.race >= 1 ? (win ? '  kept' : '') : ''), bx + len + 5, yy + rh * .68, S.race >= 1 && win ? C('--green') : C('--muted'), 10.5, 'left', win && S.race >= 1 ? 700 : 400);
        yy += rh;
      });
    });
  }
  const cv = PE.canvas(el, opts.aspect || (w => w < 560 ? 1.95 : .5), draw);
  let stop = () => {};
  function set(next) {
    if (next.stage === undefined || next.stage === S.stage) { cv.redraw(); return; }
    stop(); S.prev = S.stage; S.stage = next.stage; S.u = 0; S.race = 0;
    const rs = () => { if (S.stage === 5) stop = PE.animate(900, v => { S.race = v; cv.redraw(); }); };
    if (S.prev >= 4 && S.stage === 5) { S.u = 1; S.prev = 5; rs(); return; }
    stop = PE.animate(800, v => { S.u = v; cv.redraw(); }, () => { S.prev = S.stage; rs(); });
  }
  return { set, info: () => KIT.fuseInfo(S.stage), state: S, redraw: cv.redraw, resize: cv.resize, destroy: () => { stop(); cv.destroy(); } };
};
KIT.fuseText = i => {
  const base = `<b>Step ${i.stage}: ${i.name}.</b> ${i.kernels} kernels launched, ${i.kernels} tensors written to memory, ${KIT.bytes(i.bytes)} written per image`;
  if (i.stage === 0) return base + ' (toy tensors: 64 channels at 80x80, FP16 or INT8).';
  if (i.stage < 5) return base + `, down from ${i.kernels0} kernels and ${KIT.bytes(i.bytes0)}.`;
  return base + `. Kept: ${i.win.map(w => w[0] + ' (' + w[1].toFixed(1) + ' \u00b5s)').join(', ')}; toy total ${i.total.toFixed(1)} \u00b5s. The timings, and so the engine, belong to this GPU and this TensorRT version.`;
};
