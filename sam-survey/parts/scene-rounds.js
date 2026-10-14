/* ch-training: SAM's simulated interactive training, eleven rounds for one mask (App. A). The model is a toy:
   its predictions are perturbed rasters of the desk scene; the sampling rules are the paper's. Every round is
   precomputed from a seed, so the round slider and the buttons set state. */
const ROUNDS_KIT = (() => {
  const GX = 96, GY = 60, N = GX * GY;
  const TARGETS = ['mug', 'mugset', 'plant', 'lamp', 'book', 'window', 'saucer', 'pot', 'shade', 'leaves'];
  const cellXY = k => [((k % GX) + .5) / GX * TOY.WW, (Math.floor(k / GX) + .5) / GY * TOY.WH];
  const iou = (a, b) => TOY.iou(a, b);
  const nb4 = k => { const i = k % GX, j = (k / GX) | 0, o = []; if (i) o.push(k - 1); if (i < GX - 1) o.push(k + 1); if (j) o.push(k - GX); if (j < GY - 1) o.push(k + GX); return o; };
  const boundary = m => { const o = []; for (let k = 0; k < N; k++) if (nb4(k).some(q => m[q] !== m[k])) o.push(k); return o; };
  function morph(m, grow, times = 1) { let a = m; for (let t = 0; t < times; t++) { const b = new Uint8Array(N); for (let k = 0; k < N; k++) { const ns = nb4(k); b[k] = grow ? (a[k] || ns.some(q => a[q]) ? 1 : 0) : (a[k] && ns.every(q => a[q]) ? 1 : 0); } a = b; } return a; }
  // the toy model refining its own mask: flip cells that disagree with at least 3 of their 4 neighbours
  // (removes specks and pinholes, keeps thin parts)
  function majority(m) { const b = m.slice(); for (let k = 0; k < N; k++) { const ns = nb4(k), same = ns.filter(q => m[q] === m[k]).length; if (same <= ns.length - 3) b[k] = 1 - m[k]; } return b; }
  function noisy(m, r, p = .35, specks = 6) { const o = m.slice(); boundary(m).forEach(k => { if (r() < p) o[k] = 1 - o[k]; });
    for (let s = 0; s < specks; s++) { const k = Math.floor(r() * N); if (!m[k]) { o[k] = 1; if (k + 1 < N) o[k + 1] = 1; } } return o; }
  function pick(cells, r) { return cells.length ? cells[Math.floor(r() * cells.length)] : -1; }
  function fix(P, T, c) { // the toy model's response to a click: it repairs the error near the click
    const o = P.slice(), [ci, cj] = [c % GX, (c / GX) | 0], seen = new Uint8Array(N), q = [c]; seen[c] = 1;
    while (q.length) { const k = q.pop(), i = k % GX, j = (k / GX) | 0; if (Math.hypot(i - ci, j - cj) > 11) continue; o[k] = T[k];
      nb4(k).forEach(n => { if (!seen[n] && P[n] !== T[n]) { seen[n] = 1; q.push(n); } }); }
    for (let k = 0; k < N; k++) { const i = k % GX, j = (k / GX) | 0; if (Math.hypot(i - ci, j - cj) <= 4) o[k] = T[k]; }
    return o; }
  function run(seed) {
    const r = PE.rng(seed * 7919 + 13), node = TOY.byId[TARGETS[seed % TARGETS.length]], T = TOY.raster(node, GX, GY);
    const tcells = []; for (let k = 0; k < N; k++) if (T[k]) tcells.push(k);
    const first = r() < .5 ? 'point' : 'box', rounds = [];
    let cands, box = null, pt = null;
    if (first === 'point') {
      const c = pick(tcells, r); pt = cellXY(c);
      cands = TOY.three(pt[0], pt[1]).map(n => ({ name: n.name, m: noisy(TOY.raster(n, GX, GY), r) }));
    } else {
      const [x0, y0, x1, y1] = TOY.bbox(node), g = PE.gauss(r), cap = 20 / 1024 * TOY.WW;
      const jit = side => PE.clamp(g() * .1 * side, -cap, cap); box = [x0 + jit(x1 - x0), y0 + jit(y1 - y0), x1 + jit(x1 - x0), y1 + jit(y1 - y0)];
      cands = [{ name: 'tight to the object', m: noisy(T, r) }, { name: 'loose, with the background', m: noisy(morph(T, true, 3), r) }, { name: 'shrunk inside', m: noisy(morph(T, false, 2), r) }];
    }
    cands.forEach(cd => { cd.iou = iou(cd.m, T); cd.piou = PE.clamp(cd.iou + (r() - .5) * .16, 0, 1); });
    const lossPick = cands.reduce((b, cd, i) => cd.iou > cands[b].iou ? i : b, 0), passPick = cands.reduce((b, cd, i) => cd.piou > cands[b].piou ? i : b, 0);
    let P = cands[passPick].m;
    rounds.push({ type: first, pt, fg: true, box, P, iou: iou(P, T), cands, lossPick, passPick });
    const empty1 = 2 + Math.floor(r() * 9);
    for (let n = 2; n <= 11; n++) {
      if (n === empty1 || n === 11) { P = majority(P); rounds.push({ type: 'none', P, iou: iou(P, T) }); continue; }
      const err = []; for (let k = 0; k < N; k++) if (P[k] !== T[k]) err.push(k);
      const c = pick(err, r); if (c < 0) { rounds.push({ type: 'point', pt: null, P, iou: 1 }); continue; }
      const fg = !!T[c]; P = fix(P, T, c); rounds.push({ type: 'point', pt: cellXY(c), fg, P, iou: iou(P, T) });
    }
    return { node, T, rounds, first };
  }
  return { GX, GY, N, run, TARGETS };
})();

SCENES.rounds = ctx => {
  const el = mkCanvas(ctx.stage), slider = ctx.q('[data-round]'), newBtn = ctx.q('[data-newmask]'), sv = ctx.q('[data-round-value]');
  const SEED0 = 3, K = ROUNDS_KIT;
  const S = { seed: +(newBtn && newBtn.dataset.seed || SEED0), round: +(slider ? slider.value : 1), shown: 1 };
  let run = K.run(S.seed), cancel = () => {};
  const cv = PE.canvas(el, S3ASPECT(ctx.stage), draw);
  function setRound(n, anim = true) { S.round = PE.clamp(n, 0, 11); if (slider) slider.value = S.round; if (sv) sv.textContent = S.round;
    cancel(); const from = S.shown; cancel = anim ? PE.animate(450, u => { S.shown = PE.lerp(from, S.round, u); cv.redraw(); }) : (S.shown = S.round, cv.redraw(), () => {}); readout(); }
  const offs = [];
  if (slider) offs.push(listen(slider, 'input', () => setRound(+slider.value)));
  ctx.qa('[data-round-step]').forEach(b => offs.push(listen(b, 'click', () => setRound(S.round + +b.dataset.roundStep))));
  if (newBtn) offs.push(listen(newBtn, 'click', () => { S.seed++; newBtn.dataset.seed = S.seed; run = K.run(S.seed); setRound(1, false); }));
  ctx.qa('[data-cue-reset]').forEach(b => offs.push(listen(b, 'click', () => { if (newBtn) newBtn.dataset.seed = SEED0; if (slider) slider.value = 1; S.seed = SEED0; run = K.run(S.seed); setRound(1, false); })));
  const cellRects = (c, V, m, color, alpha) => { c.save(); c.globalAlpha = alpha; c.fillStyle = color; const cw = V.w / K.GX, ch = V.h / K.GY;
    for (let k = 0; k < K.N; k++) if (m[k]) c.fillRect(V.ox + (k % K.GX) * cw, V.oy + ((k / K.GX) | 0) * ch, cw + .4, ch + .4); c.restore(); };
  function draw(c, W, H) {
    c.clearRect(0, 0, W, H);
    const narrow = W < 620, pad = 12, top = 34, bottom = ctx.reserve || 0;
    const imgW = narrow ? W - 2 * pad : W * .64, imgH = narrow ? (H - bottom - top) * .62 : Math.min(H - bottom - top - pad, imgW / TOY.WW);
    const top2 = narrow ? top : Math.max(top, (H - bottom - imgH) / 2), V = TOY.fit(pad, top2, imgW, imgH);
    TOY.drawWorld(c, V, { dim: .3 });
    const n = Math.round(S.shown), R = run.rounds, cur = n >= 1 ? R[n - 1] : null;
    if (cur) {
      const err = new Uint8Array(K.N); for (let k = 0; k < K.N; k++) err[k] = cur.P[k] !== run.T[k] ? 1 : 0;
      cellRects(c, V, cur.P, PE.css('--green'), .5); cellRects(c, V, err, PE.css('--red'), .55);
    }
    // ground truth: dashed outline
    c.save(); c.translate(V.ox, V.oy); c.scale(V.s, V.s); c.strokeStyle = PE.css('--fg'); c.lineWidth = 1.5 / V.s; c.setLineDash([5 / V.s, 4 / V.s]); c.stroke(run.node.mask); c.restore();
    if (R[0].box && n >= 1) { const b = R[0].box; TOY.boxPrompt(c, V.X(b[0]), V.Y(b[1]), V.X(b[2]), V.Y(b[3]), n === 1 ? 1 : .55); }
    for (let i = 0; i < Math.min(n, 11); i++) { const rd = R[i]; if (!rd.pt) continue; TOY.point(c, V.X(rd.pt[0]), V.Y(rd.pt[1]), rd.fg ? 'pos' : 'neg', i === n - 1 ? 1 : .6, i === n - 1 ? 7 : 5);
      if (i === n - 1) PE.ring(c, V.X(rd.pt[0]), V.Y(rd.pt[1]), 12, PE.css('--yellow'), 1.5, .7); }
    PE.label(c, 'dashed: the ground-truth mask (' + run.node.name + ')', V.ox + 6, V.oy + V.h + 14 > H - bottom ? V.oy + 14 : V.oy + V.h + 14, PE.css('--muted'), 11, 'left', 500);
    // side panel: round list with IoU bars, and the three candidates in round 1
    const b = narrow ? { x: pad, y: top2 + imgH + 22, w: W - 2 * pad, h: H - bottom - (top2 + imgH + 22) - 4 } : { x: pad + imgW + 18, y: top2, w: W - imgW - 2 * pad - 18, h: imgH };
    panel(c, b, n, narrow);
  }
  function panel(c, b, n, narrow) {
    const R = run.rounds, M = PE.css('--muted'), F = PE.css('--fg'), G = PE.css('--green');
    if (n === 1 && !narrow) {
      const rd = R[0]; PE.label(c, 'Round 1: three masks', b.x, b.y + 12, F, 12, 'left', 700);
      rd.cands.forEach((cd, i) => { const y = b.y + 34 + i * 46;
        PE.label(c, cd.name, b.x, y, F, 11, 'left', 600);
        c.fillStyle = PE.css('--panel2'); c.fillRect(b.x, y + 6, b.w - 4, 6); c.fillStyle = G; c.fillRect(b.x, y + 6, (b.w - 4) * cd.piou, 6);
        const tags = []; if (i === rd.lossPick) tags.push('gets the gradient'); if (i === rd.passPick) tags.push('passed on');
        PE.label(c, 'predicted IoU ' + cd.piou.toFixed(2), b.x, y + 25, M, 10, 'left', 500);
        if (tags.length) PE.label(c, tags.join(', '), b.x + b.w - 4, y + 25, PE.css('--yellow'), 10, 'right', 600); });
      PE.label(c, 'gradient: lowest loss; passed on: highest predicted IoU', b.x, b.y + 34 + 3 * 46 - 4, M, 9, 'left', 400);
      const y0 = b.y + 34 + 3 * 46 + 10; chart(c, { x: b.x, y: y0, w: b.w, h: b.y + b.h - y0 }, n, false); return;
    }
    chart(c, b, n, narrow);
  }
  function chart(c, b, n, narrow) {
    const R = run.rounds, M = PE.css('--muted'), F = PE.css('--fg'), G = PE.css('--green');
    PE.label(c, 'IoU with the ground truth, per round', b.x, b.y + 12, F, 11, 'left', 700);
    const y0 = b.y + 22, h = b.h - 26; if (h < 30) return;
    if (narrow) { const bw = (b.w - 10 * 4) / 11; R.forEach((rd, i) => { const x = b.x + i * (bw + 4), bh = (h - 16) * rd.iou, on = i < n;
        c.globalAlpha = on ? 1 : .25; c.fillStyle = rd.type === 'none' ? PE.css('--brown') : G; c.fillRect(x, y0 + (h - 16) - bh, bw, bh); c.globalAlpha = 1;
        PE.label(c, String(i + 1), x + bw / 2, y0 + h - 2, i === n - 1 ? F : M, 9, 'center', i === n - 1 ? 700 : 400); }); return; }
    const rh = Math.min(20, h / 11);
    R.forEach((rd, i) => { const y = y0 + i * rh, on = i < n, lab = (i + 1) + ' ' + (i === 0 ? (rd.type === 'box' ? 'box' : 'point') : rd.type === 'none' ? 'no new point' : rd.pt ? (rd.fg ? 'point, include' : 'point, exclude') : 'no error left');
      c.globalAlpha = on ? 1 : .3; PE.label(c, lab, b.x, y + rh * .7, i === n - 1 ? F : M, 10, 'left', i === n - 1 ? 700 : 400);
      const bx = b.x + b.w * .5, bw = b.w * .5 - 34; c.fillStyle = PE.css('--panel2'); c.fillRect(bx, y + rh * .3, bw, rh * .4); c.fillStyle = rd.type === 'none' ? PE.css('--brown') : G; c.fillRect(bx, y + rh * .3, bw * rd.iou, rh * .4);
      PE.label(c, rd.iou.toFixed(2), b.x + b.w - 2, y + rh * .7, M, 10, 'right', 400); c.globalAlpha = 1; });
  }
  function readout() {
    const n = S.round, R = run.rounds;
    let t;
    if (n === 0) t = 'Round 0: the target mask is chosen. Move the slider to start the eleven rounds.';
    else if (n === 1) t = R[0].type === 'box' ? 'Round 1: the first prompt is a noisy box around the target. SAM returns three masks. Only the lowest-loss mask gets the gradient; the one with the highest predicted IoU is fed to round 2 as logits.' : 'Round 1: the first prompt is a point sampled uniformly inside the target. SAM returns three masks. Only the lowest-loss mask gets the gradient; the one with the highest predicted IoU is fed to round 2 as logits.';
    else { const rd = R[n - 1]; t = rd.type === 'none' ? `Round ${n}: no new point. The model gets only its own previous logits and must refine them.` : rd.pt ? `Round ${n}: a new point is sampled uniformly from the red error region. It is ${rd.fg ? 'an include point, because it fell where the mask is missing' : 'an exclude point, because it fell where the mask spills over'}. With more than one prompt, SAM returns one mask.` : `Round ${n}: no error is left, so no point is sampled.`; }
    ctx.hud(badge('toy', `${t} IoU now ${n ? R[n - 1].iou.toFixed(2) : '-'} (toy model).`));
  }
  if (sv) sv.textContent = S.round; S.shown = S.round; readout();
  return { flat: true, resize: () => cv.resize(), destroy: () => { cancel(); offs.forEach(f => f()); cv.destroy(); } };
};
