/* ch-sam2: a toy video. The mug slides right along the table, passes behind a box (occluded), and comes out again.
   A click on frame 1 starts the track; each frame reads a memory bank of the 6 most recent frame memories (with
   their temporal position) plus the prompted-frame memories and object pointers; the occlusion head marks
   "object absent" behind the box. Without a correction the toy track grabs the box edge for a few frames after the
   occlusion; a correction click on a later frame becomes a prompted memory and fixes the rest. Toy throughout. */
SCENES.sam2 = ctx => {
  const el = mkCanvas(ctx.stage), T = 24, NREC = 6;
  const S = { started: 0, corrected: 0, f: 0 };
  let cancel = () => {};
  const WW = 1.6, WH = 1;
  const BOX = [.78, .3, .3, .52]; // occluder x, y, w, h (world units), standing on the table at y = .82
  const mugX = f => .22 + f * (1.36 - .22) / (T - 1);
  function mugPath(f) { const x = mugX(f), p = new Path2D();
    p.moveTo(x - .1, .55); p.lineTo(x + .1, .55); p.lineTo(x + .086, .82); p.lineTo(x - .086, .82); p.closePath();
    const h = new Path2D(); const cx = x + .098, cy = .67; for (let i = 0; i <= 20; i++) { const a = -1.3 + 2.6 * i / 20; const fn = i ? 'lineTo' : 'moveTo'; h[fn](cx + .07 * Math.cos(a), cy + .07 * Math.sin(a)); }
    for (let i = 20; i >= 0; i--) { const a = -1.3 + 2.6 * i / 20; h.lineTo(cx + .042 * Math.cos(a), cy + .042 * Math.sin(a)); } h.closePath(); p.addPath(h); return p; }
  const boxPath = (() => { const p = new Path2D(); p.rect(BOX[0], BOX[1], BOX[2], BOX[3]); return p; })();
  // Visible fraction of the mug (sampled), used for the occlusion state.
  const probe = document.createElement('canvas').getContext('2d');
  const vis = []; for (let f = 0; f < T; f++) { let n = 0, v = 0; const mp = mugPath(f);
    for (let y = .5; y < .86; y += .02) for (let x = mugX(f) - .12; x < mugX(f) + .2; x += .01) if (probe.isPointInPath(mp, x, y)) { n++; if (!probe.isPointInPath(boxPath, x, y)) v++; }
    vis.push(n ? v / n : 0); }
  const firstBack = vis.findIndex((v, i) => i > 6 && v >= .12 && vis[i - 1] < .12), CORR = firstBack + 1; // correct on the second frame after the mug reappears
  // Toy model state per frame: 'none' before the click, 'ok', 'absent' (occlusion head), 'drift' (wrong mask after the occlusion).
  function state(f) { if (!S.started) return 'none'; if (vis[f] < .12) return 'absent';
    if (!S.corrected || f < CORR) { if (firstBack > 0 && f >= firstBack && f < firstBack + 5) return 'drift'; }
    return 'ok'; }
  const fit = (x, y, w, h) => { const s = Math.min(w / WW, h / WH), ox = x + (w - WW * s) / 2, oy = y + (h - WH * s) / 2; return { s, ox, oy, w: WW * s, h: WH * s, X: v => ox + v * s, Y: v => oy + v * s }; };
  const col = t => PE.mix(PE.css('--bg'), PE.css('--fg'), t);
  function frame(c, V, f, o = {}) {
    c.save(); c.beginPath(); c.rect(V.ox, V.oy, V.w, V.h); c.clip(); c.translate(V.ox, V.oy); c.scale(V.s, V.s);
    c.fillStyle = col(.07); c.fillRect(0, 0, WW, .62); c.fillStyle = col(.13); c.fillRect(0, .62, WW, WH - .62);
    c.fillStyle = col(.2); c.fillRect(.2, .1, .34, .26); c.fillStyle = col(.32); c.fillRect(.22, .12, .14, .1); c.fillRect(.38, .12, .14, .1); c.fillRect(.22, .24, .14, .1); c.fillRect(.38, .24, .14, .1);
    c.fillStyle = col(.5); c.beginPath(); c.ellipse(mugX(f), .82, .15, .03, 0, 0, 7); c.fill();
    const mp = mugPath(f); c.fillStyle = col(.66); c.fill(mp); c.fillStyle = col(.3); c.beginPath(); c.arc(mugX(f), .69, .035, 0, 7); c.fill();
    c.fillStyle = col(.42); c.fill(boxPath); c.fillStyle = col(.5); c.fillRect(BOX[0], BOX[1], BOX[2], .05);
    const st = o.mask === false ? 'none' : state(f), G = PE.css('--green');
    if (st === 'ok' || st === 'drift') {
      c.save(); if (st === 'ok') { c.beginPath(); c.rect(0, 0, WW, WH); c.rect(BOX[0], BOX[1], BOX[2], BOX[3]); c.clip('evenodd'); }
      const m = st === 'ok' ? mp : (() => { const p = new Path2D(); p.rect(BOX[0] + BOX[2] - .06, BOX[1] + .1, .06, BOX[3] - .1); p.addPath(mp); return p; })();
      const colr = st === 'ok' ? G : PE.css('--red');
      c.globalAlpha = .4; c.fillStyle = colr; c.fill(m); c.globalAlpha = 1; c.strokeStyle = colr; c.lineWidth = 2 / V.s; c.stroke(m); c.restore(); }
    c.restore();
    return st;
  }
  const btnStart = ctx.q('[data-sam2="click"]'), btnCorr = ctx.q('[data-sam2="correct"]'), btnPlay = ctx.q('[data-sam2="play"]'), slider = ctx.q('[data-sam2-frame]');
  const offs = [];
  if (btnStart) offs.push(listen(btnStart, 'click', () => { S.started = 1; press([btnStart], () => true); goTo(0); readout(); }));
  if (btnCorr) offs.push(listen(btnCorr, 'click', () => { S.started = 1; S.corrected = 1; press([btnStart, btnCorr].filter(Boolean), () => true); play(CORR); }));
  if (btnPlay) offs.push(listen(btnPlay, 'click', () => { S.started = 1; press([btnStart].filter(Boolean), () => true); play(0); }));
  if (slider) { slider.max = T - 1; offs.push(listen(slider, 'input', () => { cancel(); S.f = +slider.value; cv.redraw(); readout(); })); }
  const glow = {};
  const onLink = e => { const k = e.detail; PE.animate(1100, u => { glow[k] = Math.sin(Math.PI * u); cv.redraw(); }, () => { glow[k] = 0; cv.redraw(); }); };
  ctx.article.addEventListener('pe-link', onLink);
  const cv = PE.canvas(el, S3ASPECT(ctx.stage), draw);
  function memories(f) { // frames held in the bank when frame f is processed
    if (!S.started || f === 0) return { rec: [], pr: [] };
    const pr = [0]; if (S.corrected && f > CORR) pr.push(CORR);
    const rec = []; for (let k = f - 1; k >= 0 && rec.length < NREC; k--) if (!pr.includes(k)) rec.push(k);
    return { rec, pr };
  }
  function draw(c, W, H) {
    c.clearRect(0, 0, W, H);
    const narrow = W < 620, pad = 12, top = 40, bottom = ctx.reserve || 0, B = PE.css('--blue'), Y = PE.css('--yellow'), G = PE.css('--green'), M = PE.css('--muted'), F = PE.css('--fg'), Br = PE.css('--brown'), R = PE.css('--red');
    const stripH = 34, avail = H - bottom - top - stripH - 16;
    const mainW = narrow ? W - 2 * pad : W * .56, mainH = narrow ? avail * .52 : Math.min(avail, mainW / WW);
    const top2 = narrow ? top : top + Math.max(0, (avail - mainH) / 2);
    const V = fit(pad, top2, mainW, mainH);
    const st = frame(c, V, S.f);
    c.save(); c.strokeStyle = B; c.globalAlpha = .35 + .65 * (glow.frame || 0); c.lineWidth = 1.5 + 2 * (glow.frame || 0); c.strokeRect(V.ox, V.oy, V.w, V.h); c.restore();
    if (!narrow) PE.label(c, `frame ${S.f + 1} of ${T}`, V.ox, V.oy - 8, F, 12, 'left', 600);
    if (S.started && S.f === 0) TOY.point(c, V.X(mugX(0)), V.Y(.69), 'pos', 1, 6);
    if (S.corrected && S.f === CORR) TOY.point(c, V.X(mugX(CORR)), V.Y(.69), 'pos', 1, 6);
    const stLab = { none: 'no prompt yet', ok: 'mask from memory', absent: 'occlusion head: object absent', drift: 'error: the track grabbed the box edge' }[st];
    PE.label(c, narrow ? `frame ${S.f + 1}: ${stLab}` : stLab, narrow ? V.ox : V.ox + V.w, V.oy - 8, st === 'absent' ? Br : st === 'drift' ? R : st === 'ok' ? G : M, narrow ? 11 : 12, narrow ? 'left' : 'right', 600);
    // memory bank panel
    const mb = narrow ? { x: pad, y: V.oy + V.h + 14, w: W - 2 * pad, h: avail - V.h - 14 } : { x: pad + mainW + 18, y: V.oy, w: W - mainW - 2 * pad - 18, h: V.h };
    c.save(); c.strokeStyle = B; c.globalAlpha = .5 + .5 * (glow.memory || 0); c.lineWidth = 1 + 2 * (glow.memory || 0); c.beginPath(); c.roundRect(mb.x, mb.y, mb.w, mb.h, 10); c.stroke(); c.restore();
    PE.label(c, 'memory bank read by this frame', mb.x + 10, mb.y + 18, B, 12, 'left', 600);
    const { rec, pr } = memories(S.f), slots = narrow ? 4 : 3, rows = narrow ? 2 : 3, cell = (mb.w - 20 - (slots - 1) * 8) / slots, ch = Math.min(cell / WW, (mb.h - 46 - rows * 16) / rows);
    const thumbs = [...pr.map(k => ({ k, p: 1 })), ...rec.map((k, i) => ({ k, p: 0, t: i + 1 }))];
    thumbs.slice(0, slots * rows).forEach((m, i) => {
      const cx = mb.x + 10 + (i % slots) * (cell + 8), cy = mb.y + 28 + Math.floor(i / slots) * (ch + 16);
      const V2 = fit(cx, cy, cell, ch); frame(c, V2, m.k);
      c.save(); c.strokeStyle = m.p ? Y : B; c.lineWidth = m.p ? 2 : 1; c.strokeRect(V2.ox, V2.oy, V2.w, V2.h); c.restore();
      const occ = state(m.k) === 'absent';
      PE.label(c, narrow ? (m.p ? `prompt f${m.k + 1}` : `t-${m.t}`) : (m.p ? `prompted: frame ${m.k + 1}` : `t-${m.t}: frame ${m.k + 1}`), V2.ox, V2.oy + V2.h + 12, m.p ? Y : B, narrow ? 9 : 10, 'left', 600);
      if (occ) { PE.dot(c, V2.ox + V2.w - 6, V2.oy + 6, 4, Br); }
      c.save(); c.fillStyle = G; c.globalAlpha = occ ? .3 : .9; for (let q = 0; q < 4; q++) c.fillRect(V2.ox + 4 + q * 5, V2.oy + 4, 3.5, 6); c.restore();
    });
    if (!thumbs.length) { PE.label(c, S.started ? 'Frame 1 holds the prompt.' : 'Click the mug on frame 1 to start.', mb.x + 10, mb.y + 40, M, 11, 'left', 400); if (S.started) PE.label(c, 'The bank is empty, so SAM 2 acts like SAM.', mb.x + 10, mb.y + 56, M, 11, 'left', 400); }
    else PE.label(c, narrow ? `${rec.length} recent + ${pr.length} prompted; ticks: pointers` : `${rec.length} recent + ${pr.length} prompted; green ticks: object pointers`, mb.x + 10, mb.y + mb.h - 8, M, narrow ? 9 : 10, 'left', 400);
    // timeline strip
    const sy = H - bottom - stripH, sx = pad, sw = W - 2 * pad, cw = sw / T;
    for (let f = 0; f < T; f++) { const s = state(f), colr = s === 'ok' ? G : s === 'absent' ? Br : s === 'drift' ? R : PE.css('--rule');
      c.fillStyle = colr; c.globalAlpha = f === S.f ? 1 : .55; c.fillRect(sx + f * cw + 1, sy + 8, cw - 2, 12); c.globalAlpha = 1;
      if ((f === 0 && S.started) || (f === CORR && S.corrected)) PE.dot(c, sx + f * cw + cw / 2, sy + 4, 3.5, Y); }
    c.save(); c.strokeStyle = F; c.lineWidth = 2; c.strokeRect(sx + S.f * cw, sy + 6, cw, 16); c.restore();
    PE.label(c, 'frames: green mask, brown absent, red error, yellow dot prompt', sx, sy + 33, M, narrow ? 9 : 10, 'left', 400);
  }
  function readout() {
    const st = state(S.f), { rec, pr } = memories(S.f);
    const t = { none: 'No prompt yet. Click the mug on frame 1, or press Play.',
      ok: S.f === 0 ? 'Frame 1: the click is the prompt. The memory bank is empty, so this frame is segmented exactly as SAM would.' : `Frame ${S.f + 1}: the frame's features attend to ${rec.length} recent memories and ${pr.length} prompted ${pr.length > 1 ? 'memories' : 'memory'}, then the SAM decoder draws the mask.`,
      absent: `Frame ${S.f + 1}: the mug is behind the box. The occlusion head says "no object", so no mask is drawn, and this frame's memory carries the occlusion embedding.`,
      drift: `Frame ${S.f + 1}: in this toy the track comes back wrong for a few frames. Press "Correct with one click": a click on frame ${CORR + 1} becomes a prompted memory and fixes every later frame.` }[st];
    ctx.hud(badge('toy', t));
  }
  function goTo(f) { cancel(); S.f = f; if (slider) slider.value = f; cv.redraw(); }
  function play(from) { cancel(); S.f = from; if (slider) slider.value = from; readout();
    if (PE.instant()) { S.f = T - 1; if (slider) slider.value = S.f; cv.redraw(); readout(); return; }
    let last = -1; cancel = PE.animate((T - from) * 220, u => { const f = Math.min(T - 1, from + Math.floor(u * (T - from))); if (f !== last) { last = f; S.f = f; if (slider) slider.value = f; cv.redraw(); readout(); } }); }
  readout();
  return { flat: true, replay: () => play(0), resize: () => cv.resize(), destroy: () => { cancel(); offs.forEach(f => f()); ctx.article.removeEventListener('pe-link', onLink); cv.destroy(); } };
};
