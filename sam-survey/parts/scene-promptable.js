/* ch-promptable: click the desk scene; SAM-style outputs. Modes: one click (three masks, or the blurred average a
   single-output model learns), two clicks (the fourth token: one mask), box (toy: the object whose box fits best).
   Train view: pick the mask the user meant; per-mask loss (toy stand-in: 1 - IoU), only the lowest-loss mask
   gets gradient; the IoU head is pulled toward each mask's true IoU. Toy scores. */
SCENES.promptable = ctx => {
  const el = mkCanvas(ctx.stage); el.tabIndex = 0; el.setAttribute('role', 'img');
  el.setAttribute('aria-label', 'Desk scene. Click, or move the focus ring with the arrow keys and press Enter, to place a prompt.');
  const S = { mode: 'click', out: 'three', view: 'use', target: 1, p1: [1.115, .7], p2: null, box: null, drag: null, focus: [1.115, .7], u: 1 };
  const SC = [.79, .93, .88];
  const B = sel => ctx.qa(sel);
  const modeB = B('[data-pmode]'), outB = B('[data-pout]'), viewB = B('[data-pview]'), tgtB = B('[data-ptarget]');
  // State that must survive an atlas theme remount lives in the pressed buttons.
  const pr = (bs, k) => { const b = bs.find(x => x.getAttribute('aria-pressed') === 'true'); return b ? b.dataset[k] : null; };
  S.mode = pr(modeB, 'pmode') || S.mode; S.out = pr(outB, 'pout') || S.out; S.view = pr(viewB, 'pview') || S.view; S.target = +(pr(tgtB, 'ptarget') ?? S.target);
  let V = null, cancel = () => {}, glow = {};
  const cv = PE.canvas(el, S3ASPECT(ctx.stage), draw);
  function sync() { press(modeB, b => b.dataset.pmode === S.mode); press(outB, b => b.dataset.pout === S.out); press(viewB, b => b.dataset.pview === S.view); press(tgtB, b => +b.dataset.ptarget === S.target); }
  function fan() { cancel(); cancel = PE.animate(650, t => { S.u = t; cv.redraw(); }); }
  const offs = [
    ...modeB.map(b => listen(b, 'click', () => { S.mode = b.dataset.pmode; if (S.mode !== 'click') S.view = 'use'; if (S.mode === 'box' && !S.box) S.box = [.97, .5, 1.3, .9]; if (S.mode === 'two' && !S.p2) S.p2 = [1.26, .885]; sync(); readout(); fan(); })),
    ...outB.map(b => listen(b, 'click', () => { S.out = b.dataset.pout; S.mode = 'click'; if (S.out === 'one') S.view = 'use'; sync(); readout(); fan(); })),
    ...viewB.map(b => listen(b, 'click', () => { S.view = b.dataset.pview; if (S.view === 'train') { S.mode = 'click'; S.out = 'three'; } sync(); readout(); fan(); })),
    ...tgtB.map(b => listen(b, 'click', () => { S.target = +b.dataset.ptarget; S.view = 'train'; S.mode = 'click'; S.out = 'three'; sync(); readout(); fan(); })),
  ];
  const onLink = e => { const k = e.detail; if (S.view !== 'train') { const t = viewB.find(b => b.dataset.pview === 'train'); if (t) t.click(); }
    PE.animate(1100, u => { glow[k] = Math.sin(Math.PI * u); cv.redraw(); }, () => { glow[k] = 0; cv.redraw(); }); };
  ctx.article.addEventListener('pe-link', onLink);
  const world = ev => { const r = el.getBoundingClientRect(); return [V.ix(ev.clientX - r.left), V.iy(ev.clientY - r.top)]; };
  const inWorld = p => p[0] >= 0 && p[0] <= TOY.WW && p[1] >= 0 && p[1] <= TOY.WH;
  function place(p) {
    if (!inWorld(p)) return;
    if (S.mode === 'two') { if (!S.p2 && !S.p1) S.p1 = p; else S.p2 = p; }
    else { S.p1 = p; if (S.mode === 'box') S.mode = 'click'; }
    S.focus = p; sync(); readout(); fan();
  }
  offs.push(listen(el, 'pointerdown', ev => { if (!V) return; const p = world(ev); if (!inWorld(p)) return;
    if (S.mode === 'box') { S.drag = p; S.box = [p[0], p[1], p[0], p[1]]; el.setPointerCapture(ev.pointerId); cv.redraw(); } else place(p); }));
  offs.push(listen(el, 'pointermove', ev => { if (!S.drag) return; const p = world(ev); S.box = [Math.min(S.drag[0], p[0]), Math.min(S.drag[1], p[1]), Math.max(S.drag[0], p[0]), Math.max(S.drag[1], p[1])]; cv.redraw(); }));
  offs.push(listen(el, 'pointerup', () => { if (!S.drag) return; S.drag = null; const b = S.box; if (b[2] - b[0] < .03 || b[3] - b[1] < .03) S.box = [b[0] - .1, b[1] - .1, b[0] + .1, b[1] + .1]; readout(); fan(); }));
  offs.push(listen(el, 'keydown', ev => { const d = { ArrowLeft: [-.03, 0], ArrowRight: [.03, 0], ArrowUp: [0, -.03], ArrowDown: [0, .03] }[ev.key];
    if (d) { ev.preventDefault(); S.focus = [PE.clamp(S.focus[0] + d[0], 0, TOY.WW), PE.clamp(S.focus[1] + d[1], 0, TOY.WH)]; cv.redraw(); }
    else if (ev.key === 'Enter' || ev.key === ' ') { ev.preventDefault(); if (S.mode === 'box') { const f = S.focus; S.box = [f[0] - .12, f[1] - .12, f[0] + .12, f[1] + .12]; readout(); fan(); } else place(S.focus.slice()); } }));
  // Long-page Reset remounts the scene: clear the pressed state first so the remount starts fresh.
  ctx.qa('[data-cue-reset]').forEach(b => offs.push(listen(b, 'click', () => ctx.qa('[aria-pressed]').forEach(x => x.removeAttribute('aria-pressed')))));
  offs.push(listen(el, 'focus', () => cv.redraw()), listen(el, 'blur', () => cv.redraw()));
  // What the model returns in the current state.
  function result() {
    if (S.mode === 'two') { const a = TOY.chain(...S.p1), b = S.p2 ? TOY.chain(...S.p2) : a; const both = a.filter(n => b.includes(n));
      return { kind: 'single', nodes: both.length ? [both[0]] : [], why: both.length ? 'fourth token' : 'none' }; }
    if (S.mode === 'box') { const bx = S.box; let best = null, bi = 0;
      TOY.all.forEach(n => { if (n === TOY.root || n.hidden || !n.mask) return; const [x0, y0, x1, y1] = TOY.bbox(n);
        const ix = Math.max(0, Math.min(x1, bx[2]) - Math.max(x0, bx[0])), iy = Math.max(0, Math.min(y1, bx[3]) - Math.max(y0, bx[1])), I = ix * iy;
        const U = (x1 - x0) * (y1 - y0) + (bx[2] - bx[0]) * (bx[3] - bx[1]) - I, v = U > 0 ? I / U : 0; if (v > bi) { bi = v; best = n; } });
      return { kind: 'single', nodes: best && bi > .15 ? [best] : [], why: 'box' }; }
    const t = TOY.three(...S.p1); return { kind: S.out === 'one' ? 'avg' : 'three', nodes: t };
  }
  // The single-output average: equal-weight overlap of the valid masks, softened. Darker = more of the answers agree.
  function avgMask(c, V2, nodes, a = 1) { c.save(); try { c.filter = 'blur(' + Math.max(2, V2.s * .012) + 'px)'; } catch (e) {}
    c.translate(V2.ox, V2.oy); c.scale(V2.s, V2.s); c.fillStyle = PE.css('--green'); nodes.forEach(n => { c.globalAlpha = a * .3; c.fill(n.mask); }); c.restore(); }
  function train(nodes) { const tgt = nodes[Math.min(S.target, nodes.length - 1)], tr = TOY.raster(tgt);
    const ious = nodes.map(n => TOY.iou(TOY.raster(n), tr)), loss = ious.map(v => 1 - v); let k = 0; loss.forEach((v, i) => { if (v < loss[k]) k = i; });
    return { tgt, ious, loss, k }; }
  function draw(c, W, H) {
    c.clearRect(0, 0, W, H);
    const narrow = W < 620, pad = 12, top = 40, bottom = ctx.reserve || 0;
    const mainW = narrow ? W - 2 * pad : W * .6, mainH = narrow ? (H - bottom) * .55 : H - bottom - top - pad;
    V = TOY.fit(pad, narrow ? top - 6 : top, mainW, mainH - (narrow ? 0 : 0));
    const R = result(), u = S.u, T = S.view === 'train' && R.kind === 'three' && R.nodes.length ? train(R.nodes) : null;
    TOY.drawWorld(c, V);
    if (R.kind === 'avg' && R.nodes.length) avgMask(c, V, R.nodes, u);
    else if (R.nodes.length) { const show = T ? R.nodes[T.k] : R.kind === 'three' ? R.nodes[1] || R.nodes[0] : R.nodes[0]; TOY.drawMask(c, V, show, { alpha: u }); }
    if (T) { c.save(); c.translate(V.ox, V.oy); c.scale(V.s, V.s); c.strokeStyle = PE.css('--fg'); c.lineWidth = 2 / V.s; c.setLineDash([5 / V.s, 4 / V.s]); c.stroke(T.tgt.mask); c.restore();
      const [x0t, y0t] = TOY.bbox(T.tgt); PE.label(c, 'what the user meant', V.X(x0t), V.Y(y0t) - 8, PE.css('--fg'), 11, 'left', 600); }
    // prompts
    if (S.mode === 'box' && S.box) TOY.boxPrompt(c, V.X(S.box[0]), V.Y(S.box[1]), V.X(S.box[2]), V.Y(S.box[3]));
    else { if (S.p1) TOY.point(c, V.X(S.p1[0]), V.Y(S.p1[1]), 'pos', 1, 7); if (S.mode === 'two' && S.p2) TOY.point(c, V.X(S.p2[0]), V.Y(S.p2[1]), 'pos', 1, 7); }
    if (document.activeElement === el) PE.ring(c, V.X(S.focus[0]), V.Y(S.focus[1]), 11, PE.css('--yellow'), 1.5, .9);
    // panels
    const items = R.kind === 'three' ? R.nodes.map((n, i) => ({ n, label: n.name, score: SC[i] ?? .8, i }))
      : R.kind === 'avg' ? (R.nodes.length ? [{ avg: R.nodes, label: 'one output: the average', i: 0 }] : [])
      : R.nodes.map(n => ({ n, label: n.name, score: .95, i: 0, note: R.why === 'fourth token' ? 'fourth output token' : 'box prompt' }));
    if (!items.length) PE.label(c, S.mode === 'two' ? 'No object holds both clicks.' : 'No object under the prompt in this toy scene.', narrow ? pad : mainW + 2 * pad + 6, narrow ? mainH + top + 20 : top + 20, PE.css('--muted'), 12, 'left', 500);
    const nP = Math.max(3, items.length);
    items.forEach((it, idx) => {
      let px, py, pw, ph; const t = PE.clamp(u * 1.6 - idx * .25, 0, 1);
      if (narrow) { pw = (W - 2 * pad - 16) / 3; ph = (H - bottom) - mainH - top - pad; px = pad + idx * (pw + 8); py = mainH + top + 4; }
      else { px = mainW + 2 * pad + 6; pw = W - px - pad; ph = (H - bottom - top - 2 * pad) / nP - (pad * (nP - 1)) / nP; py = top + idx * (ph + pad); }
      drawPanel(c, it, { px, py, pw, ph, t, narrow, T });
    });
  }
  function drawPanel(c, it, o) {
    const { px, py, pw, ph, t, narrow, T } = o, lab = narrow ? 30 : 34;
    c.save(); c.globalAlpha = t;
    const isMin = T && it.i === T.k, edge = T ? (isMin ? PE.css('--red') : PE.css('--brown')) : PE.css('--rule');
    c.strokeStyle = edge; c.lineWidth = T ? 2 + (isMin ? 3 * (glow.min || 0) : 0) : 1 + 2 * (glow.masks || 0); c.beginPath(); c.roundRect(px, py, pw, ph, 8); c.stroke();
    const n = it.n || it.avg[it.avg.length - 1], [x0, y0, x1, y1] = TOY.bbox(n), m = .06, bw = x1 - x0 + 2 * m, bh = y1 - y0 + 2 * m;
    const s = Math.min((pw - 12) / bw, (ph - lab - 10) / bh), V2 = { s, ox: px + (pw - bw * s) / 2 - (x0 - m) * s, oy: py + 5 - (y0 - m) * s };
    V2.ox = Math.min(px, Math.max(px + pw - TOY.WW * s, V2.ox)); V2.X = x => V2.ox + x * s; V2.Y = y => V2.oy + y * s; V2.w = TOY.WW * s; V2.h = TOY.WH * s;
    c.save(); c.beginPath(); c.rect(px + 1, py + 1, pw - 2, ph - lab - 4); c.clip(); TOY.drawWorld(c, V2);
    if (it.avg) avgMask(c, V2, it.avg); else TOY.drawMask(c, V2, it.n, { lift: 2, lw: 2 + 2 * (glow.masks || 0) });
    if (S.p1 && S.mode !== 'box') TOY.point(c, V2.X(S.p1[0]), V2.Y(S.p1[1]), 'pos', 1, 4); c.restore();
    const by = py + ph - lab + 4, bwid = pw - 16, fs = narrow ? 10 : 12;
    c.font = `600 ${fs}px system-ui, sans-serif`; const name = narrow ? (n.short || it.label) : it.label;
    PE.label(c, it.avg ? (narrow ? 'average' : it.label) : name, px + 8, by + 8, PE.css('--fg'), fs, 'left', 600);
    if (it.score != null) {
      c.fillStyle = PE.css('--panel2'); c.fillRect(px + 8, by + 16, bwid, 5);
      c.fillStyle = PE.css('--green'); c.fillRect(px + 8, by + 16, bwid * it.score * t, 5);
      if (glow.iou) { c.save(); c.globalAlpha = glow.iou; c.strokeStyle = PE.css('--green'); c.lineWidth = 2; c.strokeRect(px + 6, by + 14, bwid + 4, 9); c.restore(); }
      if (T) { const xi = px + 8 + bwid * T.ious[it.i]; c.fillStyle = PE.css('--red'); c.fillRect(xi - 1.5, by + 12, 3, 13);
        PE.label(c, `loss ${T.loss[it.i].toFixed(2)}`, px + pw - 8, by + 8, isMin ? PE.css('--red') : PE.css('--brown'), fs, 'right', 600); }
      else if (!narrow || c.measureText(name + '  0.00').width < pw - 16) PE.label(c, it.score.toFixed(2), px + pw - 8, by + 8, PE.css('--green'), fs, 'right', 600);
    }
    c.restore();
  }
  function readout() {
    const R = result(); let s;
    if (S.mode === 'two') s = R.nodes.length ? `Two clicks: ambiguity is much rarer, so SAM returns one mask through its fourth output token: ${R.nodes[0].name}.` : 'Two clicks on different objects: no single object contains both.';
    else if (S.mode === 'box') s = R.nodes.length ? `A box says how big the thing is. In this toy, the box picks ${R.nodes[0].name}, the object whose outline fits it best.` : 'Drag a box around something.';
    else if (!R.nodes.length) s = 'No object under the click in this toy scene. Click on the mug, the plant, the lamp or the window.';
    else if (S.out === 'one') s = 'One output: a model with a single mask is trained toward all the valid answers at once, so it learns their average, a blur that is no valid mask.';
    else if (S.view === 'train') { const T = train(R.nodes); s = `Training: the user meant ${T.tgt.name}. Each output gets a loss; only the lowest (${R.nodes[T.k].name}, loss ${T.loss[T.k].toFixed(2)}) sends gradient. The score head is pulled toward each mask's true overlap (red ticks).`; }
    else s = `One click, ${R.nodes.length} valid masks: ${R.nodes.map(n => n.name).join(', ')}. Each comes with a predicted score; SAM ranks them by it.`;
    ctx.hud(badge('toy', s));
  }
  sync(); readout();
  return { flat: true, resize: () => cv.resize(), destroy: () => { cancel(); offs.forEach(f => f()); ctx.article.removeEventListener('pe-link', onLink); cv.destroy(); } };
};
