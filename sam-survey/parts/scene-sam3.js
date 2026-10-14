/* ch-sam3: concept prompts on the toy desk. Each query proposes one region with a query score q (how well it
   matches the phrase IF the phrase is in the image); one presence token gives p (is the phrase in the image
   at all). Total score = p x q (SAM 3 v2, App. C.2); total > 0.5 keeps the mask. Hard-negative training moves
   p for look-alike phrases; exemplar boxes move q for similar regions. All scores are toy values. */
SCENES.sam3 = ctx => {
  const el = mkCanvas(ctx.stage), THR = .5, NQ = 10;
  // Toy query scores per concept (regions not listed get a small seeded baseline).
  const Q = {
    leaf: { leaf1: .93, leaf2: .95, leaf3: .92, leaf4: .44, leaf5: .41, pot: .2, leaves: .3 },
    pane: { pane1: .9, pane2: .92, pane3: .88, pane4: .9, window: .62 },
    mug: { mug: .94, saucer: .3, handle: .35, logo: .2, mugset: .4 },
    shade: { shade: .93, arm: .3, lamp: .35 },
    pencil: { pencil: .9, spine: .25 },
    ymug: { mug: .82, saucer: .25, handle: .3, logo: .45 },
    cat: { plant: .14, pot: .1 },
    none: {} };
  const EX = { leaf: { leaf4: .86, leaf5: .84 }, pane: { window: .12 } }; // exemplar effect on q
  const PRES = { leaf: [.97, .97], pane: [.96, .96], mug: [.97, .97], shade: [.95, .95], pencil: [.94, .95], ymug: [.78, .1], cat: [.05, .04], none: [.05, .05] }; // [no hard negatives, with hard negatives]
  const NAMES = { leaf: 'leaf', pane: 'window pane', mug: 'mug', shade: 'lamp shade', pencil: 'pencil', ymug: 'yellow mug', cat: 'cat', none: '' };
  const POOL = ['leaf1', 'leaf2', 'leaf3', 'leaf4', 'leaf5', 'pot', 'pane1', 'pane2', 'pane3', 'pane4', 'window', 'mug', 'saucer', 'handle', 'logo', 'shade', 'arm', 'lampbase', 'pencil', 'spine', 'leaves', 'mugset', 'lamp', 'plant'];
  const base = {}; { const r = PE.rng(31); POOL.forEach(id => { base[id] = .02 + r() * .1; }); }
  const S = { c: 'leaf', hn: 1, pres: 1, ex: 0, typed: '' };
  const cur = { q: {}, p: 0, u: 1 }, prev = { q: {}, p: 0 };
  const glow = {};
  const qOf = (c, id, ex) => { const e = ex && EX[c] && EX[c][id] != null ? EX[c][id] : null; return e != null ? e : (Q[c][id] != null ? Q[c][id] : base[id]); };
  function target() { const q = {}; POOL.forEach(id => { q[id] = qOf(S.c, id, S.ex); }); return { q, p: PRES[S.c][S.hn] }; }
  function lerpState(u) { const t = target(), o = {}; POOL.forEach(id => { o[id] = PE.lerp(prev.q[id] ?? t.q[id], t.q[id], u); }); cur.q = o; cur.p = PE.lerp(prev.p ?? t.p, t.p, u); cur.u = u; }
  const total = id => (S.pres ? cur.p : 1) * cur.q[id];
  let cancel = () => {};
  const cv = PE.canvas(el, S3ASPECT(ctx.stage), draw);
  const cBtns = ctx.qa('[data-concept]'), hnBtns = ctx.qa('[data-hn]'), prBtns = ctx.qa('[data-presence]'), exBtns = ctx.qa('[data-exemplar]');
  const input = ctx.q('[data-concept-input]'), go = ctx.q('[data-concept-go]');
  // State that must survive an atlas theme remount lives in the pressed controls.
  { const pc = cBtns.find(b => b.getAttribute('aria-pressed') === 'true'); if (pc) S.c = pc.dataset.concept;
    const ph = hnBtns.find(b => b.getAttribute('aria-pressed') === 'true'); if (ph) S.hn = +ph.dataset.hn;
    const pp = prBtns.find(b => b.getAttribute('aria-pressed') === 'true'); if (pp) S.pres = +pp.dataset.presence;
    const pe = exBtns.find(b => b.getAttribute('aria-pressed') === 'true'); if (pe) { S.ex = 1; S.c = pe.dataset.exemplar; } }
  function sync() { press(cBtns, b => b.dataset.concept === S.c && !S.typed); press(hnBtns, b => +b.dataset.hn === S.hn); press(prBtns, b => +b.dataset.presence === S.pres); press(exBtns, b => S.ex && b.dataset.exemplar === S.c); }
  function change(fn) { prev.q = { ...cur.q }; prev.p = cur.p; fn(); sync(); cancel(); cancel = PE.animate(650, u => { lerpState(u); cv.redraw(); }, readout); readout(); }
  const offs = [
    ...cBtns.map(b => listen(b, 'click', () => change(() => { S.c = b.dataset.concept; S.ex = 0; S.typed = ''; }))),
    ...hnBtns.map(b => listen(b, 'click', () => change(() => { S.hn = +b.dataset.hn; }))),
    ...prBtns.map(b => listen(b, 'click', () => change(() => { S.pres = +b.dataset.presence; }))),
    ...exBtns.map(b => listen(b, 'click', () => change(() => { S.c = b.dataset.exemplar; S.ex = 1; S.typed = ''; })))];
  function mapText(t) { t = t.toLowerCase().trim();
    if (/yellow/.test(t) && /(mug|cup)/.test(t)) return 'ymug'; if (/lea(f|ves)/.test(t)) return 'leaf'; if (/(pane|glass)/.test(t)) return 'pane';
    if (/(mug|cup)/.test(t)) return 'mug'; if (/shade/.test(t)) return 'shade'; if (/pencil/.test(t)) return 'pencil'; if (/cat/.test(t)) return 'cat'; return 'none'; }
  if (go) offs.push(listen(go, 'click', () => { const t = (input && input.value) || ''; if (!t.trim()) return; change(() => { S.c = mapText(t); S.ex = 0; S.typed = t.trim().slice(0, 40); }); }));
  if (input) offs.push(listen(input, 'keydown', e => { if (e.key === 'Enter') { e.preventDefault(); go && go.click(); } }));
  const onLink = e => { const k = e.detail; PE.animate(1100, u => { glow[k] = Math.sin(Math.PI * u); cv.redraw(); }, () => { glow[k] = 0; cv.redraw(); }); };
  ctx.article.addEventListener('pe-link', onLink);
  let nShow = NQ;
  function slots() { return POOL.slice().sort((a, b) => cur.q[b] - cur.q[a]).slice(0, nShow); }
  function draw(c, W, H) {
    c.clearRect(0, 0, W, H);
    const narrow = W < 620, pad = 12, top = 34, bottom = ctx.reserve || 0;
    const imgW = narrow ? W - 2 * pad : W * .56, imgH = narrow ? Math.min((H - bottom) * .5, imgW / TOY.WW) : Math.min(H - bottom - top - pad, imgW / TOY.WW);
    const top2 = narrow ? top : Math.max(top, (H - bottom - imgH) / 2);
    const V = TOY.fit(pad, top2, imgW, imgH), F = PE.css('--fg'), M = PE.css('--muted'), Y = PE.css('--yellow'), G = PE.css('--green'), B = PE.css('--blue');
    TOY.drawWorld(c, V);
    const panelH = narrow ? H - bottom - (top2 + imgH + 14) - 6 : imgH;
    nShow = Math.max(4, Math.min(NQ, Math.floor((panelH - 70) / 17)));
    const sl = slots(), kept = sl.filter(id => total(id) > THR);
    kept.forEach(id => TOY.drawMask(c, V, TOY.byId[id], { alpha: PE.clamp(cur.u * 1.4, 0, 1), lift: 2, lw: 2 + 3 * (glow.total || 0) }));
    // exemplar boxes
    if (S.ex && S.c === 'leaf') { const [x0, y0, x1, y1] = TOY.bbox(TOY.byId.leaf4); TOY.boxPrompt(c, V.X(x0) - 3, V.Y(y0) - 3, V.X(x1) + 3, V.Y(y1) + 3); PE.label(c, '+ exemplar', V.X(x0) - 6, V.Y(y1) + 15, Y, 11, 'right'); }
    if (S.ex && S.c === 'pane') { const [x0, y0, x1, y1] = TOY.bbox(TOY.byId.window); c.save(); c.strokeStyle = PE.css('--brown'); c.lineWidth = 2; c.setLineDash([6, 4]); c.strokeRect(V.X(x0) - 4, V.Y(y0) - 4, V.X(x1) - V.X(x0) + 8, V.Y(y1) - V.Y(y0) + 8); c.restore(); PE.label(c, '- exemplar (the frame)', V.X(x1) + 8, V.Y(y1), PE.css('--brown'), 11); }
    // prompt chip on the image
    const phrase = S.typed || NAMES[S.c]; c.save(); c.font = '600 13px system-ui, sans-serif'; const tw = c.measureText(phrase).width;
    c.fillStyle = PE.css('--panel'); c.beginPath(); c.roundRect(V.ox + 8, V.oy + 8, tw + 18, 24, 12); c.fill(); c.strokeStyle = Y; c.lineWidth = 1.5; c.stroke(); c.restore();
    PE.label(c, phrase, V.ox + 17, V.oy + 25, Y, 13);
    // the score panel
    const P = narrow ? { x: pad, y: top2 + imgH + 14, w: W - 2 * pad, h: H - bottom - (top2 + imgH + 14) - 6 } : { x: pad + imgW + 18, y: top2, w: W - imgW - 2 * pad - 18, h: imgH };
    const pres = cur.p, pg = glow.presence || 0;
    PE.label(c, S.pres ? 'presence token: is the phrase in the image?' : 'no presence head: each query decides alone', P.x, P.y + 10, S.pres ? B : M, 11);
    const bx = P.x, bw = P.w - 46, by = P.y + 18;
    c.fillStyle = PE.css('--panel2'); c.fillRect(bx, by, bw, 10);
    if (S.pres) { c.fillStyle = B; c.globalAlpha = .85; c.fillRect(bx, by, bw * pres, 10); c.globalAlpha = 1; if (pg) { c.save(); c.strokeStyle = B; c.lineWidth = 1 + 3 * pg; c.strokeRect(bx - 2, by - 2, bw + 4, 14); c.restore(); }
      PE.label(c, pres.toFixed(2), bx + bw + 6, by + 10, B, 12, 'left', 700); }
    const top3 = by + 26, rows = sl.length, rh = Math.min(22, (P.h - (top3 - P.y) - 18) / rows);
    const nameW = narrow ? 70 : Math.min(92, P.w * .34), barX = P.x + nameW, barW = P.w - nameW - 40;
    PE.label(c, narrow ? 'proposal' : 'query proposals', P.x, top3 + 2, M, 10); PE.label(c, narrow ? 'query score (outline), total (fill)' : 'query score (outline) and total (fill)', barX, top3 + 2, M, 10);
    const tx = barX + barW * THR;
    sl.forEach((id, i) => {
      const y = top3 + 10 + i * rh, q = cur.q[id], t = total(id), on = t > THR, n = TOY.byId[id];
      PE.label(c, (n.short || n.name).replace('window pane', 'pane').replace('logo on the mug', 'logo'), P.x, y + rh * .62, on ? G : M, narrow ? 10 : 11, 'left', on ? 700 : 500);
      c.save(); c.strokeStyle = F; c.globalAlpha = .7 + .3 * (glow.query || 0); c.lineWidth = 1 + 2 * (glow.query || 0); c.strokeRect(barX, y + 3, barW * q, rh - 7); c.restore();
      c.fillStyle = on ? G : PE.css('--brown'); c.globalAlpha = on ? .85 : .55; c.fillRect(barX, y + 3, barW * t, rh - 7); c.globalAlpha = 1;
      if (glow.total) { c.save(); c.strokeStyle = G; c.lineWidth = 3 * glow.total; c.strokeRect(barX - 1, y + 2, barW * t + 2, rh - 5); c.restore(); }
      PE.label(c, t.toFixed(2), barX + barW + 4, y + rh * .62, on ? G : M, 10);
    });
    c.save(); c.strokeStyle = F; c.globalAlpha = .55; c.setLineDash([3, 3]); c.beginPath(); c.moveTo(tx, top3 + 8); c.lineTo(tx, top3 + 10 + rows * rh); c.stroke(); c.restore();
    PE.label(c, 'keep if total > 0.5', tx + 3, top3 + 20 + rows * rh, M, 10);
  }
  function readout() {
    const t = target(), kept = POOL.filter(id => (S.pres ? t.p : 1) * t.q[id] > THR), phrase = S.typed || NAMES[S.c];
    let s;
    if (S.c === 'none') s = `"${phrase}" is not in this toy's vocabulary, so every query score stays low and the presence score is ${t.p.toFixed(2)}. Nothing is kept.`;
    else if (!kept.length) s = `"${phrase}": presence ${t.p.toFixed(2)}. The phrase is judged absent, so the gate closes on every query and nothing is kept.`;
    else s = `"${phrase}": presence ${S.pres ? t.p.toFixed(2) : 'not used'}; ${kept.length} instance${kept.length > 1 ? 's' : ''} kept.`;
    if (S.c === 'ymug') s += S.pres ? (S.hn ? ' Trained with hard negatives, the presence token knows a grey mug is not a yellow mug.' : ' Trained without hard negatives, the presence token is fooled by the look-alike: a false positive.') : ' Without a presence head, the mug query fires on its own: a false positive.';
    if (S.c === 'leaf' && !S.ex) s += ' The two small sideways leaves score below the line. Try the exemplar.';
    if (S.c === 'leaf' && S.ex) s += ' One exemplar box on a small leaf lifts the similar leaves above the line.';
    if (S.c === 'pane' && !S.ex) s += ' The whole window frame also scores above the line. Try the negative exemplar.';
    if (S.c === 'pane' && S.ex) s += ' A negative exemplar on the frame pushes it below the line.';
    ctx.hud(badge('toy', s));
  }
  sync(); lerpState(1); readout();
  return { flat: true, resize: () => cv.resize(), destroy: () => { cancel(); offs.forEach(f => f()); ctx.article.removeEventListener('pe-link', onLink); cv.destroy(); } };
};
