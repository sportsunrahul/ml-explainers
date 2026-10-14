<script>
/* Chapter 11 figures, shared by the long page and the atlas.
   SceneVisualPrompts: a toy inspection tray. Each part has a toy region embedding of 7 numbers. The prompt is
   the text vector of the name you would type, the embedding of one boxed example (a box prompt), or their
   average (T-Rex2 Eq. 12). Score = sigmoid((cos - .82) / .05); 0.5 claims a box. The vectors were tuned in node
   so that: the name "bracket" claims the angle bracket and misses the odd parts; a box on odd part 1 finds parts
   1 and 2 but not the dark, turned part 3; both finds all three. For washers the name alone is best.
   Click a part (or arrow keys / space when focused) to make it the example; its class name becomes the text prompt.
   SceneVisualPromptsBench: sourced name-vs-box numbers from T-Rex2, RF100-VL and FSOD-VLM, one axis per panel.
   o: { aspect(w) -> h/w, glow(word) -> 0..1, onSay(text), bottom (px kept free at the bottom) } */
const SceneVisualPrompts = (el, o = {}) => {
  const PARTS = [
    { kind: 'omega', cls: 'odd', lab: 'odd part 1', box: [.04, .1, .2, .34], rot: 0, v: [1.1, 1, 0, 0, 0, 0, 0] },
    { kind: 'angle', cls: 'angle', lab: 'angle bracket', box: [.29, .1, .18, .34], v: [1.16, 0, 1, 0, 0, 0, .1] },
    { kind: 'washer', cls: 'washer', lab: 'washer 1', box: [.53, .13, .16, .28], v: [0, 0, 0, 0, 1, .12, 0] },
    { kind: 'omega', cls: 'odd', lab: 'odd part 2', box: [.75, .09, .21, .36], rot: -.22, v: [1.12, .97, .05, .02, 0, 0, .03] },
    { kind: 'nut', cls: 'nut', lab: 'hex nut', box: [.05, .62, .15, .26], v: [0, 0, 0, .1, .35, 1, 0] },
    { kind: 'omega', cls: 'odd', lab: 'odd part 3', box: [.26, .57, .21, .36], rot: Math.PI / 2, dark: true, v: [1.3, 1, .05, 1.25, 0, 0, .06] },
    { kind: 'washer', cls: 'washer', lab: 'washer 2', box: [.53, .6, .16, .28], dark: true, v: [0, 0, 0, .8, 1, .12, 0] },
    { kind: 'bolt', cls: 'bolt', lab: 'bolt', box: [.74, .63, .22, .23], v: [0, 0, .05, 0, 0, .55, 1] },
  ];
  const TEXT = { odd: ['bracket', [1.27, .87, 1.29, .5, 0, 0, 0]], angle: ['angle bracket', [1, 0, 1.2, 0, 0, 0, 0]],
    washer: ['washer', [0, 0, 0, .34, 1, .08, 0]], nut: ['hex nut', [0, 0, 0, 0, .25, 1, .1]], bolt: ['bolt', [0, 0, 0, 0, 0, .45, 1]] };
  const PLURAL = { odd: 'odd parts', angle: 'angle brackets', washer: 'washers', nut: 'hex nuts', bolt: 'bolts' };
  const C0 = .82, TAU = .05, TH = .5;
  const unit = v => { const L = Math.hypot(...v) || 1; return v.map(x => x / L); };
  const cos = (a, b) => unit(a).reduce((s, x, i) => s + x * unit(b)[i], 0);
  const sig = z => 1 / (1 + Math.exp(-z));
  const glow = o.glow || (() => 0), bottom = o.bottom || 0;
  const promptVec = (mode, ex) => {
    const t = TEXT[PARTS[ex].cls][1], v = PARTS[ex].v;
    return mode === 'text' ? unit(t) : mode === 'box' ? unit(v) : unit(unit(t).map((x, i) => x + unit(v)[i]));
  };
  const scoresFor = (mode, ex) => { const p = promptVec(mode, ex); return PARTS.map(q => sig((cos(p, q.v) - C0) / TAU)); };
  const W0 = { text: [1, 0], box: [0, 1], both: [1, 1] };
  const reset0 = () => ({ mode: 'text', ex: 0, exFrom: 0, uEx: 1, sc: scoresFor('text', 0), w: W0.text.slice() });
  let S = reset0(), stop = () => {};

  function geo(w, h) {
    const hh = h - bottom, narrow = w < 620, sw = narrow ? w : w * .6, compact = narrow && hh < w * 1.45;
    const sh = narrow ? (compact ? hh * .56 : Math.min(hh * .5, sw * .9)) : hh, band = narrow ? 66 : 40, bandY = narrow ? 46 : 20;
    const tray = { x: 8, y: band, w: sw - 16, h: sh - band - 8 };
    const pan = narrow ? { x: 12, y: sh + 8, w: w - 24, h: hh - sh - 12 } : { x: sw + 12, y: 12, w: w - sw - 24, h: hh - 24 };
    return { hh, narrow, compact, sw, sh, band, bandY, tray, pan };
  }
  const rectOf = (i, g, pad = .012) => { const [u, v, a, b] = PARTS[i].box, t = g.tray;
    return [t.x + (u - pad) * t.w, t.y + (v - pad) * t.h, (a + 2 * pad) * t.w, (b + 2 * pad) * t.h]; };

  // Flat parts drawn around (cx, cy) with half-size r; dark parts use more ink.
  function drawPart(ctx, p, x, y, w, h, k) {
    const cx = x + w / 2, cy = y + h / 2, r = Math.min(w, h) * .42;
    const f = p.dark ? DK.mixh(k.bg, k.fg, .5) : DK.mixh(k.bg, k.fg, .2), s = p.dark ? DK.mixh(k.bg, k.fg, .7) : DK.mixh(k.bg, k.fg, .48);
    const hole = k.p2;
    ctx.save(); ctx.translate(cx, cy); ctx.rotate(p.rot || 0); ctx.lineJoin = 'round'; ctx.lineCap = 'round';
    const H = (hx, hy, hr) => { ctx.beginPath(); ctx.arc(hx * r, hy * r, hr * r, 0, 7); ctx.fillStyle = hole; ctx.fill(); };
    if (p.kind === 'omega') {
      const path = () => { ctx.beginPath(); ctx.moveTo(-.95 * r, .55 * r); ctx.lineTo(-.5 * r, .55 * r); ctx.lineTo(-.5 * r, 0);
        ctx.arc(0, 0, .5 * r, Math.PI, 0); ctx.lineTo(.5 * r, .55 * r); ctx.lineTo(.95 * r, .55 * r); };
      path(); ctx.strokeStyle = s; ctx.lineWidth = .3 * r; ctx.stroke();
      path(); ctx.strokeStyle = f; ctx.lineWidth = .18 * r; ctx.stroke();
      H(-.78, .55, .07); H(.78, .55, .07);
    } else if (p.kind === 'angle') {
      ctx.beginPath(); [[-.7, -.85], [-.35, -.85], [-.35, .45], [.85, .45], [.85, .8], [-.7, .8]].forEach(([a, b], i) => i ? ctx.lineTo(a * r, b * r) : ctx.moveTo(a * r, b * r));
      ctx.closePath(); ctx.fillStyle = f; ctx.fill(); ctx.strokeStyle = s; ctx.lineWidth = 1.4; ctx.stroke(); H(-.52, -.45, .09); H(.5, .62, .09);
    } else if (p.kind === 'washer') {
      ctx.beginPath(); ctx.arc(0, 0, .8 * r, 0, 7); ctx.fillStyle = f; ctx.fill(); ctx.strokeStyle = s; ctx.lineWidth = 1.4; ctx.stroke();
      ctx.beginPath(); ctx.arc(0, 0, .36 * r, 0, 7); ctx.fillStyle = hole; ctx.fill(); ctx.stroke();
    } else if (p.kind === 'nut') {
      ctx.beginPath(); for (let j = 0; j < 6; j++) { const a = Math.PI / 6 + j * Math.PI / 3; ctx.lineTo(.82 * r * Math.cos(a), .82 * r * Math.sin(a)); }
      ctx.closePath(); ctx.fillStyle = f; ctx.fill(); ctx.strokeStyle = s; ctx.lineWidth = 1.4; ctx.stroke();
      ctx.beginPath(); ctx.arc(0, 0, .34 * r, 0, 7); ctx.fillStyle = hole; ctx.fill(); ctx.stroke();
    } else if (p.kind === 'bolt') {
      ctx.fillStyle = f; ctx.strokeStyle = s; ctx.lineWidth = 1.4;
      ctx.fillRect(-r, -.42 * r, .42 * r, .84 * r); ctx.strokeRect(-r, -.42 * r, .42 * r, .84 * r);
      ctx.fillRect(-.58 * r, -.2 * r, 1.55 * r, .4 * r); ctx.strokeRect(-.58 * r, -.2 * r, 1.55 * r, .4 * r);
      ctx.beginPath(); for (let j = 0; j < 7; j++) { const xx = (-.3 + j * .18) * r; ctx.moveTo(xx, -.2 * r); ctx.lineTo(xx + .08 * r, .2 * r); } ctx.stroke();
    }
    ctx.restore();
  }
  function pill(ctx, x, y, text, col, k, o2 = {}) {
    ctx.save(); ctx.globalAlpha = o2.alpha == null ? 1 : o2.alpha; ctx.font = `600 ${o2.fs || 12}px system-ui, sans-serif`;
    const w = ctx.measureText(text).width + 18, h = 24;
    DK.rr(ctx, x, y - h / 2, w, h, h / 2); ctx.fillStyle = k.bg; ctx.fill(); ctx.strokeStyle = col; ctx.lineWidth = 1.3 + (o2.lw || 0);
    if (o2.dash) ctx.setLineDash(o2.dash); ctx.stroke(); ctx.setLineDash([]);
    ctx.fillStyle = col; ctx.textBaseline = 'middle'; ctx.fillText(text, x + 9, y + .5); ctx.restore(); return w;
  }
  const halo = (ctx, s, x, y, k) => { ctx.lineWidth = 4; ctx.strokeStyle = k.bg; ctx.lineJoin = 'round'; ctx.strokeText(s, x, y); ctx.fillText(s, x, y); };

  function draw(ctx, w, h) {
    const k = DK.C(); ctx.clearRect(0, 0, w, h); if (w < 60 || h < 60) return;
    const g = geo(w, h), t = g.tray, ex = PARTS[S.ex], cls = ex.cls, name = TEXT[cls][0];
    const gT = glow('text'), gV = glow('visual'), gM = glow('mixed'), gE = glow('exemplar'), gF = glow('features'), gR = glow('regions'), gS = glow('scores'), gA = glow('align');
    // prompt band
    ctx.save(); ctx.font = `500 ${g.narrow ? 11 : 12}px system-ui, sans-serif`; ctx.fillStyle = k.mu; ctx.textBaseline = 'middle';
    ctx.fillText('prompt', 12, g.bandY); const px0 = 12 + ctx.measureText('prompt').width + 8; ctx.restore();
    const wT = PE.clamp(S.w[0], 0, 1), wV = PE.clamp(S.w[1], 0, 1);
    const tw = pill(ctx, px0, g.bandY, name, k.Y, k, { alpha: .3 + .7 * wT, lw: 2 * gT + 2 * gM, fs: g.narrow ? 11 : 12 });
    ctx.save(); ctx.globalAlpha = .4 + .6 * Math.min(wT, wV); ctx.fillStyle = k.mu; ctx.font = `600 13px system-ui, sans-serif`; ctx.textBaseline = 'middle';
    ctx.fillText('+', px0 + tw + 6, g.bandY); ctx.restore();
    pill(ctx, px0 + tw + 20, g.bandY, 'box on ' + ex.lab, k.Y, k, { alpha: .3 + .7 * wV, dash: [4, 3], lw: 2 * gE + 2 * gM, fs: g.narrow ? 11 : 12 });
    // tray
    DK.rr(ctx, t.x, t.y, t.w, t.h, 10); ctx.fillStyle = k.p2; ctx.fill(); ctx.strokeStyle = k.rule; ctx.lineWidth = 1; ctx.stroke();
    PARTS.forEach((p, i) => { const [x, y, bw, bh] = rectOf(i, g, -.01); drawPart(ctx, p, x, y, bw, bh, k); });
    // the example box (a prompt, so yellow); it glides when the example changes
    const r0 = rectOf(S.exFrom, g, .035), r1 = rectOf(S.ex, g, .035), R = r0.map((a, j) => PE.lerp(a, r1[j], S.uEx));
    ctx.save(); ctx.globalAlpha = .28 + .72 * wV; ctx.strokeStyle = k.Y; ctx.lineWidth = 1.5 + 2.5 * gE; ctx.setLineDash([5, 4]);
    ctx.strokeRect(R[0], R[1], R[2], R[3]); ctx.setLineDash([]); ctx.restore();
    // overlays: proposals, claims, misses
    const fs = g.narrow ? 10 : 11;
    PARTS.forEach((p, i) => {
      const [x, y, bw, bh] = rectOf(i, g), s = S.sc[i], target = p.cls === cls, on = s >= TH;
      if (on) {
        DK.brackets(ctx, x, y, bw, bh, k.G, { lw: 2.2 + gR });
        DK.tag(ctx, x, y + 1, S.mode === 'box' ? 'box' : S.mode === 'both' ? name + ' + box' : name, s.toFixed(2), target ? k.G : k.R, k, { size: fs, maxX: t.x + t.w - 2 });
      } else if (target) DK.brackets(ctx, x, y, bw, bh, k.Br, { lw: 1.6 + gR, dash: [4, 3] });
      else DK.brackets(ctx, x, y, bw, bh, k.B, { lw: 1.3 + 1.5 * gR, alpha: .45 + .5 * gR });
      if (!on) { ctx.save(); ctx.font = `500 ${fs}px system-ui, sans-serif`; ctx.fillStyle = target ? k.Br : k.mu; ctx.textAlign = 'center';
        halo(ctx, s.toFixed(2), x + bw / 2, y + bh + fs + 2 > t.y + t.h ? y + bh - 3 : y + bh + fs, k); ctx.restore(); }
    });
    if (gF > 0) { // deformable attention: a few sampling points around the example box read the image features
      const cx = R[0] + R[2] / 2, cy = R[1] + R[3] / 2, off = [[-.36, -.3], [.32, -.34], [.38, .2], [-.3, .34], [.05, -.42], [-.05, .42], [.42, -.04], [-.42, .06]];
      off.forEach(([a, b]) => { const x2 = cx + a * R[2], y2 = cy + b * R[3]; ctx.save(); ctx.globalAlpha = gF; ctx.strokeStyle = k.B; ctx.lineWidth = 1;
        ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(x2, y2); ctx.stroke(); ctx.restore(); PE.dot(ctx, x2, y2, 3.5, k.B, gF); });
      ctx.save(); ctx.globalAlpha = gF; ctx.font = `600 ${fs}px system-ui, sans-serif`; ctx.fillStyle = k.B; halo(ctx, 'sampled features', R[0], R[1] + R[3] + fs + 4, k); ctx.restore();
    }
    drawPanel(ctx, g, k, { gT, gV, gM, gS, gA, wT, wV, cls });
  }

  function drawPanel(ctx, g, k, z) {
    const P = g.pan, n = g.narrow, fs = n ? 10.5 : 11.5, ex = S.ex;
    const T = unit(TEXT[z.cls][1]), V = unit(PARTS[ex].v), M = unit(T.map((x, i) => x + V[i]));
    ctx.save(); ctx.textBaseline = 'middle';
    ctx.font = `600 ${fs + .5}px system-ui, sans-serif`; ctx.fillStyle = k.fg; if (!g.compact) ctx.fillText('Prompt vector (toy, 7 numbers)', P.x, P.y + 8);
    const rowH = n ? 24 : 30, lab = n ? 92 : 104, vy = P.y + 26, barW = Math.min(14, (P.w - lab - 70) / 7 - 3);
    const rows = g.compact ? [] : [['text T', T, k.Y, z.wT, z.gT], ['box V', V, k.B, z.wV, z.gV], ['mixed (T+V)/2', M, k.fg, Math.min(z.wT, z.wV), z.gM]];
    rows.forEach(([nm, vec, col, a, gl], j) => {
      const y = vy + j * rowH, used = (S.mode === 'text' && j === 0) || (S.mode === 'box' && j === 1) || (S.mode === 'both' && j === 2);
      ctx.globalAlpha = .35 + .65 * Math.max(a, gl); ctx.font = `${used ? 700 : 500} ${fs}px system-ui, sans-serif`; ctx.fillStyle = col; ctx.fillText(nm, P.x, y + rowH / 2 - 2);
      vec.forEach((x, i) => { const bh = Math.max(1, x * (rowH - 8)), bx = P.x + lab + i * (barW + 3);
        ctx.fillStyle = col; ctx.fillRect(bx, y + rowH - 4 - bh, barW, bh + gl * 2); });
      if (used) { ctx.globalAlpha = 1; ctx.font = `500 ${fs - .5}px system-ui, sans-serif`; ctx.fillStyle = k.mu; ctx.fillText('in use', P.x + lab + 7 * (barW + 3) + 6, y + rowH / 2 - 2); }
      ctx.globalAlpha = 1;
    });
    if (z.gA > 0 && !g.compact) { // T-Rex2 Eq. 9 pulls V toward T in training
      const x = P.x + lab - 10, y1 = vy + rowH / 2, y2 = vy + rowH * 1.5;
      ctx.save(); ctx.globalAlpha = z.gA; ctx.strokeStyle = k.R; ctx.lineWidth = 2; ctx.setLineDash([4, 3]);
      ctx.beginPath(); ctx.moveTo(x, y1); ctx.quadraticCurveTo(x - 16, (y1 + y2) / 2, x, y2); ctx.stroke(); ctx.setLineDash([]);
      PE.arrow(ctx, x - 6, y1 + 5, x, y1 + 1, k.R, 2, 6); PE.arrow(ctx, x - 6, y2 - 5, x, y2 - 1, k.R, 2, 6);
      ctx.font = `600 ${fs}px system-ui, sans-serif`; ctx.fillStyle = k.R; ctx.textAlign = 'right'; halo(ctx, 'align', x - 14, (y1 + y2) / 2, k); ctx.restore();
    }
    // scores
    const sy = g.compact ? P.y + 6 : vy + 3 * rowH + (n ? 14 : 22), avail = P.y + P.h - sy - 18, sh = Math.min(n ? 20 : 26, avail / PARTS.length);
    ctx.font = `600 ${fs + .5}px system-ui, sans-serif`; ctx.fillStyle = k.fg; ctx.fillText('Score of every box', P.x, sy);
    const bx0 = P.x + lab, bw = P.w - lab - 40, thx = bx0 + bw * TH;
    PARTS.forEach((p, i) => {
      const y = sy + 14 + i * sh + sh / 2, s = S.sc[i], target = p.cls === z.cls, on = s >= TH;
      ctx.font = `${target ? 700 : 500} ${fs}px system-ui, sans-serif`; ctx.fillStyle = target ? k.fg : k.mu; ctx.fillText(p.lab, P.x, y);
      ctx.fillStyle = on ? (target ? k.G : k.R) : (target ? k.Br : DK.mixh(k.bg, k.fg, .3));
      const L = Math.max(1.5, bw * s); ctx.fillRect(bx0, y - 5 - z.gS * 2, L, 10 + z.gS * 4);
      ctx.fillStyle = k.mu; ctx.font = `500 ${fs - .5}px system-ui, sans-serif`; ctx.fillText(s.toFixed(2), bx0 + L + 5, y);
      if (on && !target && bx0 + L + 34 + 50 < P.x + P.w) { ctx.fillStyle = k.R; ctx.fillText('false hit', bx0 + L + 34, y); }
    });
    ctx.strokeStyle = k.mu; ctx.setLineDash([3, 3]); ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(thx, sy + 10); ctx.lineTo(thx, sy + 14 + PARTS.length * sh + 2); ctx.stroke(); ctx.setLineDash([]);
    ctx.font = `500 ${fs - 1}px system-ui, sans-serif`; ctx.fillStyle = k.mu; ctx.textAlign = 'center'; ctx.fillText('0.5 claims a box', thx, sy + 14 + PARTS.length * sh + 10);
    ctx.restore();
  }

  const cv = PE.canvas(el, o.aspect || (w => (w < 620 ? 1.9 : .52)), draw);
  function describe() {
    const ex = PARTS[S.ex], cls = ex.cls, name = TEXT[cls][0], sc = scoresFor(S.mode, S.ex);
    const tgt = PARTS.map((p, i) => i).filter(i => PARTS[i].cls === cls), found = tgt.filter(i => sc[i] >= TH), missed = tgt.filter(i => sc[i] < TH);
    const fh = PARTS.map((p, i) => i).filter(i => PARTS[i].cls !== cls && sc[i] >= TH);
    const head = { text: `Text prompt "${name}"`, box: `Box prompt on ${ex.lab}`, both: `Name "${name}" and the box on ${ex.lab}, averaged` }[S.mode];
    const what = tgt.length > 1 ? `${found.length} of the ${tgt.length} ${PLURAL[cls]}` : (found.length ? `the ${ex.lab}` : `not the ${ex.lab}`);
    let s = `${head}: finds ${what}`;
    if (missed.length && tgt.length > 1) s += `; misses ${missed.map(i => `${PARTS[i].lab} (${sc[i].toFixed(2)})`).join(', ')}`;
    s += '.';
    s += fh.length ? ` False hit: ${fh.map(i => `${PARTS[i].lab} (${sc[i].toFixed(2)})`).join(', ')}.` : ' No false hits.';
    return s;
  }
  const say = () => o.onSay && o.onSay(describe());
  function retarget(ms = 700) {
    const sc0 = S.sc.slice(), sc1 = scoresFor(S.mode, S.ex), w0 = S.w.slice(), w1 = W0[S.mode], u0 = S.uEx;
    stop(); stop = PE.animate(ms, u => { S.sc = sc0.map((a, i) => PE.lerp(a, sc1[i], u)); S.w = w0.map((a, i) => PE.lerp(a, w1[i], u)); S.uEx = PE.lerp(u0, 1, u); cv.redraw(); });
  }
  // Each mode is a state: setting it twice equals setting it once.
  function set(mode) { if (!W0[mode]) return; S.mode = mode; S.exFrom = S.ex; retarget(); say(); }
  function pick(i) { if (i === S.ex) return; S.exFrom = S.ex; S.ex = i; S.uEx = 0; retarget(); say(); }
  function reset() { stop(); S = reset0(); cv.redraw(); say(); }
  const offs = [
    ['pointerdown', e => { const r = el.getBoundingClientRect(), g = geo(r.width, r.height), x = e.clientX - r.left, y = e.clientY - r.top;
      PARTS.forEach((p, i) => { const [a, b, c, d] = rectOf(i, g, .02); if (x >= a && x <= a + c && y >= b && y <= b + d) pick(i); }); }],
    ['keydown', e => { const d = { ArrowRight: 1, ArrowDown: 1, ' ': 1, ArrowLeft: -1, ArrowUp: -1 }[e.key]; if (!d) return;
      e.preventDefault(); pick((S.ex + d + PARTS.length) % PARTS.length); }],
  ].map(([ev, fn]) => { el.addEventListener(ev, fn); return () => el.removeEventListener(ev, fn); });
  say();
  return { set, pick, reset, redraw: cv.redraw, resize: () => cv.resize && cv.resize(), mode: () => S.mode,
    destroy: () => { stop(); offs.forEach(f => f()); cv.destroy && cv.destroy(); } };
};

const SceneVisualPromptsBench = (el, o = {}) => {
  // Name vs box (or image) prompts as printed. Panel 1: T-Rex2 Table 1 (Swin-T, Text vs Visual-G) and Table 8
  // (mixed, Swin-T; LVIS val and COCO only). Panel 2: RF100-VL Table 2 (MQ-GLIP-Text zero-shot, MQ-GLIP-Image
  // 10-shot, MQ-GLIP names + images 10-shot). Panel 3: FSOD-VLM Table 3 (10-shot; MQ-GLIP-Text, -Image, MQ-GLIP).
  const DATA = [
    { title: 'T-Rex2 Swin-T, its own tests', sub: 'box prompt: 16 training-set boxes per class, averaged', max: 50, ticks: [0, 25, 50], dec: 1, groups: [
      { a: 'LVIS val rare', b: '337 classes, APr', v: [29.0, 32.4, 34.3] },
      { a: 'LVIS minival rare', b: '70 classes, APr', v: [37.4, 29.9, null] },
      { a: 'Roboflow100', b: 'not RF100-VL, AP', v: [8.2, 17.4, null] },
      { a: 'COCO val', b: 'common, AP', v: [45.8, 38.8, 42.5] } ] },
    { title: 'RF100-VL, MQ-GLIP', sub: 'image prompt: 10 shots per class, frozen GLIP-L', max: 15, ticks: [0, 5, 10, 15], dec: 1, groups: [
      { a: 'all', b: '100 datasets, AP', v: [12.2, 6.4, 12.2] },
      { a: 'industrial', b: '22 datasets, AP', v: [7.8, 3.9, 7.8] } ] },
    { title: 'FSOD-VLM nuImages, MQ-GLIP', sub: 'image prompt: 10 shots per class, frozen GLIP-L', max: 25, ticks: [0, 10, 20], dec: 2, groups: [
      { a: 'Few cohort', b: 'rarest classes, AP', v: [8.41, 3.34, 10.26] },
      { a: 'all classes', b: '18 classes, AP', v: [17.01, 14.07, 21.42] } ] },
  ];
  const bottom = o.bottom || 0;
  let S = { mode: o.mixed ? 'mixed' : 'pair', uM: o.mixed ? 1 : 0 }, stop = () => {};
  function wrap(ctx, s, maxW) {
    const words = s.split(' '), out = []; let line = '';
    words.forEach(wd => { const t = line ? line + ' ' + wd : wd; if (ctx.measureText(t).width > maxW && line) { out.push(line); line = wd; } else line = t; });
    if (line) out.push(line); return out;
  }
  function legend(ctx, k, w, fs, y0) {
    let lx = 12, ly = y0;
    [['name (text prompt)', k.Y, 1], ['box or image prompt', k.B, 1], ['both, averaged', k.fg, S.uM]].forEach(([s, c, a], j) => {
      ctx.font = `500 ${fs}px system-ui, sans-serif`; const tw = ctx.measureText(s).width;
      if (lx + tw + 15 > w - 8) { lx = 12; ly += 18; }
      ctx.globalAlpha = .25 + .75 * a;
      if (j === 2) { ctx.strokeStyle = c; ctx.lineWidth = 1.4; ctx.strokeRect(lx + .7, ly - 4.3, 8.6, 8.6); } else { ctx.fillStyle = c; ctx.fillRect(lx, ly - 5, 10, 10); }
      ctx.fillStyle = k.fg; ctx.textAlign = 'left'; ctx.fillText(s, lx + 15, ly); lx += tw + 32; ctx.globalAlpha = 1; });
    return ly;
  }
  // One bar (vertical on desktop, horizontal on phones); j: 0 name, 1 box, 2 both (outlined).
  function bar(ctx, k, j, x, y, bw, bl, horiz) {
    const col = [k.Y, k.B, k.fg][j];
    if (j === 2) { ctx.strokeStyle = col; ctx.lineWidth = 1.6; if (horiz) ctx.strokeRect(x + .8, y + .8, Math.max(0, bl - 1.6), bw - 1.6); else ctx.strokeRect(x + .8, y - bl + .8, bw - 1.6, Math.max(0, bl - 1.6)); }
    else { ctx.fillStyle = col; if (horiz) ctx.fillRect(x, y, bl, bw); else ctx.fillRect(x, y - bl, bw, bl); }
  }
  function draw(ctx, w, h) {
    const k = DK.C(); ctx.clearRect(0, 0, w, h); if (w < 60 || h < 60) return;
    const hh = h - bottom, narrow = w < 620, fs = narrow ? 11 : 11.5, nG = DATA.reduce((s, p) => s + p.groups.length, 0);
    ctx.save(); ctx.textBaseline = 'middle';
    const ly = legend(ctx, k, narrow ? w - 8 : w - 110, fs, narrow ? 44 : 18), top = ly + 20;
    if (narrow && hh >= 600) {
      // phones: panels stacked, one horizontal bar cluster per group, values at the bar ends
      const head = 40, axisH = 20, gh = PE.clamp((hh - top - DATA.length * (head + axisH)) / nG, 34, 58), bw = Math.min(11, (gh - 16) / 3 - 2);
      let y = top;
      DATA.forEach(P => {
        ctx.font = `700 ${fs + 1}px system-ui, sans-serif`; ctx.fillStyle = k.fg; ctx.textAlign = 'left'; ctx.fillText(P.title, 10, y + 8);
        ctx.font = `500 ${fs - 1}px system-ui, sans-serif`; ctx.fillStyle = k.mu; ctx.fillText(wrap(ctx, P.sub, w - 20)[0], 10, y + 24);
        y += head;
        const x0 = 12, x1 = w - 46, X = v => x0 + (x1 - x0) * v / P.max;
        P.groups.forEach(G => {
          ctx.font = `600 ${fs - .5}px system-ui, sans-serif`; ctx.fillStyle = k.fg; ctx.textAlign = 'left'; ctx.fillText(G.a, x0, y + 6);
          const wa = ctx.measureText(G.a + '  ').width; ctx.font = `500 ${fs - 1}px system-ui, sans-serif`; ctx.fillStyle = k.mu; ctx.fillText(G.b, x0 + wa, y + 6);
          G.v.forEach((v, j) => {
            const a = j === 2 ? S.uM : 1, by = y + 14 + j * (bw + 2); if (a <= .01) return;
            ctx.globalAlpha = a;
            if (v == null) { ctx.font = `500 ${fs - 1}px system-ui, sans-serif`; ctx.fillStyle = k.mu; ctx.fillText('n/a', x0 + 2, by + bw / 2); ctx.globalAlpha = 1; return; }
            const bl = (X(v) - x0) * (j === 2 ? S.uM : 1); bar(ctx, k, j, x0, by, bw, bl, true);
            ctx.font = `${j < 2 && j === (G.v[1] > G.v[0] ? 1 : 0) ? 700 : 500} ${fs - 1}px system-ui, sans-serif`; ctx.fillStyle = k.fg;
            ctx.fillText(v.toFixed(P.dec), x0 + bl + 5, by + bw / 2); ctx.globalAlpha = 1;
          });
          y += gh;
        });
        ctx.strokeStyle = k.rule; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(x0, y - 2); ctx.lineTo(x1, y - 2); ctx.stroke();
        ctx.font = `500 ${fs - 1}px system-ui, sans-serif`; ctx.fillStyle = k.mu; ctx.textAlign = 'center';
        P.ticks.forEach(t => ctx.fillText(String(t), X(t), y + 8)); ctx.textAlign = 'left';
        y += axisH;
      });
      ctx.restore(); return;
    }
    if (!narrow) {
      const gap = 22;
      DATA.forEach((P, pi) => {
        const before = DATA.slice(0, pi).reduce((s, q) => s + q.groups.length, 0), pw = (w - 16 - gap * 2) * P.groups.length / nG;
        panelV(ctx, k, P, 8 + (w - 16 - gap * 2) * before / nG + gap * pi, pw, top, hh - top - 6, fs, true);
      });
    } else { // short phone stage (atlas): stacked panels, vertical bars, settings left to the panel text
      const ph = (hh - top) / DATA.length;
      DATA.forEach((P, pi) => panelV(ctx, k, P, 8, w - 16, top + pi * ph, ph - 4, 11, false));
    }
    ctx.restore();
  }
  function panelV(ctx, k, P, x0, pw, y0, ph, fs, showSub) {
    ctx.font = `700 ${fs + 1}px system-ui, sans-serif`; ctx.fillStyle = k.fg; ctx.textAlign = 'left'; ctx.fillText(P.title, x0, y0 + 8);
    ctx.font = `500 ${fs - .5}px system-ui, sans-serif`; ctx.fillStyle = k.mu;
    const subL = showSub ? wrap(ctx, P.sub, pw) : []; subL.forEach((s, j) => ctx.fillText(s, x0, y0 + 24 + j * 14));
    const axL = 24, py = y0 + (showSub ? 24 + subL.length * 14 + 14 : 30), labH = showSub ? 46 : 38, pb = y0 + ph - labH, plotH = pb - py;
    const Y = v => pb - plotH * v / P.max, gw = (pw - axL) / P.groups.length;
    ctx.strokeStyle = k.rule; ctx.lineWidth = 1; ctx.font = `500 ${fs - 1}px system-ui, sans-serif`; ctx.fillStyle = k.mu; ctx.textAlign = 'right';
    P.ticks.forEach(t => { ctx.beginPath(); ctx.moveTo(x0 + axL, Y(t)); ctx.lineTo(x0 + pw, Y(t)); ctx.stroke(); ctx.fillText(String(t), x0 + axL - 5, Y(t)); });
    P.groups.forEach((G, gi) => {
      const gx = x0 + axL + gi * gw, bw = Math.min(22, gw / 4.2), cx = gx + gw / 2, xs = [cx - bw * 1.6, cx - bw * .5, cx + bw * .6];
      const win = G.v[1] > G.v[0] ? 1 : 0;
      G.v.forEach((v, j) => {
        const a = j === 2 ? S.uM : 1, x = xs[j]; if (a <= .01) return;
        ctx.globalAlpha = a; ctx.textAlign = 'center';
        if (v == null) { ctx.font = `500 ${fs - 1}px system-ui, sans-serif`; ctx.fillStyle = k.mu; ctx.fillText('n/a', x + bw / 2, pb - 8); ctx.globalAlpha = 1; return; }
        const bl = (pb - Y(v)) * (j === 2 ? S.uM : 1); bar(ctx, k, j, x, pb, bw, bl, false);
        ctx.font = `${j === win && j < 2 ? 700 : 500} ${fs - 1}px system-ui, sans-serif`; ctx.fillStyle = k.fg;
        ctx.fillText(v.toFixed(P.dec), x + bw / 2, pb - bl - 8); ctx.globalAlpha = 1;
      });
      ctx.textAlign = 'center'; ctx.font = `600 ${fs - .5}px system-ui, sans-serif`; ctx.fillStyle = k.fg;
      const la = showSub ? wrap(ctx, G.a, gw - 4) : [G.a]; la.slice(0, 2).forEach((s, j) => ctx.fillText(s, cx, pb + 11 + j * 13));
      ctx.font = `500 ${fs - 1}px system-ui, sans-serif`; ctx.fillStyle = k.mu; const lb = wrap(ctx, G.b, gw - 4), o2 = Math.min(2, la.length) * 13;
      lb.slice(0, 2).forEach((s, j) => ctx.fillText(s, cx, pb + 11 + o2 + j * 12));
    });
    ctx.strokeStyle = k.mu; ctx.beginPath(); ctx.moveTo(x0 + axL, pb); ctx.lineTo(x0 + pw, pb); ctx.stroke();
  }
  const cv = PE.canvas(el, o.aspect || (w => (w < 620 ? 700 / w : .5)), draw);
  const TEXTS = {
    pair: 'Name against box, in 8 groups: the box prompt wins 2 (both T-Rex2: LVIS val rare and the original Roboflow100) and loses 6. Compare bars inside a group only; each group has its own metric and setting.',
    mixed: 'Name plus box, in the 6 groups that report it: better than the name alone in 3, equal in 2 (RF100-VL), worse in 1 (T-Rex2 on COCO).',
  };
  const say = () => o.onSay && o.onSay(TEXTS[S.mode]);
  function set(mode) { if (!TEXTS[mode]) return; S.mode = mode; const a = S.uM, b = mode === 'mixed' ? 1 : 0;
    stop(); stop = PE.animate(650, u => { S.uM = PE.lerp(a, b, u); cv.redraw(); }); say(); }
  function reset() { stop(); S = { mode: 'pair', uM: 0 }; cv.redraw(); say(); }
  say();
  return { set, reset, redraw: cv.redraw, resize: () => cv.resize && cv.resize(), mode: () => S.mode,
    destroy: () => { stop(); cv.destroy && cv.destroy(); } };
};
</script>
