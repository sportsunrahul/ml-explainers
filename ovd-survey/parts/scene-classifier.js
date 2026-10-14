<script>
/* Chapter 1 toy, shared by the long page and the atlas: region features (blue points) and class rows
   (arrows) in a 2D slice of the embedding space. Score = cosine / tau, softmax against a fixed background
   logit. Closed set: learned rows with no "forklift". Text: the rows are text embeddings. Add: a "forklift"
   text vector appears. Points are draggable (pointer, or arrow keys with space to pick the next box).
   o: { aspect(w) -> h/w, glow(word) -> 0..1, onSay(text), bottom (px kept free at the bottom) } */
const SceneClassifier = (el, o = {}) => {
  const TAU = .1, BG = 7.5;
  const ROWS0 = [['person', 100], ['pallet', 200], ['traffic cone', 250], ['car', 22], ['chair', 150]];
  const REG0 = [['person', 104, .82], ['pallet', 194, .8], ['traffic cone', 256, .78], ['forklift', 326, .84]];
  const deg = d => d * Math.PI / 180, glow = o.glow || (() => 0), bottom = o.bottom || 0;
  const reset0 = () => ({ mode: 'closed', uText: 0, uAdd: 0, sel: 3,
    regs: REG0.map(([n, a, r]) => ({ n, x: r * Math.cos(deg(a)), y: r * Math.sin(deg(a)) })) });
  let S = reset0(), stop = () => {};
  function rows() {
    const r = ROWS0.map(([n, a]) => ({ n, a: deg(a), w: 1 }));
    if (S.uAdd > 0) r.push({ n: 'forklift', a: deg(321), w: S.uAdd, added: true });
    return r;
  }
  function scores(reg) {
    const L = Math.hypot(reg.x, reg.y) || 1, rs = rows();
    // an added row fades in: its logit blends up from very low so the softmax changes smoothly
    const z = rs.map(r => { const v = (reg.x * Math.cos(r.a) + reg.y * Math.sin(r.a)) / L / TAU; return r.added ? PE.lerp(-20, v, r.w) : v; });
    const p = PE.softmax([BG, ...z]);
    return { names: ['background', ...rs.map(r => r.n)], p };
  }
  function geo(w, h) {
    const hh = h - bottom, narrow = w < 620, pw = narrow ? w : w * .56, ph = narrow ? hh * .58 : hh;
    const R = Math.min(pw, ph) * .4, cx = pw / 2, cy = ph / 2 + (narrow ? 8 : 4);
    return { hh, narrow, pw, ph, R, cx, cy, X: x => cx + x * R, Y: y => cy - y * R };
  }
  function draw(ctx, w, h) {
    const k = DK.C(); ctx.clearRect(0, 0, w, h); if (w < 60 || h < 60) return;
    const { hh, narrow, pw, ph, R, cx, cy, X, Y } = geo(w, h);
    ctx.strokeStyle = k.rule; ctx.lineWidth = 1; ctx.beginPath(); ctx.arc(cx, cy, R, 0, 7); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(cx - R - 8, cy); ctx.lineTo(cx + R + 8, cy); ctx.moveTo(cx, cy - R - 8); ctx.lineTo(cx, cy + R + 8); ctx.stroke();
    const halo = (s, x, y) => { ctx.lineWidth = 4; ctx.strokeStyle = k.bg; ctx.lineJoin = 'round'; ctx.strokeText(s, x, y); ctx.fillText(s, x, y); };
    const gR = glow('rows'), gT = glow('text');
    rows().forEach(r => {
      const base = DK.mixh(DK.mixh(k.bg, k.fg, .5), k.Y, r.added ? 1 : S.uText);
      const len = r.added ? r.w : 1;
      PE.arrow(ctx, cx, cy, X(Math.cos(r.a) * len), Y(Math.sin(r.a) * len), base, 2 + 2 * (S.uText > .5 || r.added ? gT : gR), 9, r.added ? r.w : 1);
      ctx.save(); ctx.globalAlpha = r.added ? r.w : 1; ctx.font = `500 ${narrow ? 11 : 12.5}px system-ui, sans-serif`;
      ctx.fillStyle = base; ctx.textAlign = Math.cos(r.a) < -.2 ? 'right' : Math.cos(r.a) > .2 ? 'left' : 'center'; ctx.textBaseline = 'middle';
      halo(r.n, X(Math.cos(r.a) * 1.12), Y(Math.sin(r.a) * 1.12)); ctx.restore();
    });
    const gO = glow('regions');
    S.regs.forEach((g, i) => {
      const sc = scores(g), j = sc.p.indexOf(Math.max(...sc.p)), col = j === 0 ? k.Br : (sc.names[j] === g.n ? k.G : k.R);
      PE.dot(ctx, X(g.x), Y(g.y), 6 + 3 * gO, k.B);
      ctx.strokeStyle = col; ctx.lineWidth = i === S.sel ? 2.6 : 1.4; ctx.beginPath(); ctx.arc(X(g.x), Y(g.y), 11 + 3 * gO, 0, 7); ctx.stroke();
      // label on the inner side of the point, so it never sits on the row label outside the circle
      const L = Math.hypot(g.x, g.y) || 1, ux = g.x / L, uy = g.y / L;
      ctx.save(); ctx.font = `600 ${narrow ? 10.5 : 11.5}px system-ui, sans-serif`; ctx.fillStyle = k.B;
      ctx.textAlign = ux > .35 ? 'right' : ux < -.35 ? 'left' : 'center'; halo(g.n + ' box', X(g.x) - ux * 26, Y(g.y) + uy * 22 + 4); ctx.restore();
    });
    const sc = scores(S.regs[S.sel]), bx = narrow ? 16 : pw + 8, by = narrow ? ph + 10 : 38, bw = narrow ? w - 32 : w - pw - 24;
    const rowH = Math.min(26, (narrow ? hh - ph - 20 : hh - 60) / (sc.names.length + 1)), gS = glow('scores');
    ctx.save(); ctx.font = `600 ${narrow ? 11.5 : 12.5}px system-ui, sans-serif`; ctx.fillStyle = k.fg; ctx.textBaseline = 'middle';
    ctx.fillText(`Scores for the ${S.regs[S.sel].n} box`, bx, by - 14);
    const lab = narrow ? 92 : 104, best = sc.p.indexOf(Math.max(...sc.p));
    sc.names.forEach((n, i) => {
      const y = by + i * rowH + rowH / 2, isText = i > 0 && (S.uText > .5 || n === 'forklift');
      ctx.font = `500 ${narrow ? 11 : 12}px system-ui, sans-serif`; ctx.fillStyle = i === 0 ? k.Br : (isText ? k.Y : k.fg); ctx.fillText(n, bx, y);
      const L = Math.max(1, (bw - lab - 44) * sc.p[i]);
      ctx.fillStyle = i === best ? (i === 0 ? k.Br : k.G) : PE.mix(k.bg, k.fg, .3); ctx.fillRect(bx + lab, y - 6 - gS * 2, L, 12 + gS * 4);
      ctx.fillStyle = k.mu; ctx.fillText(sc.p[i].toFixed(2), bx + lab + L + 6, y);
    });
    ctx.restore();
  }
  const cv = PE.canvas(el, o.aspect || (w => (w < 620 ? 1.25 : .52)), draw);
  function describe() {
    const g = S.regs[S.sel], sc = scores(g), j = sc.p.indexOf(Math.max(...sc.p));
    const mode = { add: 'Text rows plus "forklift".', text: 'Rows are text embeddings of the names.', closed: 'Closed set: one learned row per training class.' }[S.mode];
    return `${mode} The ${g.n} box scores highest on "${sc.names[j]}" (${sc.p[j].toFixed(2)})` + (j === 0 ? ', so the detector calls it background.' : '.');
  }
  const say = () => o.onSay && o.onSay(describe());
  function tw(key, to, ms = 700) { const from = S[key]; stop(); stop = PE.animate(ms, u => { S[key] = PE.lerp(from, to, u); cv.redraw(); }, say); }
  // Each mode is a state, so setting it twice equals setting it once.
  function set(mode) {
    S.mode = mode;
    if (mode === 'closed') { S.uAdd = 0; tw('uText', 0); }
    else if (mode === 'text') { S.uAdd = 0; tw('uText', 1); }
    else { S.uText = 1; S.uAdd = 0; tw('uAdd', 1, 900); }
    say();
  }
  function reset() { stop(); S = reset0(); cv.redraw(); say(); }
  let drag = -1;
  const toWorld = e => { const r = el.getBoundingClientRect(), g = geo(r.width, r.height); return [(e.clientX - r.left - g.cx) / g.R, -(e.clientY - r.top - g.cy) / g.R]; };
  const offs = [
    ['pointerdown', e => { const [x, y] = toWorld(e); let best = -1, bd = .18;
      S.regs.forEach((g, i) => { const d = Math.hypot(g.x - x, g.y - y); if (d < bd) { bd = d; best = i; } });
      if (best >= 0) { drag = best; S.sel = best; el.setPointerCapture(e.pointerId); cv.redraw(); } }],
    ['pointermove', e => { if (drag < 0) return; const [x, y] = toWorld(e), L = Math.hypot(x, y), m = Math.min(1.05, L) / (L || 1);
      S.regs[drag].x = x * m; S.regs[drag].y = y * m; cv.redraw(); }],
    ['pointerup', () => { if (drag >= 0) say(); drag = -1; }],
    ['keydown', e => { const d = { ArrowLeft: [-.05, 0], ArrowRight: [.05, 0], ArrowUp: [0, .05], ArrowDown: [0, -.05] }[e.key];
      if (!d && e.key !== ' ') return; e.preventDefault();
      if (e.key === ' ') S.sel = (S.sel + 1) % S.regs.length; else { const g = S.regs[S.sel]; g.x = PE.clamp(g.x + d[0], -1.05, 1.05); g.y = PE.clamp(g.y + d[1], -1.05, 1.05); }
      cv.redraw(); say(); }],
  ].map(([ev, fn]) => { el.addEventListener(ev, fn); return () => el.removeEventListener(ev, fn); });
  say();
  return { set, reset, redraw: cv.redraw, resize: () => cv.resize && cv.resize(), mode: () => S.mode,
    destroy: () => { stop(); offs.forEach(f => f()); cv.destroy && cv.destroy(); } };
};
</script>
