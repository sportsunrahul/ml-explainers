<script>
/* SceneOdinw(el, o): zero-shot AP on each of the 35 ODinW datasets (WHEN_DATA.odinw; OWLv2 Table A4 and
   Grounding DINO Table 14). Steps: '13' (only the 13-dataset subset most papers report, with its mean),
   '35' (all 35, with the median), 'ft' (the same model after fine-tuning on LVIS base; lines join each
   dataset), 'gdino' (Grounding DINO L as a second family). Filled dots are the ODinW-13 datasets.
   Pointer or arrow keys pick a dataset. o: { aspect, bottom, glow, onSay }. Returns { set, reset, redraw, destroy }. */
const SceneOdinw = (el, o = {}) => {
  const D = WHEN_DATA.odinw, glow = o.glow || (() => 0), bottom = o.bottom || 0;
  const in13 = new Set(D.owl13), N = D.names.length;
  const ORDER = ['13', '35', 'ft', 'gdino'];
  let S = { step: '13', u: 1, sel: -1 }, stop = () => {};
  const at = s => ORDER.indexOf(S.step) >= ORDER.indexOf(s);
  const jit = (i, s) => { const x = Math.sin((i + 1) * 12.9898 + s * 78.233) * 43758.5453; return x - Math.floor(x) - .5; };
  function layout(w, h) {
    const hh = h - bottom, narrow = w < 620, left = narrow ? 8 : 132, b = { x: left + (narrow ? 4 : 0), y: 10, w: w - left - (narrow ? 16 : 24), h: hh - 46 };
    const rows = [{ key: 'owl', label: ['OWL-ST L/14', 'self-trained'] }, { key: 'owlft', label: ['OWL-ST L/14', 'fine-tuned on LVIS'] }, { key: 'gdinoL', label: ['Grounding DINO L', 'zero-shot'] }];
    const band = b.h / 3, rowY = i => b.y + band * (i + .5) + (narrow ? 6 : 0);
    return { narrow, b, rows, rowY, band, X: v => b.x + b.w * v / 100 };
  }
  function pts(L) {
    const out = [];
    L.rows.forEach((r, ri) => {
      const s = D[r.key] || D.series[r.key], names = s.names || D.names;
      s.v.forEach((v, i) => {
        const name = names[i], m13 = r.key === 'gdinoL' ? null : in13.has(name);
        out.push({ ri, key: r.key, i, name, v, m13, x: L.X(v), y: L.rowY(ri) + jit(i, ri) * L.band * (L.narrow ? .42 : .5) });
      });
    });
    return out;
  }
  function visible(p) {
    if (p.key === 'owl') return at('35') || p.m13;
    if (p.key === 'owlft') return at('ft');
    return at('gdino');
  }
  function draw(ctx, w, h) {
    const k = DK.C(); ctx.clearRect(0, 0, w, h); if (w < 80) return;
    const L = layout(w, h), { b, rows, rowY, X, narrow } = L;
    PE.axes(ctx, b, [0, 100], [0, 1], { xt: [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100], xl: 'zero-shot AP on one ODinW dataset' });
    const P = pts(L), g13 = glow('mean13'), gmed = glow('median');
    rows.forEach((r, ri) => {
      const on = ri === 0 || (ri === 1 && at('ft')) || (ri === 2 && at('gdino')); if (!on) return;
      ctx.save(); ctx.font = `600 ${narrow ? 10.5 : 12}px system-ui, sans-serif`; ctx.fillStyle = k.fg;
      if (narrow) { ctx.textAlign = 'left'; ctx.fillText(r.label.join(', '), b.x, rowY(ri) - L.band / 2 + 4); }
      else { ctx.textAlign = 'right'; ctx.fillText(r.label[0], b.x - 12, rowY(ri) - 3); ctx.font = `${11.5}px system-ui, sans-serif`; ctx.fillStyle = k.mu; ctx.fillText(r.label[1], b.x - 12, rowY(ri) + 12); }
      ctx.restore();
    });
    // fine-tuning lines: the same dataset before and after
    if (at('ft')) {
      const a = P.filter(p => p.key === 'owl'), bb = P.filter(p => p.key === 'owlft');
      a.forEach((p, i) => { const q = bb[i], drop = q.v < p.v - .5;
        ctx.save(); ctx.globalAlpha = (S.step === 'ft' ? S.u : 1) * (drop ? .55 : .25); ctx.strokeStyle = drop ? k.R : k.mu; ctx.lineWidth = 1;
        ctx.beginPath(); ctx.moveTo(p.x, p.y); ctx.lineTo(q.x, q.y); ctx.stroke(); ctx.restore(); });
    }
    P.forEach(p => {
      if (!visible(p)) return;
      let a = 1;
      if (p.key === 'owl' && !p.m13 && S.step === '35') a = S.u;
      if (p.key === 'owlft' && S.step === 'ft') a = S.u;
      if (p.key === 'gdinoL' && S.step === 'gdino') a = S.u;
      const col = p.key === 'gdinoL' ? k.Y : k.B, sel = S.sel === p.i && p.key !== 'gdinoL';
      ctx.save(); ctx.globalAlpha = a; ctx.beginPath(); ctx.arc(p.x, p.y, sel ? 6 : 4.2, 0, 7);
      if (p.m13 === false) { ctx.strokeStyle = col; ctx.lineWidth = 1.5; ctx.fillStyle = k.bg; ctx.fill(); ctx.stroke(); } else { ctx.fillStyle = col; ctx.fill(); }
      if (sel) { ctx.strokeStyle = k.fg; ctx.lineWidth = 1.5; ctx.stroke(); }
      ctx.restore();
    });
    // summary markers
    const mark = (ri, v, label, col, gl, below) => {
      const x = X(v), half = L.band * (narrow ? .3 : .36), y0 = rowY(ri) - half, y1 = rowY(ri) + half;
      ctx.save(); ctx.strokeStyle = col; ctx.lineWidth = 2 + 2 * gl; ctx.beginPath(); ctx.moveTo(x, y0); ctx.lineTo(x, y1); ctx.stroke();
      ctx.font = `600 ${narrow ? 10.5 : 11.5}px system-ui, sans-serif`; ctx.textAlign = 'center';
      const tx = PE.clamp(x, b.x + 52, b.x + b.w - 52), ty = below ? y1 + 12 : y0 - 5;
      ctx.lineWidth = 4; ctx.strokeStyle = k.bg; ctx.lineJoin = 'round'; ctx.strokeText(label, tx, ty); ctx.fillStyle = col; ctx.fillText(label, tx, ty); ctx.restore();
    };
    const s0 = D.series.owl, s1 = D.series.owlft, f1 = v => (+v).toFixed(1);
    if (S.step === '13') mark(0, s0.mean13, `mean of 13: ${f1(s0.mean13)}`, k.G, g13, false);
    if (at('35')) { mark(0, s0.median35, `median of 35: ${f1(s0.median35)}`, k.R, gmed, true); if (!narrow || S.step === '35') mark(0, s0.mean13, `mean of 13: ${f1(s0.mean13)}`, k.G, g13, false); }
    if (at('ft')) mark(1, s1.median35, `median of 35: ${f1(s1.median35)}`, k.R, gmed, true);
    if (at('gdino')) mark(2, +D.gdinoL.median35, `median of 35: ${f1(D.gdinoL.median35)}`, k.R, gmed, true);
    // selected dataset label
    if (S.sel >= 0) {
      const p = P.find(q => q.key === 'owl' && q.i === S.sel);
      ctx.save(); ctx.font = `600 ${narrow ? 10.5 : 12}px system-ui, sans-serif`; ctx.fillStyle = k.fg; ctx.textAlign = p.x > b.x + b.w * .6 ? 'right' : 'left';
      ctx.fillText(p.name, p.x + (ctx.textAlign === 'right' ? -10 : 10), p.y - 10); ctx.restore();
    }
  }
  const cv = PE.canvas(el, o.aspect || (w => (w < 620 ? 1.5 : .5)), draw);
  function describe() {
    const s0 = D.series.owl, s1 = D.series.owlft;
    let t = { '13': `OWL-ST L/14 on the 13 ODinW datasets most papers report: mean AP ${s0.mean13}.`,
      '35': `The same model on all 35 ODinW datasets: median AP ${s0.median35}, and many datasets near zero.`,
      ft: `After fine-tuning on LVIS base classes the median over 35 falls from ${s0.median35} to ${s1.median35}; red lines mark datasets that got worse.`,
      gdino: `Grounding DINO L, zero-shot on the same 35 datasets: median AP ${D.gdinoL.median35}.` }[S.step];
    if (S.sel >= 0) { const v0 = s0.v[S.sel], v1 = s1.v[S.sel]; t += ` Selected: ${D.names[S.sel]}, ${v0} AP` + (at('ft') ? ` before, ${v1} after fine-tuning.` : '.'); }
    return t;
  }
  const say = () => o.onSay && o.onSay(describe());
  function set(step) { if (!ORDER.includes(step)) return; stop(); const same = step === S.step; S.step = step; S.u = same ? 1 : 0;
    stop = PE.animate(700, u => { S.u = same ? 1 : u; cv.redraw(); }); say(); }
  function reset() { stop(); S = { step: '13', u: 1, sel: -1 }; cv.redraw(); say(); }
  const offs = [
    ['pointermove', e => { const r = el.getBoundingClientRect(), L = layout(r.width, r.height), mx = e.clientX - r.left, my = e.clientY - r.top;
      let best = -1, bd = 14; pts(L).forEach(p => { if (p.key !== 'owl' || !visible(p)) return; const d = Math.hypot(p.x - mx, p.y - my); if (d < bd) { bd = d; best = p.i; } });
      if (best !== S.sel) { S.sel = best; cv.redraw(); if (best >= 0) say(); } }],
    ['keydown', e => { if (e.key !== 'ArrowRight' && e.key !== 'ArrowLeft') return; e.preventDefault();
      const vis = D.series.owl.v.map((v, i) => ({ v, i })).filter(q => at('35') || in13.has(D.names[q.i])).sort((a, c) => a.v - c.v).map(q => q.i);
      const j = vis.indexOf(S.sel), n = e.key === 'ArrowRight' ? (j + 1) % vis.length : (j - 1 + vis.length) % vis.length; S.sel = vis[j < 0 ? 0 : n]; cv.redraw(); say(); }],
  ].map(([ev, fn]) => { el.addEventListener(ev, fn); return () => el.removeEventListener(ev, fn); });
  say();
  return { set, reset, redraw: cv.redraw, step: () => S.step, destroy: () => { stop(); offs.forEach(f => f()); cv.destroy && cv.destroy(); } };
};
</script>
