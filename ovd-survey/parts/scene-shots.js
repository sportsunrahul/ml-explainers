<script>
/* SceneShots(el, o): GLIP Table 11, ODinW-13 average AP against labelled examples per class (0, 1, 3, 5, 10,
   all), full-model fine-tuning, mean of 3 runs. DyHead-T (O365) is the same detector family without text,
   so it has no zero-shot point. Steps: 'dyhead', 'glipt', 'glipl' add one line each; 'match' draws the
   level line from zero-shot GLIP-T across to the DyHead curve. o: { aspect, bottom, glow, onSay }. */
const SceneShots = (el, o = {}) => {
  const D = WHEN_DATA.glip, glow = o.glow || (() => 0), bottom = o.bottom || 0;
  const XS = ['0', '1', '3', '5', '10', 'all'], ORDER = ['dyhead', 'glipt', 'glipl', 'match'];
  const SHORT = { 'dyhead-o365': 'DyHead-T, no text', t: 'GLIP-T', l: 'GLIP-L' };
  const SER = [{ key: 'dyhead-o365', step: 'dyhead', col: k => k.mu, dash: null, lw: 2.2 },
    { key: 't', step: 'glipt', col: k => k.Y, dash: [6, 4], lw: 2.4 }, { key: 'l', step: 'glipl', col: k => k.Y, dash: null, lw: 3 }];
  let S = { step: 'dyhead', u: 1 }, stop = () => {};
  const at = s => ORDER.indexOf(S.step) >= ORDER.indexOf(s);
  function draw(ctx, w, h) {
    const k = DK.C(); ctx.clearRect(0, 0, w, h); if (w < 80) return;
    const hh = h - bottom, narrow = w < 620, b = { x: narrow ? 40 : 56, y: 20, w: w - (narrow ? 56 : 170), h: hh - 62 };
    const { X, Y } = PE.axes(ctx, b, [0, 5], [25, 70], { yt: [30, 40, 50, 60, 70], yl: 'ODinW-13 average AP', ylo: narrow ? 30 : 38 });
    ctx.save(); ctx.font = `${narrow ? 10.5 : 11.5}px system-ui, sans-serif`; ctx.fillStyle = k.mu; ctx.textAlign = 'center';
    XS.forEach((s, i) => ctx.fillText(s, X(i), b.y + b.h + 14)); ctx.fillText('labelled examples per class (fine-tuned), 0 = prompt only', b.x + b.w / 2, b.y + b.h + 30); ctx.restore();
    SER.forEach(s => {
      if (!at(s.step)) return;
      const a = S.step === s.step ? S.u : 1, d = D[s.key], col = s.col(k), gl = glow(s.step);
      const pts = d.v.map((v, i) => v == null ? null : [X(i), Y(v)]).filter(Boolean);
      const n = Math.max(1, Math.ceil(pts.length * a));
      PE.polyline(ctx, pts.slice(0, n), col, s.lw + 2 * gl, s.dash);
      pts.slice(0, n).forEach(([x, y]) => PE.dot(ctx, x, y, 3.6, col));
      if (a >= 1) { const [x, y] = pts[pts.length - 1]; ctx.save(); ctx.font = `600 ${narrow ? 10.5 : 12}px system-ui, sans-serif`; ctx.fillStyle = col;
        if (narrow) { ctx.textAlign = 'right'; ctx.fillText(SHORT[s.key], x - 8, y + (s.key === 't' ? 16 : -8)); } else ctx.fillText(SHORT[s.key], x + 10, y + 4); ctx.restore(); }
    });
    if (at('match')) {
      const z = D.t.v[0], i5 = 3, a = S.step === 'match' ? S.u : 1, gl = glow('match');
      ctx.save(); ctx.globalAlpha = a; ctx.strokeStyle = k.G; ctx.lineWidth = 1.8 + 2 * gl; ctx.setLineDash([3, 3]);
      ctx.beginPath(); ctx.moveTo(X(0), Y(z)); ctx.lineTo(X(i5), Y(z)); ctx.stroke(); ctx.setLineDash([]);
      ctx.beginPath(); ctx.arc(X(i5), Y(D['dyhead-o365'].v[i5]), 8, 0, 7); ctx.stroke();
      ctx.font = `600 ${narrow ? 10.5 : 11.5}px system-ui, sans-serif`; ctx.fillStyle = k.G;
      ctx.fillText(`prompt only: ${z}`, X(0) + 6, Y(z) - 8); ctx.fillText(`5 shots, no text: ${D['dyhead-o365'].v[i5]}`, X(i5) - (narrow ? 70 : 40), Y(z) + 22); ctx.restore();
    }
  }
  const cv = PE.canvas(el, o.aspect || (w => (w < 620 ? .95 : .44)), draw);
  function describe() {
    const d = D['dyhead-o365'], t = D.t, l = D.l;
    return { dyhead: `DyHead-T, a closed-set detector pretrained on Objects365: ${d.v[1]} AP with 1 example per class, ${d.v[5]} with all data.`,
      glipt: `GLIP-T reads a prompt: ${t.v[0]} AP with no examples at all, ${t.v[5]} with all data.`,
      glipl: `GLIP-L: ${l.v[0]} AP with no examples, ${l.v[1]} with one.`,
      match: `GLIP-T with only a prompt (${t.v[0]}) is level with DyHead-T trained on 5 examples per class (${d.v[3]}).` }[S.step];
  }
  const say = () => o.onSay && o.onSay(describe());
  function set(step) { if (!ORDER.includes(step)) return; stop(); const same = step === S.step; S.step = step; S.u = same ? 1 : 0;
    stop = PE.animate(800, u => { S.u = same ? 1 : u; cv.redraw(); }); say(); }
  function reset() { stop(); S = { step: 'dyhead', u: 1 }; cv.redraw(); say(); }
  say();
  return { set, reset, redraw: cv.redraw, destroy: () => { stop(); cv.destroy && cv.destroy(); } };
};
</script>
