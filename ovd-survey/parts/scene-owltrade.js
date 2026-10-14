<script>
/* SceneOwlTrade(el, o): OWLv2's fine-tuning trade-off, sourced, shared by the long page (viz-owltrade) and the atlas
   (chapter "owl", step 4). Two panels, two metrics, never one axis: LVIS val rare-class AP (standard AP, OWLv2 Table 1)
   and the median zero-shot AP over all 35 ODinW datasets (OWLv2 Table A4). States: st (self-trained only), ft (add the
   fine-tuned points), ens (add the weight average, which exists only for the curated-vocabulary L/14 model, row 21).
   o: { aspect, bottom, onSay }. API: set(state), reset(), redraw(), resize(), destroy(), state(). */
const SceneOwlTrade = (el, o = {}) => {
  // OWLv2 Table 1 rows 11-16 (LVIS val APr, standard AP) and Table A4 rows 11-16 (ODinW median over 35 datasets).
  const D = [
    { m: 'B/16', st: [29.6, 11.6], ft: [36.2, 6.0], dash: [6, 4] },
    { m: 'L/14', st: [34.9, 16.2], ft: [44.6, 6.3], dash: null },
    { m: 'G/14', st: [37.5, 17.5], ft: [47.2, 9.5], dash: [2, 3] },
  ];
  // Table A4: row 21 (N-grams + curated vocabulary, fine-tuned) median 7.0; its weight average with the
  // self-trained checkpoint (fine-tuned weight 0.4) median 10.6. No table reports the average's LVIS AP.
  const ENS = { ft21: 7.0, ens: 10.6 };
  const PAN = [
    { t: 'LVIS val, rare-class AP', t2: 'standard AP, Table 1', y: [25, 50], ticks: [30, 40, 50], j: 0 },
    { t: 'ODinW, median zero-shot AP', t2: 'over all 35 datasets, Table A4', y: [0, 20], ticks: [0, 5, 10, 15, 20], j: 1 },
  ];
  let S = { s: 'st', uF: 0, uE: 0 }, stop = () => {};
  const aspect = o.aspect || (w => (w < 620 ? 560 / w : .44));
  function draw(ctx, w, h) {
    const k = DK.C(); ctx.clearRect(0, 0, w, h); if (w < 160 || h < 160) return;
    const hh = h - (o.bottom || 0), narrow = w < 620, fs = narrow ? 11 : 12;
    const boxes = narrow ? [0, 1].map(i => ({ x: 46, y: 64 + i * (hh - 40) / 2, w: w - 46 - 64, h: (hh - 40) / 2 - 92 }))
      : [0, 1].map(i => ({ x: 52 + i * w / 2, y: 74, w: w / 2 - 52 - 70, h: hh - 74 - 54 }));
    PAN.forEach((P, pi) => {
      const b = boxes[pi], xs = [b.x + b.w * .26, b.x + b.w * .58, b.x + b.w * .9];
      ctx.save(); ctx.font = `600 ${fs + .5}px system-ui, sans-serif`; ctx.fillStyle = k.fg; ctx.fillText(P.t, b.x - 30, b.y - 38);
      ctx.font = `400 ${fs - .5}px system-ui, sans-serif`; ctx.fillStyle = k.mu; ctx.fillText(P.t2, b.x - 30, b.y - 21); ctx.restore();
      const A = PE.axes(ctx, b, [0, 1], P.y, { yt: P.ticks });
      ctx.save(); ctx.font = `500 ${fs - .5}px system-ui, sans-serif`; ctx.textAlign = 'center'; ctx.fillStyle = k.mu;
      ctx.fillText('self-trained', xs[0], b.y + b.h + 16); ctx.globalAlpha = Math.max(.35, S.uF); ctx.fillText('+ fine-tuned', xs[1], b.y + b.h + 16);
      ctx.globalAlpha = S.uE; ctx.fillText('weight average', xs[2], b.y + b.h + 16); ctx.restore();
      D.forEach(d => {
        const col = d.m === 'L/14' ? k.fg : k.mu, y0 = A.Y(d.st[P.j]), y1 = A.Y(d.ft[P.j]);
        if (S.uF > 0) PE.polyline(ctx, [[xs[0], y0], [PE.lerp(xs[0], xs[1], S.uF), PE.lerp(y0, y1, S.uF)]], col, d.m === 'L/14' ? 2.6 : 1.8, d.dash);
        PE.dot(ctx, xs[0], y0, 4.5, col);
        ctx.save(); ctx.font = `600 ${fs - .5}px system-ui, sans-serif`; ctx.fillStyle = col; ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
        const off = { 'B/16': 0, 'L/14': 0, 'G/14': 0 }; if (P.j === 1) { off['G/14'] = -6; off['L/14'] = 6; }
        ctx.fillText(`${d.m} ${d.st[P.j].toFixed(1)}`, xs[0] - 8, y0 + off[d.m]); ctx.restore();
        if (S.uF > .98) { PE.dot(ctx, xs[1], y1, 4.5, col);
          ctx.save(); ctx.font = `600 ${fs - .5}px system-ui, sans-serif`; ctx.fillStyle = col; ctx.textBaseline = 'middle';
          const o2 = P.j === 1 ? { 'B/16': -10, 'L/14': 9, 'G/14': 0 } : { 'B/16': 0, 'L/14': 0, 'G/14': 0 };
          ctx.fillText(`${d.m} ${d.ft[P.j].toFixed(1)}`, xs[1] + 8, y1 + o2[d.m]); ctx.restore(); }
      });
      if (S.uE > 0) {
        ctx.save(); ctx.globalAlpha = S.uE;
        if (P.j === 0) { ctx.font = `400 ${fs - .5}px system-ui, sans-serif`; ctx.fillStyle = k.mu; ctx.textAlign = 'center';
          ctx.fillText('no LVIS number', xs[2], b.y + b.h * .8); ctx.fillText('in any table', xs[2], b.y + b.h * .8 + 14); }
        else {
          const yR = A.Y(16.2); ctx.setLineDash([4, 4]); ctx.strokeStyle = k.fg; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(xs[0], yR); ctx.lineTo(b.x + b.w, yR); ctx.stroke(); ctx.setLineDash([]);
          ctx.font = `400 ${fs - 1}px system-ui, sans-serif`; ctx.fillStyle = k.mu; ctx.textAlign = 'right'; ctx.fillText('self-trained L/14', b.x + b.w, yR - 6);
          const yA = A.Y(ENS.ft21), yB = A.Y(ENS.ens);
          ctx.strokeStyle = k.fg; ctx.lineWidth = 1.5; ctx.beginPath(); ctx.arc(xs[2], yA, 4.5, 0, 7); ctx.stroke();
          PE.arrow(ctx, xs[2], yA - 6, xs[2], yB + 7, k.fg, 1.4, 6);
          ctx.fillStyle = k.fg; ctx.beginPath(); ctx.moveTo(xs[2], yB - 6); ctx.lineTo(xs[2] + 6, yB); ctx.lineTo(xs[2], yB + 6); ctx.lineTo(xs[2] - 6, yB); ctx.closePath(); ctx.fill();
          ctx.font = `600 ${fs - .5}px system-ui, sans-serif`; ctx.textAlign = 'left'; ctx.textBaseline = 'middle';
          ctx.fillText(ENS.ens.toFixed(1), xs[2] + 10, yB); ctx.fillStyle = k.mu; ctx.font = `400 ${fs - 1}px system-ui, sans-serif`;
          ctx.fillText(`${ENS.ft21.toFixed(1)} before`, xs[2] + 10, yA + 2);
          ctx.textAlign = 'center'; ctx.fillText('curated-vocabulary L/14', xs[2], b.y + b.h + 30);
        }
        ctx.restore();
      }
    });
  }
  const cv = PE.canvas(el, aspect, draw);
  const SAY = {
    st: 'Self-trained only (OWL-ST), B/16, L/14, G/14: LVIS val rare-class AP 29.6, 34.9, 37.5; ODinW median over 35 datasets 11.6, 16.2, 17.5.',
    ft: 'After fine-tuning on LVIS base: rare-class AP rises at every size (36.2, 44.6, 47.2); the ODinW median falls at every size (6.0, 6.3, 9.5).',
    ens: 'Weight average of the curated-vocabulary L/14 model: ODinW median 10.6, up from 7.0 after fine-tuning but below the self-trained L/14\'s 16.2. Its ODinW-13 mean, 56.3, is the paper\'s best. No table gives its LVIS AP.',
  };
  const say = () => o.onSay && o.onSay(SAY[S.s]);
  function set(s) {
    stop(); S.s = s; const f0 = S.uF, e0 = S.uE, f1 = s === 'st' ? 0 : 1, e1 = s === 'ens' ? 1 : 0;
    stop = PE.animate(800, u => { S.uF = PE.lerp(f0, f1, u); S.uE = PE.lerp(e0, e1, u); cv.redraw(); }); say();
  }
  function reset() { stop(); S = { s: 'st', uF: 0, uE: 0 }; cv.redraw(); say(); }
  say();
  return { set, reset, redraw: cv.redraw, resize: () => (cv.resize ? cv.resize() : cv.redraw()), state: () => S.s,
    destroy: () => { stop(); cv.destroy && cv.destroy(); } };
};
</script>
